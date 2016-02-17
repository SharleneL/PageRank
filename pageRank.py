from scipy import sparse
import os
import sys

__author__ = 'luoshalin'

# COMMAND LINE INPUT FORMAT:
# python pageRank.y [GPR/QTSPR/PTSPR] [NS/WS/CM] -prw [prWeight] -srw [srWeight] -a [alpha] -b [beta] -g [gamma] [output_filepath]

import numpy as np
from numpy import zeros
from scipy.sparse import *
from scipy import *

def main(argv):
    # GET TERMINAL INPUT
    pr_method = sys.argv[1]
    combination_method = sys.argv[2]
    if sys.argv[3] == '-prw':
        pr_weight = float(sys.argv[4])
    if sys.argv[5] == '-srw':
        sr_weight = float(sys.argv[6])
    if sys.argv[7] == '-a':
        alpha = float(sys.argv[8])
    if sys.argv[9] == '-b':
        beta = float(sys.argv[10])
    if sys.argv[11] == '-g':
        gamma = float(sys.argv[12])
    output_filepath = sys.argv[-1]

    # PARAMETER INITIALIZATION
    size = 81433  # total number of docs # FINISHED
    topic_size = 12
    query_size = 6
    # DATA IMPORT
    M = get_M("../../data/hw3-resources/transition.txt", size)  # FINISHED
    p0 = np.zeros(size).transpose()  # create a [81433*1] vector, with 1/size as value # FINISHED
    p0.fill(1/float(size))
    pt_dict = get_pt("../../data/hw3-resources/doc_topics.txt", size)
    qt_coeff_dict = get_coeff("../../data/hw3-resources/query-topic-distro.txt", size, topic_size, query_size)
    ut_coeff_dict = get_coeff("../../data/hw3-resources/user-topic-distro.txt", size, topic_size, query_size)

    # PAGERANK CALCULATION
    # GPR
    if pr_method == 'GPR':
        pr_scores = get_gpr_score(alpha, M, p0, size)
    # print gpr_score
    # np.savetxt('gpr.txt', gpr_score)

    # TSPR
    tspr_matrix = get_tspr_matrix(alpha, beta, gamma,  M, pt_dict, p0, size)
    # np.savetxt('tspr_matrix.txt', tspr_matrix)
    # TSPR - QTSPR  # a tspr_score_matrix_dict, saves <uid, tspr_score_matrix>; tspr_score_matrix=[doc, qid]
    if pr_method == 'QTSPR':
        pr_scores = get_tspr_score_matrix_dict(tspr_matrix, qt_coeff_dict)
    # TSPR - PTSPR  # a tspr_score_matrix_dict, saves <uid, tspr_score_matrix>; tspr_score_matrix=[doc, qid]
    if pr_method == 'PTSPR':
        pr_scores = get_tspr_score_matrix_dict(tspr_matrix, ut_coeff_dict)

    # COMBINING SCORE CALCULATION & OUTPUT
    # read in file -> get file name<uid, qid> -> read in line
    # -> get<uid, qid, docid> -> search in matrix_dict -> combine scores -> write to file
    dir_path = '../../data/hw3-resources/indri-lists'
    print_combined_score(dir_path, combination_method, pr_method, pr_weight, sr_weight, pr_scores, output_filepath)


def get_M(file_path, size):
    # M = zeros((size, size))  # size * size
    row = []
    col = []
    data = []

    with open(file_path) as f:
        line = f.readline().strip()
        while line != '':
            row.append(int(line.split()[0]) - 1)  # ? -1
            col.append(int(line.split()[1]) - 1)
            data.append(float(line.split()[2]))
            line = f.readline().strip()
        # add diagnol
        for i in range(size):
            row.append(i)
            col.append(i)
            data.append(1)
    M = csr_matrix((data, (row, col)), shape=(size, size))

    # normalize
    M_row_sums = np.array(M.sum(1))[:, 0]  # get the row sum of M and save into an array (1 means axis)
    row_ids, _ = M.nonzero()  # the non-zero row ids
    M.data /= M_row_sums[row_ids]  # normalize the non-zero rows
    return M


def get_pt(file_path, size):
    pt_dict = dict()  # save <topicid, vector(pt1, pt2, ..., ptn)>

    # save <topicid, list(docid)> into a dict
    dic = dict()
    with open(file_path) as f:
        line = f.readline().strip()
        while line != '':
            docid = int(line.split()[0])
            topic = int(line.split()[1])
            if topic not in dic:
                dic[topic] = [docid]
            else:
                dic[topic].append(docid)
            line = f.readline().strip()

    # construct pt_dict <topicid, vector<pt1, pt2, ..., ptn>>
    for topic, docid_list in dic.iteritems():
        pt = np.zeros(size).transpose()  # create a [81433*1] vector for current topic
        for docid in docid_list:
            num_sum = len(docid_list)
            if num_sum != 0:
                pt[docid-1] = 1.0/float(num_sum)
        pt_dict[topic] = pt
    return pt_dict


def get_coeff(file_path, size, topic_size, query_size):
    coeff_dict = dict()  # <userid, coeff_matrix> & coeff_matrix=[topics * queries]

    with open(file_path) as f:
        line = f.readline().strip()
        while line != '':
            uid = int(line.split()[0])
            qid = int(line.split()[1])
            coeff_v = np.zeros(topic_size).transpose()  # create a [12*1] vector for current query
            for i in range(topic_size):
                coeff_v[int(line.split()[2+i].split(":")[0]) - 1] = float(line.split()[2+i].split(":")[1])
            # save into the data structure
            if uid not in coeff_dict:
                coeff_dict[uid] = np.zeros(shape=(topic_size, query_size))
            coeff_dict[uid][:, qid-1] = coeff_v
            line = f.readline().strip()
    return coeff_dict


def get_gpr_score(alpha, M, p0, size):
    gpr = np.ones(size).transpose()  # create a [81433*1] vector, with 1 as value
    gpr.fill(1.0/float(size))
    # gpr = (1-alpha) * sparse.csr_matrix.transpose(M) * gpr + alpha * p0
    for i in range(10):
        gpr = (1-alpha) * sparse.csr_matrix.transpose(M) * gpr + alpha * p0
        print "GPR ITERATION " + str(i+1)
    return gpr


def get_tspr_matrix(alpha, beta, gamma,  M, pt_dict, p0, size):
    tspr_matrix = np.zeros(shape=(size, 12))
    for tid in range(1, 13):
        tspr = np.ones(size).transpose()
        tspr.fill(1.0/float(size))
        for i in range(10):
            tspr = alpha * sparse.csr_matrix.transpose(M) * tspr + beta * pt_dict[tid] + gamma * p0
            print "TSPR ITERATION " + str(i+1) + " of TID #" + str(tid)
        tspr_matrix[:, tid-1] = tspr
    return tspr_matrix


# tspr_score_matrix_dict saves <uid, tspr_score_matrix>; tspr_score_matrix=[doc, qid]
def get_tspr_score_matrix_dict(tspr_matrix, coeff_dict):
    tspr_score_matrix_dict = dict()
    for uid, coeff_matrix in coeff_dict.iteritems():
        tspr_score_matrix_dict[uid] = np.dot(tspr_matrix, coeff_matrix)
    return tspr_score_matrix_dict


def print_combined_score(dir_path, combination_method, pr_method, pr_weight, sr_weight, pr_scores, output_filepath):
    res_f = open(output_filepath, 'w')
    for filename in os.listdir(dir_path):
        file_path = dir_path + "/" + filename
        filename = filename.split(".")[0]
        uid = int(filename.split("-")[0])
        qid = int(filename.split("-")[1])
        with open(file_path) as f:
            docid_score_list = []
            line = f.readline().strip()
            while line != '':
                docid = int(line.split()[2])
                sr_score = float(line.split()[4])  # search relevance score
                line = f.readline().strip()
                if pr_method == 'GPR':
                    pr_score = pr_scores[docid-1]
                else:
                    pr_score = pr_scores[uid][:, qid-1][docid-1]
                # combine score here if needed
                if combination_method == 'NS':
                    combined_score = pr_score
                if combination_method == 'WS':
                    combined_score = pr_score * pr_weight + sr_score * sr_weight
                if combination_method == 'CM':
                    # TO-DO
                    combined_score = pr_score
                docid_score_list.append((docid, round(combined_score, 7)))  # round to 7 decimal digits

            docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)
            # write to file
            rank = 1
            for docid_score in docid_score_list:
                res_line = str(uid) + "-" + str(qid) + " Q0 " + str(docid_score[0]) + " " + str(rank) + " " + str(docid_score[1]) + " shalinl" + "\n"
                res_f.write(res_line)
                rank += 1
    res_f.close()

if __name__ == '__main__':
    main(sys.argv[1:])