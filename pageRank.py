from scipy import sparse

__author__ = 'luoshalin'

import numpy as np
from numpy import zeros
from scipy.sparse import *
from scipy import *

def main():
    # INITIALIZATION
    alpha = 0.2
    beta = 0.4
    gamma = 0.4
    ####### REAL ########
    size = 81433  # total number of docs # FINISHED
    print("size")
    print(size)
    M = get_M("../data/hw3-resources/transition.txt", size)  # FINISHED
    print("M got")
    p0 = np.zeros(size).transpose()  # create a [81433*1] vector, with 1/size as value # FINISHED
    p0.fill(1/float(size))
    print ("p0")
    print len(p0)
    pt_dict = get_pt("../data/hw3-resources/doc_topics.txt", size)
    print "pt_dict"
    print len(pt_dict)
    pu_dict = get_pu("../data/hw3-resources/user-topic-distro.txt")
    print "pu_dict"
    print len(pu_dict)
    ####### REAL ########

    ####### TEST ########
    # size = 4  # total number of docs # FINISHED
    # M = get_M("../data/hw3-resources/small_transition.txt", size)  # FINISHED
    # p0 = np.zeros(size).transpose()  # create a [81433*1] vector, with 1/size as value # FINISHED
    # p0.fill(1/float(size))
    # pt_dict = get_pt("../data/hw3-resources/small_doc_topics.txt", size)
    # pu_dict = get_pu("../data/hw3-resources/user-topic-distro.txt")
    ####### TEST ########

    # PAGERANK CALCULATION
    gpr = get_gpr(alpha, M, p0, size)
    print gpr
    np.savetxt('res.txt', gpr)
    # f1 = open(, 'w+')
    # for (x, ), value in np.ndenumerate(gpr):
    #     f1.write(str(value)+'\n')
    # f1.close()
    # qtspr = get_qtspr(alpha, beta, gamma,  M, pt_dict, p0, size, tid)
    # ptspr = get_ptspr(alpha, M, pu_dict, size, qid, uid)

    # COMBINING SCORE CALCULATION
    # NS
    # WS
    # CM

    # OUTPUT TO FILE


def get_M(file_path, size):
    # M = zeros((size, size))  # size * size
    row = []
    col = []
    data = []
    print file_path

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
    # print dic

    # construct pt_dict <topicid, vector<pt1, pt2, ..., ptn>>
    for topic, docid_list in dic.iteritems():
        pt = np.zeros(size).transpose()  # create a [81433*1] vector for current topic
        for docid in docid_list:
            num_sum = len(docid_list)
            if num_sum != 0:
                pt[docid-1] = 1.0/float(num_sum)
        pt_dict[topic] = pt
    return pt_dict


def get_pu(file_path):
    pu_dict = dict()  # pu_dict<qid, uv_dict<userid, vector(pu1, pu2, ..., pun))>>
    with open(file_path) as f:
        line = f.readline().strip()
        while line != '':
            userid = line.split()[0]
            qid = line.split()[1]
            pu = np.zeros(12).transpose()  # create a [81433*1] vector for current topic
            for i in range(12):
                pu[int(line.split()[2+i].split(":")[0]) - 1] = float(line.split()[2+i].split(":")[1])
            if qid not in pu_dict:
                uv_dict = dict()
                uv_dict[userid] = pu
                pu_dict[qid] = uv_dict
            else:
                pu_dict[qid][userid] = pu
            line = f.readline().strip()
    # print pu_dict
    return pu_dict


def get_gpr(alpha, M, p0, size):
    gpr = np.ones(size).transpose()  # create a [81433*1] vector, with 1 as value
    gpr.fill(1.0/float(size))
    # gpr = (1-alpha) * sparse.csr_matrix.transpose(M) * gpr + alpha * p0
    for i in range(10):
        gpr = (1-alpha) * sparse.csr_matrix.transpose(M) * gpr + alpha * p0
        print "ITERATION!"
    return gpr


def get_qtspr(alpha, beta, gamma,  M, pt_dict, p0, size, tid):
    gtspr = np.ones(1.0/float(size)).transpose()
    gtspr = alpha * np.dot(M.transpose(), gtspr) + beta * pt_dict[tid] + gamma * p0
    # while iteration & stop criteria - !!!!!


def get_ptspr(alpha, M, pu_dict, size, qid, uid):  # pu_dict<qid, uv_dict<userid, vector(pu1, pu2, ..., pun))>>
    ptspr = np.ones(1.0/float(size)).transpose()
    ptspr = (1-alpha) * np.dot(M.transpose(), ptspr) + alpha * pu_dict[qid][uid]
    # while iteration & stop criteria - !!!!!

if __name__ == '__main__':
    main()