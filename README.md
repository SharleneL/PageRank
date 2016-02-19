# A PageRank Algorithm 
**Supporting:**
- Global PageRank
- Query-based Topic Sensitive PageRank
- Personalized Topic Sensitive PageRank
- Weighted combination score with relevance score.

**Run:**
In terminal, run the following command:

`$ python pageRank.y [GPR/QTSPR/PTSPR] [NS/WS/CM] -prw [prWeight] -srw [srWeight] -a [alpha] -b [beta] -g [gamma] [output_filepath]`

**Example**:

`$ python pageRank.py GPR NS -prw 0.5 -srw 0.5 -a 0.8 -b 0.1 -g 0.1 output.txt`

**Parameters:**
- [GPR/QTSPR/PTSPR]: choose one of the three PageRank methods.
- [NS/WS/CM]: choose one of the three combination schemes.
- [prWeight]: PageRank weight for WS scheme; a number from 0.0 to 1.0. In NS and CM, any value entered here will be ignored.
- [srWeight]: Search relevance weight for WS scheme; a number from 0.0 to 1.0. In NS and CM, any value entered here will be ignored.
- [alpha]: The value of alpha. 
- [beta]: The value of beta. If beta value will not be used in current algorithm, any value entered here will be ignored.
- [gamma]: The value of gamma. If beta value will not be used in current algorithm, any value entered here will be ignored.
- [output_filepath]: the path of output file.
