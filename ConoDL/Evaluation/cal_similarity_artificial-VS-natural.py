'''Function to calculate the similarity between two sets of input sequences.'''

import os
from math import log
from Bio import pairwise2
from Bio.Align import substitution_matrices


file1 = 'input_sequence_1.txt'
file2 = 'input_sequence_2.txt'


result1 = []
with open(file1) as f1:
    result1 = f1.read().split('\n')

result2 = []
with open(file2) as f2:
    result2 = f2.read().split('\n')

with open('output_similarity_2.txt', mode='w') as Note:
    Note.write("sequence,similarity\n")

def cluster(seq_list, seq):
    max_similarity = -1000.0
    for each in seq_list:
        _, a = similarity(each, seq)
        if a > max_similarity:
            max_similarity = a
    print(max_similarity)
    with open('output_similarity_2.txt', mode='a') as Note:
        Note.write(f"{seq},{max_similarity}\n")


def similarity(seqs_lst1, seqs_lst2):
    gap_open = -10
    gap_extend = -1
    matrix = substitution_matrices.load("BLOSUM62")
    results_dict = {}
    sim_lst = []
    rec = seqs_lst1
    rec1 = seqs_lst2
    if len(rec) > 1:
        if len(rec1) > 1:
            if str(rec) != str(rec1):
                alns = pairwise2.align.globalds(str(rec), str(rec1), matrix, gap_open, gap_extend)
                top_aln = alns[0]
                al1, al2, score, begin, end = top_aln
                sim_lst.append(score / log(len(rec)))
            else:
                print('seqs_lst1 is the same as seqs_lst2')
    results_dict['sim'] = sim_lst
    av_sim = sum(sim_lst) / len(sim_lst) if len(sim_lst) > 0 else 0.0
    return results_dict, av_sim



for i in range(0, len(result1)):
    try:
        cluster(result2, result1[i])
        print(i)
    except:
        break

