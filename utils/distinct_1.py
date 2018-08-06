import numpy as np

def distinct_1(base1, base2, base3):
    base = [open(base1, 'r').readlines(), open(base2, 'r').readlines(), open(base3, 'r').readlines()]

    tot = [0, 0, 0]
    count = [0, 0, 0]
    vocabulary = [set(), set(), set()]
    for i in range(len(base[0])):
        for j in range(len(base)):
            base[j][i] = base[j][i].strip().split(" ")
            for token in base[j][i]:
                tot[j] += 1
                if token not in vocabulary[j]:
                    count[j] += 1
                    vocabulary[j].add(token)

    for j in range(len(base)):
        print(count[j] / tot[j])

if __name__ == '__main__':
    hanl_bi = '../results/HANL_bi_part/test50.txt'
    hanl = '../results/base_plus/test200.txt'
    hred = '../results/HRED/test200.txt'
    vhred = '../results/VHRED/test200.txt'
    s2s = '../results/seq2seq/test200.txt'
    s2s_att = '../results/seq2seq_att/test200.txt'
    ground = '../results/base_plus/ground200.txt'

    distinct_1(hanl_bi,hred,vhred)
    distinct_1(s2s, s2s_att, ground)