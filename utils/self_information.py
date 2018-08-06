import numpy as np

def length(filename):
    with open(filename,'r') as fr:
        sents = fr.readlines()

    count = 0
    for sent in sents:
        num = len(sent.strip().split())
        count += num

    avg_num = count/len(sents)
    print(avg_num)

def word_h(distribution, word_idxs, valid_num):
    wh = 0
    for i in range(valid_num):
        p = distribution[i,word_idxs[i]]
        wh += -np.log(p)*p

    return wh

if __name__ == '__main__':
    hanl = '../results/base_plus/test300.txt'
    hred = '../results/HRED/test200.txt'
    vhred = '../results/VHRED/test200.txt'
    s2s = '../results/seq2seq/test200.txt'
    s2s_att = '../results/seq2seq_att/test200.txt'
    ground = '../results/base_plus/ground200.txt'

    length(hanl)
    length(hred)
    length(vhred)
    length(s2s)
    length(s2s_att)
    length(ground)