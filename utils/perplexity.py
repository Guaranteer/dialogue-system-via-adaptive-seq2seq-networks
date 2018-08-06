import numpy as np


def calculate_perplexity(distribution, word_idxs, valid_num):

    dists = 1
    for i in range(valid_num):
        dists *= distribution[i,word_idxs[i]]
    log_perplexity = -np.log(dists)/valid_num
    perplexity = np.exp(log_perplexity)
    return perplexity



