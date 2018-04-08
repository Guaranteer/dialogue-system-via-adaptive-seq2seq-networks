"""
Word embedding based evaluation metrics for dialogue.

This method implements three evaluation metrics based on Word2Vec word embeddings, which compare a target utterance with a model utterance:
1) Computing cosine-similarity between the mean word embeddings of the target utterance and of the model utterance
2) Computing greedy meatching between word embeddings of target utterance and model utterance (Rus et al., 2012)
3) Computing word embedding extrema scores (Forgues et al., 2014)

We believe that these metrics are suitable for evaluating dialogue systems.

Example run:

    python embedding_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

The script assumes one example per line (e.g. one dialogue or one sentence per line), where line n in 'path_to_ground_truth.txt' matches that of line n in 'path_to_predictions.txt'.

NOTE: The metrics are not symmetric w.r.t. the input sequences. 
      Therefore, DO NOT swap the ground truths with the predicted responses.

References:

A Comparison of Greedy and Optimal Assessment of Natural Language Student Input Word Similarity Metrics Using Word to Word Similarity Metrics. Vasile Rus, Mihai Lintean. 2012. Proceedings of the Seventh Workshop on Building Educational Applications Using NLP, NAACL 2012.

Bootstrapping Dialog Systems with Word Embeddings. G. Forgues, J. Pineau, J. Larcheveque, R. Tremblay. 2014. Workshop on Modern Machine Learning and Natural Language Processing, NIPS 2014.


"""
__docformat__ = 'restructedtext en'
__authors__ = ("Chia-Wei Liu", "Iulian Vlad Serban")

from random import randint
from gensim.models import KeyedVectors
import numpy as np
import argparse

def greedy_match(fileone, filetwo, w2v):
    res1 = greedy_score(fileone, filetwo, w2v)
    res2 = greedy_score(filetwo, fileone, w2v)
    res_sum = (res1 + res2)/2.0

    return np.mean(res_sum), 1.96*np.std(res_sum)/float(len(res_sum)), np.std(res_sum)


def greedy_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()
    dim = 300 # embedding dimensions

    scores = []
    length = min(len(r1),len(r2))
    for i in range(length):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X= np.zeros((dim,))
        y_count = 0
        x_count = 0
        o = 0.0
        Y = np.zeros((dim,1))
        for tok in tokens2:
            if tok in w2v:
                temp_y = (w2v[tok].reshape((dim,1)))
                norm_y = temp_y / np.linalg.norm(temp_y)
                Y = np.hstack((Y, norm_y))
                y_count += 1

        for tok in tokens1:
            if tok in w2v:
                tmp_x = w2v[tok].reshape((1,dim))
                tmp = tmp_x / np.linalg.norm(tmp_x)
                tmp_res = tmp.dot(Y)
                tmp_max = np.max(tmp_res)
                # if tmp_max > 1:
                #     print(tmp_res)
                #     print(tmp_max)
                #     print(tokens1)
                #     print(tokens2)
                o += tmp_max
                x_count += 1

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0)
            continue

        o /= float(x_count)
        scores.append(o)


    return np.asarray(scores)


def extrema_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()

    scores = []
    length = min(len(r1), len(r2))
    for i in range(length):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X= []
        for tok in tokens1:
            if tok in w2v:
                X.append(w2v[tok])
        Y = []
        for tok in tokens2:
            if tok in w2v:
                Y.append(w2v[tok])

        # if none of the words have embeddings in ground truth, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X,0)  # get abs of min
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)   # get extrema

        ymax = np.max(Y, 0)
        ymin = np.min(Y,0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)


def average(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()
    dim = 300 # dimension of embeddings

    scores = []
    length = min(len(r1), len(r2))
    for i in range(length):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X= np.zeros((dim,))
        for tok in tokens1:
            if tok in w2v:
                X+=w2v[tok]
        Y = np.zeros((dim,))
        for tok in tokens2:
            if tok in w2v:
                Y += w2v[tok]

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        X = np.array(X)/np.linalg.norm(X)
        Y = np.array(Y)/np.linalg.norm(Y)
        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)


if __name__ == "__main__":

    word2vec =  './word2vec/word2vec.bin'
    ground_file = 'data/ResponseContextPairs/raw_testing_responses.txt'
    #result_file = 'data/ResponseContextPairs/ModelPredictions/HRED_Baseline/HRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
    result_file = 'output_rl/test35.txt'
    # result_file = 'data/ResponseContextPairs/ModelPredictions/MrRNN_HRED_ActEntAbstractions_GRUEncoder/Test_GeneratedSamples_BeamSearch_5.txt_First.txt'
    #result_file = './data/ResponseContextPairs/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
    print("loading embeddings file...")
    w2v = KeyedVectors.load_word2vec_format(word2vec, binary=True)

    r = average(ground_file, result_file, w2v)
    print("Embedding Average Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    r = greedy_match(ground_file, result_file, w2v)
    print("Greedy Matching Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    r = extrema_score(ground_file, result_file, w2v)
    print("Extrema Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))


