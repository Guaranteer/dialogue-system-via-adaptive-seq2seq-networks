import numpy as np
import nltk

chencherry = nltk.translate.bleu_score.SmoothingFunction()


def bleu_smooth(truth, base, smooth_function=chencherry.method7):
    truth = open(truth, 'r')
    base = open(base, 'r')

    truth = truth.readlines()
    base = base.readlines()
    # length = min(len(base),len(truth))
    # print(length)
    # truth = truth[0:length]
    # base = truth[0:length]
    for i in range(len(truth)):
        truth[i] = [truth[i].strip().split(" ")]
        base[i] = base[i].strip().split(" ")

    print(nltk.translate.bleu_score.corpus_bleu(truth, base, smoothing_function=smooth_function,weights=(0.5,0.5,0,0)))



def bleu_val(truth, base1):
    truth = open(truth, 'r')
    base1 = open(base1, 'r')
    truth = truth.readlines()
    base1 = base1.readlines()

    count = 0
    # print(len(base1))
    length = min(len(truth),len(base1))
    for i in range(length):
        t = [truth[i].strip().split(" ")]
        b = base1[i].strip().split(" ")

        count += nltk.translate.bleu_score.sentence_bleu(t, b, weights=(0.5,0.5,0,0),smoothing_function=chencherry.method7)

    print(count / len(base1))

def bleu_single_sent(truth,base):
    t = [truth.strip().split(" ")]
    b = base.strip().split(" ")
    score = nltk.translate.bleu_score.sentence_bleu(t, b, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method7)
    return score

kk = 'output6/test55.txt'
result_file = 'data/ResponseContextPairs/ModelPredictions/LSTM_Baseline/LSTM_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
result_file3 = './data/ResponseContextPairs/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
result_file2 = 'data/ResponseContextPairs/ModelPredictions/HRED_Baseline/HRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
result_file4 = 'data/ResponseContextPairs/ModelPredictions/MrRNN_HRED_ActEntAbstractions_GRUEncoder/Test_GeneratedSamples_BeamSearch_5.txt_First.txt'
truth = 'data/ResponseContextPairs/raw_testing_responses.txt'


bleu_val(truth,kk)
