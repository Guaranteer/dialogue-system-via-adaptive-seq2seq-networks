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

kk = 'result.txt'
result_file = 'data/ResponseContextPairs/ModelPredictions/LSTM_Baseline/LSTM_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
result_file3 = './data/ResponseContextPairs/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
result_file2 = 'data/ResponseContextPairs/ModelPredictions/HRED_Baseline/HRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt'
truth = 'data/ResponseContextPairs/raw_testing_responses.txt'

distinct_1(kk,result_file2,result_file3)