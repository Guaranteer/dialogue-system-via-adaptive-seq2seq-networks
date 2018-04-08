import pickle
from gensim.models import KeyedVectors

def get_data():
    with open('data/Test.dialogues.pkl', 'rb') as f:
        data_list = pickle.load(f)
    max1 = 0
    max2 = 0
    all1 = 0
    all2 = 0
    for items in data_list:
        eos_pos = [i for i, key in enumerate(items) if key == 1]
        split_pos = eos_pos[-1]
        context, response = items[:split_pos + 1], items[split_pos + 1:]
        all1 += len(context)
        all2 += len(response)

        if len(context)> max1:
            max1 = len(context)
        if len(response) > max2:
            max2 = len(response)
        print(len(context),len(response))
    return max1,max2,all1/len(data_list),all2/len(data_list),len(data_list)


def clean_output():
    word2vec = './word2vec/word2vec.bin'
    result_file = 'output/test10.txt'
    f1 = open(result_file, 'r')
    r1 = f1.readlines()

    w2v = KeyedVectors.load_word2vec_format(word2vec, binary=True)
    sents = list()
    for i in range(len(r1)):
        tokens = r1[i].strip().split(" ")
        clean_tokens = [token for token in tokens if token in w2v]
        sent = ' '.join(clean_tokens)
        sents.append(sent)
    sents = '\n'.join(sents)
    file_w = open('result.txt', 'w')
    file_w.write(sents)
    file_w.close()


def clean_output2():

    result_file = 'output_rl/test20.txt'
    f1 = open(result_file, 'r')
    r1 = f1.readlines()

    sents = list()
    for i in range(len(r1)):
        tokens = r1[i].strip().split(" ")
        clean_tokens = list()
        for token in tokens:
            clean_tokens.append(token)
            if token == '__eou__':
                break
        sent = ' '.join(clean_tokens)
        sents.append(sent)
    sents = '\n'.join(sents)
    file_w = open('result.txt', 'w')
    file_w.write(sents)
    file_w.close()

clean_output2()