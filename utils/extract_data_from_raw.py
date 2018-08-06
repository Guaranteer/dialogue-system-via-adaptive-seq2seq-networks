import json
import re
import nltk
import gensim
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

def extract_data(config):
    movie_lines_file = config["movie_lines"]
    movie_conversation_file = config["movie_conversation"]

    with open(movie_conversation_file,'r',encoding='utf-8') as fr:
        movie_conversation = fr.readlines()
    with open(movie_lines_file,'r',encoding='utf-8',errors='ignore') as fr:
        movie_lines = fr.readlines()

    lines_dict = dict()
    for line in movie_lines:
        line = line.strip('\n')
        items = line.split(' +++$+++ ')
        lines_dict[int(items[0][1:])] = items[4]

    pattern = re.compile(r'\'\w*?\'')
    conversation_list = list()
    for conversation in movie_conversation:
        conversation = conversation.strip('\n')
        items = conversation.split(' +++$+++ ')
        utterances  = pattern.findall(items[3])
        utterances = [int(idx[2:-1]) for idx in utterances]
        if len(conversation_list) == 0:
            conversation_list.append(utterances)
        elif conversation_list[-1][-1]+1 == utterances[0] :
            conversation_list[-1].extend(utterances)
        else:
            conversation_list.append(utterances)

    return lines_dict, conversation_list


def statistics(conversation_list):
    print(len(conversation_list))
    count = dict()
    for conversation in conversation_list:
        if len(conversation) in count:
            count[len(conversation)] += 1
        else:
            count[len(conversation)] = 1

    print(count)

    conversation_list = [conversation for conversation in conversation_list if len(conversation) > 7]
    all = map(lambda x:len(x),conversation_list)
    print(len(conversation_list))
    print(sum(all))

    conversation_list = [conversation for conversation in conversation_list if len(conversation) > 10]
    all = map(lambda x:len(x),conversation_list)
    print(len(conversation_list))
    print(sum(all))

def statistics_processed_data(config):
    with open(config['train_data_process'],'r') as fr:
        train_data = json.load(fr)
    with open(config['test_data_process'],'r') as fr:
        test_data = json.load(fr)
    with open(config['val_data_process'],'r') as fr:
        val_data = json.load(fr)

    print('train data:',len(train_data))
    print('val data:',len(val_data))
    print('test_data:',len(test_data))
    all_sample = len(train_data) + len(val_data) + len(test_data)
    print('all data:', all_sample)

    all_sent_num = 0
    all_word_num = 0
    for dataset in [train_data, val_data, test_data]:
        for sents in dataset:
            all_sent_num += len(sents)
            for sent in sents:
                all_word_num += len(sent.split())
    print('avg sents:', all_sent_num/all_sample)
    print('avg words:', all_word_num/all_sent_num)
    print('words:', all_word_num/(len(train_data) + len(val_data)+ len(test_data)))

    sent_num = 0
    word_num = 0
    for sents in train_data:
        sent_num += len(sents)
        for sent in sents:
            word_num += len(sent.split())
    print('avg sents:', sent_num/len(train_data))
    print('avg words:', word_num/sent_num)
    print('words:', word_num/len(train_data))

    sent_num = 0
    word_num = 0
    for sents in val_data:
        sent_num += len(sents)
        for sent in sents:
            word_num += len(sent.split())
    print('avg sents:', sent_num/len(val_data))
    print('avg words:', word_num/sent_num)
    print('words:', word_num/len(val_data))

    sent_num = 0
    word_num = 0
    for sents in test_data:
        sent_num += len(sents)
        for sent in sents:
            word_num += len(sent.split())
    print('avg sents:', sent_num/len(test_data))
    print('avg words:', word_num/sent_num)
    print('words:', word_num/len(test_data))



def search_important_word(config):
    with open(config['train_data_process'],'r') as fr:
        train_data = json.load(fr)
    with open(config['test_data_process'],'r') as fr:
        test_data = json.load(fr)
    with open(config['val_data_process'],'r') as fr:
        val_data = json.load(fr)
    with open(config['word2index'], 'rb') as fr:
        word2index = pickle.load(fr)

    all_sents = list()
    for dataset in [train_data, test_data, val_data]:
        for sents in dataset:
            all_sents.extend(sents)
    vectorizer = CountVectorizer()
    countVec = vectorizer.fit_transform(all_sents)
    word_set = vectorizer.get_feature_names()
    tfidfer = TfidfTransformer()
    tfidfVec = tfidfer.fit_transform(countVec)
    tfidfArr = tfidfVec.toarray()

    train_data_bi, test_data_bi, val_data_bi = list(), list(), list()
    stop_words = set(stopwords.words('english'))
    stop_words.add('\'m')
    stop_words.add('n\'t')
    stop_words.add('\'re')
    stop_words.add('\'s')

    print(stop_words)
    count = 0
    corpus_bi = [train_data_bi, test_data_bi, val_data_bi]
    for corpus_id, corpus in enumerate([train_data, test_data, val_data]):
        for sents in corpus:
            index = count + len(sents) - 1
            res_sent = tfidfArr[index]
            value_pair = [(i, res_sent[i], word_set[i]) for i in range(len(res_sent)) if res_sent[i] > 0]
            sorted_pair = sorted(value_pair, key=lambda x: x[1], reverse=True)
            # print(value_pair)
            words = nltk.word_tokenize(sents[-1])
            words = [word.lower() for word in words if word not in ['.',',','?','...','--','!','\'','\"','(',')',':','-',';']]

            flag = 0
            for pair in sorted_pair:
                if word_set[pair[0]] not in stop_words and word_set[pair[0]] in word2index and word_set[pair[0]] in words:
                    important_word = word_set[pair[0]]
                    important_index = words.index(important_word.lower())
                    # print(words, important_word, important_index)
                    flag = 1
                    break

            if flag == 0:
                for pair in sorted_pair:
                    if word_set[pair[0]] in word2index and word_set[pair[0]] in words:
                        important_word = word_set[pair[0]]
                        important_index = words.index(important_word.lower())
                        # print(words, important_word, important_index)
                        flag = 1
                        break

            if flag == 0:
                sents[-1] = [sents[-1],0,'<unk>']
            else:
                sents[-1] = [sents[-1],important_index,important_word]
            corpus_bi[corpus_id].append(sents)
            count += len(sents)

    print(train_data_bi[0:10])
    print(len(train_data_bi))
    print(train_data[0:10])
    print(len(train_data))
    print(test_data_bi[0:10])
    print(len(test_data_bi))
    print(test_data[0:10])
    print(len(test_data))
    print(val_data_bi[0:10])
    print(len(val_data_bi))
    print(val_data[0:10])
    print(len(val_data))

    with open(config['train_data_bi'],'w') as fw:
        json.dump(train_data_bi,fw)
    with open(config['test_data_bi'],'w') as fw:
        json.dump(test_data_bi,fw)
    with open(config['val_data_bi'],'w') as fw:
        json.dump(val_data_bi,fw)



def create_dataset(lines_dict,conversation_list,config):
    conversation_list = [conversation for conversation in conversation_list if len(conversation) > 9]
    train = len(conversation_list)*8//10
    test = len(conversation_list)//10
    val =  len(conversation_list) - train - test

    train_data, test_data, val_data = list(), list(), list()
    for i,conversation in enumerate(conversation_list):
        sentences = [lines_dict[idx] for idx in conversation]
        expands = list()
        if len(sentences) > 20:
            for start in range(len(sentences)-20):
                expands.append(sentences[start:start+20])
        elif len(sentences) > 15:
            expands.extend([sentences,sentences[:-1],sentences[:-2]])
        else:
            expands.extend([sentences,sentences[:-1]])

        if i <= train:
            train_data.extend(expands)
        elif i <= train + test:
            test_data.extend(expands)
        else:
            val_data.extend(expands)

    with open(config['train_data_process'],'w') as fw:
        json.dump(train_data,fw)
    with open(config['test_data_process'],'w') as fw:
        json.dump(test_data,fw)
    with open(config['val_data_process'],'w') as fw:
        json.dump(val_data,fw)

def count_dataset(dataset,count_dict):
    stopwords = ['.',',','?','...','--','!','\'','\"','(',')',':','-',';']
    for sample in dataset:
        for sent in sample:
            words = nltk.word_tokenize(sent)
            words = [word.lower() for word in words if word not in stopwords]
            for word in words:
                if word in count_dict:
                    count_dict[word] += 1
                else:
                    count_dict[word] = 1

def load_word_2_vec(filepath):
    print('load word2vec...')
    w2v = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
    return  w2v


def process_data(config):
    with open(config['train_data'],'r') as fr:
        train_data = json.load(fr)
    with open(config['val_data'],'r') as fr:
        val_data = json.load(fr)
    with open(config['test_data'],'r') as fr:
        test_data = json.load(fr)

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))

    count_words = dict()
    count_dataset(train_data,count_words)
    count_dataset(val_data, count_words)
    count_dataset(test_data, count_words)
    sorted_words = sorted(count_words.items(),key=lambda x:x[1],reverse=True)
    print(sorted_words[0:20000])

    vocab = list()
    word2index = dict()
    index2word = dict()
    word2index['<start>'] = 20000
    word2index['<end>'] = 20001
    word2index['<pad>'] = 20002
    word2index['<unk>'] = 20003
    index2word[20000] = '<start>'
    index2word[20001] = '<end>'
    index2word[20002] = '<pad>'
    index2word[20003] = '<unk>'
    idx = 0

    wv = load_word_2_vec(config["word2vec"])
    for word, freq in sorted_words:
        if idx >= 20000:
            break
        if word in wv:
            vocab.append(wv[word])
            word2index[word] = idx
            index2word[idx] = word
            idx += 1
    vocab.append(0.6 * np.random.rand(300) - 0.3)
    vocab.append(0.6 * np.random.rand(300) - 0.3)
    vocab.append(0.6 * np.random.rand(300) - 0.3)
    vocab.append(0.6 * np.random.rand(300) - 0.3)
    embedding = np.stack(vocab)
    assert len(embedding) == 20004


    with open(config['word2index'], 'wb') as fw:
        pickle.dump(word2index, fw)
    with open(config['index2word'], 'wb') as fw:
        pickle.dump(index2word, fw)
    with open(config['embedding'], 'wb') as fw:
        pickle.dump(embedding, fw)



if __name__ == '__main__':
    config_file = '../configs/configs_HANL.json'
    with open(config_file, 'r') as fr:
        config = json.load(fr)
    # lines_dict, conversation_list = extract_data(config)
    # statistics(conversation_list)
    # create_dataset(lines_dict, conversation_list,config)
    # print('process data...')
    # process_data(config)
    statistics_processed_data(config)
    # search_important_word(config)




