import json
import pickle
import numpy as np
import random
import nltk

def load_file(filename):
    with open(filename,'rb') as fw:
        return pickle.load(fw)

def load_json(filename):
    with open(filename, 'r') as fr:
        return json.load(fr)

class Batcher(object):
    def __init__(self, params, data_file, mode, reshuffle=False):
        # general
        self.reshuffle = reshuffle
        self.data_file = data_file
        self.params = params
        self.mode = mode
        self.next_idx = 0

        self.max_n_words = params['max_n_words']
        self.max_n_sentences = params['max_n_sentences']
        self.max_r_words = params['max_r_words']
        self.max_r_f_words = params['max_r_f_words']
        self.input_dim = params['input_dim']
        self.max_batch_size = params['batch_size']

        # dataset
        self.word_to_idx, self.idx_to_word = load_file(self.params["word2index"]), load_file(self.params["index2word"])
        self.embedding = load_file(self.params["embedding"])
        self.data = load_json(data_file)

        self.data_index = list(range(len(self.data)))
        self.sample_num = len(self.data_index)
        if self.reshuffle:
            random.shuffle(self.data_index)

        self.stopwords = ['.', ',', '?', '...', '--', '!', '\'', '\"', '(', ')', ':', '-', ';']

    def reset(self):
        # reset for next epoch
        self.next_idx = 0
        if self.reshuffle:
            random.shuffle(self.data_index)

    def process_data(self, sents):


        context = np.zeros((self.max_n_sentences, self.max_n_words, self.input_dim), dtype=float)
        sent_len = np.zeros((self.max_n_sentences), dtype=int)
        res_vecs = np.zeros((self.max_r_words, self.input_dim), dtype=float)
        res_idx = np.zeros((self.max_r_words), dtype=int)

        res_vecs_forward = np.zeros((self.max_r_f_words, self.input_dim), dtype=float)
        res_idx_forward = np.zeros((self.max_r_f_words), dtype=int)

        if len(sents) > self.max_n_sentences + 1:
            sents = sents[:self.max_n_sentences] + [sents[-1]]

        # context
        sent_num = len(sents) - 1
        for idx in range(sent_num):
            words = nltk.word_tokenize(sents[idx])
            words = [word.lower() for word in words if word not in self.stopwords]
            words_vec = list()
            for word in words:
                if word in self.word_to_idx:
                    word_vec = self.embedding[self.word_to_idx[word]]
                    words_vec.append(word_vec)
            if len(words_vec) > self.max_n_words:
                words_vec = words_vec[:self.max_n_words]
            if len(words_vec) == 0:
                words_vec.append(self.embedding[self.word_to_idx['<end>']])
            sent_len[idx] = len(words_vec)
            # print(sent_len)
            context[idx,:len(words_vec),:] = words_vec

        # response
        words = nltk.word_tokenize(sents[-1][0])
        important_idx, important_word = sents[-1][1], sents[-1][2]
        words = [word.lower() for word in words if word not in self.stopwords]
        words_vec = list()
        words_idx = list()
        words_reserve = list()
        for word in words:
            if word in self.word_to_idx:
                word_vec = self.embedding[self.word_to_idx[word]]
                word_idx = self.word_to_idx[word]
                words_vec.append(word_vec)
                words_idx.append(word_idx)
                words_reserve.append(word)
        words_vec.append(self.embedding[self.word_to_idx['<end>']])
        words_idx.append(self.word_to_idx['<end>'])
        res_num = len(words_vec)
        if len(words_vec) > self.max_r_words:
            words_vec = words_vec[:self.max_r_words-1] + words_vec[-1:]
            words_idx = words_idx[:self.max_r_words-1] + words_idx[-1:]
            res_num = self.max_r_words
        if len(words_vec) < self.max_r_words:
            words_vec.extend([self.embedding[self.word_to_idx['<pad>']]]*(self.max_r_words - len(words_vec)))
            words_idx.extend([self.word_to_idx['<pad>']] * (self.max_r_words - len(words_idx)))
        res_vecs[:,:] = words_vec
        res_idx[:] = words_idx

        # forward response
        words_vec_forward = list()
        words_idx_forward = list()
        if important_word in words_reserve:
            important_idx = words_reserve.index(important_word)
        else:
            important_idx = len(words_reserve)//2

        if len(words_reserve) == 0:
            important_idx = -1

        for i in range(self.max_r_f_words):
            if important_idx - i >= 0:
                word = words_reserve[important_idx - i]
                word_vec = self.embedding[self.word_to_idx[word]]
                word_idx = self.word_to_idx[word]
                words_vec_forward.append(word_vec)
                words_idx_forward.append(word_idx)
        words_vec_forward.append(self.embedding[self.word_to_idx['<start>']])
        words_idx_forward.append(self.word_to_idx['<start>'])
        res_num_forward = len(words_vec_forward)
        if len(words_vec_forward) > self.max_r_f_words:
            words_vec_forward = words_vec_forward[:self.max_r_f_words - 1] + words_vec_forward[-1:]
            words_idx_forward = words_idx_forward[:self.max_r_f_words - 1] + words_idx_forward[-1:]
            res_num = self.max_r_words
        if len(words_vec_forward) < self.max_r_f_words:
            words_vec_forward.extend([self.embedding[self.word_to_idx['<pad>']]] * (self.max_r_f_words - len(words_vec_forward)))
            words_idx_forward.extend([self.word_to_idx['<pad>']] * (self.max_r_f_words - len(words_idx_forward)))
        res_vecs_forward[:, :] = words_vec_forward
        res_idx_forward[:] = words_idx_forward

        return context, sent_len, sent_num, res_vecs, res_idx, res_num, res_vecs_forward, res_idx_forward, res_num_forward



    def generate(self):

        batch_size = self.max_batch_size

        context_vecs = np.zeros((batch_size,self.max_n_sentences, self.max_n_words, self.input_dim), dtype=float)
        context_sent_len = np.zeros((batch_size,self.max_n_sentences), dtype=int)
        context_conv_len = np.zeros((batch_size), dtype=int)
        response_vecs = np.zeros((batch_size, self.max_r_words, self.input_dim), dtype=float)
        response_idx = np.zeros((batch_size, self.max_r_words), dtype=int)
        response_n  = np.zeros((batch_size), dtype=int)
        response_vecs_forward = np.zeros((batch_size, self.max_r_f_words, self.input_dim), dtype=float)
        response_idx_forward = np.zeros((batch_size, self.max_r_f_words), dtype=int)
        response_n_forward = np.zeros((batch_size), dtype=int)

        for i in range(batch_size):
            curr_data_index = self.data_index[self.next_idx]
            context, sent_len, sent_num, res_vecs, res_idx, res_num, \
                res_vecs_forward, res_idx_forward, res_num_forward = self.process_data(self.data[curr_data_index])

            context_vecs[i, :, :] = context
            context_sent_len[i, :] =sent_len
            context_conv_len[i] = sent_num
            response_vecs[i, :, :] = res_vecs
            response_idx[i, :] = res_idx
            response_n[i] = res_num
            response_vecs_forward[i, :, :] = res_vecs_forward
            response_idx_forward[i, :] = res_idx_forward
            response_n_forward[i] = res_num_forward

            self.next_idx += 1

            if self.next_idx == self.sample_num:
                self.next_idx = 0

        context_vecs = np.reshape(context_vecs,[batch_size*self.max_n_sentences, self.max_n_words, self.input_dim])
        context_sent_len = np.reshape(context_sent_len,[batch_size*self.max_n_sentences])

        return context_vecs, context_sent_len, context_conv_len, response_vecs, response_idx, response_n, response_vecs_forward, response_idx_forward, response_n_forward

if __name__ == '__main__':
    config_file = '../configs/configs_HANL_bi.json'
    with open(config_file, 'r') as fr:
        config = json.load(fr)

    batcher = Batcher(config,config['train_data_bi'],mode=1)

    for i in range(3):
        batcher.reset()
        num_per_epoch = batcher.sample_num // 100
        for _ in range(num_per_epoch):
            context_vecs, context_sent_len, context_conv_len, response_vecs, response_idx, response_n, \
                response_vecs_forward, response_idx_forward, response_n_forward = batcher.generate()

            # print(context_vecs.shape)
            print(response_idx)
            print(response_n)
            print(response_idx_forward)
            print(response_n_forward)
            # print(context_vecs[125,:,:])
            # print(context_sent_len[120:130])

