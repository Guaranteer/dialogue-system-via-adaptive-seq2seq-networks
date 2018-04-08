import json
import pickle
import numpy as np
import random
import os
import gensim


class Batcher(object):
    def __init__(self, params, data_file, mode, reshuffle=False):
        # general
        self.reshuffle = reshuffle
        self.max_batch_size = params['batch_size']
        self.data_file = data_file
        self.next_idx = 0
        self.params = params
        self.max_n_context = params['max_n_context']
        self.max_n_response = params['max_n_response']
        self.dim = params['word_dim']

        # dataset
        self.word_to_idx, self.idx_to_word = self.get_vocab()
        self.data = self.get_data()



        self.data_index = list(range(len(self.data)))
        if self.reshuffle:
            random.shuffle(self.data_index)


    def get_vocab(self):
        with open(self.params['dataset'],'rb') as f:
            data = pickle.load(f)

        idx_to_word = dict()
        word_to_idx = dict()
        for item in data:
            idx_to_word[item[1]] = item[0]
            word_to_idx[item[0]] = item[1]
        idx_to_word[20000] = '<start>'
        idx_to_word[20001] = '<pad>'
        word_to_idx['<start>'] = 20000
        word_to_idx['<pad>'] = 20001

        return  word_to_idx, idx_to_word


    def get_data(self):
        with open(self.data_file,'rb') as f:
            data_list = pickle.load(f)

        result = list()
        for items in data_list:
            eos_pos = [i for i, key in enumerate(items) if key == self.word_to_idx['__eot__']] #1
            split_pos = eos_pos[-1]
            context, target = items[:split_pos+1],items[split_pos+1:]

            mask = len(target)+1
            if mask > self.max_n_response:
                mask = self.max_n_response

            if len(context) < self.max_n_context:
                context.extend([20001]*(self.max_n_context - len(context)))
            else:
                context = context[:self.max_n_context]

            if len(target) < self.max_n_response:
                target.extend([20001]*(self.max_n_response - len(target)))
            else:
                target = target[:self.max_n_response-1] + [18575] #__eou__

            response = [20000]+target[:self.max_n_response-1]
            result.append((context,response,target,mask))

        return result

    def reset(self):
        # reset for next epoch
        self.next_idx = 0
        if self.reshuffle:
            random.shuffle(self.data_index)

    def generate(self, embedding):

        batch_size = min(self.max_batch_size, len(self.data_index) - self.next_idx)
        if batch_size <= 0:
            return None, None, None, None

        context_vecs = np.zeros((batch_size, self.max_n_context, self.dim), dtype=float)
        response_vecs = np.zeros((batch_size, self.max_n_response, self.dim), dtype=float)
        target_idx = np.zeros((batch_size, self.max_n_response), dtype=int)
        target_mask = np.zeros((batch_size), dtype=int)

        for i in range(batch_size):
            curr_data_index = self.data_index[self.next_idx + i]
            context, response,target, mask = self.data[curr_data_index]

            tmp_context_vec = np.vstack([embedding[idx] for idx in context])
            tmp_response_vec = np.vstack([embedding[idx] for idx in response])

            context_vecs[i, :, :] = tmp_context_vec
            response_vecs[i, :, :] = tmp_response_vec
            target_idx[i, :] = np.array(target)
            target_mask[i] = mask

        self.next_idx += batch_size
        return context_vecs, response_vecs, target_idx, target_mask




