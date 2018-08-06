import sys
sys.path.append('../utils/')
sys.path.append('../configs/')
sys.path.append('../dataloaders/')
sys.path.append('../models/')
from data_loader_seq2seq import Batcher
from model_seq2seq_attention import Model
import time
import tensorflow as tf
import numpy as np
import os
import gensim
import json
import pickle
import util
import embedding_metrics
import perplexity

def create_path(path):
    if not os.path.exists(path):
        print('create path: ', path)
        os.makedirs(path)

class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.model = Model(params)
        self.model.build_model()
        print('load word2vec...')
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(params['word2vec'], binary=True)

    def train(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config = sess_config)

        self.model_path = os.path.join(self.params['model_dir'] + 'seq2seq_att/')
        self.result_path = os.path.join(self.params['result_dir'] + 'seq2seq_att/')
        self.summary_path = os.path.join(self.params['summary_dir'] + 'seq2seq_att/')
        create_path(self.model_path)
        create_path(self.result_path)
        create_path(self.summary_path)

        self.last_checkpoint = None

        self.train_batcher = Batcher(self.params, self.params['train_data_process'], 'train', self.params['epoch_reshuffle'])
        self.valid_batcher = Batcher(self.params, self.params['val_data_process'], 'valid')
        self.test_batcher = Batcher(self.params, self.params['test_data_process'], 'test')
        self.model_saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

        print('Trainnning begins......')
        self._train(sess)

        # testing
        print('Evaluating best model in file', self.last_checkpoint, '...')
        if self.last_checkpoint is not None:
            self.model_saver.restore(sess, self.last_checkpoint)
            self._test(sess, 10000)
        else:
            print ('ERROR: No checkpoint available!')
        sess.close()

    def _train(self, sess):

        init_proc = tf.global_variables_initializer()
        sess.run(init_proc)
        best_epoch_acc = 0
        best_epoch_id = 0

        print('****************************')
        print('Trainning datetime:', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('training params')
        print (self.params)
        util.count_total_variables()
        print ('****************************')
        for i_epoch in range(self.params['max_epoches']):
            t_begin = time.time()
            t1 = time.time()
            self.train_batcher.reset()
            loss_sum = 0
            num_per_epoch = self.train_batcher.sample_num // 100 + 1
            print('the number per epoch',num_per_epoch)

            for i_batch in range(num_per_epoch):
                context_vecs, context_sent_len, context_conv_len, response_vecs, response_idx, response_n = self.train_batcher.generate()

                mask_matrix = np.zeros([np.shape(response_n)[0], self.params['max_r_words']], np.int32)
                for ind, row in enumerate(mask_matrix):
                    row[:response_n[ind]] = 1
                batch_data ={
                    self.model.encode_input: context_vecs,
                    self.model.encode_sent_len: context_sent_len,
                    self.model.encode_conv_len: context_conv_len,
                    self.model.is_training: True,
                    self.model.ans_vec: response_vecs,
                    self.model.y: response_idx,
                    self.model.y_mask: mask_matrix
                }

                _, loss, train_ans, g_step = sess.run([self.model.train_proc, self.model.train_loss, self.model.answer_word_train, self.model.global_step], feed_dict = batch_data)

                train_ans = np.transpose(np.array(train_ans), (1, 0))
                sent_list = list()
                for i in range(len(response_n)):
                    ground_a = list()
                    for l in range(self.params['max_r_words']):
                        word = response_idx[i][l]
                        ground_a.append(self.train_batcher.idx_to_word[word])
                    ground_sent = ' '.join(ground_a)


                    generate_a = list()
                    for l in range(self.params['max_r_words']):
                        word = train_ans[i][l]
                        generate_a.append(self.train_batcher.idx_to_word[word])
                    generate_sent = ' '.join(generate_a)

                    sent_list.append([ground_sent,generate_sent])


                loss_sum += loss
                if i_batch % self.params['display_batch_interval'] == 0 and i_batch != 0 :
                    t2 = time.time()
                    print ('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % (i_epoch, i_batch, loss , (t2-t1)/self.params['display_batch_interval']))
                    print('sample of sentences:', sent_list[3])
                    t1 = t2




            avg_batch_loss = loss_sum/num_per_epoch
            t_end = time.time()
            if i_epoch % self.params['evaluate_interval'] == 0:
                print ('****************************')
                print ('Overall evaluation')
                print ('****************************')
                valid_acc, _ = self._test(sess,i_epoch)
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                print ('****************************')
            else:
                print ('****************************')
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                valid_acc = self._evaluate(sess, self.valid_batcher, self.result_path+'ground'+str(i_epoch)+'.txt', self.result_path+'valid'+str(i_epoch)+'.txt')
                print ('****************************')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print ('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.last_checkpoint = self.model_saver.save(sess, self.model_path+timestamp, global_step=self.model.global_step)
                print ('Saved at', self.last_checkpoint)
            else:
                if i_epoch-best_epoch_id >= self.params['early_stopping']:
                    print ('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

    def _test(self, sess, epoch):
        print ('Validation set:')
        valid_acc = self._evaluate(sess, self.valid_batcher, self.result_path + 'ground' + str(epoch) + '.txt',
                                   self.result_path + 'valid' + str(epoch) + '.txt')
        print ('Test set:')
        test_acc = self._evaluate(sess, self.test_batcher, self.result_path + 'ground' + str(epoch) + '.txt',
                                   self.result_path+'test'+str(epoch)+'.txt')
        return valid_acc, test_acc


    def _evaluate(self, sess, batcher, ground_file, result_file):
        batcher.reset()
        num_per_epoch = batcher.sample_num // 100
        print('number per epoch', num_per_epoch)
        ground_sent_list = list()
        generate_sent_list = list()
        ppl = 0
        for _ in range(num_per_epoch):
            context_vecs, context_sent_len, context_conv_len, response_vecs, response_idx, response_n = batcher.generate()

            mask_matrix = np.zeros([np.shape(response_n)[0], self.params['max_r_words']], np.int32)
            for ind, row in enumerate(mask_matrix):
                row[:response_n[ind]] = 1
            batch_data = {
                self.model.encode_input: context_vecs,
                self.model.encode_sent_len: context_sent_len,
                self.model.encode_conv_len: context_conv_len,
                self.model.is_training: False,
                self.model.ans_vec: response_vecs,
                self.model.y: response_idx,
                self.model.y_mask: mask_matrix
            }

            loss, test_ans, test_dist = sess.run([ self.model.test_loss, self.model.answer_word_test, self.model.distribution_word_test], feed_dict=batch_data)

            test_ans = np.transpose(np.array(test_ans), (1,0))
            for i in range(len(response_n)):
                ground_a = list()
                for l in range(self.params['max_r_words']):
                    word = response_idx[i][l]
                    ground_a.append(batcher.idx_to_word[word])
                    if batcher.idx_to_word[word] == '<end>':
                        break
                ground_sent = ' '.join(ground_a)
                ground_sent_list.append(ground_sent)

                generate_a = list()
                for l in range(self.params['max_r_words']):
                    word = test_ans[i][l]
                    generate_a.append(batcher.idx_to_word[word])
                    if batcher.idx_to_word[word] == '<end>':
                        break
                generate_sent = ' '.join(generate_a)
                generate_sent_list.append(generate_sent)

            test_dist = np.transpose(np.array(test_dist),(1,0,2))
            for i in range(len(response_n)):
                ppl += perplexity.calculate_perplexity(test_dist[i],response_idx[i],response_n[i])

        ppl = ppl/(num_per_epoch*100)


        ground_sents = '\n'.join(ground_sent_list)
        generate_sents = '\n'.join(generate_sent_list)

        with open(result_file, 'w') as fw:
            fw.write(generate_sents)

        with open(ground_file, 'w') as fw:
            fw.write(ground_sents)


        avg_r = embedding_metrics.average(ground_file, result_file, self.w2v)
        print("Embedding Average Score: %f +/- %f ( %f )" % (avg_r[0], avg_r[1], avg_r[2]))

        greedy_r = embedding_metrics.greedy_match(ground_file,result_file, self.w2v)
        print("Greedy Matching Score: %f +/- %f ( %f )" % (greedy_r[0], greedy_r[1], greedy_r[2]))

        extrema_r = embedding_metrics.extrema_score(ground_file,result_file, self.w2v)
        print("Extrema Score: %f +/- %f ( %f )" % (extrema_r[0], extrema_r[1], extrema_r[2]))

        print("perplexity: %f" % (ppl))

        # bleu = BLEU.bleu_val(ground_file, result_file)
        # print("BLEU Score: %f" % bleu)
        return avg_r[0] + greedy_r[0] + extrema_r[0]

if __name__ == '__main__':
    config_file = '../configs/configs_seq2seq.json'
    with open(config_file, 'r') as fr:
        config = json.load(fr)
    trainer = Trainer(config)
    trainer.train()