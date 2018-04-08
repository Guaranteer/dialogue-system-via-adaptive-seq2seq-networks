from data_feed import Batcher
from model_base import Model
import time
import tensorflow as tf
import numpy as np
import os
import gensim
import json
import pickle
import utils
import embedding_metrics
import BLEU

class Trainer(object):

    def __init__(self, data_params):
        self.data_params = data_params
        self.wv = gensim.models.KeyedVectors.load_word2vec_format(self.data_params['word2vec'], binary=True)
        self.embedding = self.get_embedding()

        with tf.name_scope('training'):
            self.train_model = Model(data_params, self.embedding, forward_only=False)
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('test'):
            self.test_model = Model(data_params,self.embedding, forward_only=True)

    def train(self):

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config = sess_config)

        self.model_path = os.path.join(self.data_params['cache_dir'], 'tfmodel')
        self.last_checkpoint = None

        self.train_batcher = Batcher(self.data_params, self.data_params['train_data'], 'train', self.data_params['epoch_reshuffle'])
        self.valid_batcher = Batcher(self.data_params, self.data_params['valid_data'], 'valid')
        self.test_batcher = Batcher(self.data_params, self.data_params['test_data'], 'test')

        print ('Trainnning begins......')
        self._train(sess)

        # testing
        print ('Evaluating best model in file', self.last_checkpoint, '...')
        if self.last_checkpoint is not None:
            self.model_saver.restore(sess, self.last_checkpoint)
            self._test(sess, 10000)
        else:
            print ('ERROR: No checkpoint available!')
        sess.close()

    def _train(self, sess):

        self.summary_writer = tf.summary.FileWriter(self.data_params['summary_dir'], sess.graph)
        self.model_saver = tf.train.Saver()

        with open('./model_list.json','r') as fj:
            saved_model = json.load(fj)['best_model']

        if saved_model == "":
            init_proc = tf.global_variables_initializer()
            sess.run(init_proc)
        else:
            print('restore the mest model')
            self.model_saver.restore(sess, saved_model)

        best_epoch_acc = 0
        best_epoch_id = 0

        print('****************************')
        print('Trainning datetime:', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('Data params')
        print (self.data_params)
        utils.count_total_variables()
        print ('****************************')
        for i_epoch in range(self.data_params['max_epoches']):
            t_begin = time.time()
            t1 = time.time()
            self.train_batcher.reset()
            i_batch = 0
            loss_sum = 0
            num_per_epoch = len(self.train_batcher.data) // 100
            print('num_per_epoch',num_per_epoch)
            for _ in range(num_per_epoch):
                context_vecs, response_vecs, target_idx, target_mask  = self.train_batcher.generate(self.embedding)

                if context_vecs is None:
                    break

                batch_data ={
                    self.train_model.encode_input: context_vecs,
                    self.train_model.decode_input: response_vecs,
                    self.train_model.target: target_idx,
                    self.train_model.mask: target_mask,
                    self.train_model.is_training: True
                }

                indics = sess.run( self.train_model.indices, feed_dict = batch_data)

                sent_list = list()
                for idxs in indics:
                    sent = [self.train_batcher.idx_to_word[idx[0]] for idx in idxs if idx[0] != 20001]
                    sent = ' '.join(sent)
                    sent_list.append(sent)

                truth_sent_list = list()
                for idxs in target_idx:
                    truth_sent = [self.train_batcher.idx_to_word[idx] for idx in idxs if idx != 20001]
                    truth_sent = ' '.join(truth_sent)
                    truth_sent_list.append(truth_sent)

                reward = np.zeros(len(sent_list),dtype=float)
                for i in range(len(sent_list)):
                    reward[i] = BLEU.bleu_single_sent(truth_sent_list[i],sent_list[i])

                batch_data[self.train_model.reward] = reward

                _, loss, summary, g_step= sess.run([self.train_model.train_proc, self.train_model.loss,
                                                    self.train_model.summary_proc, self.train_model.global_step], feed_dict = batch_data)

                i_batch += 1
                loss_sum += loss
                if i_batch % self.data_params['display_batch_interval'] == 0:
                    t2 = time.time()
                    print ('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % (i_epoch, i_batch, loss, (t2-t1)/self.data_params['display_batch_interval']))
                    t1 = t2

            # do summaries and evaluations
            #if i_epoch % train_params['summary_interval'] == 0:
            #    summary_str = sess.run(summary_proc, feed_dict=batch_data)
            #    summary_writer.add_summary(summary_str, i_batch)


            avg_batch_loss = loss_sum/i_batch
            t_end = time.time()
            if i_epoch % self.data_params['evaluate_interval'] == 0:
                print ('****************************')
                print ('Overall evaluation')
                print ('****************************')
                valid_acc, _ = self._test(sess,i_epoch)
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                print ('****************************')
            else:
                print ('****************************')
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                valid_acc = self._evaluate(sess, self.test_model, self.valid_batcher, self.data_params['ground_valid'], self.data_params['output_dir']+'valid'+str(i_epoch)+'.txt')
                print ('****************************')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print ('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.last_checkpoint = self.model_saver.save(sess, self.model_path+timestamp, global_step=self.train_model.global_step)
                print ('Saved at', self.last_checkpoint)
            else:
                if i_epoch-best_epoch_id >= self.data_params['early_stopping']:
                    print ('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

    def _test(self, sess, epoch):
        # make prediction and evaluation for all sets
        # print ('Train set:')
        # train_acc = self._evaluate(sess, self.model, self.train_batcher)
        print ('Validation set:')
        valid_acc = self._evaluate(sess, self.test_model, self.valid_batcher, self.data_params['ground_valid'],  self.data_params['output_dir']+'valid'+str(epoch)+'.txt')
        print ('Test set:')
        test_acc = self._evaluate(sess, self.test_model, self.test_batcher, self.data_params['ground_test'],  self.data_params['output_dir']+'test'+str(epoch)+'.txt')
        return valid_acc, test_acc


    def _evaluate(self, sess, model, batcher, ground_file, result_file):

        batcher.reset()
        num_per_epoch = len(batcher.data) // 100
        print('num_per_epoch', num_per_epoch)
        sent_list = list()
        for _ in range(num_per_epoch):
            context_vecs, response_vecs, target_idx, target_mask = batcher.generate(self.embedding)
            # train a batch
            if context_vecs is None:
                break

            batch_data = {
                model.encode_input: context_vecs,
                model.decode_input: response_vecs,
                model.target: target_idx,
                model.mask: target_mask,
                model.is_training: False
            }
            indics = sess.run( model.indices, feed_dict = batch_data)
            for idxs in indics:
                sent = [batcher.idx_to_word[idx[0]] for idx in idxs if idx[0] != 20001]
                sent = ' '.join(sent)
                sent_list.append(sent)
        sents = '\n'.join(sent_list)

        file_w = open(result_file, 'w')
        file_w.write(sents)
        file_w.close()

        avg_r = embedding_metrics.average(ground_file, result_file, self.wv)
        print("Embedding Average Score: %f +/- %f ( %f )" % (avg_r[0], avg_r[1], avg_r[2]))

        greedy_r = embedding_metrics.greedy_match(ground_file,result_file, self.wv)
        print("Greedy Matching Score: %f +/- %f ( %f )" % (greedy_r[0], greedy_r[1], greedy_r[2]))

        extrema_r = embedding_metrics.extrema_score(ground_file,result_file, self.wv)
        print("Extrema Score: %f +/- %f ( %f )" % (extrema_r[0], extrema_r[1], extrema_r[2]))

        bleu = BLEU.bleu_val(ground_file, result_file)
        print("BLEU Score: %f" % bleu)
        # return avg_r[0] + greedy_r[0] + extrema_r[0]
        return  bleu

    def get_embedding(self):
        with open(self.data_params['dataset'],'rb') as f:
            vocab = pickle.load(f)

        embedding_list = list()
        for item in vocab:
            word = item[0]
            if word in self.wv:
                embedding_list.append(self.wv[word])
            else:
                w_vec = np.random.uniform(-0.1, 0.1, size=300)
                embedding_list.append(w_vec)
        embedding_list.append(np.random.uniform(-0.1, 0.1, size=300))
        embedding_list.append(np.random.uniform(-0.1, 0.1, size=300))

        return  np.array(embedding_list).reshape(20002,300)