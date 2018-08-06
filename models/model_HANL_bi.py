import sys
sys.path.append('../utils/')
import tensorflow as tf
import random
import pickle
import numpy as np
import util
import layers
import json

def load_file(filename):
    with open(filename,'rb') as fr:
        return pickle.load(fr)


class Model(object):
    def __init__(self, params):
        self.params = params

        self.batch_size = params['batch_size']
        self.n_words = params['n_words']

        self.max_n_sentences = params['max_n_sentences']
        self.max_n_words = params['max_n_words']
        self.max_r_words = params['max_r_words']
        self.max_r_f_words = params['max_r_f_words']
        self.max_r_b_words = params['max_r_b_words']

        self.ref_dim = params['ref_dim']
        self.word_lstm_dim = params['word_lstm_dim']
        self.lstm_dim = params['lstm_dim']
        self.second_lstm_dim = params['second_lstm_dim']
        self.attention_dim = params['attention_dim']
        self.decode_dim = params['decode_dim']
        self.input_dim = params['input_dim']

        self.regularization_beta = params['regularization_beta']
        self.dropout_prob = params['dropout_prob']


    def define_var(self):
        # answer->word predict
        self.embed_word_W = tf.get_variable('embed_word_W', shape=[self.decode_dim, self.n_words], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.embed_word_b = tf.get_variable('embed_word_b', shape=[self.n_words], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        # # word dim -> decode_dim
        # self.word_to_lstm_w = tf.get_variable('word_to_lstm_W', shape=[self.input_dim, self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # self.word_to_lstm_b = tf.get_variable('word_to_lstm_b', shape=[self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


        # decoder attention layer
        with tf.variable_scope('forward_decoder_attention'):
            self.forward_attention_w_x = tf.get_variable('attention_w_x', shape=[self.lstm_dim, self.attention_dim], dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.forward_attention_w_h = tf.get_variable('attention_w_h', shape=[self.decode_dim, self.attention_dim], dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.forward_attention_b = tf.get_variable('attention_b', shape=[self.attention_dim], dtype=tf.float32,
                                                       initializer=tf.contrib.layers.xavier_initializer())
            self.forward_attention_a = tf.get_variable('attention_a', shape=[self.attention_dim, 1], dtype=tf.float32,
                                                       initializer=tf.contrib.layers.xavier_initializer())
            self.forward_attention_to_decoder = tf.get_variable('attention_to_decoder', shape=[self.lstm_dim, self.decode_dim], dtype=tf.float32,
                                                                initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('entire_decoder_attention'):
            self.entire_attention_w_x = tf.get_variable('attention_w_x', shape=[self.lstm_dim, self.attention_dim], dtype=tf.float32,
                                                      initializer=tf.contrib.layers.xavier_initializer())
            self.entire_attention_w_h = tf.get_variable('attention_w_h', shape=[self.decode_dim, self.attention_dim], dtype=tf.float32,
                                                      initializer=tf.contrib.layers.xavier_initializer())
            self.entire_attention_b = tf.get_variable('attention_b', shape=[self.attention_dim], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.entire_attention_a = tf.get_variable('attention_a', shape=[self.attention_dim, 1], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.entire_attention_to_decoder = tf.get_variable('attention_to_decoder', shape=[self.lstm_dim, self.decode_dim], dtype=tf.float32,
                                                             initializer=tf.contrib.layers.xavier_initializer())
        # decoder

        with tf.variable_scope('forward_decoder'):
            self.forward_decoder_r = tf.get_variable('decoder_r', shape=[self.decode_dim * 4, self.decode_dim], dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer())
            self.forward_decoder_z = tf.get_variable('decoder_z', shape=[self.decode_dim * 4, self.decode_dim], dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer())
            self.forward_decoder_w = tf.get_variable('decoder_w', shape=[self.decode_dim * 4, self.decode_dim], dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('entire_decoder'):
            self.entire_decoder_r = tf.get_variable('decoder_r', shape=[self.decode_dim * 4, self.decode_dim], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.entire_decoder_z = tf.get_variable('decoder_z', shape=[self.decode_dim * 4, self.decode_dim], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.entire_decoder_w = tf.get_variable('decoder_w', shape=[self.decode_dim * 4, self.decode_dim], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())


    def build_train_proc(self):
        # input layer (batch_size, n_steps, input_dim)
        self.encode_input = tf.placeholder(tf.float32, [None, self.max_n_words, self.input_dim])
        self.encode_sent_len = tf.placeholder(tf.int32, [None])
        self.encode_conv_len = tf.placeholder(tf.int32, [self.batch_size])
        self.is_training = tf.placeholder(tf.bool)
        self.reward = tf.placeholder(tf.float32, [None])

        self.ans_vec_forward = tf.placeholder(tf.float32, [None, self.max_r_f_words, self.input_dim])
        self.y_forward = tf.placeholder(tf.int32, [None, self.max_r_f_words])
        self.y_mask_forward = tf.placeholder(tf.float32, [None, self.max_r_f_words])

        self.ans_vec_entire = tf.placeholder(tf.float32, [None, self.max_r_words, self.input_dim])
        self.y_entire = tf.placeholder(tf.int32, [None, self.max_r_words])
        self.y_mask_entire = tf.placeholder(tf.float32, [None, self.max_r_words])

        self.y_forward_generation = tf.placeholder(tf.int32, [None, self.max_r_words])
        # self.batch_size = tf.placeholder(tf.int32, [])

        self.encode_input = tf.contrib.layers.dropout(self.encode_input, self.dropout_prob, is_training=self.is_training)

        sent_outputs, sent_state = layers.dynamic_origin_bilstm_layer(self.encode_input, self.word_lstm_dim, scope_name = 'sent_level_bilstm_rnn', input_len=self.encode_sent_len)
        sent_last_state = tf.concat([sent_state[0][1],sent_state[1][1]],axis=1)
        # sent_last_state = tf.contrib.layers.dropout(sent_last_state, self.dropout_prob, is_training=self.is_training)
        sent_outputs = tf.reshape(sent_outputs, shape=[self.batch_size, self.max_n_sentences, self.max_n_words, self.lstm_dim])
        ind = tf.stack([tf.range(self.batch_size), self.encode_conv_len - 1], axis=1)
        sent_last_outputs = tf.gather_nd(sent_outputs,indices=ind)

        conv_sents = tf.reshape(sent_last_state,shape = [self.batch_size, self.max_n_sentences, self.lstm_dim])
        self.sent_last_state_trun = tf.gather_nd(conv_sents, indices=ind)
        conv_outputs, conv_state = layers.dynamic_origin_lstm_layer(conv_sents, self.lstm_dim, 'conv_level_rnn', input_len=self.encode_conv_len)
        self.conv_last_state = conv_state[1]

        self.sent_features = sent_last_outputs
        self.conv_features = conv_outputs

        self.sent_features = tf.contrib.layers.dropout(self.sent_features, self.dropout_prob, is_training=self.is_training)
        self.conv_features = tf.contrib.layers.dropout(self.conv_features, self.dropout_prob, is_training=self.is_training)



        # decoder

        with tf.variable_scope('linear'):
            sent_and_conv_last = tf.concat([self.sent_last_state_trun,self.conv_last_state],axis=1)
            decoder_input_W = tf.get_variable('sw', shape=[self.sent_last_state_trun.shape[1] + self.conv_last_state.shape[1], self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())
            decoder_input_b = tf.get_variable('sb', shape=[self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())

            self.decoder_input = tf.matmul(sent_and_conv_last, decoder_input_W) + decoder_input_b

        self.define_var()

        # embedding layer
        embedding = load_file(self.params['embedding'])
        self.Wemb = tf.constant(embedding, dtype=tf.float32)

        # generate training
        forward_answer_train, forward_train_loss, forward_distribution_train = self.generate_forward_answer_on_training()
        entire_answer_train, entire_train_loss, entire_distribution_train = self.generate_entire_answer_on_training()
        forward_answer_test, forward_test_loss, forward_distribution_test = self.generate_forward_answer_on_testing()
        entire_answer_test, entire_test_loss, entire_distribution_test = self.generate_entire_answer_on_testing()

        # final
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        self.forward_answer_word_train = forward_answer_train
        self.answer_word_train = entire_answer_train
        self.forward_train_loss = forward_train_loss + self.regularization_beta * regularization_cost
        self.train_loss = entire_train_loss  + self.regularization_beta * regularization_cost
        self.distribution_word_train = entire_distribution_train

        self.forward_answer_word_test = forward_answer_test
        self.answer_word_test = entire_answer_test
        self.test_loss = entire_test_loss
        self.distribution_word_test = entire_distribution_test

        self.forward_global_step = tf.get_variable('forward_global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        forward_learning_rates = tf.train.exponential_decay(self.params['learning_rate'], self.global_step, decay_steps=self.params['lr_decay_n_iters'],
                                                    decay_rate=self.params['lr_decay_rate'], staircase=True)
        learning_rates = tf.train.exponential_decay(self.params['learning_rate'], self.global_step, decay_steps=self.params['lr_decay_n_iters'],
                                                    decay_rate=self.params['lr_decay_rate'], staircase=True)
        forward_optimizer = tf.train.AdamOptimizer(forward_learning_rates)
        optimizer = tf.train.AdamOptimizer(learning_rates)
        self.forward_train_proc = forward_optimizer.minimize(self.forward_train_loss, global_step=self.forward_global_step)
        self.train_proc = optimizer.minimize(self.train_loss, global_step=self.global_step)

        # tf.summary.scalar('global_step', self.global_step)
        # tf.summary.scalar('training cross entropy', self.train_loss)
        # tf.summary.scalar('test cross entropy', self.test_loss)
        # self.summary_proc = tf.summary.merge_all()

    def generate_forward_answer_on_training(self):
        with tf.variable_scope("forward_decoder"):
            forward_answer_train = []
            forward_distribution_train =[]
            decoder_state = self.decoder_input 
            loss = 0.0

            with tf.variable_scope("decoder_lstm") as scope:
                for i in range(self.max_r_f_words):
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        scope.reuse_variables()
                        # next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        # current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)
                        # current_emb = tf.nn.xw_plus_b(self.ans_vec[:, i - 1, :], self.word_to_lstm_w, self.word_to_lstm_b)
                        # next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        # current_emb = next_word_vec
                        current_emb = self.ans_vec_entire[:, i - 1, :]

                    # attention sent
                    s_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_words, 1]))
                    s_attention_input = tf.tanh(util.tensor_matmul(self.sent_features, self.forward_attention_w_x)
                                              + util.tensor_matmul(s_tiled_decoder_state_h, self.forward_attention_w_h)
                                              + self.forward_attention_b)
                    s_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(s_attention_input, self.forward_attention_a), axis=[2]))
                    s_attention_output = tf.reduce_sum(tf.multiply(self.sent_features, tf.expand_dims(s_attention_score, 2)), 1)
                    s_attention_decoder = tf.matmul(s_attention_output, self.forward_attention_to_decoder)

                    # attention conv
                    c_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_sentences, 1]))
                    c_attention_input = tf.tanh(util.tensor_matmul(self.conv_features, self.forward_attention_w_x)
                                              + util.tensor_matmul(c_tiled_decoder_state_h, self.forward_attention_w_h)
                                              + self.forward_attention_b)
                    c_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(c_attention_input, self.forward_attention_a), axis=[2]))
                    c_attention_output = tf.reduce_sum(tf.multiply(self.conv_features, tf.expand_dims(c_attention_score, 2)), 1)
                    c_attention_decoder = tf.matmul(c_attention_output, self.forward_attention_to_decoder)

                    # attention_decoder = (s_attention_decoder + c_attention_decoder)/2

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, s_attention_decoder, c_attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.forward_decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.forward_decoder_z))
                    decoder_middle = tf.concat([tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, s_attention_decoder), tf.multiply(decoder_r_t, c_attention_decoder), current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.forward_decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t, decoder_state_)

                    output = decoder_state

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    soft_logit_words = tf.nn.softmax(logit_words,axis=1)
                    max_prob_word = tf.argmax(logit_words, 1)
                    forward_answer_train.append(max_prob_word)
                    forward_distribution_train.append(soft_logit_words)

                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_forward[:,i], logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask_forward[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask_forward)
            return forward_answer_train, loss, forward_distribution_train

    def generate_entire_answer_on_training(self):
        with tf.variable_scope("entire_decoder"):
            entire_answer_train = []
            entire_distribution_train =[]
            decoder_state = self.decoder_input
            loss = 0.0

            with tf.variable_scope("decoder_lstm") as scope:
                for i in range(self.max_r_words):
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        scope.reuse_variables()
                        # next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        # current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)
                        # current_emb = tf.nn.xw_plus_b(self.ans_vec[:, i - 1, :], self.word_to_lstm_w, self.word_to_lstm_b)
                        # next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        # current_emb = next_word_vec
                        current_emb = self.ans_vec_entire[:, i - 1, :]

                    # attention sent
                    s_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_words, 1]))
                    s_attention_input = tf.tanh(util.tensor_matmul(self.sent_features, self.entire_attention_w_x)
                                              + util.tensor_matmul(s_tiled_decoder_state_h, self.entire_attention_w_h)
                                              + self.entire_attention_b)
                    s_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(s_attention_input, self.entire_attention_a), axis=[2]))
                    s_attention_output = tf.reduce_sum(tf.multiply(self.sent_features, tf.expand_dims(s_attention_score, 2)), 1)
                    s_attention_decoder = tf.matmul(s_attention_output, self.entire_attention_to_decoder)

                    # attention conv
                    c_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_sentences, 1]))
                    c_attention_input = tf.tanh(util.tensor_matmul(self.conv_features, self.entire_attention_w_x)
                                              + util.tensor_matmul(c_tiled_decoder_state_h, self.entire_attention_w_h)
                                              + self.entire_attention_b)
                    c_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(c_attention_input, self.entire_attention_a), axis=[2]))
                    c_attention_output = tf.reduce_sum(tf.multiply(self.conv_features, tf.expand_dims(c_attention_score, 2)), 1)
                    c_attention_decoder = tf.matmul(c_attention_output, self.entire_attention_to_decoder)

                    # attention_decoder = (s_attention_decoder + c_attention_decoder)/2

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, s_attention_decoder, c_attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.entire_decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.entire_decoder_z))
                    decoder_middle = tf.concat(
                        [tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, s_attention_decoder), tf.multiply(decoder_r_t, c_attention_decoder),
                         current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.entire_decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t,
                                                                                                decoder_state_)

                    output = decoder_state


                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    soft_logit_words = tf.nn.softmax(logit_words,axis=1)
                    max_prob_word = tf.argmax(logit_words, 1)
                    entire_answer_train.append(max_prob_word)
                    entire_distribution_train.append(soft_logit_words)

                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_entire[:,i], logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask_entire[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask_entire)
            return entire_answer_train, loss, entire_distribution_train

    def generate_forward_answer_on_testing(self):
        with tf.variable_scope("forward_decoder"):
            forward_answer_test = []
            forward_distribution_test = []
            decoder_state = self.decoder_input
            loss = 0.0

            with tf.variable_scope("decoder_lstm") as scope:
                for i in range(self.max_r_f_words):
                    scope.reuse_variables()
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        current_emb = next_word_vec
                        # current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)

                    # attention sent
                    s_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_words, 1]))
                    s_attention_input = tf.tanh(util.tensor_matmul(self.sent_features, self.forward_attention_w_x)
                                                + util.tensor_matmul(s_tiled_decoder_state_h, self.forward_attention_w_h)
                                                + self.forward_attention_b)
                    s_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(s_attention_input, self.forward_attention_a), axis=[2]))
                    s_attention_output = tf.reduce_sum(tf.multiply(self.sent_features, tf.expand_dims(s_attention_score, 2)), 1)
                    s_attention_decoder = tf.matmul(s_attention_output, self.forward_attention_to_decoder)

                    # attention conv
                    c_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_sentences, 1]))
                    c_attention_input = tf.tanh(util.tensor_matmul(self.conv_features, self.forward_attention_w_x)
                                                + util.tensor_matmul(c_tiled_decoder_state_h, self.forward_attention_w_h)
                                                + self.forward_attention_b)
                    c_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(c_attention_input, self.forward_attention_a), axis=[2]))
                    c_attention_output = tf.reduce_sum(tf.multiply(self.conv_features, tf.expand_dims(c_attention_score, 2)), 1)
                    c_attention_decoder = tf.matmul(c_attention_output, self.forward_attention_to_decoder)

                    # attention_decoder = (s_attention_decoder + c_attention_decoder)/2

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, s_attention_decoder, c_attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.forward_decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.forward_decoder_z))
                    decoder_middle = tf.concat([tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, s_attention_decoder), tf.multiply(decoder_r_t, c_attention_decoder), current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.forward_decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t, decoder_state_)

                    output = decoder_state


                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    soft_logit_words = tf.nn.softmax(logit_words, axis=1)
                    max_prob_word = tf.argmax(logit_words, 1)
                    forward_answer_test.append(max_prob_word)
                    forward_distribution_test.append(soft_logit_words)

                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_forward[:, i], logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask_forward[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask_forward)
            return forward_answer_test, loss, forward_distribution_test

    def generate_entire_answer_on_testing(self):
        with tf.variable_scope("entire_decoder"):
            entire_answer_test = []
            entire_distribution_test = []
            decoder_state = self.decoder_input
            loss = 0.0

            with tf.variable_scope("decoder_lstm") as scope:
                for i in range(self.max_r_words):
                    scope.reuse_variables()
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        current_emb_forward = self.ans_vec_entire[:, i - 1, :]
                        current_emb_back = next_word_vec
                        current_emb = tf.where(self.y_forward_generation[:,i-1] > 0, current_emb_forward, current_emb_back)
                        # current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)

                    # attention sent
                    s_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_words, 1]))
                    s_attention_input = tf.tanh(util.tensor_matmul(self.sent_features, self.entire_attention_w_x)
                                                + util.tensor_matmul(s_tiled_decoder_state_h, self.entire_attention_w_h)
                                                + self.entire_attention_b)
                    s_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(s_attention_input, self.entire_attention_a), axis=[2]))
                    s_attention_output = tf.reduce_sum(tf.multiply(self.sent_features, tf.expand_dims(s_attention_score, 2)), 1)
                    s_attention_decoder = tf.matmul(s_attention_output, self.entire_attention_to_decoder)

                    # attention conv
                    c_tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1), tf.stack([1, self.max_n_sentences, 1]))
                    c_attention_input = tf.tanh(util.tensor_matmul(self.conv_features, self.entire_attention_w_x)
                                                + util.tensor_matmul(c_tiled_decoder_state_h, self.entire_attention_w_h)
                                                + self.entire_attention_b)
                    c_attention_score = tf.nn.softmax(tf.squeeze(util.tensor_matmul(c_attention_input, self.entire_attention_a), axis=[2]))
                    c_attention_output = tf.reduce_sum(tf.multiply(self.conv_features, tf.expand_dims(c_attention_score, 2)), 1)
                    c_attention_decoder = tf.matmul(c_attention_output, self.entire_attention_to_decoder)

                    # attention_decoder = (s_attention_decoder + c_attention_decoder)/2


                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, s_attention_decoder, c_attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.entire_decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.entire_decoder_z))
                    decoder_middle = tf.concat(
                        [tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, s_attention_decoder), tf.multiply(decoder_r_t, c_attention_decoder),
                         current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.entire_decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t,
                                                                                                decoder_state_)

                    output = decoder_state

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    soft_logit_words = tf.nn.softmax(logit_words, axis=1)
                    max_prob_word = tf.argmax(logit_words, 1)
                    entire_answer_test.append(max_prob_word)
                    entire_distribution_test.append(soft_logit_words)

                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_entire[:, i], logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask_entire[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask_entire)
            return entire_answer_test, loss, entire_distribution_test

    def build_model(self):
        self.build_train_proc()


if __name__ == '__main__':
    config_file = '../configs/configs_HANL.json'
    with open(config_file, 'r') as fr:
        config = json.load(fr)

    model = Model(config)
    model.build_model()
    sess = tf.InteractiveSession()
    init_proc = tf.global_variables_initializer()
    sess.run(init_proc)

    encode_input = np.random.rand(3000,15,300)
    encode_sent_len = np.random.randint(5,14,size=[100*30])
    encode_conv_len = np.random.randint(5,14,size=[100])
    is_training = True
    ans_vec = np.random.rand(100,15,300)
    y = np.random.randint(5,14,size=[100,15])
    y_mask = np.random.rand(100,15)
    res,ans = sess.run([model.train_loss,model.answer_word_train], feed_dict={
        model.encode_input : encode_input,
        model.encode_sent_len : encode_sent_len,
        model.encode_conv_len : encode_conv_len,
        model.is_training:is_training,
        model.ans_vec:ans_vec,
        model.y:y,
        model.y_mask:y_mask
    })
    print(res)
    print(ans)
