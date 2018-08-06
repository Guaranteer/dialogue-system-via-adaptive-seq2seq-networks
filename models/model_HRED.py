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

        self.ref_dim = params['ref_dim']
        self.word_lstm_dim = params['word_lstm_dim']
        self.lstm_dim = params['lstm_dim']
        self.second_lstm_dim = params['second_lstm_dim']
        self.attention_dim = params['attention_dim']
        self.decode_dim = params['decode_dim']
        self.input_dim = params['input_dim']

        self.regularization_beta = params['regularization_beta']
        self.dropout_prob = params['dropout_prob']



    def build_train_proc(self):
        # input layer (batch_size, n_steps, input_dim)
        self.encode_input = tf.placeholder(tf.float32, [None, self.max_n_words, self.input_dim])
        self.encode_sent_len = tf.placeholder(tf.int32, [None])
        self.encode_conv_len = tf.placeholder(tf.int32, [self.batch_size])
        self.decode_input = tf.placeholder(tf.float32, [None, self.max_r_words, self.input_dim])
        self.decode_sent_len = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.reward = tf.placeholder(tf.float32, [None])
        self.ans_vec = tf.placeholder(tf.float32, [None, self.max_r_words, self.input_dim])
        self.y = tf.placeholder(tf.int32, [None, self.max_r_words])
        self.y_mask = tf.placeholder(tf.float32, [None, self.max_r_words])

        self.encode_input = tf.contrib.layers.dropout(self.encode_input, self.dropout_prob, is_training=self.is_training)
        self.decode_input = tf.contrib.layers.dropout(self.decode_input, self.dropout_prob, is_training=self.is_training)

        sent_outputs, sent_state = layers.dynamic_origin_bilstm_layer(self.encode_input, self.word_lstm_dim, scope_name = 'sent_level_bilstm_rnn', input_len=self.encode_sent_len)
        sent_last_state = tf.concat([sent_state[0][1],sent_state[1][1]],axis=1)
        # sent_outputs = tf.reshape(sent_outputs, shape=[self.batch_size, self.max_n_sentences, self.max_n_words, self.lstm_dim])
        # ind = tf.stack([tf.range(self.batch_size), self.encode_conv_len - 1], axis=1)
        # sent_last_outputs = tf.gather_nd(sent_outputs,indices=ind)

        conv_sents = tf.reshape(sent_last_state,shape = [self.batch_size, self.max_n_sentences, self.lstm_dim])
        # self.sent_last_state_trun = tf.gather_nd(conv_sents, indices=ind)
        conv_outputs, conv_state = layers.dynamic_origin_lstm_layer(conv_sents, self.lstm_dim, 'conv_level_rnn', input_len=self.encode_conv_len)
        self.conv_last_state = conv_state[1]


        # self.sent_features = sent_last_outputs
        # self.conv_features = conv_outputs

        # self.sent_features = tf.contrib.layers.dropout(self.sent_features, self.dropout_prob, is_training=self.is_training)
        # self.conv_features = tf.contrib.layers.dropout(self.conv_features, self.dropout_prob, is_training=self.is_training)


        # decoder


        with tf.variable_scope('linear'):
            decoder_input_W = tf.get_variable('sw', shape=[self.conv_last_state.shape[1], self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())
            decoder_input_b = tf.get_variable('sb', shape=[self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())

            self.decoder_input = tf.matmul(self.conv_last_state, decoder_input_W) + decoder_input_b


        # answer->word predict
        self.embed_word_W = tf.get_variable('embed_word_W', shape=[self.decode_dim, self.n_words], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.embed_word_b = tf.get_variable('embed_word_b', shape=[self.n_words], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


        # # word dim -> decode_dim
        # self.word_to_lstm_w = tf.get_variable('word_to_lstm_W', shape=[self.input_dim, self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # self.word_to_lstm_b = tf.get_variable('word_to_lstm_b', shape=[self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        # decoder
        with tf.variable_scope('decoder'):
            self.decoder_r = tf.get_variable('decoder_r', shape=[self.decode_dim * 2, self.decode_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_z = tf.get_variable('decoder_z', shape=[self.decode_dim * 2, self.decode_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_w = tf.get_variable('decoder_w', shape=[self.decode_dim * 2, self.decode_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())

        # embedding layer
        embedding = load_file(self.params['embedding'])
        self.Wemb = tf.constant(embedding, dtype=tf.float32)

        # generate training
        answer_train, train_loss, distribution_train = self.generate_answer_on_training()
        answer_test, test_loss, distribution_test = self.generate_answer_on_testing()

        # final
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        self.answer_word_train = answer_train
        self.train_loss = train_loss + self.regularization_beta * regularization_cost
        self.distribution_word_train = distribution_train

        self.answer_word_test = answer_test
        self.test_loss = test_loss + self.regularization_beta * regularization_cost
        self.distribution_word_test = distribution_test

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rates = tf.train.exponential_decay(self.params['learning_rate'], self.global_step, decay_steps=self.params['lr_decay_n_iters'],
                                                    decay_rate=self.params['lr_decay_rate'], staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rates)
        self.train_proc = optimizer.minimize(self.train_loss, global_step=self.global_step)


    def generate_answer_on_training(self):
        with tf.variable_scope("decoder"):
            answer_train = []
            distribution_train =[]
            decoder_state = self.decoder_input
            loss = 0.0

            with tf.variable_scope("decoder_lstm") as scope:
                for i in range(self.max_r_words):
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        scope.reuse_variables()
                        current_emb = self.ans_vec[:, i - 1, :]


                    # decoder : GRU
                    decoder_input = tf.concat([decoder_state, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_z))
                    decoder_middle = tf.concat([tf.multiply(decoder_r_t, decoder_state), current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t, decoder_state_)

                    output = decoder_state

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    soft_logit_words = tf.nn.softmax(logit_words,axis=1)
                    max_prob_word = tf.argmax(logit_words, 1)
                    answer_train.append(max_prob_word)
                    distribution_train.append(soft_logit_words)

                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[:,i], logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask)
            return answer_train, loss, distribution_train

    def generate_answer_on_testing(self):
        with tf.variable_scope("decoder"):
            answer_test = []
            distribution_test = []
            decoder_state = self.decoder_input
            loss = 0.0

            with tf.variable_scope("decoder_lstm") as scope:
                for i in range(self.max_r_words):
                    scope.reuse_variables()
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        current_emb = next_word_vec


                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_z))
                    decoder_middle = tf.concat([tf.multiply(decoder_r_t, decoder_state), current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t, decoder_state_)

                    output = decoder_state

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    soft_logit_words = tf.nn.softmax(logit_words, axis=1)
                    max_prob_word = tf.argmax(logit_words, 1)
                    answer_test.append(max_prob_word)
                    distribution_test.append(soft_logit_words)

                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[:, i], logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask)
            return answer_test, loss, distribution_test

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
