import sys
sys.path.append('../utils/')
import tensorflow as tf
import random
import pickle
import numpy as np
import util
import layers
import json
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

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
        self.ans_vec = tf.placeholder(tf.float32, [None, self.max_r_words-1, self.input_dim])
        self.y = tf.placeholder(tf.int32, [None, self.max_r_words])
        self.y_mask = tf.placeholder(tf.float32, [None, self.max_r_words])
        # self.batch_size = tf.placeholder(tf.int32, [])

        # self.encode_input = tf.contrib.layers.dropout(self.encode_input, self.dropout_prob, is_training=self.is_training)
        # self.decode_input = tf.contrib.layers.dropout(self.decode_input, self.dropout_prob, is_training=self.is_training)

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

        self.sent_outputs = sent_outputs
        self.ind = ind
        self.sent_features = sent_last_outputs
        self.conv_features = conv_outputs





        # decoder

        self.decode_cell = tf.contrib.rnn.BasicLSTMCell(self.decode_dim, state_is_tuple=True)

        with tf.variable_scope('linear'):
            sent_and_conv_last = tf.concat([self.sent_last_state_trun,self.conv_last_state],axis=1)
            decoder_input_W = tf.get_variable('sw', shape=[self.sent_last_state_trun.shape[1] + self.conv_last_state.shape[1], self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())
            decoder_input_b = tf.get_variable('sb', shape=[self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())

            self.decoder_input = tf.matmul(sent_and_conv_last, decoder_input_W) + decoder_input_b


        # answer->word predict
        self.embed_word_W = tf.get_variable('embed_word_W', shape=[self.decode_dim, self.n_words], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.embed_word_b = tf.get_variable('embed_word_b', shape=[self.n_words], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


        # word dim -> decode_dim
        self.word_to_lstm_w = tf.get_variable('word_to_lstm_W', shape=[self.input_dim, self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.word_to_lstm_b = tf.get_variable('word_to_lstm_b', shape=[self.decode_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


        # decoder attention layer
        with tf.variable_scope('decoder_attention'):
            self.attention_w_x = tf.get_variable('attention_w_x', shape=[self.lstm_dim, self.attention_dim], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.attention_w_h = tf.get_variable('attention_w_h', shape=[self.decode_dim, self.attention_dim], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.attention_b = tf.get_variable('attention_b', shape=[self.attention_dim], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.attention_a = tf.get_variable('attention_a', shape=[self.attention_dim, 1], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.attention_to_decoder = tf.get_variable('attention_to_decoder', shape=[self.lstm_dim, self.decode_dim], dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer())
        # decoder
        with tf.variable_scope('decoder'):
            self.decoder_r = tf.get_variable('decoder_r', shape=[self.decode_dim * 3, self.decode_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_z = tf.get_variable('decoder_z', shape=[self.decode_dim * 3, self.decode_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_w = tf.get_variable('decoder_w', shape=[self.decode_dim * 3, self.decode_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())

        # embedding layer
        embedding = load_file(self.params['embedding'])
        self.Wemb = tf.constant(embedding, dtype=tf.float32)

        # generate training
        answer_train, train_loss = self.generate_answer_on_training()
        answer_test, test_loss = self.generate_answer_on_testing()

        # final
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        self.answer_word_train = answer_train
        self.train_loss = train_loss + self.regularization_beta * regularization_cost

        self.answer_word_test = answer_test
        self.test_loss = test_loss + self.regularization_beta * regularization_cost


        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rates = tf.train.exponential_decay(self.params['learning_rate'], self.global_step, decay_steps=self.params['lr_decay_n_iters'],
                                                    decay_rate=self.params['lr_decay_rate'], staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rates)
        self.train_proc = optimizer.minimize(self.train_loss, global_step=self.global_step)

        # tf.summary.scalar('global_step', self.global_step)
        # tf.summary.scalar('training cross entropy', self.train_loss)
        # tf.summary.scalar('test cross entropy', self.test_loss)
        # self.summary_proc = tf.summary.merge_all()

    def generate_answer_on_training(self):
        with tf.variable_scope("decoder") as scope:
            decoder_state = self.decoder_input # self.decoder_cell.zero_state(self.batch_size, tf.float32)

            ans_start = tf.expand_dims(self.decoder_input, axis=1)
            sliced_decode_inputs = tf.concat([ans_start,self.ans_vec],axis=1)
            sliced_decode_inputs = tf.unstack(sliced_decode_inputs, self.max_r_words, axis=1)
            decode_output, decode_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=sliced_decode_inputs,
                                                                                      initial_state=decoder_state,
                                                                                      attention_states=self.sent_features,
                                                                                      cell=self.decode_cell,
                                                                                      output_size=None,
                                                                                      num_heads=1,
                                                                                      loop_function=None,
                                                                                      dtype=tf.float32,
                                                                                      scope=None,
                                                                                      initial_state_attention=False)
            decode_output = tf.stack(values=decode_output, axis=1)
            logit_words = util.tensor_matmul(decode_output, self.embed_word_W) + self.embed_word_b
            answer_train = tf.argmax(logit_words,axis=2)

            loss = tf.contrib.seq2seq.sequence_loss(logits=logit_words,
                                                         targets=self.y,
                                                         weights=self.y_mask,
                                                         average_across_timesteps=True,
                                                         average_across_batch=True,
                                                         softmax_loss_function=None,
                                                         name=None)
            return answer_train, loss

    def generate_answer_on_testing(self):
        with tf.variable_scope("decoder") as scope:
            scope.reuse_variables()
            decoder_state = self.decoder_input # self.decoder_cell.zero_state(self.batch_size, tf.float32)

            def _extract_argmax_and_embed(embedding,
                                          output_projection=None,
                                          update_embedding=True):

                def loop_function(prev, _):
                    if output_projection is not None:
                        prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
                    prev_symbol = math_ops.argmax(prev, 1)
                    # Note that gradients will not propagate through the second parameter of
                    # embedding_lookup.
                    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
                    if not update_embedding:
                        emb_prev = array_ops.stop_gradient(emb_prev)
                    return emb_prev

                return loop_function

            self.loop_f = _extract_argmax_and_embed(self.Wemb, output_projection=(self.word_to_lstm_w, self.word_to_lstm_b), update_embedding=False)
            ans_start = tf.expand_dims(self.decoder_input, axis=1)
            sliced_decode_inputs = tf.concat([ans_start,self.ans_vec],axis=1)
            sliced_decode_inputs = tf.unstack(sliced_decode_inputs, self.max_r_words, axis=1)
            decode_output, decode_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=sliced_decode_inputs,
                                                                                      initial_state=decoder_state,
                                                                                      attention_states=self.sent_features,
                                                                                      cell=self.decode_cell,
                                                                                      output_size=None,
                                                                                      num_heads=1,
                                                                                      loop_function=self.loop_f,
                                                                                      dtype=tf.float32,
                                                                                      scope=None,
                                                                                      initial_state_attention=False)
            decode_output = tf.stack(values=decode_output, axis=1)
            logit_words = util.tensor_matmul(decode_output, self.embed_word_W) + self.embed_word_b
            answer_test = tf.argmax(logit_words,axis=2)

            loss = tf.contrib.seq2seq.sequence_loss(logits=logit_words,
                                                         targets=self.y,
                                                         weights=self.y_mask,
                                                         average_across_timesteps=True,
                                                         average_across_batch=True,
                                                         softmax_loss_function=None,
                                                         name=None)

            return answer_test, loss

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