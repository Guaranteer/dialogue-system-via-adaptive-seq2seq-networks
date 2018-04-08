import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
import utils
import numpy as np

class Model(object):

    def __init__(self, data_params, embedding, forward_only = False):
        self.data_params = data_params
        self.batch_size = data_params['batch_size']
        self.vacab_size = data_params['vacab_size']

        self.word_dim = data_params['word_dim']
        self.max_n_context = data_params['max_n_context']
        self.max_n_response = data_params['max_n_response']
        self.ref_dim = data_params['ref_dim']
        self.lstm_dim = data_params['lstm_dim']
        self.attention_dim = data_params['attention_dim']

        self.regularization_beta = data_params['regularization_beta']
        self.dropout_prob = data_params['dropout_prob']
        self.embedding = embedding

        self.forward_only = forward_only

        self.encode_input = tf.placeholder(tf.float32, [None, self.max_n_context, self.word_dim])
        self.decode_input = tf.placeholder(tf.float32, [None, self.max_n_response, self.word_dim])
        self.target = tf.placeholder(tf.int32, [None, self.max_n_response])
        self.mask = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.reward = tf.placeholder(tf.float32, [None])

        with tf.variable_scope("ref_var"):
            self.Wsi = tf.get_variable(initializer=tf.truncated_normal(shape=[self.word_dim, self.ref_dim], stddev=5e-2), dtype=tf.float32, name='Wsi')
            self.Wsh = tf.get_variable(initializer=tf.truncated_normal(shape=[self.lstm_dim, self.ref_dim], stddev=5e-2), dtype=tf.float32, name='Wsh')
            self.Wsq = tf.get_variable(initializer=tf.truncated_normal(shape=[self.lstm_dim, self.ref_dim], stddev=5e-2), dtype=tf.float32, name='Wsq')
            self.bias = tf.get_variable(initializer=tf.truncated_normal(shape=[self.ref_dim], stddev=5e-2), dtype=tf.float32, name='bias')
            self.Vs = tf.get_variable(initializer=tf.truncated_normal(shape=[self.ref_dim, 1], stddev=5e-2), dtype=tf.float32, name='Vs')

        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("word_embeddings", initializer=self.embedding.astype(np.float32), trainable=False)

        encode_input = tf.contrib.layers.dropout(self.encode_input, self.dropout_prob, is_training=self.is_training)
        decode_input = tf.contrib.layers.dropout(self.decode_input, self.dropout_prob, is_training=self.is_training)

        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, forget_bias=0.0, state_is_tuple=True)
        state = cell.zero_state(self.batch_size, tf.float32)

        cell_second = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, forget_bias=0.0, state_is_tuple=True)
        state_second = cell_second.zero_state(self.batch_size, tf.float32)

        cell_output_second = tf.zeros([self.batch_size, self.lstm_dim])
        first_outputs = []
        second_outputs = []

        with tf.variable_scope("Adaptive_RNN"):
            for time_step in range(self.max_n_context):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                ref = tf.matmul(state[1], self.Wsh) + tf.matmul(encode_input[:, time_step, :], self.Wsi) + self.bias
                condition = tf.sigmoid(tf.matmul(ref, self.Vs))
                prod = tf.squeeze(condition, 1) > 0.3

                state = (tf.where(prod, state[0], tf.zeros_like(state[0])), tf.where(prod, state[1], tf.zeros_like(state[1])))
                # state = (condition * state[0], condition * state[1])
                (cell_output, state) = cell(encode_input[:, time_step, :], state)

                with tf.variable_scope("second_layer"):
                    (cell_output_second_tmp, state_second_tmp) = cell_second(cell_output, state_second)
                cell_output_second = tf.where(prod, cell_output_second, cell_output_second_tmp)
                state_second = (tf.where(prod, state_second[0], state_second_tmp[0]),
                                tf.where(prod, state_second[1], state_second_tmp[1]))

                first_outputs.append(cell_output)
                second_outputs.append(cell_output_second)
                # att_outputs.append((1-condition) * input_x[:, time_step, :])

        first_lstm_output = tf.reshape(tf.concat(first_outputs, 1), [self.batch_size, -1, self.lstm_dim])

        def _extract_argmax_and_embed(embedding,
                                      output_projection=None,
                                      update_embedding=True):
            """Get a loop_function that extracts the previous symbol and embeds it.
            Args:
              embedding: embedding tensor for symbols.
              output_projection: None or a pair (W, B). If provided, each fed previous
                output will first be multiplied by W and added B.
              update_embedding: Boolean; if False, the gradients will not propagate
                through the embeddings.
            Returns:
              A loop function.
            """

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

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', [self.lstm_dim, self.vacab_size], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [self.vacab_size], dtype=tf.float32)

        if self.forward_only:
            self.loop_f = _extract_argmax_and_embed(embedding, output_projection=(softmax_w, softmax_b),
                                                    update_embedding=False)
        else:
            self.loop_f = None

        print(self.loop_f)

        sliced_decode_inputs = tf.unstack(value=decode_input, axis=1)
        decode_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
        decode_output, decode_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=sliced_decode_inputs,
                                                                                  initial_state=state,
                                                                                  attention_states=first_lstm_output,
                                                                                  cell=decode_cell,
                                                                                  output_size=None,
                                                                                  num_heads=1,
                                                                                  loop_function=self.loop_f,
                                                                                  dtype=tf.float32,
                                                                                  scope=None,
                                                                                  initial_state_attention=False)

        decode_output = tf.stack(values=decode_output, axis=1)
        self.outputs = utils.tensor_matmul(decode_output, softmax_w) + softmax_b
        self.values, self.indices = tf.nn.top_k(self.outputs, k=1, sorted=True, name='top_k')

        seq_mask = tf.sequence_mask(self.mask, self.max_n_response, dtype=tf.float32)
        seq_mask = tf.multiply(seq_mask,tf.expand_dims(self.reward,1))

        self.variables = tf.trainable_variables()
        print(self.variables)
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in self.variables])
        main_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                     targets=self.target,
                                                     weights=seq_mask,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True,
                                                     softmax_loss_function=None,
                                                     name=None)

        self.loss = main_loss + self.regularization_beta * regularization_cost
        tf.summary.scalar('cross entropy', self.loss)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        tf.summary.scalar('global_step', self.global_step)
        learning_rates = tf.train.exponential_decay(self.data_params['learning_rate'], self.global_step,
                                                    decay_steps=self.data_params['lr_decay_n_iters'],
                                                    decay_rate=self.data_params['lr_decay_rate'], staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rates)
        self.train_proc = optimizer.minimize(self.loss, global_step=self.global_step)

        self.summary_proc = tf.summary.merge_all()





