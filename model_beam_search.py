import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
import utils
import numpy as np

class Model(object):

    def __init__(self, data_params, embedding):
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





        dec_cell = tf.contrib.rnn.BasicLSTMCell( self.lstm_dim )
        beam_cel = tf.contrib.rnn.BasicLSTMCell( self.lstm_dim )
        out_layer = tf.layers.Dense(self.vacab_size)
        beam_width = 5

        with tf.variable_scope("Decoder"):
                    
            attn_mech = tf.contrib.seq2seq.BahdanauAttention( num_units = self.lstm_dim,  memory = first_lstm_output, normalize=True)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper( cell = dec_cell,attention_mechanism = attn_mech ) 
        
            batch_size = tf.shape(first_lstm_output)[0]
            initial_state = attn_cell.zero_state( batch_size = batch_size , dtype=tf.float32 )
            initial_state = initial_state.clone(cell_state = state)
        
            helper = tf.contrib.seq2seq.TrainingHelper( inputs = decode_input , sequence_length = self.mask )
            decoder = tf.contrib.seq2seq.BasicDecoder( cell = attn_cell, helper = helper, initial_state = initial_state ,output_layer=out_layer ) 
            outputs, final_state, final_sequence_lengths= tf.contrib.seq2seq.dynamic_decode(decoder=decoder,impute_finished=True)

        
            training_logits = tf.identity(outputs.rnn_output )
            self.training_pred = tf.identity(outputs.sample_id )


            print(training_logits.shape)


        # for var in tf.trainable_variables():
        #     print(var, var.shape)

        with tf.variable_scope("Decoder" , reuse=True):
            enc_rnn_out_beam   = tf.contrib.seq2seq.tile_batch( first_lstm_output, beam_width )
            seq_len_beam       = tf.contrib.seq2seq.tile_batch( self.mask, beam_width )
            enc_rnn_state_beam = tf.contrib.seq2seq.tile_batch( state, beam_width )
        
            batch_size_beam      = tf.shape(enc_rnn_out_beam)[0]   # now batch size is beam_width times
        
            # start tokens mean be the original batch size so divide
            start_tokens = tf.tile(tf.constant([20000], dtype=tf.int32), [ batch_size_beam//beam_width ] )
            end_token = 20001
        
            attn_mech_beam = tf.contrib.seq2seq.BahdanauAttention( num_units = self.lstm_dim,  memory = enc_rnn_out_beam, normalize=True)
            cell_beam = tf.contrib.seq2seq.AttentionWrapper(cell=beam_cel, attention_mechanism=attn_mech_beam)  
        
            initial_state_beam = cell_beam.zero_state(batch_size=batch_size_beam,dtype=tf.float32).clone(cell_state=enc_rnn_state_beam)
        
            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = cell_beam,
                                                               embedding = embedding,
                                                               start_tokens = start_tokens,
                                                               end_token = end_token,
                                                               initial_state = initial_state_beam,
                                                               beam_width = beam_width,
                                                               output_layer=out_layer)
        
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=self.max_n_response)
            self.beam_output = output.predicted_ids

                



        seq_mask = tf.sequence_mask(self.mask, self.max_n_response, dtype=tf.float32)
        self.seq_mask = tf.multiply(seq_mask,tf.expand_dims(self.reward,1))

        self.variables = tf.trainable_variables()
        for var in self.variables:
            print(var,var.shape)
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in self.variables])
        main_loss = tf.contrib.seq2seq.sequence_loss(logits=training_logits,
                                                     targets=self.target,
                                                     weights=self.seq_mask,
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






