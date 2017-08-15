# -*- coding:utf-8 -*-
import tensorflow as tf
import json
from tensorflow.contrib import rnn

class Generator:

    def __init__(self, hidden_size=2000):
        # lstm's hidden layer contains 2000 neurons
        self.hidden_size = hidden_size
        self.reuse = False

    def __call__(self, imgFeatures, sentences, input_lens, wordFeatures):
        # imgFeatures : a placeholder, shape is [batch_size, dim]
        # wordFeatures : a placeholder, shape is [vocabulary_size, dim]
        # sentences : a placeholder, shape is [batch_size, max_step], content is word indices
        # input_lens : a placeholder, shape is [batch_size], content is sentence lengths(including punctuation)
        reuse = self.reuse
        # get how many examples contained in this batch
        batch_size = imgFeatures.get_shape()[0].value
        # get the lengths of sentences padded with non-sense word
        max_step = sentences.get_shape()[1].value
        # get the feature vector's dimension
        dim = imgFeatures.get_shape()[1].value
        # get the number of words in the vocabulary
        vocabulary_size = wordFeatures.get_shape()[0].value

        with tf.variable_scope('G_RNN', reuse=reuse) as scope:
            # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)

            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)

            # **步骤5：用全零来初始化state
            init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

            _, init_state = lstm_cell(imgFeatures, init_state)

            inputs = tf.nn.embedding_lookup(wordFeatures, sentences)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                tf.get_variable_scope().reuse_variables()
                outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=inputs, sequence_length=input_lens,
                                           initial_state=init_state, time_major=False, scope=scope)

            outputs = tf.reshape(outputs[:, :-1, :], [batch_size * (max_step - 1), self.hidden_size])
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size, vocabulary_size], tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable("softmax_b", [vocabulary_size], tf.float32,
                                        initializer=tf.constant_initializer(0.0))
            weights = tf.nn.softmax(tf.matmul(outputs, softmax_w) + softmax_b)
            wordVector = tf.matmul(weights, wordFeatures)
            wordVector = tf.reshape(wordVector, [batch_size, max_step - 1, dim])
            generated_data = tf.concat(values=[tf.expand_dims(imgFeatures, 1), tf.expand_dims(inputs[:, 0, :],1), wordVector], axis=1)
            real_data = tf.concat([tf.expand_dims(imgFeatures, 1), inputs], axis=1)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_RNN')
        return generated_data, real_data


    #def __call__(self, imgFeatures, sentences, input_lens, wordFeatures):
        # imgFeatures : a placeholder, shape is [batch_size, dim]
        # wordFeatures : a placeholder, shape is [vocabulary_size, dim]
        # sentences : a placeholder, shape is [batch_size, max_step], content is word indices
        # input_lens : a placeholder, shape is [batch_size], content is sentence lengths(including punctuation)

        # get how many examples contained in this batch
    #    batch_size = imgFeatures.get_shape()[0].value
        # get the lengths of sentences padded with non-sense word
    #    max_step = sentences.get_shape()[1].value
        # get the feature vector's dimension
    #    dim = imgFeatures.get_shape()[1].value
        # get the number of words in the vocabulary
    #    vocabulary_size = wordFeatures.get_shape()[0].value

    #    with tf.variable_scope('G_RNN', reuse=self.reuse) as scope:

            # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            #lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=self.reuse)

            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            #lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

            # **步骤5：用全零来初始化state
            #init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

            #outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=X, initial_state=init_state, time_major=False)
            #h_state = outputs[:, -1, :]
            #return h_state

    #        lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=self.reuse)
    #        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    #        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    #        (_, init_state) = lstm_cell(imgFeatures, init_state)
    #        inputs = tf.nn.embedding_lookup(wordFeatures, sentences)
    #        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=inputs,sequence_length=input_lens, initial_state=init_state, time_major=False)
    #        outputs = tf.reshape(outputs[:,:-1,:], [batch_size*(max_step-1), dim])
    #        softmax_w = tf.get_variable('softmax_w', [self.hidden_size, vocabulary_size], tf.float32,
    #                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    #        softmax_b = tf.get_variable("softmax_b", [vocabulary_size], tf.float32, initializer=tf.constant_initializer(0.0))
    #        weights = tf.nn.softmax(tf.matmul(outputs, softmax_w) + softmax_b)
    #        wordVector = tf.matmul(weights, wordFeatures)
    #        wordVector = tf.reshape(wordVector, [batch_size, max_step-1, dim])
    #        generated_data = tf.concat(values=[tf.stack(imgFeatures, inputs[:,0,:]) ,wordVector], axis=1)
    #        real_data = tf.concat([tf.expand_dims(imgFeatures, 1),inputs])
    #    self.reuse = True
    #    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_RNN')
    #    return generated_data, real_data

    def generate(self, imgFeature, wordFeatures, vocabulary, sess):
        # imgFeature is a placeholder, it's shape is [dim]
        # wordFeatures is a placeholder, it's shape is [vocabulary_size, dim]
        # vocabulary is a placeholder, it's shape is [vocabulary_size]
        # sess is used to get a corresponding word

        imgFeature = tf.expand_dims(imgFeature, 0)
        wordFeature = tf.expand_dims(wordFeatures[0,:], 0)
        vocabulary_size = wordFeatures.get_shape()[0].value

        sentence = ['START']
        timestep_size = 20

        with tf.variable_scope('G_RNN', reuse=self.reuse) as scope:

            lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(1, dtype=tf.float32)
            (_, state) = lstm_cell(imgFeature, init_state)

            for timestep in range(timestep_size):
                if sentence[-1] == 'END':
                    break
                else:
                    (cell_output, state) = lstm_cell(wordFeature, state)
                    softmax_w = tf.get_variable('softmax_w', [self.hidden_size, vocabulary_size], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
                    softmax_b = tf.get_variable("softmax_b", [vocabulary_size], tf.float32, initializer=tf.constant_initializer(0.0))
                    weights = tf.nn.softmax(tf.matmul(cell_output, softmax_w) + softmax_b)
                    wordVector = tf.matmul(weights, wordFeatures)
                    index = sess.run(tf.argmax(weights, 1)[0])
                    sentence.append(vocabulary[index])
        return sentence


class Discriminator:
    def __init__(self, hidden_size=2000):

        self.hidden_size = hidden_size
        self.reuse = False

    def __call__(self, inputs, input_lens, name=''):
        reuse = self.reuse
        inputs = tf.convert_to_tensor(inputs)
        batch_size = inputs.get_shape()[0].value

        with tf.name_scope('D_RNN' + name), tf.variable_scope('D_RNN', reuse=reuse) as scope:
            lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)
            init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            (_, init_state) = lstm_cell(inputs[:, 0, :], init_state)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                tf.get_variable_scope().reuse_variables()
                (_, init_state) = lstm_cell(inputs[:, 1, :], init_state)
                outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=inputs[:, 2:, :], sequence_length=input_lens,
                                               initial_state=init_state, time_major=False, scope=scope)

            outputs = outputs[:, -1, :]


            with tf.variable_scope('classify'):
                outputs = tf.layers.dense(outputs, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_RNN')
        return outputs


class DCGAN:
    def __init__(self,g_hidden_size=2000,d_hidden_size=2000):

        self.g = Generator(g_hidden_size)
        self.d = Discriminator(d_hidden_size)


    def loss(self, imgFeatures, idxSentences, input_lens, wordFeatures):

        batch_size = tf.convert_to_tensor(imgFeatures).get_shape()[0].value

        generated, real = self.g(imgFeatures, idxSentences,
                                 input_lens, wordFeatures)
        g_outputs = self.d(generated, input_lens, name='g')

        t_outputs = self.d(real, input_lens, name='t')
        # add each losses to collection
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([batch_size], dtype=tf.int64),
                    logits=t_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        return {
            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }

    def train(self, losses, learning_rate=0.0002, beta1=0.5):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    def sample_images(self, imgNameList, imgFeatures, wordFeatures, vocabulary, sess):
        # imgNameList : a list of image names
        # imgFeatures : a dictionary of image features
        # wordFeatures : a list of word features
        # vocabulary : a list of wordss
        sentences = {}

        for imgName in imgNameList:
            imgFeature = imgFeatures[imgName]
            sentence = self.g.generate(imgFeature, wordFeatures, vocabulary, sess)
            sentences[imgName] = sentence

        with open("generated_sentences.json", 'w') as f:
            json.dump(sentences, f)