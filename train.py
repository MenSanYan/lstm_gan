from dcgan import DCGAN
import tensorflow as tf
import json
import math
import numpy as np



MAX_EPOCH = 1
BATCH_SIZE = 100


with open('/home/yanzehang/PycharmProjects/2017.8.13/gan_lstm/data/dataset','r') as f:
    data = json.load(f)
print "-------loading data finished--------"
def getimgFeatures(permut, setp):
    return np.array(data['imgFeatures'])[permut[step*BATCH_SIZE:(step+1)*BATCH_SIZE]]
def getidxSentences(permut, setp):
    return np.array(data['idxSentences'])[permut[step*BATCH_SIZE:(step+1)*BATCH_SIZE]]
def getinput_lens(permut, setp):
    return np.array(data['input_lens'])[permut[step*BATCH_SIZE:(step+1)*BATCH_SIZE]]
def getwordFeatures():
    return np.array(data['wordFeatures'])

EXAMPLE_NUM = len(data['imgNames'])
FEATURE_DIM = len(data['imgFeatures'][0])
MAX_TIME_STEP = len(data['idxSentences'][0])
VOCABULARY_SIZE = len(data['vocabulary'])
BATCH_NUM = int(math.ceil(EXAMPLE_NUM / float(100)))

imgFeatures = tf.placeholder(tf.float32, [BATCH_SIZE, FEATURE_DIM])
idxSentences = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_TIME_STEP])
input_lens = tf.placeholder(tf.int32, [BATCH_SIZE])
wordFeatures = tf.placeholder(tf.float32, [VOCABULARY_SIZE, FEATURE_DIM])

dcgan = DCGAN()

losses = dcgan.loss(imgFeatures, idxSentences, input_lens, wordFeatures)
train_op = dcgan.train(losses)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    permut = np.array(range(EXAMPLE_NUM))
    for epoch in range(MAX_EPOCH):
        np.random.shuffle(permut)

        for step in range(BATCH_NUM):

            _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]],
                                                     feed_dict={imgFeatures: getimgFeatures(permut, data),
                                                                idxSentences: getidxSentences(permut, data),
                                                                input_lens: getinput_lens(permut, data),
                                                                wordFeatures: getwordFeatures()})
            print str(step + 1) + '/' + str(BATCH_NUM) + '\t',
            print 'g loss: ' + str(g_loss_value) + '\t',
            print 'd loss: ' + str(d_loss_value)


    # save trained variables
    saver.save(sess, "/home/yan/PycharmProjects/gan_lstm/save/model")