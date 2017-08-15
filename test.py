from dcgan import DCGAN
import tensorflow as tf

dcgan = DCGAN()
images = dcgan.sample_images()

saver = tf.train.Saver()

with tf.Session() as sess:
    # restore trained variables
    saver.restore(sess, "/home/yan/PycharmProjects/gan_lstm/save/model")

    generated = sess.run(images)
    with open('img.jpg', 'wb') as f:
        f.write(generated)