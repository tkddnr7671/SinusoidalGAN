# import packages
import argparse
import tensorflow as tf
from ops import *
from utilities import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--phones', type=str, default='mono')
    parser.add_argument('--freq', type=str, default='1.0kHz')
    parser.add_argument('--snr', type=str, default='99dB')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=100000)
    args = parser.parse_args()

# real data loading
phones = args.phones
freq = args.freq
snr = args.snr
wavDir = "./database/{0:s}/{1:s}/{2:s}".format(phones, snr, freq)
WavData, nData, nLength = WaveRead(wavDir)
WavLabel = np.ones(shape=[nData, 1])
WavData = WaveNormalization(WavData)

# audio parameters
FS = 16000
FrmLeng = 512
FrmOver = int(FrmLeng * 3 / 4)
total_epochs = args.max_epoch
maxValue = 32767                              # max value of short integer(2 byte)

# transform from wave to spectrogram
SpecData, nFre, nFrm = wav2spec(WavData, FS, FrmLeng, FrmOver)

# training parameters
batch_size = args.batch_size
learning_rate = 0.000001

# generating parameters
random_dim = 128

# module 1: Generator
def generator(z):
    with tf.variable_scope(name_or_scope="G") as scope:
        # define weights for generator
        weights = {
            'gw1': tf.get_variable(name='gw1', shape=[random_dim, FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gw2': tf.get_variable(name='gw2', shape=[FrmLeng, FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gw3': tf.get_variable(name='gw3', shape=[FrmLeng, int(FrmLeng)], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gw4': tf.get_variable(name='gw4', shape=[int(FrmLeng/2), nLength], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        }
        bias = {
            'gb1': tf.get_variable(name='gb1', shape=[FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gb2': tf.get_variable(name='gb2', shape=[FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gb3': tf.get_variable(name='gb3', shape=[FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        }

        fc = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(z, weights['gw1']), bias['gb1'])))
        fc = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(fc, weights['gw2']), bias['gb2'])))
        fc = tf.cos(tf.layers.batch_normalization(tf.add(tf.matmul(fc, weights['gw3']), bias['gb3'])))

        fc1 = tf.slice(input_=fc, begin=[0, 0], size=[batch_size, int(FrmLeng/2)])
        fc2 = tf.slice(input_=fc, begin=[0, int(FrmLeng/2)], size=[batch_size, int(FrmLeng/2)])

        fc = tf.add(tf.matmul(fc1, weights['gw4']), tf.matmul(fc2, weights['gw4']))

    return tf.nn.tanh(fc)

# module 2: Discriminator
def discriminator(x, reuse=False):
    if reuse == False:
        with tf.variable_scope(name_or_scope="D") as scope:
            weights = {
                'dw1': tf.get_variable(name='dw1', shape=[17 * 4 * 16, 1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            bias = {
                'db1': tf.get_variable(name='db1', shape=[1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, 2, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 4, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 8, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 16, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
    else:
        with tf.variable_scope(name_or_scope="D", reuse=True) as scope:
            weights = {
                'dw1': tf.get_variable(name='dw1', shape=[17 * 4 * 16, 1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            bias = {
                'db1': tf.get_variable(name='db1', shape=[1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, 2, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 4, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 8, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 16, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])

    hconv = tf.reshape(hconv, shape=[-1, 17 * 4 * 16])
    output = tf.nn.sigmoid(tf.add(tf.matmul(hconv, weights['dw1']), bias['db1']))

    return output


# module 3: Random noise as an input
def random_noise(batch_size):
    return np.random.normal(size=[batch_size, random_dim]), np.zeros(shape=[batch_size, 1])

# Make a graph
g = tf.Graph()
with g.as_default():
    # input node
    X = tf.placeholder(tf.float32, [batch_size, nFre, nFrm, 1]) # for real data
    R = tf.placeholder(tf.float32, [batch_size, 1])             # for real data label
    Z = tf.placeholder(tf.float32, [batch_size, random_dim])    # for generated samples
    F = tf.placeholder(tf.float32, [batch_size, 1])             # for generated data label

    # Results in each module; G and D
    fake_x = generator(Z)
    fake_spec = tensor_stft(fake_x, FrmLeng=FrmLeng, FrmOver=FrmOver)

    result_of_fake = discriminator(fake_spec)
    result_of_real = discriminator(X, True)

    # for LSGAN: Loss function in each module: G and D
    g_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=R, predictions=result_of_fake))
    d_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=R, predictions=result_of_real)
                            + tf.losses.mean_squared_error(labels=F, predictions=result_of_fake))

    # Optimization procedure
    t_vars = tf.trainable_variables()

    gr_vars = [var for var in t_vars if "gw4" in var.name]
    g_vars = [var for var in t_vars if "G" in var.name]
    d_vars = [var for var in t_vars if "D" in var.name]
    w_vars = [var for var in t_vars if ("D" or "G") in var.name]

    # Regularization for weights
    gr_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l1_regularizer(0.5e-6),
                                                     weights_list=gr_vars)
    g_loss_reg = g_loss + gr_loss
    d_loss_reg = d_loss

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    g_train = optimizer.minimize(g_loss_reg, var_list=g_vars)
    gw_train = optimizer.minimize(g_loss_reg, var_list=gr_vars)
    d_train = optimizer.minimize(d_loss_reg, var_list=d_vars)


# Training graph g
saver = tf.train.Saver(var_list=w_vars)
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./model/LSGAN')
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join('./model/LSGAN', ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        counter = 0

    total_batchs = int(WavData.shape[0] / batch_size)

    logPath = "./result/LSGAN/GAN_result.log"
    log_fp = open(logPath, 'w')
    log = "Class: %s, nData: %d, max_epoch: %d, batch_size: %d, random_dim: %d" \
          % (phones, nData, total_epochs, batch_size, random_dim)
    print(log)
    log_fp.write(log + "\n")

    for epoch in range(counter, total_epochs):
        avg_G_loss = 0
        avg_D_loss = 0

        data_indices = np.arange(nData)
        np.random.shuffle(data_indices)
        SpecData = SpecData[data_indices]
        WavLabel = WavLabel[data_indices]
        for batch in range(total_batchs):
            batch_x = SpecData[batch*batch_size:(batch+1)*batch_size]
            batch_r = WavLabel[batch*batch_size:(batch+1)*batch_size]

            noise, nlabel = random_noise(batch_size)
            sess.run(d_train, feed_dict={X: batch_x, R: batch_r, Z: noise, F: nlabel})

            sess.run(g_train, feed_dict={X: batch_x, R: batch_r, Z: noise, F: nlabel})
            sess.run(gw_train, feed_dict={X: batch_x, R: batch_r, Z: noise, F: nlabel})
            sess.run(gw_train, feed_dict={X: batch_x, R: batch_r, Z: noise, F: nlabel})

            gl, dl = sess.run([g_loss_reg, d_loss_reg], feed_dict={X: batch_x, R: batch_r, Z: noise, F: nlabel})

            avg_G_loss += gl
            avg_D_loss += dl

        avg_G_loss /= total_batchs
        avg_D_loss /= total_batchs

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            log = "=========Epoch : %d ======================================" % (epoch + 1)
            print(log)
            log_fp.write(log + "\n")
            log = "G_loss : %.15f" % avg_G_loss
            print(log)
            log_fp.write(log + "\n")
            log = "D_loss : %.15f" % avg_D_loss
            print(log)
            log_fp.write(log + "\n")

            # Generating wave
            sample_input, _ = random_noise(batch_size)
            generated = sess.run(fake_x, feed_dict={Z: sample_input})

            # Writing the generated wave
            savePath = './wave_log/{}.wav'.format(str(epoch + 1).zfill(3))
            WriteWave(savePath, 1, 2, FS, generated[5], maxValue)
            log = "Writing generated audio to %s" % savePath
            print(log)

        if (epoch + 1) % 5000 == 0 or epoch == 0:
            # save model
            modelPath = "./model/LSGAN/{0:s}_{1:s}_{2:s}".format(phones, freq, snr)
            saver.save(sess=sess, save_path=modelPath, global_step=(epoch+1))

    # Generating wave
    sample_noise, _ = random_noise(batch_size)
    generated = sess.run(fake_x, feed_dict={Z: sample_noise})
    # Writing the generated wave
    for i in range(batch_size):
        savePath = './wave/LSGAN/{}.wav'.format(str(i).zfill(3))
        WriteWave(savePath, 1, 2, FS, generated[i], maxValue)
        print("Writing generated audio to " + savePath)

    log = "Complete Audio GAN"
    print(log)
    log_fp.write(log + "\n")
    log_fp.close()
