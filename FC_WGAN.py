# import packages
import argparse
from ops import *
from utilities import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--phones', type=str, default='mono')
    parser.add_argument('--freq', type=str, default='1.0kHz')
    parser.add_argument('--snr', type=str, default='99dB')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=200000)
    args = parser.parse_args()


# real data loading
phones = args.phones
freq = args.freq
snr = args.snr
wavDir = "./database/{0:s}/{1:s}/{2:s}".format(phones, snr, freq)
WavData, nData, nLength = WaveRead(wavDir)
WavData = WaveNormalization(WavData)

# audio parameters
FS = 16000
FrmLeng = 512
total_epochs = args.max_epoch
maxValue = 32767                              # max value of short integer(2 byte)

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
            'w1': tf.get_variable(name='w1', shape=[random_dim, FrmLeng], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'w2': tf.get_variable(name='w2', shape=[FrmLeng, FrmLeng], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'w3': tf.get_variable(name='w3', shape=[FrmLeng, int(FrmLeng)], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'w4': tf.get_variable(name='w4', shape=[int(FrmLeng/2), nLength], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        }
        bias = {
            'b1': tf.get_variable(name='b1', shape=[FrmLeng], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'b2': tf.get_variable(name='b2', shape=[FrmLeng], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'b3': tf.get_variable(name='b3', shape=[FrmLeng], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'b4': tf.get_variable(name='b4', shape=[nLength], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        }

        fc = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(z, weights['w1']), bias['b1'])))
        fc = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(fc, weights['w2']), bias['b2'])))
        fc = tf.cos(tf.layers.batch_normalization(tf.add(tf.matmul(fc, weights['w3']), bias['b3'])))

        fc1 = tf.slice(input_=fc, begin=[0, 0], size=[batch_size, int(FrmLeng/2)])
        fc2 = tf.slice(input_=fc, begin=[0, int(FrmLeng/2)], size=[batch_size, int(FrmLeng/2)])

        fc = tf.add(tf.matmul(fc1, weights['w4']), tf.matmul(fc2, weights['w4']))
        fc = tf.nn.tanh(tf.layers.batch_normalization(tf.add(fc, bias['b4'])))

    return fc

# module 2: Discriminator
def discriminator(x, reuse=False):
    if reuse == False:
        with tf.variable_scope(name_or_scope="D") as scope:
            fc = x
            fc = tf.contrib.layers.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.relu
            )
            fc = tf.contrib.layers.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.relu
            )
            fc = tf.contrib.layers.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.relu
            )
            fc = tf.contrib.layers.fully_connected(
                fc, 1,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.sigmoid
            )
    else:
        with tf.variable_scope(name_or_scope="D", reuse=True) as scope:
            fc = x
            fc = tf.contrib.layers.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.relu
            )
            fc = tf.contrib.layers.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.relu
            )
            fc = tf.contrib.layers.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.relu
            )
            fc = tf.contrib.layers.fully_connected(
                fc, 1,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                activation_fn=tf.nn.sigmoid
            )

    return fc

# module 3: Random noise as an input
def random_noise(batch_size):
    return np.random.normal(size=[batch_size, random_dim]), np.zeros(shape=[batch_size, 1])

# Make a graph
g = tf.Graph()
with g.as_default():
    # input node
    X = tf.placeholder(tf.float32, [batch_size, nLength])       # for real data
    Z = tf.placeholder(tf.float32, [batch_size, random_dim])    # for generated samples

    # Results in each module; G and D
    fake_x = generator(Z)

    # Probability in discriminator
    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X, True)

    # for WGAN: Loss function in each module: G and D => it must be maximize
    g_loss = tf.reduce_mean(result_of_fake)
    d_loss = tf.reduce_mean(result_of_real) - tf.reduce_mean(result_of_fake)

    # Optimization procedure
    t_vars = tf.trainable_variables()

    g_vars = [var for var in t_vars if "w4" in var.name]
    d_vars = [var for var in t_vars if "D" in var.name]
    w_vars = [var for var in t_vars if ("D" or "G") in var.name]

    # Regularization for weights
    gr_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l1_regularizer(2.5e-6),
                                                     weights_list=g_vars)
    g_loss_reg = g_loss - gr_loss
    d_loss_reg = d_loss

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    g_train = optimizer.minimize(-g_loss_reg, var_list=g_vars)
    d_train = optimizer.minimize(-d_loss_reg, var_list=d_vars)

    # Clipping weight to [-0.01 0.01]
    d_clip = [v.assign(tf.clip_by_value(v, -0.0015, 0.0015)) for v in d_vars]

# Training graph g
saver = tf.train.Saver(var_list=w_vars)
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./model/WGAN')
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join('./model/WGAN', ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        counter = 0

    total_batchs = int(WavData.shape[0] / batch_size)

    logPath = "./result/WGAN/GAN_result.log"
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
        WavData = WavData[data_indices]
        for batch in range(total_batchs):
            batch_x = WavData[batch*batch_size:(batch+1)*batch_size]

            noise, nlabel = random_noise(batch_size)
            sess.run(d_train, feed_dict={X: batch_x, Z: noise})
            sess.run(d_clip)

            sess.run(g_train, feed_dict={Z: noise})
            sess.run(g_train, feed_dict={Z: noise})

            gl, dl = sess.run([g_loss_reg, d_loss_reg], feed_dict={X: batch_x, Z: noise})

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
            modelPath = "./model/WGAN/{0:s}_{1:s}_{2:s}".format(phones, freq, snr)
            saver.save(sess=sess, save_path=modelPath, global_step=(epoch+1))

    # Generating wave
    sample_noise, _ = random_noise(batch_size)
    generated = sess.run(fake_x, feed_dict={Z: sample_noise})
    # Writing the generated wave
    for i in range(batch_size):
        savePath = './wave/WGAN/{}.wav'.format(str(i).zfill(3))
        WriteWave(savePath, 1, 2, FS, generated[i], maxValue)
        print("Writing generated audio to " + savePath)

    log = "Complete Audio GAN"
    print(log)
    log_fp.write(log + "\n")
    log_fp.close()
