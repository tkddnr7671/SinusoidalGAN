import tensorflow as tf
from tensorflow.python.ops import spectral_ops

def conv1d(x, nfilter, filter_size, step):
    return tf.layers.conv1d(inputs=x, filters=nfilter, kernel_size=filter_size, strides=step, padding='SAME')

def conv2d(x, nfilter, filter_size, step):
    return tf.layers.conv2d(inputs=x, filters=nfilter, kernel_size=filter_size, strides=step, padding='SAME')

def conv2d_transpose(x, filter_size, out_shape, step, name='deconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', filter_size, initializer=tf.random_normal_initializer(stddev=0.01))
    return tf.nn.conv2d_transpose(value=x, filter=w, output_shape=out_shape, strides=step)

def ResBlock(x, nfilters, filter_size, step):
    output = x
    hidden = tf.layers.conv1d(inputs=x, filters=nfilters, kernel_size=filter_size, strides=step, padding='SAME')
    return tf.add(output, hidden)

def compose(x, nfilter, kernel_size, out_size):
    temp = tf.layers.conv1d(inputs=x, filters=nfilter, kernel_size=kernel_size, strides=1, padding='SAME')
    temp = tf.nn.leaky_relu(temp)

    in_size = x.shape[1].value
    avg_kernel_size = in_size * nfilter // out_size
    temp = tf.transpose(temp, perm=[0, 2, 1])
    temp = avgpool(temp, avg_kernel_size, avg_kernel_size)

    output = tf.reshape(temp, shape=[-1, out_size, 1])
    return output

def avgpool1d(x, pool_size, step):
    return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')

def maxpool1d(x, pool_size, step):
    return tf.layers.max_pooling1d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')

def maxpool2d(x, pool_size, step):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')

def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0] // 2
    return tf.nn.top_k(v, m).values[m-1]

def tensor_stft(x, FrmLeng, FrmOver):
    FrmStep = FrmLeng-FrmOver
    tensor_spec = tf.contrib.signal.stft(signals=x, frame_length=FrmLeng, frame_step=FrmStep,
                                         fft_length=FrmLeng, window_fn=tf.contrib.signal.hamming_window)
    tensor_spec = tf.abs(tensor_spec)
    tensor_spec = tf.transpose(a=tensor_spec, perm=[0, 2, 1])
    nFre, nFrm = tensor_spec.shape[1], tensor_spec.shape[2]
    return tf.reshape(tensor=tensor_spec, shape=[-1, nFre, nFrm, 1])

def tensor_fft(x, FrmLeng):
    nData, nFrm = x.shape[0], x.shape[1]
    nFre = int(FrmLeng/2)+1
    tensor_spectrogram = []
    for dter in range(nData):
        tensor_spectrum = []
        for fter in range(nFrm):
            temp_spec = spectral_ops.rfft(x[dter, fter, :FrmLeng], [FrmLeng])
            temp_spec = tf.abs(temp_spec)
            tensor_spectrum.append(temp_spec)
        tensor_spectrum = tf.convert_to_tensor(tensor_spectrum, dtype=tf.float32)
        tensor_spectrogram.append(tensor_spectrum)
    tensor_spectrogram = tf.convert_to_tensor(tensor_spectrogram)
    return tf.reshape(tensor=tensor_spectrogram, shape=[-1, nFrm, nFre, 1])

def tensor_normalize(input_):
    nData, nDim = input_.shape[0], input_.shape[1]

    maxVal = tf.reduce_max(input_tensor=tf.abs(input_), axis=1)
    maxVal = tf.reshape(tensor=maxVal, shape=[nData, 1])
    maxVal = tf.matmul(maxVal, tf.ones(shape=[1, nDim]))

    output = tf.div(input_, maxVal)

    return output
