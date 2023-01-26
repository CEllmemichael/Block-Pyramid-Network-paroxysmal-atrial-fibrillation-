from tensorflow.keras.layers import Dropout, Lambda, Input, average, Reshape, UpSampling1D, Multiply, Concatenate
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Add, AveragePooling1D
from tensorflow.keras.layers import ZeroPadding1D, Cropping1D, BatchNormalization, MaxPooling1D
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras as keras

def convblock1(m,layername , drop=0.5,):
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=3, name=layername + '_conv1')(m)
    x = tf.keras.layers.BatchNormalization (momentum=0.95, epsilon=0.001)(x)
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=3, name=layername + '_conv2')(m)
    x = tf.keras.layers.MaxPool1D(padding='valid',pool_size=2,strides= 2,trainable=True)(x)
    m = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True)(m)
    return tf.keras.layers.add([x,m])

def convblock2(m, layername , drop=0.5,res = 0,):
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(m)
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Conv1D(activation= 'linear',dilation_rate=1,filters=32,padding='same',
                               strides=1,kernel_size=17,name=layername + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)

    if res ==0:
        x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                                   strides=1, kernel_size=17, name=layername + '_conv2')(x)
        return tf.keras.layers.add([x,m])
    elif res ==1:
        x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                                   strides=1, kernel_size=17, name=layername + '_conv2')(x)
        x = tf.keras.layers.MaxPool1D(padding='valid',pool_size=2,strides= 2,trainable=True)(x)
        m = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True)(m)
        return tf.keras.layers.add([x,m])
    elif res == 2:
        x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                                   strides=1, kernel_size=17, name=layername + '_conv2')(x)
        x = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True,name=layername +'pooling11')(x)
        m = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True,name=layername +'pooling33')(m)
        m = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                                   strides=1, kernel_size=17, name=layername + '_conv3')(m)
        return tf.keras.layers.add([x, m])

def covnblocktype1(x, layername, drop=0.5):
    # 前者为变化少的那一个
    m = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True)(x)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=7, name=layername + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.001)(x)
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=7, name=layername + '_conv2')(x)
    x = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True)(x)

    return m, x

def covnblocktype2(x, layername, drop=0.5):
    m = x
    m = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True)(m)
    m = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=17, name=layername + '_conv2')(m)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=17, name=layername + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.001)(x)
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=17, name=layername + '_conv3')(x)
    x = tf.keras.layers.MaxPool1D(padding='valid', pool_size=2, strides=2, trainable=True)(x)
    return m, x

def covnblocktype3(x, layername, drop=0.5):
    m = x
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=27, name=layername + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.001)(x)
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='same',
                               strides=1, kernel_size=27, name=layername + '_conv2')(x)
    x = tf.keras.layers.MaxPool1D(padding='valid', pool_size=1, strides=1, trainable=True)(x)
    x = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='valid',
                               strides=3, kernel_size=33, name=layername + '_conv3')(x)
    m = tf.keras.layers.Conv1D(activation='linear', dilation_rate=1, filters=32, padding='valid',
                               strides=3, kernel_size=33, name=layername + '_conv4')(m)
    return m, x

def layer_type_1(x, y, layername, res=1):
    if res == 1:
        xx, xm = covnblocktype1(x, layername)
        yx, ym = covnblocktype1(y, layername + 'x')
        return tf.keras.layers.add([xx, ym]), tf.keras.layers.add([xm, yx])
    if res == 2:
        xx, xm = covnblocktype2(x, layername)
        yx, ym = covnblocktype2(y, layername + 'x')
        return tf.keras.layers.add([xx, ym]), tf.keras.layers.add([xm, yx])
    if res == 3:
        xx, xm = covnblocktype3(x, layername)
        yx, ym = covnblocktype3(y, layername + 'x')
        return tf.keras.layers.add([xx, ym]), tf.keras.layers.add([xm, yx])

def mix_layer(input_shape):
    datas = Input(shape = input_shape, dtype='float', name='data')
    data1 = datas[:, 0:1024]
    data2 = datas[:, 1024:2048]
    data1 = Conv1D(activation='linear', dilation_rate=1, filters=32,
                   kernel_size=17, padding='same', strides=1)(data1)
    data1 = BatchNormalization(epsilon=0.001, momentum=0.99)(data1)
    data1 = keras.layers.Activation('relu')(data1)
    data1 = convblock1(data1, layername='layers1')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data2 = Conv1D(activation='linear', dilation_rate=1, filters=32,
                   kernel_size=17, padding='same', strides=1)(data2)
    data2 = BatchNormalization(epsilon=0.001, momentum=0.99)(data2)
    data2 = keras.layers.Activation('relu')(data2)
    data2 = convblock1(data2, layername='layers2')
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # data1, data2 = layer_type_1(data1, data2, 'layer19', res=3)
    # # data1 = UpSampling1D(size=2)(data1)
    # # data2 = UpSampling1D(size=2)(data2)
    data1, data2 = layer_type_1(data1, data2, 'layer20', res=1)
    mix1 = tf.keras.layers.add([data1, data2])
    mix1 = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(mix1)
    data1, data2 = layer_type_1(data1, data2, 'layer21', res=2)
    mix2 = tf.keras.layers.add([data1, data2])
    mix2 = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(mix2)
    mix2 = UpSampling1D(size=2)(mix2)
    data1, data2 = layer_type_1(data1, data2, 'layer22', res=3)
    mix3 = tf.keras.layers.add([data1, data2])
    mix3 = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(mix3)
    mix3 = UpSampling1D(size=8)(mix3)
    mix = tf.keras.layers.add([mix1,mix2,mix3])
    # data1, data2 = layer_type_1(data1, data2, 'layer23', res=1)
    # data1, data2 = layer_type_1(data1, data2, 'layer24', res=2)
    # data1, data2 = layer_type_1(data1, data2, 'layer25', res=3)
    # data1, data2 = layer_type_1(data1, data2, 'layer26', res=2)
    # data1, data2 = layer_type_1(data1, data2, 'layer27', res=2)
    # data1, data2 = layer_type_1(data1, data2, 'layer28', res=2)
    # data1, data2 = layer_type_1(data1, data2, 'layer29', res=2)


    #mix = tf.keras.layers.add([data1, data2])
    data = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(mix)
    data = keras.layers.Activation('relu')(data)
    data = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(activation='tanh', dropout=0.3, implementation=1, recurrent_activation='hard_sigmoid',
                             recurrent_dropout=0.5, units=64))(data)
    # data = data[None]
    # data = GlobalMaxPool1D()(data)
    data = tf.keras.layers.Dense(activation='relu', units=32, name='dense0')(data)
    data = tf.keras.layers.Dense(activation='linear', units=2, name='dense1')(data)
    predictions = tf.keras.layers.Activation(activation='sigmoid', name='dense2')(data)
    model = Model(inputs=datas, outputs=predictions)
    return model