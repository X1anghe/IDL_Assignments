from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


def preprocessing(train_data, test_data):
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]
    if backend.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
        test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1) 

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255
    test_data /= 255

    return train_data, test_data, input_shape

def label_transform(time_label):
    trans_label = np.zeros(shape = (len(time_label), 4))
    for i in range(len(time_label)):
        hour = time_label[i][0]
        minute = time_label[i][1]
        hour_sincos=(np.sin(hour*(1/6)*np.pi+minute*(1/360)*np.pi),np.cos(hour*(1/6)*np.pi+minute*(1/360)*np.pi))
        minute_sincos = (np.sin(minute * (1/3) * np.pi), np.cos(minute * (1/3)*np.pi))
        time = [hour_sincos[0], hour_sincos[1], minute_sincos[0], minute_sincos[1]]
        trans_label[i] = time
    return trans_label

def custom_mse(y_true, y_pred):
    return tf.sqrt((tf.reduce_sum(tf.square(y_true - y_pred)* [3600, 3600, 1, 1]) ) / 3601)
# def custom_mse(y_true, y_pred):
#     return tf.sqrt((tf.reduce_sum(tf.math.abs(y_true - y_pred)) * [3600, 3600, 1, 1] ) / 3601)


def custom_mae(y_true, y_pred):
    abs_diff1 = tf.math.abs(y_true - y_pred)
    abs_diff2 = tf.math.abs(y_true - y_pred + 12.0)
    abs_diff3 = tf.math.abs(y_true - y_pred - 12.0)
    min_diff1 = tf.minimum(abs_diff1, abs_diff3)
    min_diff = tf.minimum(min_diff1, abs_diff2)
    return tf.reduce_mean(min_diff)

def label_trans_regression(x_train, y_train, x_test, y_test, batch_size, epochs, input_shape):

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(units=256,
                    activation='relu',
                    kernel_regularizer='l2'))
    model.add(Dense(units=128,
                    activation='relu',
                    kernel_regularizer='l2'))
    model.add(Dense(units=4, activation='linear'))
    # model.add(Flatten())
    # model.add(Dense(units=4096,
    #                 activation='relu',
    #                 kernel_regularizer='l2'))
    # model.add(Dense(units=4096,
    #                 activation='relu',
    #                 kernel_regularizer='l2'))
    # model.add(Dense(units=1, activation='linear'))

    model.compile(loss=custom_mse,
                  optimizer='adam',
                  metrics='mae')

    his = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test mse:', score[1])
    return model, his



