from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
import tensorflow as tf
import keras


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

# def custom_mae(y_true, y_pred):
#     abs_diff1 = tf.math.abs(y_true - y_pred)
#     min_diff1 = tf.minimum(abs_diff1, tf.math.abs(12.0 + y_true - y_pred))
#     min_diff2 = tf.minimum(abs_diff1, tf.math.abs(y_true - y_pred - 12.0))
#     return tf.reduce_mean(tf.maximum(min_diff1,min_diff2))  

def custom_mae(y_true, y_pred):
    j_0 = tf.minimum(y_pred, 0)
    j_1 = tf.maximum(y_pred-12, 0)
    diff1 = tf.math.square(y_true - y_pred)
    diff2 = tf.math.square(y_true - y_pred - 12.0)
    return tf.reduce_mean(tf.minimum(diff1, diff2)) + 5.0 * (j_1 - j_0)

def regression(x_train, y_train, x_test, y_test, batch_size, epochs, lr, input_shape):

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
    model.add(Dense(units=1, activation='linear'))

    model.compile(loss=custom_mae,
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
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