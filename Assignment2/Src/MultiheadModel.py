from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
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

def minutes_transform(time_label):
    trans_label = np.zeros(shape = (len(time_label), 2))
    for i in range(len(time_label)):
        minute = time_label[i]
        minute_sincos = (np.sin(minute * (1/3) * np.pi), np.cos(minute * (1/3)*np.pi))
        time = [minute_sincos[0], minute_sincos[1]]
        trans_label[i] = time
    return trans_label

def custom_diff(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))


# def custom_mae(y_true, y_pred):
#     abs_diff = tf.math.abs(y_true - y_pred)
#     min_diff = tf.minimum(tf.math.add(12.0, abs_diff), abs_diff)
#     return tf.reduce_mean(min_diff)

def multi_classfication(x_train, y_train, x_test, y_test, batch_size, epochs, lr, num_classes, input_shape):

    input = Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), 
                    activation='relu', 
                    kernel_regularizer='l2')(input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    normal1 = BatchNormalization()(pool1)
    drop1 = Dropout(0.25)(normal1)

    conv3 = Conv2D(64, (3, 3), 
                    activation='relu',
                    kernel_regularizer='l2')(drop1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    normal2 = BatchNormalization()(pool2)
    drop2 = Dropout(0.25)(normal2)

    conv3 = Conv2D(128, (3, 3), 
                    activation='relu',
                    kernel_regularizer='l2')(drop2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    normal3 = BatchNormalization()(pool3)
    drop3 = Dropout(0.25)(normal3)

    flatten = Flatten()(drop3)
    Dense1 = Dense(256,
                    activation='relu',
                    kernel_regularizer='l2')(flatten)
    Dense2 = Dense(128,
                activation='relu',
                kernel_regularizer='l2')(Dense1)
    # Multiclass
    output_class = Dense(num_classes, activation='softmax', name='hours')(Dense2)
    # Regression
    output_regress = Dense(1, activation='linear', name='minutes')(Dense2)

    multi_model = Model(inputs = input, outputs = [output_class, output_regress])

    multi_model.compile(loss=['categorical_crossentropy', 'mae'],
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=['categorical_accuracy', 'mse'])

    his = multi_model.fit(x_train, [y_train[0], y_train[1]],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, [y_test[0], y_test[1]]))

              
    score = multi_model.evaluate(x_test, [y_test[0], y_test[1]], verbose=0)
    print('Test loss:', score[0])
    print('Test mse:', score[1])
    return multi_model, his



# from keras.utils.np_utils import to_categorical
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
# from keras import backend
# import keras
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf


# def preprocessing(train_data, test_data):
#     img_rows, img_cols = train_data.shape[1], train_data.shape[2]
#     if backend.image_data_format() == 'channels_first':
#         train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
#         test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
#         input_shape = (1, img_rows, img_cols)
#     else:
#         train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
#         test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1) 

#     train_data = train_data.astype('float32')
#     test_data = test_data.astype('float32')
#     train_data /= 255
#     test_data /= 255

#     return train_data, test_data, input_shape

# def minutes_transform(time_label):
#     trans_label = np.zeros(shape = (len(time_label), 2))
#     for i in range(len(time_label)):
#         minute = time_label[i]
#         minute_sincos = (np.sin(minute * (1/3) * np.pi), np.cos(minute * (1/3)*np.pi))
#         time = [minute_sincos[0], minute_sincos[1]]
#         trans_label[i] = time
#     return trans_label

# def custom_diff(y_true, y_pred):
#     return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))


# # def custom_mae(y_true, y_pred):
# #     abs_diff = tf.math.abs(y_true - y_pred)
# #     min_diff = tf.minimum(tf.math.add(12.0, abs_diff), abs_diff)
# #     return tf.reduce_mean(min_diff)

# def multi_classfication(x_train, y_train, x_test, y_test, batch_size, epochs, lr, num_classes, input_shape):

#     input = Input(shape=input_shape)

#     conv1 = Conv2D(32, (3, 3), 
#                     activation='relu', 
#                     kernel_regularizer='l2')(input)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     normal1 = BatchNormalization()(pool1)
#     drop1 = Dropout(0.2)(normal1)

#     conv3 = Conv2D(64, (3, 3), 
#                     activation='relu',
#                     kernel_regularizer='l2')(drop1)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     normal2 = BatchNormalization()(pool2)
#     drop2 = Dropout(0.2)(normal2)

#     conv3 = Conv2D(128, (3, 3), 
#                     activation='relu',
#                     kernel_regularizer='l2')(drop2)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     normal3 = BatchNormalization()(pool3)
#     drop3 = Dropout(0.2)(normal3)

#     conv4 = Conv2D(256, (3, 3), 
#                     activation='relu',
#                     kernel_regularizer='l2')(drop3)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#     normal4 = BatchNormalization()(pool4)
#     drop4 = Dropout(0.2)(normal4)

#     flatten = Flatten()(drop4)
#     Dense1 = Dense(256,
#                     activation='relu',
#                     kernel_regularizer='l2')(flatten)
#     drop5 = Dropout(0.2)(Dense1)
#     Dense2 = Dense(128,
#                 activation='relu',
#                 kernel_regularizer='l2')(drop5)
#     drop6 = Dropout(0.2)(Dense2)
#     # Multiclass
#     output_class = Dense(num_classes, activation='softmax', name='hours')(drop6)
#     # Regression
#     output_regress = Dense(1, activation='linear', name='minutes')(drop6)

#     multi_model = Model(inputs = input, outputs = [output_class, output_regress])

#     multi_model.compile(loss=['categorical_crossentropy', 'mae'],
#                 optimizer=keras.optimizers.Adam(learning_rate=lr),
#                 metrics=['categorical_accuracy', 'mse'])

#     his = multi_model.fit(x_train, [y_train[0], y_train[1]],
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(x_test, [y_test[0], y_test[1]]))

              
#     score = multi_model.evaluate(x_test, [y_test[0], y_test[1]], verbose=0)
#     print('Test loss:', score[0])
#     print('Test mse:', score[1])
#     return multi_model, his
