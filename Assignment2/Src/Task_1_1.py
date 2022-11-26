#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
import Mnist_cnn
import Mnist_mlp
import Mnist_cnn_custom
import Mnist_mlp_custom
import Cifar10_cnn
import matplotlib.pyplot as plt


# ---

# ### Data Visualization

# In[18]:


def Data_inform(data):
    print("Sample size: ",data.size)     
    print("Sample Shape: ",data.shape)  
    print("Classes num: ",len(np.unique(data)), '\n')


# In[20]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('minist x_train')
Data_inform(x_train)
print('minist y_train')
Data_inform(y_train)
print('minist x_test')
Data_inform(x_test)
print('minist y_test')
Data_inform(y_test)


# In[19]:


# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('minist x_train')
Data_inform(x_train)
print('minist y_train')
Data_inform(y_train)
print('minist x_test')
Data_inform(x_test)
print('minist y_test')
Data_inform(y_test)


# ---

# ### Minist CNN

# ### Original model

# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

batch_size = 128
num_classes = 10
epochs = 20
ini = ['Ones', 'Zeros', 'RandomUniform', 'RandomNormal']
regu = ['l1', 'l2', 'l1', 'l2']; drop = [True, True, False, False]

acti = ['sigmoid', 'relu', 'tanh', 'softmax']
opti = ['SGD', "Adadelta", 'Adam', 'RMSprop']
cnn_parameters = [ [ini[0], acti[0], opti[0], regu[0], drop[0]],
                    [ini[0], acti[1], opti[1], regu[1], drop[1]],
                    [ini[0], acti[2], opti[2], regu[2], drop[2]],
                    [ini[0], acti[3], opti[3], regu[3], drop[3]],
                    [ini[1], acti[0], opti[1], regu[2], drop[2]],
                    [ini[1], acti[1], opti[0], regu[3], drop[3]],
                    [ini[1], acti[2], opti[3], regu[0], drop[0]],
                    [ini[1], acti[3], opti[2], regu[1], drop[1]],
                    [ini[2], acti[0], opti[2], regu[3], drop[3]],
                    [ini[2], acti[1], opti[3], regu[2], drop[2]],
                    [ini[2], acti[2], opti[0], regu[1], drop[1]],
                    [ini[2], acti[3], opti[1], regu[0], drop[0]],
                    [ini[3], acti[0], opti[3], regu[1], drop[1]],
                    [ini[3], acti[1], opti[2], regu[0], drop[0]],
                    [ini[3], acti[2], opti[1], regu[3], drop[3]],
                    [ini[3], acti[3], opti[0], regu[2], drop[2]],]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist cnn
print('Minist CNN')
for i in range(len(cnn_parameters)):
    print('=====================', i ,'=====================')
    print(cnn_parameters[i])
    Mnist_cnn.network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = cnn_parameters[i][0],
                        regularizer = cnn_parameters[i][3],
                        activation = cnn_parameters[i][1],
                        optimizer = cnn_parameters[i][2],
                        dropout = cnn_parameters[i][4],
                        plot_num = i)


# ---

# ### Custom model

# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

batch_size = 128
num_classes = 10
epochs = 20
ini = ['Ones', 'Zeros', 'RandomUniform', 'RandomNormal']
regu = ['l1', 'l2', 'l1', 'l2']; drop = [True, True, False, False]

acti = ['sigmoid', 'relu', 'tanh', 'softmax']
opti = ['SGD', "Adadelta", 'Adam', 'RMSprop']
cnn_parameters = [ [ini[0], acti[0], opti[0], regu[0], drop[0]],
                    [ini[0], acti[1], opti[1], regu[1], drop[1]],
                    [ini[0], acti[2], opti[2], regu[2], drop[2]],
                    [ini[0], acti[3], opti[3], regu[3], drop[3]],
                    [ini[1], acti[0], opti[1], regu[2], drop[2]],
                    [ini[1], acti[1], opti[0], regu[3], drop[3]],
                    [ini[1], acti[2], opti[3], regu[0], drop[0]],
                    [ini[1], acti[3], opti[2], regu[1], drop[1]],
                    [ini[2], acti[0], opti[2], regu[3], drop[3]],
                    [ini[2], acti[1], opti[3], regu[2], drop[2]],
                    [ini[2], acti[2], opti[0], regu[1], drop[1]],
                    [ini[2], acti[3], opti[1], regu[0], drop[0]],
                    [ini[3], acti[0], opti[3], regu[1], drop[1]],
                    [ini[3], acti[1], opti[2], regu[0], drop[0]],
                    [ini[3], acti[2], opti[1], regu[3], drop[3]],
                    [ini[3], acti[3], opti[0], regu[2], drop[2]],]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist cnn
print('Minist CNN')
for i in range(len(cnn_parameters)):
    print('=====================', i ,'=====================')
    print(cnn_parameters[i])
    Mnist_cnn_custom.custom_network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = cnn_parameters[i][0],
                        regularizer = cnn_parameters[i][3],
                        activation = cnn_parameters[i][1],
                        optimizer = cnn_parameters[i][2],
                        dropout = cnn_parameters[i][4],
                        plot_num = i)


# ---

# ### Minist MLP

# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

batch_size = 128
num_classes = 10
epochs = 20

ini = ['Ones', 'Zeros', 'RandomUniform', 'RandomNormal']
regu = ['l1', 'l2', 'l1', 'l2']; drop = [True, True, False, False]

acti = ['sigmoid', 'relu', 'tanh', 'softmax']
opti = ['SGD', "Adadelta", 'Adam', 'RMSprop']
mlp_parameters = [ [ini[0], acti[0], opti[0], regu[0], drop[0]],
                    [ini[0], acti[1], opti[1], regu[1], drop[1]],
                    [ini[0], acti[2], opti[2], regu[2], drop[2]],
                    [ini[0], acti[3], opti[3], regu[3], drop[3]],
                    [ini[1], acti[0], opti[1], regu[2], drop[2]],
                    [ini[1], acti[1], opti[0], regu[3], drop[3]],
                    [ini[1], acti[2], opti[3], regu[0], drop[0]],
                    [ini[1], acti[3], opti[2], regu[1], drop[1]],
                    [ini[2], acti[0], opti[2], regu[3], drop[3]],
                    [ini[2], acti[1], opti[3], regu[2], drop[2]],
                    [ini[2], acti[2], opti[0], regu[1], drop[1]],
                    [ini[2], acti[3], opti[1], regu[0], drop[0]],
                    [ini[3], acti[0], opti[3], regu[1], drop[1]],
                    [ini[3], acti[1], opti[2], regu[0], drop[0]],
                    [ini[3], acti[2], opti[1], regu[3], drop[3]],
                    [ini[3], acti[3], opti[0], regu[2], drop[2]],]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist MLP
print('Minist MLP')
for i in range(len(mlp_parameters)):
    print('=====================', i ,'=====================')
    print(mlp_parameters[i])
    Mnist_mlp.network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = mlp_parameters[i][0],
                        regularizer = mlp_parameters[i][3],
                        activation = mlp_parameters[i][1],
                        optimizer = mlp_parameters[i][2],
                        dropout = mlp_parameters[i][4],
                        plot_num = i)


# ---

# ### custom mlp

# In[4]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

batch_size = 128
num_classes = 10
epochs = 20
# RandomUniform
# tanh
# SGD
# l2
# True
ini = ['Ones', 'Zeros', 'RandomUniform', 'RandomNormal']
regu = ['l1', 'l2', 'l1', 'l2']; drop = [True, True, False, False]

acti = ['sigmoid', 'relu', 'tanh', 'softmax']
opti = ['SGD', "Adadelta", 'Adam', 'RMSprop']
mlp_parameters = [ [ini[0], acti[0], opti[0], regu[0], drop[0]],
                    [ini[0], acti[1], opti[1], regu[1], drop[1]],
                    [ini[0], acti[2], opti[2], regu[2], drop[2]],
                    [ini[0], acti[3], opti[3], regu[3], drop[3]],
                    [ini[1], acti[0], opti[1], regu[2], drop[2]],
                    [ini[1], acti[1], opti[0], regu[3], drop[3]],
                    [ini[1], acti[2], opti[3], regu[0], drop[0]],
                    [ini[1], acti[3], opti[2], regu[1], drop[1]],
                    [ini[2], acti[0], opti[2], regu[3], drop[3]],
                    [ini[2], acti[1], opti[3], regu[2], drop[2]],
                    [ini[2], acti[2], opti[0], regu[1], drop[1]],
                    [ini[2], acti[3], opti[1], regu[0], drop[0]],
                    [ini[3], acti[0], opti[3], regu[1], drop[1]],
                    [ini[3], acti[1], opti[2], regu[0], drop[0]],
                    [ini[3], acti[2], opti[1], regu[3], drop[3]],
                    [ini[3], acti[3], opti[0], regu[2], drop[2]],]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist MLP
print('Minist MLP')
for i in range(len(mlp_parameters)):
    print('=====================', i ,'=====================')
    print(mlp_parameters[i])
    Mnist_mlp_custom.custom_network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = mlp_parameters[i][0],
                        regularizer = mlp_parameters[i][3],
                        activation = mlp_parameters[i][1],
                        optimizer = mlp_parameters[i][2],
                        dropout = mlp_parameters[i][4],
                        plot_num = i)


# In[3]:


batch_size = 32
num_classes = 10
epochs = 20

mlp_parameters = [  ['RandomUniform', 'l2', 'relu', 'SGD'],
                    ['RandomUniform', 'l2', 'relu', 'Adam']]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist MLP
print('Minist MLP')
for i in range(len(mlp_parameters)):
    print('=====================', i ,'=====================')
    print(mlp_parameters[i])
    Mnist_mlp.network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = mlp_parameters[i][0],
                        regularizer = mlp_parameters[i][1],
                        activation = mlp_parameters[i][2],
                        optimizer = mlp_parameters[i][3],
                        plot_num = i)


# ---

# ### CIFAR-10 cnn

# In[ ]:


batch_size = 32
num_classes = 10
epochs = 20

cnn_parameters = [  ['RandomUniform', 'l2', 'relu', 'SGD'],
                    ['RandomUniform', 'l2', 'relu', 'Adam']]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist cnn
print('Minist CNN')
for i in range(len(cnn_parameters)):
    j = i + 10
    print('=====================', i ,'=====================')
    print(cnn_parameters[i])
    Cifar10_cnn.network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = cnn_parameters[i][0],
                        regularizer = cnn_parameters[i][1],
                        activation = cnn_parameters[i][2],
                        optimizer = cnn_parameters[i][3],
                        plot_num = j)


# In[6]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
batch_size = 32
num_classes = 10
epochs = 20

cnn_parameters = [  ['RandomUniform', 'l2', 'relu', 'SGD'],
                    ['RandomUniform', 'l2', 'relu', 'Adam']]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist cnn
print('Minist CNN')
for i in range(len(cnn_parameters)):
    j = i + 10
    print('=====================', i ,'=====================')
    print(cnn_parameters[i])
    Cifar10_cnn.network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = cnn_parameters[i][0],
                        regularizer = cnn_parameters[i][1],
                        activation = cnn_parameters[i][2],
                        optimizer = cnn_parameters[i][3],
                        plot_num = j)


# In[7]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
batch_size = 128
num_classes = 10
epochs = 20

cnn_parameters = [  ['RandomUniform', 'l2', 'relu', 'SGD'],
                    ['RandomUniform', 'l2', 'relu', 'Adam']]


# mlp_setups = ['glorot_normal', 'relu', 'rmsprop', 'l2']

# mnist cnn
print('Minist CNN')
for i in range(len(cnn_parameters)):
    j = i + 10
    print('=====================', i ,'=====================')
    print(cnn_parameters[i])
    Cifar10_cnn.network(  x_train, y_train, 
                        x_test, y_test,
                        batch_size, 
                        num_classes, 
                        epochs,
                        initializer = cnn_parameters[i][0],
                        regularizer = cnn_parameters[i][1],
                        activation = cnn_parameters[i][2],
                        optimizer = cnn_parameters[i][3],
                        plot_num = j)


# ---

# ### Model Structure

# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# #### MINIST CNN

# In[11]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='PReLU',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# #### MINIST MLP

# In[12]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


# #### MINIST Custom CNN

# In[13]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='PReLU',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# #### MINIST Custom MLP

# In[14]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


# #### CIFAR CNN

# In[15]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='PReLU',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# #### CIFAR MLP

# In[21]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1024 * 3,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


# #### CIFAR custom CNN

# In[22]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='PReLU',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# #### CIFAR custom MLP

# In[23]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1024 * 3,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

