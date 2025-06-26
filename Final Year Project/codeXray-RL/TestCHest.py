import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import gym
import EnvRLforClassification
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras.constraints import maxnorm
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## set image size 
image_size = 150

labels = ['PNEUMONIA', 'NORMAL']
def get_data(path):
    data = list()
    for label in labels:
        image_dir = os.path.join(path, label)
        class_num = labels.index(label)
        for img in os.listdir(image_dir):
            try:
                img_arr = cv2.imread(os.path.join(image_dir, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_arr, (image_size, image_size))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data('chest_xray/chest_xray/train')
test = get_data('chest_xray/chest_xray/test')
val = get_data('chest_xray/chest_xray/val')

## lets generate data

## help function to create dataset

def seperate_feature_and_label(dataset):
    """ seperate target and feature values"""
    
    X, Y = list(), list()
    for x, y in dataset:
        X.append(x)
        Y.append(y)
    return X, Y


def normalize(X):
    """ to normalize the data"""
    
    return np.array(X)/255


def reshape(X, Y, fig_size):
    """ TO resize the data"""
    
    X = X.reshape(-1, fig_size[0], fig_size[1], 1)
    Y = np.array(Y)
    return X, Y

def data_agumentation(datagen=None):
    """
    This function first check, is the data is dataGen object and than process.
    """
    
    if datagen is None:
        return ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rotation_range = 30)
    else:
        return datagen
        
## lets apply all the functions

X_train, y_train = seperate_feature_and_label(train)
X_test, y_test = seperate_feature_and_label(test)

# lets apply normalization function

X_train = normalize(X_train)
X_test = normalize(X_test)

X_train, y_train = reshape(X_train, y_train, (image_size, image_size))
X_test, y_test = reshape(X_test, y_test, (image_size, image_size))
x_train = X_train
x_test=X_test

#datagenerator = data_agumentation(None)
#datagenerator.fit(x_train)


####################
X = x_train
y = y_train


batch_size = 100

input_shape = (X.shape[1],X.shape[2],1)

 # Initialization of the enviroment
env = gym.make('EnvRLforClassification:RLClassification-v0')

# Fill values
env.init_dataset(X,y,batch_size=batch_size,output_shape=input_shape)


# parameters of RL
valid_actions = env.action_space
num_actions = valid_actions.n
epsilon = .1  # exploration
num_episodes = 400  #best test
iterations_episode = 100

decay_rate = 0.99
gamma = 0.001


# CNN 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())               # add new

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
####
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

####
model.add(Flatten())
model.add(Dense(128, activation='relu')) #128
model.add(Dropout(0.5))
model.add(Dense(num_actions, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])


print(model.summary())


############ Training the the RL agent ###########
reward_chain = []
loss_chain = []
for epoch in range(num_episodes):
    loss = 0.
    total_reward_by_episode = 0
    # Reset enviromet, actualize the data batch
    states = env.reset()

    done = False

    # Define exploration to improve performance
    exploration = 1
    # Iteration in one episode
    q = np.zeros([batch_size,num_actions])
   
    i_iteration = 0
    while not done:
        i_iteration += 1

        # get next action
        if exploration > 0.001:
            exploration = epsilon*decay_rate**(epoch*i_iteration)            

        if np.random.rand() <= exploration:
            actions = np.random.randint(0, num_actions,batch_size)
        else:
            q = model.predict(states)
            actions = np.argmax(q,axis=1)

        # apply actions, get rewards and new state
        next_states, reward, done, _ = env.step(actions)

        done = done[-1]
        next_states = next_states
        
        q_prime = model.predict(next_states)

        indx = np.argmax(q_prime,axis=1)
        sx = np.arange(len(indx))
        # Update q values
        targets = reward + gamma * q[sx,indx]   
        q[sx,actions] = targets

        # Train network, update loss
        loss += model.train_on_batch(states, q)[0]

        # Update the state
        states = next_states
        #print(reward)
        total_reward_by_episode += int(sum(reward))

    if next_states.shape[0] != batch_size:
            break # finished df
    reward_chain.append(total_reward_by_episode)    
    loss_chain.append(loss)

    print("\rEpoch {:03d}/{:03d} | Loss {:4.4f} |  Rewards {:03d} ".format(epoch,
          num_episodes ,loss, total_reward_by_episode))
    
model_name = "24-4modelChest"+str(num_episodes)
model.save(model_name)


