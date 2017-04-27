import gym
import tensorflow as tf
import numpy as np
import site
import cv2
import argparse
import keras
import random
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from collections import deque
import json

# import universe
NUM_ACTIONS = 0
IMAGE_HEIGHT = 84
IMAGE_WIDTG = 84
IMAGE_CHANNEL = 3
MEMORY_CAPACITY = 1000000
FINAL_EXPLORE_FRAME = 1000000
REPLAY_START_SIZE = 50000
BATCH_SIZE = 32
STATE_HISTORY_LENGTH = 4
UPDATE_FREQUENCY = 10000
LEARNING_RATE = 0.00025
INITIAL_EXPLORE = 1.0
FINAL_EXPLORE = 0.1
GAMMA = 0.99
RENDER = True
N_EPOCH = 50000
EPOCH = 100
def get_env(game_name):
    env = gym.make(game_name)
    return env
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (80, 80), interpolation = cv2.INTER_LINEAR)
    return image

def get_initial_state(observation):
    grey_image = process_image(observation)
    #state = grey_image.reshape(1, 1, grey_image.shape[0], grey_image.shape[1])
    state = np.stack((grey_image, grey_image, grey_image, grey_image), axis=2)
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
    return state

def start_game(env, mode):
    if mode == 'train':
        model = build_model(NUM_ACTIONS)
        epsilon = INITIAL_EXPLORE
    observation = env.reset()
    memory = deque()
    prev_state = get_initial_state(observation)
    count_frame = 0
    for i in range(REPLAY_START_SIZE):
        action = env.action_space.sample()
        observation, reward, terminal, _ = env.step(action)
        if RENDER:
            env.render()
        next_state = get_next_state(prev_state, observation)
        memory.append((prev_state, action, reward, next_state, terminal))
        prev_state = next_state
        if terminal:
            observation = env.reset()
            prev_state = get_initial_state(observation)
        print("Observing:%d", i)
    for epoch in range(1, EPOCH + 1):
        updates = 0
        loss = 0
        count_episode = 0
        total_reward = 0
        while (updates < N_EPOCH):

            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action_values = model.predict(prev_state)
                action = np.argmax(action_values)
            observation, reward, terminal, _ = env.step(action)
            total_reward += reward
            if RENDER:
                env.render()
            count_frame += 1
            if count_frame < FINAL_EXPLORE_FRAME:
                epsilon -= ((INITIAL_EXPLORE- FINAL_EXPLORE) / FINAL_EXPLORE_FRAME)
            next_state = get_next_state(prev_state, observation)
            memory.append((prev_state, action, reward, next_state, terminal))
            if (len(memory) > MEMORY_CAPACITY):
                memory.popleft()
            prev_state = next_state
            if terminal:
                observation = env.reset()
                prev_state = get_initial_state(observation)
                count_episode += 1
            minibatch = random.sample(memory, BATCH_SIZE)
            input = np.zeros(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTG, STATE_HISTORY_LENGTH)
            Q_target = np.zeros((BATCH_SIZE, NUM_ACTIONS))
            for i in range(0, BATCH_SIZE):
                S_t = minibatch[i][0]
                A = minibatch[i][1]
                R = minibatch[i][2]
                S_t1 = minibatch[i][3]
                T = minibatch[i][4]
                input[i] = S_t
                predict = model.predict(S_t)
                Q_target[i] = predict
                Q_S = predict
                if T:
                    Q_target[i, A] = R
                else:
                    Q_target[i, A] = R + GAMMA * np.max(Q_S)
            loss += model.train_on_batch(input, Q_target)
            updates += 1
            print('Epoch: {}, updates: {}, memory_size: {}, epsilon: {}'.format(epoch, updates, len(memory), epsilon) )
        print("total loss: {}, avg reward per epsiode: {}".format(loss, total_reward/count_episode))
        print("Now we save model")
        model.save_weights(str(epoch).join("-model.h5"), overwrite=True)
        with open(str(epoch).join("-model.json"), "w") as outfile:
            json.dump(model.to_json(), outfile)


def get_next_state(prev_state, observation):
    grey_image = process_image(observation)
    state = grey_image.reshape(1, grey_image.shape[0], grey_image.shape[1], 1)
    next_state = np.append(state, prev_state[:, :, :, :3], axis=3)
    return next_state

def build_model(num_actions):
    # (batch, height, width, channels)
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTG, STATE_HISTORY_LENGTH)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions))
    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    return model

def main():
    parser = argparse.ArgumentParser(description='train/test an agent.')
    parser.add_argument('-m', '--mode', help='train/test', required=True)
    parser.add_argument('-g', '--game', help='game name', required=True)
    args = vars(parser.parse_args())
    env = get_env(args['game'])
    NUM_ACTIONS = env.action_space.n
    start_game(env, args['mode'])

if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
    # env.configure(remotes=1)  # automatically creates a local docker container
    # observation_n = env.reset()
    #
    # while True:
    #   action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
    #   observation_n, reward_n, done_n, info = env.step(action_n)
    #   if not observation_n[0] == None:
    #   	print (observation_n[0]['vision'].shape)
    #   env.render()
