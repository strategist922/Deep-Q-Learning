import argparse
import json
import random
from collections import deque
import os
import gym
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# import universe
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
TEST_EXPLORE = 0.05
GAMMA = 0.99
RENDER = True
N_EPOCH = 50000
EPOCH = 100
TEST_EPISODE = 30
NO_OP_STEPS = 30
MOMENTUM = 0.95
MIN_GRAD = 0.01


# def rgb2gray(rgb):
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     return gray

def get_env(game_name):
    env = gym.make(game_name)
    return env


def process_image(image):
    image = np.uint8(resize(rgb2gray(image), (IMAGE_WIDTG, IMAGE_HEIGHT)) * 255)
    # image = rgb2gray(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = transform.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTG))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTG), interpolation = cv2.INTER_LINEAR)
    return image


def copy_model(model, num_actions):
    temp_file = "target_copy.h5"
    model.save_weights(temp_file, overwrite=True)
    copied_model = build_model(num_actions)
    copied_model.load_weights(temp_file)
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
    copied_model.compile(loss='mse', optimizer=optimizer)
    return copied_model

def get_initial_state(observation, last_observation):
    observation = np.maximum(observation, last_observation)
    grey_image = process_image(observation)
    # state = grey_image.reshape(1, 1, grey_image.shape[0], grey_image.shape[1])
    state = np.stack((grey_image, grey_image, grey_image, grey_image), axis=2)
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
    return state


def test_model(model, n_episode, env, epsilon):
    observation = env.reset()
    last_observation = observation
    state = get_initial_state(observation, last_observation)
    for _ in range(random.randint(1, NO_OP_STEPS)):
        last_observation = observation
        observation, _, _, _ = env.step(0)  # Do nothing
    n = 0
    terminal = False
    total_reward = 0
    while True:
        if terminal:
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = get_initial_state(observation, last_observation)
            n += 1
            # print(str(n) + " episodes completed")
            if n >= n_episode:
                break
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action_values = model.predict(state)
            action = np.argmax(action_values)
        last_observation = observation
        observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        state = get_next_state(state, observation, last_observation)
        if RENDER:
            env.render()
    avg_reward = total_reward / n_episode
    print ("avg reward per episode: " + str(avg_reward))
    return avg_reward

def start_game(mode, file, game):
    save_dir = game
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    env = get_env(game)
    NUM_ACTIONS = env.action_space.n
    model = build_model(NUM_ACTIONS)
    if mode == 'train':
        epsilon = INITIAL_EXPLORE
    elif mode == 'test':
        epsilon = TEST_EXPLORE
        print ("Now we load weight")
        model.load_weights(file)
        # rmsprop = RMSprop(lr=LEARNING_RATE, epsilon = 0.01)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        model.compile(loss='mse', optimizer=optimizer)
        print ("Weight load successfully, start testing")
        test_model(model, TEST_EPISODE, env, epsilon)
        return
    elif mode =='record':
    	epsilon = TEST_EXPLORE
        print ("Now we load weight")
        model.load_weights(file)
        # rmsprop = RMSprop(lr=LEARNING_RATE, epsilon = 0.01)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        model.compile(loss='mse', optimizer=optimizer)
        print ("Weight load successfully, start testing")
        record_model(model, TEST_EPISODE, env, epsilon, save_dir + "/results")
    observation = env.reset()
    last_observation = None
    for _ in range(random.randint(1, NO_OP_STEPS)):
        last_observation = observation
        observation, _, _, _ = env.step(0)  # Do nothing
    memory = deque()
    prev_state = get_initial_state(observation, last_observation)
    count_frame = 0
    for i in range(REPLAY_START_SIZE):
        action = env.action_space.sample()
        last_observation = observation
        observation, reward, terminal, _ = env.step(action)
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        if RENDER:
            env.render()
        next_state = get_next_state(prev_state, observation, last_observation)
        memory.append((prev_state, action, reward, next_state, terminal))
        prev_state = next_state
        if terminal:
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            prev_state = get_initial_state(observation, last_observation)
        print("Observing: {}".format(i))
    model.save_weights(save_dir + "/initial-model.h5", overwrite=True)
    with open(save_dir + "/initial-model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
    target_model = copy_model(model, NUM_ACTIONS)
    for epoch in range(1, EPOCH + 1):
        updates = 0
        loss = 0
        while (updates < N_EPOCH):

            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action_values = model.predict(prev_state)
                action = np.argmax(action_values)
            last_observation = observation
            observation, reward, terminal, _ = env.step(action)
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1
            if RENDER:
                env.render()
            count_frame += 1
            if count_frame < FINAL_EXPLORE_FRAME:
                epsilon -= ((INITIAL_EXPLORE - FINAL_EXPLORE) / FINAL_EXPLORE_FRAME)
            next_state = get_next_state(prev_state, observation, last_observation)
            memory.append((prev_state, action, reward, next_state, terminal))
            if (len(memory) > MEMORY_CAPACITY):
                memory.popleft()
            prev_state = next_state
            if terminal:
                observation = env.reset()
                for _ in range(random.randint(1, NO_OP_STEPS)):
                    last_observation = observation
                    observation, _, _, _ = env.step(0)  # Do nothing
                prev_state = get_initial_state(observation, last_observation)
            minibatch = random.sample(memory, BATCH_SIZE)
            input = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTG, STATE_HISTORY_LENGTH))
            Q_target = np.zeros((BATCH_SIZE, NUM_ACTIONS))
            for i in range(0, BATCH_SIZE):
                S_t = minibatch[i][0]
                A = minibatch[i][1]
                R = minibatch[i][2]
                S_t1 = minibatch[i][3]
                T = minibatch[i][4]
                input[i] = S_t
                Q_target[i] = model.predict(S_t)
                Q_S = target_model.predict(S_t1)
                if T:
                    Q_target[i, A] = R
                else:
                    Q_target[i, A] = R + GAMMA * np.max(Q_S)
            loss += model.train_on_batch(input, Q_target)
            updates += 1
            if updates % UPDATE_FREQUENCY == 0:
                target_model = copy_model(model, NUM_ACTIONS)
            print('Epoch: {}, updates: {}, memory_size: {}, epsilon: {}'.format(epoch, updates, len(memory), epsilon))
            # if updates % 10000 == 0:
            #     avg_reward = test_model(model, TEST_EPISODE, get_env(game), TEST_EXPLORE)
            #     print('Epoch: {}, updates: {}, memory_size: {}, epsilon: {}, avg_reward: {}'.format(epoch, updates, len(memory), epsilon, avg_reward))
        avg_reward = test_model(model, TEST_EPISODE, get_env(game), TEST_EXPLORE)
        print("total loss: {}, avg reward per epsiode: {}".format(loss, avg_reward))
        with open(save_dir + "/" + 'training.log', 'a') as log:
            log.write("total loss: {}, avg reward per epsiode: {}\n".format(loss, avg_reward))
        print("Now we save model")
        model.save_weights(save_dir + "/" + str(epoch) + "-model.h5", overwrite=True)
        with open(save_dir + "/" + str(epoch) + "-model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)


def get_next_state(prev_state, observation, last_observation):
    observation = np.maximum(observation, last_observation)
    grey_image = process_image(observation)
    state = grey_image.reshape(1, grey_image.shape[0], grey_image.shape[1], 1)
    next_state = np.append(state, prev_state[:, :, :, :3], axis=3)
    return next_state


def build_model(num_actions):
    # (batch, height, width, channels)
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTG, STATE_HISTORY_LENGTH)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions))
    # rmsprop = RMSprop(lr=LEARNING_RATE, epsilon=MIN_GRAD)
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def main():
    parser = argparse.ArgumentParser(description='train/test an agent.')
    parser.add_argument('-m', '--mode', help='train/test', required=True)
    parser.add_argument('-g', '--game', help='game name', required=True)
    parser.add_argument('-f', '--file', help='model file', required=True)
    args = vars(parser.parse_args())
    start_game(args['mode'], args['file'], args['game'])


if __name__ == '__main__':
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
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
