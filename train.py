import gym
import tensorflow as tf
import site
import cv2
import argparse

#import universe

env = gym.make('VideoPinball-v0')
observation = env.reset()

def main():
	pass

if __name__ == '__main__':
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