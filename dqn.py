import tensorflow as tf
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import cv2
import re

GAMMA = 0.99
ACTIONS = 2
OBSERVATION = 2000
EXPLORE = 1000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
FRAME_PER_ACTION = 2
LEARNING_RATE = 1e-6
UPDATE_TIME = 100

MODEL_PATH = "saved_networks_dqn"

class DQN:
    def __init__(self, actions):
        # relay memory
        self.replay_memory = deque()
        # parameters
        self.time_step = 0
        self.pre_time_step = 0
        self.last_model_time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        # q network
        self.state_input,self.Q_value,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.create_network()
        # target q network
        self.state_input_T,self.Q_value_T,self.W_conv1_T,self.b_conv1_T,self.W_conv2_T,self.b_conv2_T,self.W_conv3_T,self.b_conv3_T,self.W_fc1_T,self.b_fc1_T,self.W_fc2_T,self.b_fc2_T = self.create_network()

        self.copy_target_network_op = [self.W_conv1_T.assign(self.W_conv1),self.b_conv1_T.assign(self.b_conv1),self.W_conv2_T.assign(self.W_conv2),self.b_conv2_T.assign(self.b_conv2),self.W_conv3_T.assign(self.W_conv3),self.b_conv3_T.assign(self.b_conv3),self.W_fc1_T.assign(self.W_fc1),self.b_fc1_T.assign(self.b_fc1),self.W_fc2_T.assign(self.W_fc2),self.b_fc2_T.assign(self.b_fc2)]

        self.action_input = tf.placeholder(tf.float32, [None,self.actions])
        self.target_input = tf.placeholder(tf.float32, [None])
        self.loss = tf.reduce_sum(tf.square(self.target_input-tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices=1)))
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                self.last_model_time_step = int(re.search(re.compile('\\d+'), checkpoint.model_checkpoint_path).group())
                print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
                print "Could not find old network weights"

    def copy_target_network(self):
        self.session.run(self.copy_target_network_op)

    def create_network(self):
        # network weights
        # convolution
        W_conv1 = tf.Variable(tf.truncated_normal([8,8,4,32],stddev=0.01))
        b_conv1 = tf.Variable(tf.constant(0.01,shape=[32]))

        W_conv2 = tf.Variable(tf.truncated_normal([4,4,32,64],stddev=0.01))
        b_conv2 = tf.Variable(tf.constant(0.01,shape=[64]))

        W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,64],stddev=0.01))
        b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))

        # DQN
        W_fc1 = tf.Variable(tf.truncated_normal([1600,512],stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01,shape=[512]))

        W_fc2 = tf.Variable(tf.truncated_normal([512,self.actions],stddev=0.01))
        b_fc2 = tf.Variable(tf.constant(0.01,shape=[self.actions]))

        # input layer
        state_input = tf.placeholder(tf.float32,[None,80,80,4])

        # hidden layers
        h_conv1 = tf.nn.relu(tf.nn.conv2d(state_input,W_conv1,strides=[1,4,4,1],padding="SAME") + b_conv1) # [none,20,20,32]
        h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") # [none,10,10,32]

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,2,2,1],padding="SAME") + b_conv2) # [none,5,5,64]

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding="SAME") + b_conv3) # [none,5,5,64]

        h_conv3_flat = tf.reshape(h_conv3,[-1,1600]) # [1,1600=5x5x64]
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

        # Q Value layer
        Q_value = tf.matmul(h_fc1,W_fc2) + b_fc2

        return state_input,Q_value,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    def train_network(self):
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        target_batch = []
        # Q_value_batch = self.session.run(self.Q_value_T, feed_dict={self.state_input:next_state_batch})
        Q_value_batch = self.Q_value_T.eval(feed_dict={self.state_input_T:next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                target_batch.append(reward_batch[i])
            else:
                target_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.session.run(self.train_op, feed_dict={self.target_input:target_batch,self.action_input:action_batch,self.state_input:state_batch})

        if self.time_step % 10000 == 0:
            self.saver.save(self.session,MODEL_PATH + "/weights-",global_step=self.time_step+self.last_model_time_step)

        if self.time_step % UPDATE_TIME == 0:
            self.copy_target_network()

    def process(self, next_state, action, reward, done):
        new_state = np.append(self.current_state[:,:,1:],next_state,axis=2)
        self.replay_memory.append((self.current_state,action,reward,new_state,done))
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time_step > OBSERVATION:
            self.train_network()

        self.current_state = new_state
        self.time_step += 1

    def get_action(self):
        Q_value = self.session.run(self.Q_value, feed_dict={self.state_input:[self.current_state]})
        action = np.zeros(self.actions)
        action_index = 0
        if self.time_step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
            else:
                action_index = np.argmax(Q_value[0])
        action[action_index] = 1

        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVATION:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action

    def initialize(self, flappyBird):
        action = np.array([1,0])
        state, reward, done, _ = flappyBird.frame_step(action)
        state = preprocess(state)
        state = state.reshape((80,80))
        self.current_state = np.stack((state,state,state,state),axis=2)

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def print_info(network,score):
    print "TIMESTEP", network.time_step, "EPSILON", network.epsilon, "SCORE", score
    network.pre_time_step = network.time_step

def play():
    network = DQN(ACTIONS)

    flappyBird = game.GameState()
    network.initialize(flappyBird);

    _score = 0
    total = 0.0
    num_game = 0
    while True:
        action = network.get_action()
        next_state, reward, done, score = flappyBird.frame_step(action)
        if score != 0: _score = score
        if done:
            print_info(network,_score)
            total += _score
            num_game += 1
            score = 0

            if num_game == 10:
                print "\nlast 10 game avg score", total/num_game, "\n"
                total = 0.0
                num_game = 0

        next_state = preprocess(next_state)
        network.process(next_state, action, reward, done)

if len(sys.argv) > 1 and sys.argv[1]=="test":
    INITIAL_EPSILON = 0
    OBSERVATION = 99999999999999
    FRAME_PER_ACTION = 1
    if len(sys.argv) > 2:
        if sys.argv[2] == '140k':
            MODEL_PATH = "140k-saved_networks_dqn"
        elif sys.argv[2] == '200k':
            MODEL_PATH = "200k-saved_networks_dqn"
        elif sys.argv[2] == '460k':
            MODEL_PATH = "460k-saved_networks_dqn"
play()
