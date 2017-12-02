import tensorflow as tf
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import cv2

IMG_ROWS = 80
IMG_COLS = 80
IMG_CHANNELS = 4
GAMMA = 0.99
ACTIONS = 2
OBSERVATION = 3200
EXPLORE = 3000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
FRAME_PER_ACTION = 4
LEARNING_RATE = 1e-4

# create network
state_input = tf.placeholder(tf.float32, [None,IMG_ROWS,IMG_COLS,IMG_CHANNELS]) # [None,80,80,4]

# convolution
conv_1 = tf.layers.conv2d(
    inputs=state_input,
    filters=32,
    kernel_size=8,
    strides=4,
    padding="same",
    activation=tf.nn.relu
) # [None,20,20,32]
pool_1 = tf.layers.max_pooling2d(
    inputs=conv_1,
    pool_size=2,
    strides=2,
    padding="same"
) # [None,10,10,32]
conv_2 = tf.layers.conv2d(
    inputs=pool_1,
    filters=64,
    kernel_size=4,
    strides=2,
    padding="same",
    activation=tf.nn.relu
) # [None,5,5,64]
conv_3 = tf.layers.conv2d(
    inputs=conv_2,
    filters=64,
    kernel_size=3,
    strides=1,
    padding="same",
    activation=tf.nn.relu
) # [None,5,5,64]
flatten_1 = tf.layers.flatten(conv_3) # [None,640]

#DQN
dense_1 = tf.layers.dense(
    inputs=flatten_1,
    units=512,
    activation=tf.nn.relu
)
dense_2 = tf.layers.dense(
    inputs=dense_1,
    units=2
)

targets_input = tf.placeholder(tf.float32, [None,2])

loss = tf.reduce_mean(tf.square(dense_2-targets_input))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (IMG_ROWS,IMG_COLS)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(1,IMG_ROWS,IMG_COLS,1))

def train_network(session):
    flappy_bird = game.GameState()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks_dqn")
    if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
            print "Could not find old network weights"

    # replay memory
    replay_memory = deque()

    # get first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    observation0, reward0, done = flappy_bird.frame_step(do_nothing)
    observation0 = preprocess(observation0)

    # stacked 4 fromes
    states = np.stack((observation0,observation0,observation0,observation0), axis=3)
    states = states.reshape(1, states.shape[1], states.shape[2], states.shape[3]) # 1x80x80x4

    epsilon = INITIAL_EPSILON
    observe = OBSERVATION

    t = 0
    t_ = 0
    while True:
        loss = 0
        Q_sa = 0
        action_index = 0
        action = np.zeros(ACTIONS)

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
            else:
                action_dist = session.run(dense_2, feed_dict={state_input:states})
                action_index = action_dist[0].argmax()

        action[action_index] = 1

        # reduce epsilon
        if epsilon > FINAL_EPSILON and t > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run flappy_bird and get a new observation
        observation0, reward0, done = flappy_bird.frame_step(action)
        observation0 = preprocess(observation0) # 1x80x80x1

        states_ = np.append(observation0, states[:,:,:,:3], axis=3)

        replay_memory.append((states,action_index,reward0,states_,done))
        if len(replay_memory) > REPLAY_MEMORY:
            replay_memory.popleft()

        if t > observe and t % FRAME_PER_ACTION == 0:
            minibatch = random.sample(replay_memory, BATCH_SIZE)

            inputs = np.zeros((BATCH_SIZE,states.shape[1],states.shape[2],states.shape[3]))
            targets = np.zeros((inputs.shape[0],ACTIONS)) # 32x2

            for i in range(len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                done_t = minibatch[i][4]

                inputs[i:i + 1] = state_t

                targets[i] = session.run(dense_2, feed_dict={state_input:state_t})
                Q_sa = session.run(dense_2, feed_dict={state_input:state_t1})

                if done_t:
                    targets[i,action_t] = reward_t
                else:
                    targets[i,action_t] = reward_t + GAMMA*np.max(Q_sa)

            session.run([train_op], feed_dict={state_input:inputs, targets_input:targets})
        states = states_
        t += 1

        if t > observe and t%1000 == 0:
            saver.save(session, 'saved_networks_dqn/' + 'network' + '-dqn', global_step=t)

        if done:
            print "TIMESTEP", t, "SURVIVAL TIME", t-t_, "Q_MAX", np.max(Q_sa)
            t_ = t
session = tf.Session()
train_network(session)
