import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np
import tensorflow as tf

hidden_size = 512
gamma = 0.99
learning_rate = 1e-5

EPSILON_INIT = 0.3
EPSILON_MIN = 1e-4
NUM_EXPLORE = 1e3

FRAME_PER_ACTION = 4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def preprocess(observation, reshape=True):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    if reshape:
        return np.reshape(observation,(80,80,1))
    else:
        return observation
def DR(history_state, history_action, history_reward, j, gamma):
    discounted_reward = [None] * j

    j -= 1
    discounted_reward[j] = history_reward[j]
    j -= 1
    while j >= 0:
        discounted_reward[j] = history_reward[j] + gamma * discounted_reward[j+1]
        j -= 1

    return discounted_reward

def main():
    flappyBird = game.GameState()

    # initial state
    curr_state, reward0, done = flappyBird.frame_step(np.array([1,0]))
    curr_state = preprocess(curr_state, False)
    curr_state = np.stack((curr_state,curr_state,curr_state,curr_state), axis=2)

    stateInput = tf.placeholder(tf.float32, [None,80,80,4])
    # convolution layers
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([64])

    h_conv1 = tf.nn.relu(conv2d(stateInput,W_conv1,4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2,W_conv3,1) + b_conv3)

    # normal DQN
    state = tf.reshape(h_conv3,[-1,1600])
    W = tf.Variable(tf.random_uniform([1600,hidden_size]))
    hidden_layer = tf.nn.relu(tf.matmul(state, W)) # [None, hidden_size]
    O = tf.Variable(tf.random_uniform([hidden_size, 2]))
    output = tf.nn.softmax(tf.matmul(hidden_layer, O)) # [None, 2]

    rewards = tf.placeholder(tf.float32, [None]) # [step+1]
    actions = tf.placeholder(tf.int32, [None]) # [step+1]
    indicies = tf.range(0, tf.shape(output)[0]) * 2 + actions
    action_probs = tf.gather(tf.reshape(output, [-1]), indicies)

    # actor critic
    V1 = tf.Variable(tf.random_normal([1600,hidden_size],dtype=tf.float32,stddev=0.1))
    v1Out = tf.nn.relu(tf.matmul(state,V1)) # [None,4]x[4,hidden_size] = [None, hidden_size]
    V2 = tf.Variable(tf.random_normal([hidden_size,1],dtype=tf.float32,stddev=0.1))
    vOut = tf.matmul(v1Out,V2) # [None,1]
    vLoss = tf.reduce_mean(tf.square(rewards-vOut))

    loss = -tf.reduce_mean(tf.log(action_probs) * rewards) + vLoss
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # saving and loading networks
    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    session.run(tf.initialize_all_variables())

    checkpoint = tf.train.get_checkpoint_state("saved_networks_a2c")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    total_rewards = []
    history_state = []
    history_action = []
    history_reward = []
    num_frame = 0
    num_game = 0
    step = 0
    epsilon = EPSILON_INIT
    while True:
        action_dist = session.run(output, feed_dict={stateInput: [curr_state]}) # compute action distribution
        action_index = action_dist[0].argmax() if epsilon < np.random.uniform() else np.random.randint(0,2) # pick an action
        action = np.zeros(2)
        action[action_index] = 1
        next_state, reward, done = flappyBird.frame_step(action)
        next_state = preprocess(next_state)

        history_state.append(curr_state)
        history_action.append(action_index)
        history_reward.append(reward)

        curr_state = np.append(curr_state[:,:,1:], next_state, axis=2)

        step += 1
        num_frame += 1
        if done or step%2==0:
            discounted_reward = DR(history_state, history_action, history_reward, step, gamma)
            _v = session.run(vOut, feed_dict={
                stateInput: history_state,
                rewards: discounted_reward
            })

            discounted_reward -= _v.reshape(-1)

            session.run(train_op, feed_dict={
                stateInput: history_state,
                actions: history_action,
                rewards: discounted_reward
            })

        if done:
            print "NUM_FRAME:", num_frame, "survival time:", step

            num_game += 1
            total_rewards.append(step)

            if (num_game+1) % 100 == 0:
                avg = np.mean(total_rewards[-100:])
                print "\navg for last 100: ", avg, '\n'

            history_state = []
            history_action = []
            history_reward = []
            step = 0

        if num_frame % 10000 == 0:
            saver.save(session, 'saved_networks_a2c/' + 'network' + '-a2c', global_step = num_frame)

        # update epsilon
        if step > NUM_EXPLORE and epsilon >= EPSILON_MIN:
            epsilon -= (EPSILON_INIT - EPSILON_MIN) / NUM_EXPLORE

if __name__ == '__main__':
    main()
