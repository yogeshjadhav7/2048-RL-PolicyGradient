"""
Policy Gradient Reinforcement Learning
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.05,
        reward_decay=0.95,
        epochs=1,
        load_path=None,
        save_path=None
    ):

        self.quiet = False
        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epochs = epochs

        self.save_path = save_path
        self.load_path = load_path

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.cost_history = []

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if self.load_path is not None:
            try:
                self.saver.restore(self.sess, self.load_path)
            except:
                print("Saved model not found at ", load_path, " Creating a new model at ", load_path)


    def store_transition(self, s, a, r):
        """
            Store play memory for training
            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """

        self.episode_rewards.append(r)
        self.episode_observations.append(s)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)


    def choose_action(self, observation):
        """
            Choose action based on observation
            Arguments:
                observation: array of state, has shape (num_features)
            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        observation = observation[np.newaxis, :]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Normalize the observations
        episode_observations_normalize = self.normalize_observations()

        # Train on episode
        for epoch in range(self.epochs):
            self.sess.run(self.train_op, feed_dict={
                 self.X: np.vstack(episode_observations_normalize),
                 self.Y: np.vstack(np.array(self.episode_actions)),
                 self.discounted_episode_rewards_norm: discounted_episode_rewards_norm
            })
        
        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            if not self.quiet:
                print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    def normalize_observations(self):
        episode_observations_normalize = []
        for episode_observation in self.episode_observations:
            observation = []
            max_val = 0
            min_val = 99999999
            for ob in episode_observation:
                if ob > 0:
                    ob = np.log2(ob)
                    if max_val < ob: max_val = ob
                    if min_val > ob: min_val = ob

                observation.append(ob)

            observation /= np.amax(observation)
            episode_observations_normalize.append(observation)

        return episode_observations_normalize


    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards, dtype=np.float64)
        cumulative = 0.0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(None, self.n_x), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.n_y), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        '''
        # Initialize parameters
        units_layer_1 = 512
        units_layer_2 = 256
        units_layer_3 = 256
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [units_layer_3, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [units_layer_3, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W4 = tf.get_variable("W4", [units_output_layer, units_layer_3], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b4 = tf.get_variable("b4", [units_output_layer, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.softmax(Z3)
        with tf.name_scope('layer_4'):
            Z4 = tf.add(tf.matmul(W4, A3), b4)
            A4 = tf.nn.softmax(Z4)
        '''

        units_input_layer = self.n_x
        units_output_layer = self.n_y

        from keras.layers import  Dense, Dropout, BatchNormalization
        from keras import initializers

        kerner_initializer = initializers.glorot_uniform(seed=1)
        A1 = Dense(units=512, activation='elu', kernel_initializer=kerner_initializer, input_shape=(units_input_layer,))(self.X)
        A1 = BatchNormalization()(A1)

        A2 = Dense(units=256, activation='elu', kernel_initializer=kerner_initializer)(A1)
        A2 = BatchNormalization()(A2)
        A2 = Dropout(0.5)(A2)

        A3 = Dense(units=128, activation='elu', kernel_initializer=kerner_initializer)(A2)
        A3 = BatchNormalization()(A3)
        A3 = Dropout(0.6)(A3)

        A4 = Dense(units=64, activation='elu', kernel_initializer=kerner_initializer)(A3)
        A4 = BatchNormalization()(A4)
        A4 = Dropout(0.7)(A4)

        Z = Dense(units=units_output_layer, kernel_initializer=kerner_initializer, activation=None)(A4)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = Z
        labels = self.Y
        self.outputs_softmax = tf.nn.softmax(logits, name='A4')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def plot(self, y_data, y_label, n_episode, window=100, dir='outputs/plots/'):
        filename = dir + y_label + "_" + str(n_episode + 1) + ".pdf"
        y_data_mean = []
        index = window
        while True:
            if index > len(y_data):
                break

            fr = np.int(index - window)
            to = np.int(index)
            w = y_data[fr:to]
            y_data_mean.append(sum(w) * 1.0 / window)
            index = index + 1

        x_data = [(x+1) for x in range(len(y_data_mean))]
        plt.plot(x_data, y_data_mean, linewidth=1)
        plt.xlabel('Episodes')
        plt.ylabel(y_label)
        plt.savefig(filename)
        #plt.show()
