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

        self.max_val_observation = 0

        self.alpha = 0.7

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


    def is_action_valid(self, observation, observation_):
        o = np.array(observation)
        o_ = np.array(observation_)
        return not np.array_equal(o, o_)


    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Normalize the observations
        episode_observations_normalize = self.normalize_observations()

        # saving best val yet
        best_here = np.amax(self.episode_observations)
        if self.max_val_observation < best_here: self.max_val_observation = best_here

        # create a batch from data
        batches = self.create_batches(episode_observations_normalize, self.episode_actions, discounted_episode_rewards_norm)

        # Train on episode
        for n_epoch in range(self.epochs):
            n_batch = 0
            for batch in batches:
                f = batch[0]
                l = batch[1]
                r = batch[2]

                self.sess.run(self.train_op, feed_dict={
                     self.X: f,
                     self.Y: l,
                     self.discounted_episode_rewards_norm: r
                })

                print("Loss: ", self.sess.run(self.loss, feed_dict={self.X: f, self.Y: l,
                                                                    self.discounted_episode_rewards_norm: r}))

                '''
                epi = self.episode_observations[n_batch]
                board = np.reshape(epi, (self.n_y, self.n_y))
                print("\n\nboard")
                print(board)
                print("action ", self.episode_actions[n_batch])
                print("rewards ", self.episode_rewards[n_batch])
                print("normalized rewards ", discounted_episode_rewards_norm[n_batch])
                print("normalized observation ", episode_observations_normalize[n_batch])
                '''

                n_batch += 1

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            if not self.quiet:
                print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    def create_batches(self, x, y, z):
        batch = []
        x = np.vstack(np.array(x))
        y = np.vstack(np.array(y))
        z = z

        '''
        z = np.vstack(np.array(z))
        size = len(x)
        for i in range(size):
            x_ = np.array([x[i]])
            y_ = np.array([y[i]])
            z_ = np.array(z[i])
            batch.append((x_, y_, z_))
        '''

        batch.append((x, y, z))
        return batch

    def normalize_observations(self):
        episode_observations_normalize = []
        for episode_observation in self.episode_observations:
            observation = []
            for ob in episode_observation:
                if ob > 0: ob = np.log2(ob)
                observation.append(ob)

            observation /= np.amax(observation)
            episode_observations_normalize.append(observation)

        return episode_observations_normalize


    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards, dtype=np.float64)
        best_here = np.amax(self.episode_observations)
        best_yet = self.max_val_observation
        cumulative = 0  #best_here - (2 * best_yet)

        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        #discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        discounted_episode_rewards -= np.amax(discounted_episode_rewards)
        discounted_episode_rewards = np.abs(discounted_episode_rewards)

        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(None, self.n_x), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.n_y), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        units_input_layer = self.n_x
        units_output_layer = self.n_y

        from keras.layers import  Dense, Dropout, BatchNormalization
        from keras import initializers

        kerner_initializer = initializers.glorot_uniform(seed=1)
        dropout = self.epochs / 20.0

        n_neurons = np.int(1024)
        A1 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer, input_shape=(units_input_layer,))(self.X)
        A1 = BatchNormalization()(A1)
        A1 = Dropout(dropout)(A1)

        n_neurons = np.int(n_neurons / 2)
        A2 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A1)
        A2 = BatchNormalization()(A2)
        A2 = Dropout(dropout)(A2)

        A3 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A2)
        A3 = BatchNormalization()(A3)
        A3 = Dropout(dropout)(A3)

        n_neurons = np.int(n_neurons / 2)
        A4 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A3)
        A4 = BatchNormalization()(A4)
        A4 = Dropout(dropout)(A4)

        A5 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A4)
        A5 = BatchNormalization()(A5)
        A5 = Dropout(dropout)(A5)

        n_neurons = np.int(n_neurons / 2)
        A6 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A5)
        A6 = BatchNormalization()(A6)
        A6 = Dropout(dropout)(A6)

        A7 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A6)
        A7 = BatchNormalization()(A7)
        A7 = Dropout(dropout)(A7)

        n_neurons = np.int(n_neurons / 2)
        A8 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A7)
        A8 = BatchNormalization()(A8)

        A9 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A8)
        A9 = BatchNormalization()(A9)

        n_neurons = np.int(n_neurons / 2)
        A10 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A9)
        A10 = BatchNormalization()(A10)

        n_neurons = np.int(n_neurons / 2)
        A11 = Dense(units=n_neurons, activation='elu', kernel_initializer=kerner_initializer)(A10)
        A11 = BatchNormalization()(A11)

        Z = Dense(units=units_output_layer, kernel_initializer=kerner_initializer, activation=None)(A11)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = Z
        labels = self.Y
        self.outputs_softmax = tf.nn.softmax(logits, name='A12')

        with tf.name_scope('loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            self.loss = tf.reduce_mean(tf.add(tf.multiply(self.cross_entropy, 1 - self.alpha), tf.multiply(self.discounted_episode_rewards_norm, self.alpha)))  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def plot(self, y_data, y_label, n_episode, window=100, windowshift=100, dir='outputs/plots/'):
        filename = dir + y_label + "_" + str(n_episode + 1) + ".pdf"
        y_data_mean = [0]
        index = window
        while True:
            if index > len(y_data):
                break

            fr = np.int(index - window)
            to = np.int(index)
            w = y_data[fr:to]
            y_data_mean.append(sum(w) * 1.0 / window)
            index = index + windowshift

        x_data = [(x+1) for x in range(len(y_data_mean))]
        plt.plot(x_data, y_data_mean, linewidth=1)
        plt.xlabel('Episodes')
        plt.ylabel(y_label)
        plt.savefig(filename)
        #plt.show()
