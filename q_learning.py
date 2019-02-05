"""
Q Learning Reinforcement Learning
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model, clone_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from sklearn.utils.extmath import softmax

class QLearning:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.9,
        epochs=10,
        random_exploration=0.1,
        q_save_path=None,
        q_weights_save_path=None,
        t_save_path=None,
        total_episodes=1,
        restore_model=False,
        quiet=True,
        T=1,
        is_training_on=True,
    ):

        self.quiet = quiet # logging flag
        self.restore_model=restore_model # flag to tell whether to load existing model or create a new one
        self.q_save_path = q_save_path
        self.q_weights_save_path = q_weights_save_path
        self.t_save_path = t_save_path

        self.batch_size = 100
        self.sample_size_percent = 90.0

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.q_lr = 0.8
        self.reward_decay = reward_decay # reward decay parameter
        self.epochs = epochs
        self.random_exploration = random_exploration # random exploration
        self.fixed_random_exploration = 0.9
        self.use_dropout = False

        self.T = T

        self.total_episodes = total_episodes
        self.curr_episode = 0

        self.decay_rate = self.lr / self.epochs # decay rate is computed

        self.replay_experiences = []
        self.replay_experiences_size_limit = 10000

        self.is_training_on = is_training_on

        self.episodic_highest_tiles_track = []

        self.q_model = None
        self.t_model = None

        if not restore_model:
            self.build_network()
        else:
            load_status = self.load_q_model()
            if not load_status:
                self.print_log("Cannot load the model at path " + self.q_save_path + ". Creating a new model!")
                self.build_network()

        self.save_q_model()
        self.load_t_model()


    # tries to load existing q model
    # if failed then creates new
    def load_q_model(self):
        status = True
        try:
            self.q_model = load_model(self.q_save_path)
        except:
            status = False

        return status


    def save_q_model(self):
        self.q_model.save(self.q_save_path)
        self.q_model.save_weights(self.q_weights_save_path)


    def load_t_model(self):
        self.t_model = clone_model(self.q_model)
        self.t_model.load_weights(self.q_weights_save_path)


    def print_log(self, message):
        if not self.quiet:
            print(message)

        return


    def transfer_model(self):
        if self.curr_episode <= 0 or self.curr_episode % self.T != 0: return
        self.save_q_model()
        self.load_t_model()
        return


    # builds new model and initialize self.q_model with it
    def build_network(self):
        self.q_model = None

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same', input_shape=(self.n_x, self.n_x, 1)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        if self.use_dropout: model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        if self.use_dropout: model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        if self.use_dropout: model.add(Dropout(0.2))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        if self.use_dropout: model.add(Dropout(0.2))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        if self.use_dropout: model.add(Dropout(0.2))

        model.add(Dense(self.n_y, activation='linear'))
        model.compile(loss=keras.losses.mse,
                      optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay_rate),
                      metrics=['accuracy'])

        if not self.quiet: model.summary()

        self.q_model = model


    def get_random_exploration(self):
        if not self.is_training_on: return 1.0
        spisodic_factor = self.curr_episode / self.total_episodes
        adjusted_random_exploration = self.random_exploration + (spisodic_factor * (1 - self.random_exploration))
        return min(adjusted_random_exploration, self.fixed_random_exploration)


    def get_highest_tile_value(self, observation):
        return np.amax(observation.flatten())


    def observation_to_state(self, observation):
        reshaped_observation = np.reshape(observation, (1, self.n_x, self.n_x, 1))
        return np.divide(np.trunc(np.log2(np.add(reshaped_observation, 1))), 20.0)


    def predict(self, state, model):
        preds = model.predict(state)
        preds_classes = np.argmax(preds, axis=1)
        preds_probs = softmax(preds)
        return preds, preds_classes, preds_probs


    def choose_action(self, observation):
        state = self.observation_to_state(observation=observation)
        preds, preds_classes, preds_probs = self.predict(state=state, model=self.t_model)
        s = len(preds_probs.ravel())
        true_action = np.random.choice(range(s), p=preds_probs.ravel())
        random_action = np.random.randint(0, s - 1)
        actions = [true_action, random_action]
        curr_random_exploration = self.get_random_exploration()
        action_index = np.random.choice(range(2), p=[curr_random_exploration, 1 - curr_random_exploration])
        return actions[action_index]


    def calculate_reward(self, is_valid, is_game_over, raw_reward, observation):
        if not is_valid: return 0
        if is_game_over:
            highest_tile_value = self.get_highest_tile_value(observation=observation)
            reward = (np.log2(highest_tile_value) - np.log2(512.0)) / np.log2(512.0)
            return reward

        #if raw_reward == 0: return 0
        #return np.log2(raw_reward) / np.log2(512.0)
        return 0


    def save_experience(self, observation, action, reward, observation_, is_game_over, is_move_valid):
        state = self.observation_to_state(observation=observation)
        state_ = self.observation_to_state(observation=observation_)
        exp_size = len(self.replay_experiences)
        if exp_size == self.replay_experiences_size_limit:

            index = 0

            '''
            can_remove = False
            while not can_remove:
                index = np.random.randint(0, int(exp_size / 2) - 1)
                can_remove = not self.replay_experiences[index][4]
                
            '''

            self.replay_experiences.pop(index)

        self.replay_experiences.append((state, action, reward, state_, is_game_over, is_move_valid))


    def sample_from_experience(self):
        experience_size = len(self.replay_experiences)
        n_sample = int(min(self.sample_size_percent * experience_size / 100, self.batch_size))
        features = []
        labels = []
        indices = np.random.choice(experience_size, n_sample, replace=False)
        for idx in range(len(indices)):
            experience = self.replay_experiences[idx]
            state = experience[0]
            action = experience[1]
            reward = experience[2]
            state_ = experience[3]
            is_game_over = experience[4]
            is_move_valid = experience[5]

            states = np.concatenate((state, state_), axis=0)
            preds, preds_classes, _ = self.predict(state=states, model=self.t_model)

            label = np.array([preds[0]])

            future_reward = 0
            if not is_game_over and is_move_valid: future_reward = self.reward_decay * preds[1, preds_classes[1]]

            delta = reward + future_reward

            label[0, action] = ((1 - self.q_lr) * label[0, action]) + (self.q_lr * delta)

            if len(features) == 0: features = state
            else: np.concatenate((features, state), axis=0)

            if len(labels) == 0: labels = label
            else: np.concatenate((labels, label), axis=0)

        return features, labels


    def train_model(self, features, labels):
        if not self.is_training_on: return
        if len(features) == 0: return
        verbose = 1
        if self.quiet: verbose = 0
        self.q_model.fit(features, labels,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=verbose,
                  validation_data=(features, labels))

        self.save_q_model()


    def plot_progress(self, y_data, y_label, n_episode, window_size=10, stride=1, dir='outputs/plots/'):
        episode_str = "" #"_" + str(n_episode + 1)
        filename = dir + y_label \
                   + episode_str \
                   + ".pdf"

        y_data_mean = [0]
        index = window_size

        while True:
            if index > len(y_data): break

            fr = np.int(index - window_size)
            to = np.int(index)
            w = y_data[fr:to]
            y_data_mean.append(sum(w) * 1.0 / window_size)
            index = index + stride

        if len(y_data_mean) == 1: return

        x_data = [(x+1) for x in range(len(y_data_mean))]
        plt.plot(x_data, y_data_mean, linewidth=1)
        plt.xlabel('Episodes #')
        plt.ylabel(y_label)
        plt.savefig(filename)