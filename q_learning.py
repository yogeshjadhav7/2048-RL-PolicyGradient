"""
Q Learning Reinforcement Learning
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')

from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from sklearn.utils.extmath import softmax

class QLearning:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.001,
        reward_decay=0.97,
        epochs=10,
        random_exploration=0.1,
        save_path=None,
        total_episodes=1,
        restore_model=False,
        quiet=True,
        is_training_on=True,
    ):

        self.quiet = quiet # logging flag
        self.restore_model=restore_model # flag to tell whether to load existing model or create a new one
        self.save_path = save_path

        self.batch_size = 100
        self.sample_size_percent = 80.0

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.reward_decay = reward_decay # reward decay parameter
        self.epochs = epochs
        self.random_exploration = random_exploration # random exploration
        self.fixed_random_exploration = 0.9

        self.total_episodes = total_episodes
        self.curr_episode = 0

        self.decay_rate = self.lr / self.epochs # decay rate is computed

        self.replay_experiences = []
        self.replay_experiences_size_limit = 10000


        self.model = None
        if not restore_model:
            self.build_network()
        else:
            load_status = self.load_model()
            if not load_status:
                self.print_log("Cannot load the model at path " + self.save_path + ". Creating a new model!")
                self.build_network()

    def load_model(self):
        status = True
        try:
            self.model = load_model(self.save_path)
        except:
            status = False

        return status


    def save_model(self):
        self.model.save(self.save_path)


    def print_log(self, message):
        if not self.quiet:
            print(message)

        return


    def build_network(self):
        self.model = None

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='elu', padding='same', input_shape=(self.n_x, self.n_x, 1)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='elu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='elu', padding='same'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='elu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(256, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(128, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(64, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(self.n_y, activation='linear'))
        model.compile(loss=keras.losses.mse,
                      optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay_rate),
                      metrics=['accuracy'])

        if not self.quiet: model.summary()
        self.model = model


    def get_random_exploration(self):
        spisodic_factor = self.curr_episode / self.total_episodes
        adjusted_random_exploration = self.random_exploration + (spisodic_factor * (1 - self.random_exploration))
        return min(adjusted_random_exploration, self.fixed_random_exploration)


    def get_highest_tile_value(self, observation):
        return np.amax(observation.flatten())


    def observation_to_state(self, observation):
        reshaped_observation = np.reshape(observation, (1, self.n_x, self.n_x, 1))
        return np.divide(np.trunc(np.log2(np.add(reshaped_observation, 1))), 20.0)


    def predict(self, state):
        preds = self.model.predict(state)
        preds_classes = np.argmax(preds, axis=1)
        preds_probs = softmax(preds)
        return preds, preds_classes, preds_probs


    def choose_action(self, observation):
        state = self.observation_to_state(observation=observation)
        preds, preds_classes, preds_probs = self.predict(state=state)
        s = len(preds_probs.ravel())
        true_action = np.random.choice(range(s), p=preds_probs.ravel())
        random_action = np.random.randint(0, s - 1)
        actions = [true_action, random_action]
        curr_random_exploration = self.get_random_exploration()
        action_index = np.random.choice(range(2), p=[curr_random_exploration, 1 - curr_random_exploration])
        return actions[action_index]


    def calculate_reward(self, is_valid, raw_reward):
        if not is_valid: return -1.0
        if raw_reward == 0: return 0
        return np.log2(raw_reward) / 12


    def save_experience(self, observation, action, reward, observation_, is_game_over, is_move_valid):
        state = self.observation_to_state(observation=observation)
        state_ = self.observation_to_state(observation=observation_)
        exp_size = len(self.replay_experiences)
        if exp_size == self.replay_experiences_size_limit:
            index = np.random.randint(0, int(exp_size / 10) - 1)
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
            preds, preds_classes, _ = self.predict(state=states)

            label = np.array([preds[0]])
            target_value = reward

            if not is_game_over and is_move_valid: target_value = target_value + self.reward_decay * preds[1, preds_classes[1]]

            label[0, action] = target_value

            if len(features) == 0: features = state
            else: np.concatenate((features, state), axis=0)

            if len(labels) == 0: labels = label
            else: np.concatenate((labels, label), axis=0)

        return features, labels


    def train_model(self, features, labels):
        if len(features) == 0: return
        verbose = 1
        if self.quiet: verbose = 0
        self.model.fit(features, labels,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=verbose,
                  validation_data=(features, labels))

        self.save_model()
