import numpy as np
import os
import random
import shutil
from statistics import mean
from game_models.base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork

GAMMA = 0.999
# MEMORY_SIZE = 900000
MEMORY_SIZE = 20000
BATCH_SIZE = 64
TRAINING_FREQUENCY = 32
TARGET_NETWORK_UPDATE_FREQUENCY = TRAINING_FREQUENCY * 50
# TARGET_NETWORK_UPDATE_FREQUENCY = 40000
#can change MODEL_PERSISTENCE_UPDATE_FREQUENCY if want to update(save) model more frequntly
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 4000
# MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = min(1000,MEMORY_SIZE)
# REPLAY_START_SIZE = 50000

# EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.1
EXPLORATION_MAX = 0.1
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS =   850000
# EXPLORATION_STEPS = 850000

# EXPLORATION_MAX = 0.02
# EXPLORATION_MIN = 0.01
# EXPLORATION_TEST = 0.02
# EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS
import pdb

class DDQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path,start_from_0):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        # self.ddqn.summary()
        if os.path.isfile(self.model_path) and not start_from_0:
            print("Loading Pre Trained Model")
            self.ddqn.load_weights(self.model_path)
        else :
            print("Model starting with random weights")

    def _save_model(self,extra_str=""):
        self.ddqn.save_weights(self.model_path+extra_str)


class DDQNSolver(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ddqn/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN testing",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",
                               testing_model_path)

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        # q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        q_values = self.ddqn.predict(np.expand_dims(state, axis=0), batch_size=1)
        return np.argmax(q_values[0])


class DDQNTrainer(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space,start_from_0,steps):
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN training",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/training/" + self._get_date() + "/",
                               "./output/neural_nets/" + game_name + "/ddqn/" +  "my_model.h5",start_from_0)
                               # "./output/neural_nets/" + game_name + "/ddqn/" + self._get_date() + "/model.h5")

        # if os.path.exists(os.path.dirname(self.model_path)):
            # shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        # os.makedirs(os.path.dirname(self.model_path))
        self._save_model()

        self.ddqn_target = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        self._reset_target_network()
        self.epsilon = EXPLORATION_MAX
        if not start_from_0:
            self.epsilon = EXPLORATION_MAX - ((EXPLORATION_MAX-EXPLORATION_MIN)*steps/EXPLORATION_STEPS)
        self.memory = []

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        # q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        q_values = self.ddqn.predict(np.expand_dims(state, axis=0), batch_size=1)
        # print(q_values[0],q_values[0].shape,np.argmax(q_values[0]))
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.append({"current_state": current_state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                            "terminal": terminal})
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def step_update(self, total_step):
        if len(self.memory) < REPLAY_START_SIZE:
            return

        if total_step % TRAINING_FREQUENCY == 0:
            loss, accuracy, average_max_q = self._train()
            self.logger.add_loss(loss)
            self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)

        self._update_epsilon()

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            print("\nsaving model after",total_step,"steps")
            self._save_model()
        if total_step % 10000 == 0:
            self._save_model("_"+str(total_step))

        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            # print("updating ddqn_target weights model after",total_step,"steps")
            self._reset_target_network()
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon),end=" & ")
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))

    def _train(self):
        batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
        if len(batch) < BATCH_SIZE:
            return

        # current_states = np.zeros((len(batch),37,165))
        current_states = []
        q_values = []
        max_q_values = []

        for entry in batch:
            current_state = np.expand_dims(entry["current_state"], axis=0)
            current_states.append(current_state)
            next_state = np.expand_dims(entry["next_state"], axis=0)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.ddqn.predict(current_state)[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))
        # print("fit to be called")
        # pdb.set_trace()
        fit = self.ddqn.fit(np.asarray(current_states).reshape((BATCH_SIZE,37,165,1)),
                            np.asarray(q_values).squeeze(),
                            batch_size=BATCH_SIZE,
                            verbose=0)
        # pdb.set_trace()
        loss = fit.history["loss"][0]
        accuracy = fit.history["accuracy"][0]
        return loss, accuracy, mean(max_q_values)

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())