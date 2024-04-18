# DEPENDENCIES
import random
import time
from datetime import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from typing import Dict, List, Tuple
from config import MODEL_PATH, SEED, VNF_PERIOD, DIVISION_FACTOR, TRAINING_EDGES, DQN_LOG_LEVEL, DQN_LOG_FILE_NAME, \
    PLOTS_PATH, SAVE_PLOTS, TIMESTEPS_LIMIT
import os
import logging
from colorlog import ColoredFormatter
import sys

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# FUNCTIONS
def max_tensor_value_and_its_idx(main_tensor, indices):
    """
    Find the maximum value and its corresponding index in a given tensor based on a list of indices.

    Parameters:
    - main_tensor (list or tuple): The tensor containing the values.
    - indices (list): List of indices to consider.

    Returns:
    - tuple: A tuple containing the maximum value and its index.

    Raises:
    - IndexError: If any index in the provided list is out of range.
    """

    try:
        selected_values = [(idx, main_tensor[idx]) for idx in indices]
        max_index, max_value = max(selected_values, key=lambda x: x[1])
        return max_value, max_index
    except IndexError:
        print("Tensor value index out of range. Please check your list of action space.")


def seed_torch(seed: int) -> None:
    """
    Set random seeds for reproducibility in PyTorch.

    Args:
    - seed (int): The seed value to use for randomization.

    Note:
    - This function sets the random seed for CPU and GPU (if available).
    - Disables CuDNN benchmarking and enables deterministic mode for GPU.
    """

    # Set the random seed for the CPU
    torch.manual_seed(seed)

    # Check if the CuDNN (CUDA Deep Neural Network library) is enabled
    if torch.backends.cudnn.enabled:
        # If CuDNN is enabled, set the random seed for CUDA (GPU)
        torch.cuda.manual_seed(seed)

        # Disable CuDNN benchmarking to ensure reproducibility
        torch.backends.cudnn.benchmark = False

        # Enable deterministic mode in CuDNN for reproducibility
        torch.backends.cudnn.deterministic = True


def get_highest_score_model():
    """
    Get the file path of the model with the highest score from a directory.

    Returns:
        str: The file path of the model with the highest score.

    Raises:
        FileNotFoundError: If no model files are found in the specified directory.
        Exception: If an unexpected error occurs during the process.

    Note:
        Make sure to handle the returned exceptions appropriately when calling this method.

    """
    try:
        # Get a list of model files in the directory
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pt")]

        if not model_files:
            raise FileNotFoundError(f"No model files found in {MODEL_PATH}")

        # Extract scores from file names and find the highest score
        scores = [int(file.split('_')[2].split('.')[0]) for file in model_files]
        highest_score = 0

        for i in range(1, len(scores)):
            if scores[highest_score] < scores[i]:
                highest_score = i

        # Construct and return the file path
        model_path = os.path.join(MODEL_PATH, model_files[highest_score])

        return model_path

    except FileNotFoundError as e:
        # Handle FileNotFoundError
        print(e)
    except Exception as e:
        # Handle unexpected errors during the process
        print(f"An unexpected error occurred when finding the highest score model: {e}")


def save_evaluation(evaluation_num: int, obs: np.ndarray, reward: float, done: bool, step: int) -> pd.DataFrame:
    """
    Save evaluation results to a DataFrame.

    Parameters:
        evaluation_num (int): The evauation iteration
        obs (np.ndarray): The observation array.
        reward (float): The reward for the current step.
        done (bool): Flag indicating if the episode is done.
        step (int): The current step number.

    Returns:
        pd.DataFrame: Updated DataFrame with the new evaluation result.
    """

    df = pd.DataFrame(columns=[
        'eval.',
        's_n', 't_n', 'gpu', 'ram', 'bw',
        'current_n', 'cav_fec', 'timesteps', 'Cost_2_t_n',
        'FEC0_gpu', 'FEC0_ram', 'FEC0_bw',
        'FEC1_gpu', 'FEC1_ram', 'FEC1_bw',
        'FEC2_gpu', 'FEC2_ram', 'FEC2_bw',
        'FEC3_gpu', 'FEC3_ram', 'FEC3_bw',
        'Reward', 'Done'
    ])

    # Create a DataFrame with one row
    new_elements = np.concatenate([[reward], [done]])
    result = np.concatenate([[evaluation_num], obs])
    result = np.concatenate([result, new_elements])

    # Update the DataFrame with the new row
    df.loc[step] = result

    # Now df contains the new evaluation result
    return df


def get_action_space(obs, previous_node):
    length = obs[2]
    current_node = obs[5]

    # Get those actions that are feasible: just useful positions of useful edges
    action_space = []
    a = 0
    i = 7
    for e in TRAINING_EDGES:
        if current_node == e[0]:
            for _ in range(16 * DIVISION_FACTOR):
                if obs[i] > length:
                    action_space.append(a)
                i += 1
                a += 1
        else:
            a += (16 * DIVISION_FACTOR)
    if action_space == []:
        return [a]
    else:
        return action_space


# CLASS
class Network(nn.Module):
    """A simple neural network module."""

    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize the Network.
        Args:
            in_dim (int): Dimensionality of the input.
            out_dim (int): Dimensionality of the output.
        """
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """

        return self.layers(x)


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_space: int, replay_buffer_size: int, batch_size: int = 32):
        """
        Initialize the ReplayBuffer.

        Args:
            obs_space (int): Dimensionality of the observations.
            replay_buffer_size (int): Maximum size of the buffer.
            batch_size (int): Size of the mini-batch for sampling. Default is 32.
        """
        self.observations_buf = np.zeros([replay_buffer_size, obs_space], dtype=np.float32)
        self.next_observations_buf = np.zeros([replay_buffer_size, obs_space], dtype=np.float32)
        self.actions_buf = np.zeros([replay_buffer_size], dtype=np.float32)
        self.rewards_buf = np.zeros([replay_buffer_size], dtype=np.float32)
        self.is_terminated_buf = np.zeros(replay_buffer_size, dtype=np.float32)
        self.max_size, self.batch_size = replay_buffer_size, batch_size
        self.ptr, self.size, = 0, 0

    # Class methods
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.

        Args:
            obs (np.ndarray): Current observation.
            act (np.ndarray): Action taken.
            rew (float): Reward received.
            next_obs (np.ndarray): Next observation.
            done (bool): Whether the episode is done.
        """
        self.observations_buf[self.ptr] = obs
        self.next_observations_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = act
        self.rewards_buf[self.ptr] = rew
        self.is_terminated_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """
        Sample a mini-batch from the replay buffer.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing arrays for observations, actions,
                                  rewards, next observations, and done flags.
        """
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(obs=self.observations_buf[idxs],
                    next_obs=self.next_observations_buf[idxs],
                    acts=self.actions_buf[idxs],
                    rews=self.rewards_buf[idxs],
                    done=self.is_terminated_buf[idxs])

    def __len__(self) -> int:
        """Get the current size of the replay buffer."""
        return self.size


class DQNAgent:
    """DQN Agent interacting with environment."""

    def __init__(
            self,
            env: gym.Env,
            replay_buffer_size: int = 2500,
            batch_size: int = 16,
            target_update: int = 400,
            epsilon_decay: float = 1 / 55000,
            seed=None,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.001,
            gamma: float = 0.9,
            learning_rate: float = 0.001,
            tau: float = 0.001):

        # Spaces
        # obs_space = len(env.observation_space)
        obs_space = env.observation_space.n
        action_space = env.action_space.n

        # Attributes
        self.env = env
        self.replay_buffer_size = ReplayBuffer(obs_space, replay_buffer_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.update_target_every_steps = target_update
        self.gamma = gamma
        self.tau = tau
        self.evaluation = 1

        # Logging settings
        self.logger = logging.getLogger('dqn')
        self.logger.setLevel(DQN_LOG_LEVEL)
        self.logger.addHandler(logging.FileHandler(DQN_LOG_FILE_NAME, mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)

        # Initialize UPC scenario graph
        self.graph = env.get_graph()

        # device: cpu / gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(self.device) == 'cpu':
            self.logger.warning('[!] WARNING! Processing device: ' + str(self.device))
        else:
            self.logger.info('[I] Processing device: ' + str(self.device))

        # networks: online, target
        self.online_net = Network(obs_space, action_space).to(self.device)
        self.target_net = Network(obs_space, action_space).to(self.device)

        # Update target net. params. from online net.
        self.target_net.load_state_dict(self.online_net.state_dict())
        # self.target_net.eval()  # only activate for inference

        # optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # transition to store in memory
        self.experience = list()

        # If SEED is available
        if SEED is not None:
            seed_torch(SEED)

    # Class methods
    @staticmethod
    def _plot(step: int, max_steps: int, scores: List[float], mean_scores: List[float], losses: List[float],
              mean_losses: List[float], epsilons: List[float]):
        """
        Plot the training progress.

        Parameters:
        - episode (int): The current episode number.
        - scores (List[float]): List of scores over episodes.
        - mean_scores (List[float]): List of mean scores over episodes.
        - losses (List[float]): List of losses over episodes.
        - mean_losses (List[float]): List of mean losses over episodes.
        - epsilons (List[float]): List of epsilon values over episodes.

        This static method plots the training progress, including scores, losses, and epsilon values.

        """
        clear_output(True)
        plt.figure(figsize=(20, 5))

        # Plot scores
        plt.subplot(131)
        plt.title(f'Step {step}. Average Score: {np.mean(scores[-100:]):.2f}')
        plt.plot(scores, label='Real')
        plt.plot(mean_scores, label='Promig')

        # Plot losses
        plt.subplot(132)
        plt.title('Loss Over Steps')
        plt.plot(losses, label='Real')
        plt.plot(mean_losses, label='Promig')

        # Plot epsilons
        plt.subplot(133)
        plt.title('Epsilons Over Steps')
        plt.plot(epsilons)

        if SAVE_PLOTS and step == max_steps:
            date = datetime.now().strftime('%Y%m%d%H%M')
            plt.savefig(PLOTS_PATH + date + '_' + str(step) + '.png')

        plt.show()

    def select_action(self, obs: np.ndarray, previous_node: int) -> int:
        """
        Selects only possible actions within the observation based on the E-Greedy strategy.

        Parameters:
        - obs (np.ndarray): The input observation array.

        Returns:
        - int: The selected action.

        E-Greedy strategy is used for exploration and exploitation. If a random number
        is less than the exploration rate (epsilon), a random action is chosen. Otherwise,
        the action with the maximum tensor value from the neural network is selected.

        The observation and the selected action are stored in the experience list.

        """

        # All possible combinations of edges and positions (actions) to move to
        action_space = get_action_space(obs, previous_node)
        self.logger.debug('[D] ACTION SPACE (' + str(len(action_space)) + ' options): ' + str(action_space))
        # if action_space == [16 * 24]:
        #     self.logger.debug('[D] Null action space detected!')
        #     self.logger.debug('[D] obs = ' + str(obs[0:7]))
        #     for i in range(24):
        #         self.logger.debug('[D] Edge ' + str(i) + ' | dist: ' +
        #                           str(obs[7 + (i * (16 * DIVISION_FACTOR + 2))]) + ', edge availability: ' +
        #                           str(obs[8 + (i * (16 * DIVISION_FACTOR + 2))]) + ', position availabilities: ' +
        #                           str(obs[9 + (i * (16 * DIVISION_FACTOR + 2)):7 + (
        #                                   (i + 1) * (16 * DIVISION_FACTOR + 2))]))
        #     self.logger.debug('[D] Action space (' + str(len(action_space)) + ' options): ' + str(action_space))

        # exploration
        if self.epsilon > np.random.rand():
            action = random.choice(action_space)
        # exploitation
        else:
            # Get all Q_values for the observation as a tensor
            tensor = self.online_net(torch.FloatTensor(obs).to(self.device))
            # Based on the possible action space within the observation, select the max Q_value (action)
            max_tensor_value, action = max_tensor_value_and_its_idx(tensor,
                                                                    action_space)  # max_tensor_value only for debug purposes

        # Save obs and action into experience list
        self.experience = [obs, action]

        return action

    def execute_action(self, action: int) -> Tuple[np.ndarray, np.float16, bool, dict[str, any]]:
        """
        Takes an action in the environment and receives a response.

        Parameters:
        - action (np.ndarray): The action to be taken in the environment.

        Returns:
        - Tuple[np.ndarray, np.float64, bool]: A tuple containing the next observation,
          reward, and a flag indicating whether the episode is done.

        This function sends the specified action to the environment, retrieves the
        next observation, reward, termination status, truncation status, and additional
        information. It considers both termination and truncation to determine if the
        episode is done. The reward, next observation, and done flag are then saved in
        the experience list, and the experience is stored in the replay buffer.

        """
        # Send action to env and get response
        next_obs, reward, is_terminated, is_truncated, info = self.env.step(action)

        # Consider terminated or truncated
        done = is_terminated or is_truncated

        # Save reward, next_obs and done in experience
        self.experience += [reward, next_obs, done]

        # Save experience in replay buffer
        self.replay_buffer_size.store(*self.experience)

        return next_obs, reward, done, info

    def update_model(self) -> float:
        """
        Update the model parameters using gradient descent.

        Returns:
        - float: The value of the computed loss.

        This method samples a batch from the replay buffer, computes the DQN loss,
        performs gradient descent, and updates the model parameters. The loss value
        is then returned.

        """
        # Sample a batch from the replay buffer
        samples = self.replay_buffer_size.sample_batch()

        # Compute DQN loss (receives a tensor)
        loss = self._compute_dqn_loss(samples)

        # Zero the gradients, perform backward pass, and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Computes the DQN loss using the given batch of samples.

        Parameters:
        - samples (Dict[str, np.ndarray]): A dictionary containing arrays for observations,
          next observations, actions, rewards, and done flags.

        Returns:
        - torch.Tensor: The computed DQN loss.

        This method calculates the DQN loss using the online and target networks, employing
        smooth L1 loss, and applying a discount factor (gamma). PyTorch's tensors are used
        for efficient computation on the specified device.

        """
        # Convert arrays to PyTorch tensors and move them to the specified device
        observations = torch.FloatTensor(samples["obs"]).to(self.device)  # obs
        next_observations = torch.FloatTensor(samples["next_obs"]).to(self.device)  # next obs
        actions = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)  # actions
        rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)  # rewards
        dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)  # dones

        # Q-values from the online network for the taken actions
        curr_q_value = self.online_net(observations).gather(1, actions)

        # Q-values from the target network for the next observation
        # next_q_value = self.target_net(next_observations).max(dim=1, keepdim=True)[0].detach()  # Standard DQN

        # DQN
        next_q_value = self.target_net(next_observations).gather(1, self.online_net(next_observations).argmax(dim=1,
                                                                                                              keepdim=True)).detach()

        # Mask for terminal states (dones)
        mask = 1 - dones
        # Compute the target Q-value using the Bellman equation
        target = (rewards + self.gamma * next_q_value * mask).to(self.device)

        # Calculate the DQN loss using smooth L1 loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        # loss = F.mse_loss(curr_q_value, target)
        # loss = F.huber_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """
        Perform a hard update of the target network.

        This method copies the weights from the online network to the target network.

        """

        # Copy the state dictionary from the online network to the target network
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _target_soft_update(self, tau: float = 0.01):
        """
        Perform a soft update of the target network.

        Parameters:
        - tau (float, optional): The interpolation parameter for the soft update.
          A smaller value results in a slower update. Default is 0.01.

        This method updates the target network parameters as a weighted average
        of the online network parameters.

        """

        # Get the state dictionaries of the online and target networks
        online_params = dict(self.online_net.named_parameters())
        target_params = dict(self.target_net.named_parameters())

        # Update the target network parameters using a weighted average
        for name in target_params:
            target_params[name].data.copy_(tau * online_params[name].data + (1.0 - tau) * target_params[name].data)

    def load_model(self):
        """
        Load a pre-trained model from a file and update the online and target networks.

        This method attempts to load a pre-trained model from the specified file path. If successful,
        it updates the weights of both the online and target networks with the loaded state dictionary.
        Additionally, the target network is updated with the state dictionary of the online network.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            Exception: If an unexpected error occurs during the model loading process.

        Note:
            Make sure to handle the returned exceptions appropriately when calling this method.

        """
        try:
            # Try to open the model file and load its state dictionary
            model_name = get_highest_score_model()
            with open(model_name, 'rb') as model_file:
                state_dict = torch.load(model_file)
                # Update the online network with the loaded state dictionary
                self.online_net.load_state_dict(state_dict)
                # Update the target network with the state dictionary of the online network
                self.target_net.load_state_dict(self.online_net.state_dict())
        except FileNotFoundError:
            # Handle the case where the model file is not found
            self.logger.error('[!] Model file not found at ' + MODEL_PATH)
        except Exception as e:
            # Handle unexpected errors during the model loading process
            self.logger.error('An unexpected error occurred when loading the saved model: ' + str(e))

    def load_custom_model(self, model_name):
        """
        Load a pre-trained model from a file and update the online and target networks.

        This method attempts to load a pre-trained model from the specified file path. If successful,
        it updates the weights of both the online and target networks with the loaded state dictionary.
        Additionally, the target network is updated with the state dictionary of the online network.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            Exception: If an unexpected error occurs during the model loading process.

        Note:
            Make sure to handle the returned exceptions appropriately when calling this method.

        """
        try:
            # Try to open the model file and load its state dictionary
            with open(model_name, 'rb') as model_file:
                state_dict = torch.load(model_file)
                # Update the online network with the loaded state dictionary
                self.online_net.load_state_dict(state_dict)
                # Update the target network with the state dictionary of the online network
                self.target_net.load_state_dict(self.online_net.state_dict())
        except FileNotFoundError:
            # Handle the case where the model file is not found
            self.logger.error('[!] Model file not found at ' + MODEL_PATH)
        except Exception as e:
            # Handle unexpected errors during the model loading process
            self.logger.error('An unexpected error occurred when loading the saved model: ' + str(e))

    def train(self, max_steps: int, plotting_interval: int = 1000, evaluation_interval: int = 10000,
              monitor_training: int = 1000):
        """
        Train the agent using the DQN algorithm.

        Parameters:
        - max_steps (int): The maximum number of training steps.
        - plotting_interval (int, optional): The interval for printing training progress. Default is 1000.

        This method initializes the training process, iteratively selects actions using E-Greedy
        strategy, updates the model, and tracks training metrics such as scores, losses, and epsilons.
        Hard updates to the target network are performed periodically. Training progress is printed at
        specified intervals.
        """
        # Capture timestamp
        start = time.time()
        # result_df = []
        # Initial observation from the initialized environment
        obs, info = self.env.reset(seed=self.seed)

        # Initialize variables
        score = 0
        epsilons = []
        losses = []
        mean_losses = []
        scores = []
        mean_scores = []
        max_score = -99999
        update_cnt = 0
        episodes_count = 0
        step = 0
        model_name = None

        # Start training
        try:
            for step in range(1, max_steps + 1):
                # E-Greedy action selection
                action = self.select_action(obs, info['previous_node'])

                # Execute action in the environment and receive response
                next_obs, reward, done, info = self.execute_action(action)

                obs = next_obs
                score += reward

                # if episode ends, reset the environment and record scores
                if done:
                    obs, info = self.env.reset(seed=self.seed)
                    scores.append(score)
                    if len(scores) < 100:
                        mean_scores.append(np.mean(scores[0:]))
                    else:
                        mean_scores.append(np.mean(scores[-100:]))
                    score = 0
                    episodes_count += 1

                # if training is ready
                if len(self.replay_buffer_size) >= self.batch_size:
                    # Update the model and record the loss
                    loss = self.update_model()
                    losses.append(loss)
                    if len(losses) < 100:
                        mean_losses.append(np.mean(losses[0:]))
                    else:
                        mean_losses.append(np.mean(losses[-100:]))
                    update_cnt += 1

                    # linearly decay epsilon
                    self.epsilon = max(self.min_epsilon,
                                       self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                    epsilons.append(self.epsilon)

                    # if hard update is needed
                    if update_cnt % self.update_target_every_steps == 0:
                        # self._target_hard_update()
                        self._target_soft_update(self.tau)

                # Print training progress at specified intervals
                if step % plotting_interval == 0:
                    self._plot(step, max_steps, scores, mean_scores, losses, mean_losses, epsilons)
                    plt.close()

                if step % monitor_training == 0:
                    self.logger.info(f"[I] Step: {step} | "
                                     f"Rewards: {round(np.mean(scores[-100:]), 3)} | "
                                     f"Loss: {round(np.mean(losses[-100:]), 3)} | "
                                     f"Epsilons: {round(np.mean(epsilons[-100:]), 3)} | "
                                     f"Episodes: {episodes_count}")

                # if update_cnt % evaluation_interval == 0:
                #     result_df.append(self.evaluate(copy.deepcopy(self.env)))
        except KeyboardInterrupt:
            self.logger.info(f"[I] Step: {step} | "
                             f"Rewards: {round(np.mean(scores[-100:]), 3)} | "
                             f"Loss: {round(np.mean(losses[-100:]), 3)} | "
                             f"Epsilons: {round(np.mean(epsilons[-100:]), 3)} | "
                             f"Episodes: {episodes_count}")
            if episodes_count > 1000:
                self._plot(step, max_steps, scores, mean_scores, losses, mean_losses, epsilons)
                plt.close()

        end = time.time()

        self.logger.info('[I] Total run time steps: ' + str(step))
        self.logger.info('[I] Total run episodes: ' + str(episodes_count))
        self.logger.info('[I] Total elapsed time: ' + str(end - start))
        self.logger.info('[I] Final average reward: ' + str(round(np.mean(scores[-100:]), 3)))

        # Save model
        if episodes_count > 1000:
            date = datetime.now().strftime('%Y%m%d%H%M')
            model_name = 'model_' + date + '_' + str(int(np.mean(scores[-100:]))) + '.pt'
            torch.save(self.online_net.state_dict(), MODEL_PATH + model_name)

        # Close env
        self.env.close()

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 50) -> List[pd.DataFrame]:
        """
        Evaluate the agent's performance over multiple episodes.

        Parameters:
            eval_env (gym.Env): The environment for evaluation.
            n_episodes (int, optional): The number of episodes to run the evaluation. Default is 10.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing evaluation results for each episode.

        This method temporarily sets the exploration rate (epsilon) to 0 to evaluate the agent's behavior.
        It runs the agent in the provided environment for the specified number of episodes, calculates the mean score,
        and prints the result. The original epsilon value is restored after evaluation.

        """

        # Temporarily store the current epsilon value
        original_epsilon = self.epsilon

        # Set epsilon to 0 for evaluation
        self.epsilon = 0

        # Episode reward list
        episode_rewards = []

        num_lost = 0
        num_one_hop_perfect = 0
        num_one_hop_arrived_w_errors_now = 0
        num_two_hop_perfect = 0
        num_two_hop_arrived_w_errors = 0
        num_two_hop_arrived_w_errors_before = 0
        num_two_hop_arrived_w_errors_now = 0
        num_three_hop_perfect = 0
        num_three_hop_arrived_w_errors = 0
        num_three_hop_arrived_w_errors_before = 0
        num_three_hop_arrived_w_errors_now = 0
        num_arrived_wo_routing = 0
        num_arrived_w_errors = 0

        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            previous_reward = 0
            step = 0
            reward = 0

            while not done:
                previous_node = info['previous_node']
                action = self.select_action(obs, previous_node)
                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                obs = next_obs
                done = terminated or truncated
                step += 1
                if done:
                    if step > TIMESTEPS_LIMIT:
                        num_lost += 1
                    elif step == 1 and reward >= 150.0:
                        num_one_hop_perfect += 1
                    elif step == 1 and previous_reward == 0 and reward < 50:
                        num_one_hop_arrived_w_errors_now += 1
                    elif step == 2 and reward >= 150.0:
                        num_two_hop_perfect += 1
                    elif step == 2 and reward == previous_reward:
                        num_two_hop_arrived_w_errors_before += 1
                    elif step == 2 and previous_reward == 0 and reward < 50:
                        num_two_hop_arrived_w_errors_now += 1
                    elif step == 2:
                        num_two_hop_arrived_w_errors += 1
                    elif step == 3 and reward >= 150.0:
                        num_three_hop_perfect += 1
                    elif step == 3 and reward == previous_reward:
                        num_three_hop_arrived_w_errors_before += 1
                    elif step == 3 and previous_reward == 0 and reward < 50:
                        num_three_hop_arrived_w_errors_now += 1
                    elif step == 3:
                        num_three_hop_arrived_w_errors += 1
                    elif step > 3:
                        num_arrived_wo_routing += 1
                    else:
                        num_arrived_w_errors += 1

                previous_reward = reward

            # Episode reward
            # episode_rewards.append(previous_reward)
            episode_rewards.append(reward)

        # Calculate and print the mean score over episodes
        mean_score = np.mean(episode_rewards)
        self.logger.info(f"\n[I] Evaluation mean score over {n_episodes} episodes: {mean_score:.2f}\n")
        self.logger.info(f'[I] Number of episodes where flow has lost: {num_lost}')
        self.logger.info(f'[I] Number of episodes where 1-hop flow has been perfectly routed and scheduled: {num_one_hop_perfect}')
        self.logger.info(f'[I] Number of episodes where 1-hop flow reached its target with schedule faults: {num_one_hop_arrived_w_errors_now}')
        self.logger.info(f'[I] Number of episodes where 2-hop flow has been perfectly routed and scheduled: {num_two_hop_perfect}')
        self.logger.info(f'[I] Number of episodes where 2-hop flow reached its target with schedule faults only before arriving: {num_two_hop_arrived_w_errors_before}')
        self.logger.info(f'[I] Number of episodes where 2-hop flow reached its target with schedule faults only when arriving: {num_two_hop_arrived_w_errors_now}')
        self.logger.info(f'[I] Number of episodes where 2-hop flow reached its target with schedule faults when and before arriving: {num_two_hop_arrived_w_errors}')
        self.logger.info(f'[I] Number of episodes where 3-hop flow has been perfectly routed and scheduled: {num_three_hop_perfect}')
        self.logger.info(f'[I] Number of episodes where 3-hop flow reached its target with schedule faults only before arriving: {num_three_hop_arrived_w_errors_before}')
        self.logger.info(f'[I] Number of episodes where 3-hop flow reached its target with schedule faults only when arriving: {num_three_hop_arrived_w_errors_now}')
        self.logger.info(f'[I] Number of episodes where 3-hop flow reached its target with schedule faults when and before arriving: {num_three_hop_arrived_w_errors}')
        self.logger.info(f'[I] Number of episodes where multihop flow reached its target with routing faults: {num_arrived_wo_routing}')
        self.logger.info(f'[I] Number of not able to describe cases: {num_arrived_w_errors}')

        # Reset epsilon to its original value after evaluation for continuing with training
        self.epsilon = original_epsilon

        # Close the environment
        eval_env.close()
        self.evaluation += 1
