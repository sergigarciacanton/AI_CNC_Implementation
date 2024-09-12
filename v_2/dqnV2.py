# DEPENDENCIES
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from typing import Dict, List
from v_2.configV2 import MODEL_PATH, SEED, PLOTS_PATH, SAVE_PLOTS
import os

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


# CLASS
class Network(nn.Module):
    """A simple neural network module."""

    def __init__(self, in_dim: int, out_dim: int, neurons: int):
        """
        Initialize the Network.
        Args:
            in_dim (int): Dimensionality of the input.
            out_dim (int): Dimensionality of the output.
        """
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, neurons),
            nn.ReLU(),
            nn.Linear(neurons, out_dim)
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
            replay_buffer_size: int = 2500,
            batch_size: int = 16,
            max_epsilon: float = 1.0,
            gamma: float = 0.9,
            learning_rate: float = 0.001,
            obs_space: int = 10,
            action_space: int = 10,
            neurons: int = 10):

        # Attributes
        self.replay_buffer_size = ReplayBuffer(obs_space, replay_buffer_size, batch_size)
        self.epsilon = max_epsilon
        self.gamma = gamma

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # networks: online, target
        self.online_net = Network(obs_space, action_space, neurons).to(self.device)
        self.target_net = Network(obs_space, action_space, neurons).to(self.device)

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
    def plot(step: int, max_steps: int, scores: List[float], mean_scores: List[float], losses: List[float],
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
            plt.savefig(PLOTS_PATH + 'plot_' + date + '_' + str(int(np.mean(scores[-100:]))) + '.png')

        plt.show()

    def select_action(self, obs: np.ndarray, action_space: list) -> int:
        """
        Selects only possible actions within the observation based on the E-Greedy strategy.

        Parameters:
        - obs (np.ndarray): The input observation array.
        - action_space (list): Array of possible actions to take

        Returns:
        - int: The selected action.

        E-Greedy strategy is used for exploration and exploitation. If a random number
        is less than the exploration rate (epsilon), a random action is chosen. Otherwise,
        the action with the maximum tensor value from the neural network is selected.

        The observation and the selected action are stored in the experience list.

        """

        # exploration
        if self.epsilon > np.random.rand():
            action = random.choice(action_space)
        # exploitation
        else:
            # Get all Q_values for the observation as a tensor
            tensor = self.online_net(torch.FloatTensor(obs).to(self.device))
            # Based on the possible action space within the observation, select the max Q_value (action)
            max_tensor_value, action = max_tensor_value_and_its_idx(tensor, action_space)

        # Save obs and action into experience list
        self.experience = [obs, action]

        return action

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
        next_q_value = self.target_net(next_observations).gather(1, self.online_net(next_observations).
                                                                 argmax(dim=1, keepdim=True)).detach()

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

    def target_soft_update(self, tau: float = 0.01):
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
            print('[!] Model file not found at ' + MODEL_PATH)
        except Exception as e:
            # Handle unexpected errors during the model loading process
            print('An unexpected error occurred when loading the saved model: ' + str(e))

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
            print('[!] Model file not found at ' + MODEL_PATH)
        except Exception as e:
            # Handle unexpected errors during the model loading process
            print('An unexpected error occurred when loading the saved model: ' + str(e))
