import logging
import sys
import time
from datetime import datetime
from typing import Any, SupportsFloat
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from colorlog import ColoredFormatter
from v_2.EnvironmentV2 import EnvironmentTSN
from v_2.configV2 import MODEL_PATH, DQN_LOG_FILE_NAME, DQN_LOG_LEVEL, DIVISION_FACTOR
from v_2.dqnV2 import DQNAgent


class DoubleAgent:
    def __init__(
            self,
            env: EnvironmentTSN,
            replay_buffer_size: int = 2500,
            batch_size: int = 16,
            target_update: int = 400,
            epsilon_decay: float = 1 / 55000,
            seed=None,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.001,
            gamma: float = 0.9,
            learning_rate: float = 0.001,
            tau: float = 0.001,
            log_file_id: str = '0'):

        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.update_target_every_steps = target_update
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau

        self.routing_agent = DQNAgent(
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            max_epsilon=max_epsilon,
            gamma=gamma,
            learning_rate=learning_rate,
            obs_space=6,
            action_space=4,
            neurons=20
        )

        self.scheduling_agent = DQNAgent(
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            max_epsilon=max_epsilon,
            gamma=gamma,
            learning_rate=learning_rate,
            obs_space=17,
            action_space=17,
            neurons=20
        )

        # Logging settings
        self.logger = logging.getLogger('dqn')
        self.logger.setLevel(DQN_LOG_LEVEL)
        self.logger.addHandler(
            logging.FileHandler(DQN_LOG_FILE_NAME + log_file_id + '.log', mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)

    def get_routing_action_space(self, obs):
        # Get those actions that are feasible: just useful edges
        action_space = []
        neighbors = list(nx.neighbors(self.env.get_graph(), obs[1]))
        for i in range(len(neighbors)):
            if obs[i + 3] < 300:
                action_space.append(i)
        if not action_space:
            return [3]
        else:
            return action_space

    def get_scheduling_action_space(self, obs):
        length = obs[0]

        # Get those actions that are feasible: just useful positions
        action_space = []
        for a in range(self.env.hyperperiod * DIVISION_FACTOR):
            if obs[a + 1] > length:
                action_space.append(a)
        if not action_space:
            return [self.env.hyperperiod * DIVISION_FACTOR]
        else:
            return action_space

    def execute_action(self, next_node: int, position_id: int) -> tuple[Any, SupportsFloat, bool, dict[str, Any]]:
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
        routing_next_obs, routing_reward, \
            is_terminated, is_truncated, info = self.env.step(next_node, position_id)

        # Consider terminated or truncated
        done = is_terminated or is_truncated
        # Save reward, next_obs and done in experience
        self.routing_agent.experience += [routing_reward, routing_next_obs, done]
        self.routing_agent.replay_buffer_size.store(*self.routing_agent.experience)
        # print(self.routing_agent.experience)

        return routing_next_obs, routing_reward, done, info

    def train(self, max_steps: int, plotting_interval: int = 1000, monitor_training: int = 1000):
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

        # Initial observation from the initialized environment
        routing_obs, info = self.env.reset()

        # Initialize variables
        routing_score = 0
        routing_epsilons = []
        routing_losses = []
        routing_mean_losses = []
        routing_scores = []
        routing_mean_scores = []
        routing_update_cnt = 0

        scheduling_score = 0
        scheduling_epsilons = []
        scheduling_losses = []
        scheduling_mean_losses = []
        scheduling_scores = []
        scheduling_mean_scores = []
        scheduling_update_cnt = 0

        episodes_count = 0
        step = 0
        steps_in_episode = 0

        # Start training
        try:
            for step in range(1, max_steps + 1):
                # E-Greedy next_node selection
                # print('Routing action space: ' + str(self.get_routing_action_space(routing_obs)))
                next_node = self.routing_agent.select_action(routing_obs,
                                                             self.get_routing_action_space(routing_obs))

                # print('Next node: ' + str(next_node))
                # print('Scheduling action space: ' + str(self.get_scheduling_action_space(scheduling_obs,
                #                                                                         routing_obs[1],
                #                                                                         next_node)))
                if steps_in_episode == 0:
                    scheduling_obs, info = self.env.scheduling_reset(next_node)
                else:
                    scheduling_next_obs, scheduling_reward, info = self.env.scheduling_step(next_node)

                    # Consider terminated or truncated
                    # Save reward, next_obs and done in experience
                    self.scheduling_agent.experience += [scheduling_reward, scheduling_next_obs, done]
                    self.scheduling_agent.replay_buffer_size.store(*self.scheduling_agent.experience)
                    # print(self.scheduling_agent.experience)

                    scheduling_obs = scheduling_next_obs
                    scheduling_score += scheduling_reward

                position_id = self.scheduling_agent.select_action(scheduling_obs,
                                                                  self.get_scheduling_action_space(scheduling_obs))

                # Execute next_node in the environment and receive response
                routing_next_obs, routing_reward, \
                    done, info = self.execute_action(next_node, position_id)

                routing_obs = routing_next_obs
                routing_score += routing_reward

                steps_in_episode += 1

                # if episode ends, reset the environment and record scores
                if done:
                    scheduling_next_obs, scheduling_reward, info = self.env.scheduling_step(next_node)

                    # Consider terminated or truncated
                    # Save reward, next_obs and done in experience
                    self.scheduling_agent.experience += [scheduling_reward, scheduling_next_obs, done]
                    self.scheduling_agent.replay_buffer_size.store(*self.scheduling_agent.experience)
                    # print(self.scheduling_agent.experience)

                    scheduling_score += scheduling_reward

                    routing_obs, info = self.env.reset()
                    routing_scores.append(routing_score)
                    scheduling_scores.append(scheduling_score)
                    if len(routing_scores) < 100:
                        routing_mean_scores.append(float(np.mean(routing_scores[0:])))
                    else:
                        routing_mean_scores.append(float(np.mean(routing_scores[-100:])))
                    if len(scheduling_scores) < 100:
                        scheduling_mean_scores.append(float(np.mean(scheduling_scores[0:])))
                    else:
                        scheduling_mean_scores.append(float(np.mean(scheduling_scores[-100:])))
                    routing_score = 0
                    scheduling_score = 0
                    episodes_count += 1
                    steps_in_episode = 0

                # if training is ready
                if len(self.routing_agent.replay_buffer_size) >= self.batch_size:
                    # Update the model and record the loss
                    loss = self.routing_agent.update_model()
                    routing_losses.append(loss)
                    if len(routing_losses) < 100:
                        routing_mean_losses.append(float(np.mean(routing_losses[0:])))
                    else:
                        routing_mean_losses.append(float(np.mean(routing_losses[-100:])))
                    routing_update_cnt += 1

                    # linearly decay epsilon
                    self.epsilon = max(self.min_epsilon,
                                       self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                    routing_epsilons.append(self.epsilon)

                    # if hard update is needed
                    if routing_update_cnt % self.update_target_every_steps == 0:
                        # self._target_hard_update()
                        self.routing_agent.target_soft_update(self.tau)
                if len(self.scheduling_agent.replay_buffer_size) >= self.batch_size:
                    # Update the model and record the loss
                    loss = self.scheduling_agent.update_model()
                    scheduling_losses.append(loss)
                    if len(scheduling_losses) < 100:
                        scheduling_mean_losses.append(float(np.mean(scheduling_losses[0:])))
                    else:
                        scheduling_mean_losses.append(float(np.mean(scheduling_losses[-100:])))
                    scheduling_update_cnt += 1

                    # linearly decay epsilon
                    # self.epsilon = max(self.min_epsilon,
                    #                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                    scheduling_epsilons.append(self.epsilon)

                    # if hard update is needed
                    if scheduling_update_cnt % self.update_target_every_steps == 0:
                        # self._target_hard_update()
                        self.scheduling_agent.target_soft_update(self.tau)

                # Print training progress at specified intervals
                if step % plotting_interval == 0:
                    self.routing_agent.plot(step, max_steps, routing_scores, routing_mean_scores, routing_losses,
                                            routing_mean_losses, routing_epsilons)
                    self.scheduling_agent.plot(step, max_steps, scheduling_scores, scheduling_mean_scores,
                                               scheduling_losses, scheduling_mean_losses, scheduling_epsilons)
                    plt.close()

                if step % monitor_training == 0:
                    self.logger.info(f"[I] Step: {step} | "
                                     f"Routing rewards: {round(float(np.mean(routing_scores[-100:])), 3)} | "
                                     f"Routing loss: {round(float(np.mean(routing_losses[-100:])), 3)} | "
                                     f"Routing epsilons: {round(float(np.mean(routing_epsilons[-100:])), 3)} | "
                                     f"Scheduling rewards: {round(float(np.mean(scheduling_scores[-100:])), 3)} | "
                                     f"Scheduling loss: {round(float(np.mean(scheduling_losses[-100:])), 3)} | "
                                     f"Scheduling epsilons: {round(float(np.mean(scheduling_epsilons[-100:])), 3)} | "
                                     f"Episodes: {episodes_count}")

                # if update_cnt % evaluation_interval == 0:
                #     result_df.append(self.evaluate(copy.deepcopy(self.env)))
        except KeyboardInterrupt:
            self.logger.info(f"[I] Step: {step} | "
                             f"Routing rewards: {round(float(np.mean(routing_scores[-100:])), 3)} | "
                             f"Routing loss: {round(float(np.mean(routing_losses[-100:])), 3)} | "
                             f"Routing epsilons: {round(float(np.mean(routing_epsilons[-100:])), 3)} | "
                             f"Scheduling rewards: {round(float(np.mean(scheduling_scores[-100:])), 3)} | "
                             f"Scheduling loss: {round(float(np.mean(scheduling_losses[-100:])), 3)} | "
                             f"Scheduling epsilons: {round(float(np.mean(scheduling_epsilons[-100:])), 3)} | "
                             f"Episodes: {episodes_count}")
            if episodes_count > 1000:
                self.routing_agent.plot(step, max_steps, routing_scores, routing_mean_scores, routing_losses,
                                        routing_mean_losses, routing_epsilons)
                self.scheduling_agent.plot(step, max_steps, scheduling_scores, scheduling_mean_scores,
                                           scheduling_losses, scheduling_mean_losses, scheduling_epsilons)
                plt.close()

        end = time.time()

        self.logger.info('[I] Total run time steps: ' + str(step))
        self.logger.info('[I] Total run episodes: ' + str(episodes_count))
        self.logger.info('[I] Total elapsed time: ' + str(end - start))
        self.logger.info('[I] Final average routing reward: ' +
                         str(round(float(np.mean(routing_scores[-100:])), 3)))
        self.logger.info('[I] Final average scheduling reward: ' +
                         str(round(float(np.mean(scheduling_scores[-100:])), 3)))

        # Save model
        if episodes_count > 1000:
            date = datetime.now().strftime('%Y%m%d%H%M')
            model_name = 'model_' + date + '_routing_' + str(int(np.mean(routing_scores[-100:]))) + '.pt'
            torch.save(self.routing_agent.online_net.state_dict(), MODEL_PATH + model_name)
            model_name = 'model_' + date + '_scheduling_' + str(int(np.mean(scheduling_scores[-100:]))) + '.pt'
            torch.save(self.scheduling_agent.online_net.state_dict(), MODEL_PATH + model_name)

        # Close env
        # self.env.close()

    def evaluate(self, eval_env: EnvironmentTSN, n_episodes: int = 50):
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
        routing_episode_rewards = []
        scheduling_episode_rewards = []

        num_lost = 0
        num_no_resources = 0
        num_delayed = 0
        # num_bad_schedule = 0
        num_arrived = 0
        num_arrived_wo_scheduling = 0
        num_perfect = 0
        num_arrived_wo_routing = 0
        num_error_cases = 0

        for episode in range(n_episodes):
            routing_obs, info = eval_env.reset()
            done = False
            step = 0
            routing_reward = 0
            scheduling_reward = 0

            while not done:
                next_node = self.routing_agent.select_action(routing_obs, self.get_routing_action_space(routing_obs))

                if step == 1:
                    scheduling_obs, info = self.env.scheduling_reset(next_node)
                else:
                    scheduling_obs, scheduling_reward, info = self.env.scheduling_step(next_node)

                position_id = self.scheduling_agent.select_action(scheduling_obs,
                                                                  self.get_scheduling_action_space(scheduling_obs))
                routing_obs, routing_reward,\
                    terminated, truncated, info = eval_env.step(next_node, position_id)
                done = terminated or truncated
                step += 1
                if done:
                    exit_code = info['exit_code']
                    if exit_code == 0:
                        num_perfect += 1
                    elif exit_code == 1:
                        num_arrived_wo_routing += 1
                    elif exit_code == 2:
                        num_arrived_wo_scheduling += 1
                    elif exit_code == 3:
                        num_arrived += 1
                    elif exit_code == -1:
                        num_delayed += 1
                    # elif exit_code == -2:
                    #     num_bad_schedule += 1
                    elif exit_code == -3:
                        num_no_resources += 1
                    elif exit_code == -4:
                        num_lost += 1
                    else:
                        num_error_cases += 1

            # Episode reward
            # episode_rewards.append(previous_reward)
            routing_episode_rewards.append(routing_reward)
            scheduling_episode_rewards.append(scheduling_reward)

        # Calculate and print the mean score over episodes
        mean_routing_score = np.mean(routing_episode_rewards)
        mean_scheduling_score = np.mean(scheduling_episode_rewards)
        self.logger.info(f"\n[I] Evaluation mean score over {n_episodes} episodes: {mean_routing_score:.2f} (routing) | {mean_scheduling_score:.2f} (scheduling)\n")
        self.logger.info(f'[I] Number of episodes where flow has been perfectly routed and scheduled: {num_perfect}')
        self.logger.info(f'[I] Number of episodes where flow reached its target with routing faults: {num_arrived_wo_routing}')
        self.logger.info(f'[I] Number of episodes where flow reached its target with schedule faults: {num_arrived_wo_scheduling}')
        self.logger.info(f'[I] Number of episodes where flow reached its target with routing and scheduling faults: {num_arrived}')
        # self.logger.info(f'[I] Number of episodes where agent chose a bad position: {num_bad_schedule}')
        self.logger.info(f'[I] Number of episodes where flow delay exceeded the allowed maximum: {num_delayed}')
        self.logger.info(f'[I] Number of episodes where edges had not enough resources: {num_no_resources}')
        self.logger.info(f'[I] Number of episodes where flow has lost: {num_lost}')
        self.logger.info(f'[I] Number of not able to describe cases: {num_error_cases}')

        # Reset epsilon to its original value after evaluation for continuing with training
        self.epsilon = original_epsilon
