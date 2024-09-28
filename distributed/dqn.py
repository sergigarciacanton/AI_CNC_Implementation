# DEPENDENCIES
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import List

from distributed.config import MAX_EPSILON
from dqn_A import DQNAgentA
from dqn_B import DQNAgentB


class DQNAgent:
    """DQN Agent interacting with environment."""

    def __init__(
            self,
            env: gym.Env,
            log_file_id: str = 'dqn'):

        # Attributes
        self.env = env
        self.epsilon = MAX_EPSILON
        self.evaluation = 1

        self.agent_A = DQNAgentA(
            env=env,
            log_file_id=log_file_id + '_A'
        )

        self.agent_B = DQNAgentB(
            env=env,
            log_file_id=log_file_id + '_B'
        )

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 50):
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
        self.agent_A.epsilon = 0
        self.agent_B.epsilon = 0

        # Episode reward list
        episode_rewards = []

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
            obs, info = eval_env.reset()
            done = False
            step = 0
            reward = 0

            while not done:
                previous_node = info['previous_node']
                current_node = obs[2]
                if current_node <= 7:
                    action = self.agent_A.select_action(obs, previous_node)
                else:
                    action = self.agent_B.select_action(obs, previous_node)
                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                obs = next_obs
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
                    elif exit_code == -3:
                        num_no_resources += 1
                    elif exit_code == -4:
                        num_lost += 1
                    else:
                        num_error_cases += 1

            # Episode reward
            # episode_rewards.append(previous_reward)
            episode_rewards.append(reward)

        # Calculate and print the mean score over episodes
        mean_score = np.mean(episode_rewards)
        print(f"\n[I] Evaluation mean score over {n_episodes} episodes: {mean_score:.2f}\n")
        print(f'[I] Number of episodes where flow has been perfectly routed and scheduled: {num_perfect}')
        print(f'[I] Number of episodes where flow reached its target with routing faults: {num_arrived_wo_routing}')
        print(f'[I] Number of episodes where flow reached its target with schedule faults: {num_arrived_wo_scheduling}')
        print(f'[I] Number of episodes where flow reached its target with routing and scheduling faults: {num_arrived}')
        print(f'[I] Number of episodes where flow delay exceeded the allowed maximum: {num_delayed}')
        print(f'[I] Number of episodes where edges had not enough resources: {num_no_resources}')
        print(f'[I] Number of episodes where flow has lost: {num_lost}')
        print(f'[I] Number of not able to describe cases: {num_error_cases}')

        # Reset epsilon to its original value after evaluation for continuing with training
        self.epsilon = original_epsilon

        # Close the environment
        eval_env.close()
        self.evaluation += 1

    def evaluate_routes(self, eval_env: gym.Env, n_episodes: int = 50):
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
        for i in list(self.env.get_graph().nodes):
            for j in list(self.env.get_graph().nodes):

                # Temporarily store the current epsilon value
                original_epsilon = self.epsilon

                # Set epsilon to 0 for evaluation
                self.epsilon = 0

                # Episode reward list
                episode_rewards = []

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
                    obs, info = eval_env.reset(options={'route': [i, j]})
                    done = False
                    step = 0
                    reward = 0

                    while not done:
                        action = self.select_action(obs, info['previous_node'])
                        next_obs, reward, terminated, truncated, info = eval_env.step(action)
                        obs = next_obs
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
                    episode_rewards.append(reward)

                # Calculate and print the mean score over episodes
                mean_score = np.mean(episode_rewards)
                print(
                    f"\n[I] Evaluation mean score over {n_episodes} episodes for route {i} --> {j}: {mean_score:.2f}\n")
                print(
                    f'[I] Number of episodes where flow has been perfectly routed and scheduled: {num_perfect}')
                print(
                    f'[I] Number of episodes where flow reached its target with routing faults: {num_arrived_wo_routing}')
                print(
                    f'[I] Number of episodes where flow reached its target with schedule faults: {num_arrived_wo_scheduling}')
                print(
                    f'[I] Number of episodes where flow reached its target with routing and scheduling faults: {num_arrived}')
                # print(f'[I] Number of episodes where agent chose a bad position: {num_bad_schedule}')
                print(
                    f'[I] Number of episodes where flow delay exceeded the allowed maximum: {num_delayed}')
                print(f'[I] Number of episodes where edges had not enough resources: {num_no_resources}')
                print(f'[I] Number of episodes where flow has lost: {num_lost}')
                print(f'[I] Number of not able to describe cases: {num_error_cases}')

                # Reset epsilon to its original value after evaluation for continuing with training
                self.epsilon = original_epsilon

                # Close the environment
                eval_env.close()
                self.evaluation += 1
