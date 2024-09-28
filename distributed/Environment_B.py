import random
from typing import Any
import gymnasium as gym
import networkx as nx
import logging
import sys
from colorlog import ColoredFormatter
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from vnf_generator_B import VNF
import numpy as np
from itertools import chain
from config import ENV_LOG_LEVEL, ENV_LOG_FILE_NAME, SLOT_CAPACITY, DIVISION_FACTOR, \
    NODES_B, EDGES_B, BACKGROUND_STREAMS, ALL_ROUTES, VNF_PERIOD


class EnvironmentB(gym.Env):
    # Environment initialization
    def __init__(self, log_file_id, evaluate):
        self.graph = nx.DiGraph()  # Graph containing the network topology. Training topology is given at config.py
        self.background_traffic = {}  # Dict of current VNFs generated for background traffic and their states
        self.edges_info = {}  # Dict containing all edges' information: source, destination, delay and schedule
        self.hyperperiod = None  # Time duration of a hyperperiod [ms]. Set as maximum VNF period (see config.py)
        self.reward = 0  # Cumulative reward of the episode
        self.optimal_positions = True  # Set to false if agent chooses non-optimal positions. Used in reward_function
        self.terminated = False  # Check whether episode has ended
        self.current_vnf = None  # Current VNF that is being served. Iterates over vnf_list when not training
        self.current_node = None  # Current node where the current stream is located at
        self.current_delay = None  # Current calculated delay for the current stream to follow the calculated path [ms]
        self.current_position = None  # Position ID where flow is transmitted at previous edge
        self.route = []  # Route that the current stream has been assigned to follow
        self.evaluate = evaluate  # Flag that is set in case of evaluation (not set if training)

        # Logging settings
        self.logger = logging.getLogger('env')
        self.logger.setLevel(ENV_LOG_LEVEL)
        self.logger.addHandler(
            logging.FileHandler(ENV_LOG_FILE_NAME + log_file_id + '.log', mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)

        # Init procedure: get network topology and VNFs list
        self.logger.info('[I] Reading topology from config...')
        # Generate graph adding given topology (see config.py)
        self.graph.add_nodes_from(NODES_B)
        for edge, data in EDGES_B.items():
            source, target = edge
            self.graph.add_edge(source, target, weight=data['delay'])
        self.logger.info('[I] Received network topology: ' + str(self.graph.number_of_nodes()) + ' nodes and '
                         + str(self.graph.number_of_edges()) + ' edges')

        # Hyperperiod setting. Use maximum period of a given set (see config.py)
        self.hyperperiod = max(VNF_PERIOD)

        # Create edges info. Contains source, destination, delay and schedule (available bytes of each slot)
        id_edge = 0
        for edge, delay in EDGES_B.items():
            self.edges_info[id_edge] = dict(source=edge[0], destination=edge[1],
                                            schedule=[SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR,
                                            delay=delay['delay'])
            id_edge += 1

        # Observation space
        num_obs_features = 5 + ((self.hyperperiod * DIVISION_FACTOR) * 3)  # Num of obs features
        self.observation_space_B = Discrete(num_obs_features)

        # Action space
        self.action_space_B = Discrete(len(EDGES_B) * self.hyperperiod * DIVISION_FACTOR + 1)

        self.logger.info('[I] Environment ready to operate')

    # Returns the graph. Called during agent's initialization
    def get_graph(self):
        return self.graph

    # Background traffic generator. Called during reset
    def generate_background_traffic(self):
        # Iterate until having created all desired background streams (see config.py)
        num_abortions = 0
        for i in range(BACKGROUND_STREAMS):
            # Create random VNF (see vnf_generator.py) and get the route that will follow (shortest path)
            # VNF = {source, destination, length, period, max_delay, actions}
            while True:
                self.background_traffic[i] = VNF(list(self.graph.nodes)).get_request()
                if self.background_traffic[i]['source'] >= 8:
                    break
            path = random.choice(list(nx.all_shortest_paths(self.graph,
                                                            source=self.background_traffic[i]['source'],
                                                            target=self.background_traffic[i]['destination'])))

            # Iterate along the path nodes until having assigned resources at all intermediate edges
            for j in range(len(path) - 1):
                # Search for the edge that has the desired source and destination nodes
                for edge_id, edge in self.edges_info.items():
                    # If edge has the desired source and destination, schedule resources and go to next hop
                    if edge['source'] == path[j] and edge['destination'] == path[j + 1]:
                        # Choose best position of the edge (the one with the highest minimum availability)
                        # action = (e, t)  e = edge ID  t = position ID
                        position_availabilities = self.get_edge_positions(edge_id)
                        position = position_availabilities.index(max(position_availabilities))
                        action = (edge_id, position)

                        # Schedule action
                        # If action is done (return True), pass to next hop
                        # If action is impossible, create cent VNF and try to schedule it. If looping too much, stop
                        if self.schedule_background_stream(action, i):
                            break
                        else:
                            num_abortions += 1
                            i -= 1
                            if num_abortions >= 1000:
                                # If looping too much, network has collapsed. Stop creating background traffic
                                self.logger.warning(
                                    '[!] Background traffic could not be allocated! Ask for less '
                                    'background streams (see config.py --> BACKGROUND_STREAMS)')
                                return
                            break

    # Returns an array which contains the availability (in percentage) of all positions of the specified edge
    # Called during generate_background_traffic
    def get_edge_positions(self, edge_id):
        position_availabilities = [0] * self.current_vnf['period'] * DIVISION_FACTOR
        for c in range(self.current_vnf['period'] * DIVISION_FACTOR):
            free_bytes = self.get_position_availability(self.edges_info[edge_id]['schedule'], c,
                                                        self.current_vnf['period'])
            position_availabilities[c] = int(100 * free_bytes / SLOT_CAPACITY)
        return position_availabilities

    # Allocate resources for a background stream. Called during generate_background_traffic
    # Returns True if scheduling is possible and False otherwise
    def schedule_background_stream(self, action, stream_id):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources and return True. Otherwise, return False
        if self.get_position_availability(self.edges_info[action[0]]['schedule'],
                                          action[1],
                                          self.background_traffic[stream_id]['period']
                                          ) >= self.background_traffic[stream_id]['length']:
            time_slot = action[1]
            # Loop along all time slots of the position subtracting the requested resources
            while time_slot < self.hyperperiod * DIVISION_FACTOR:
                self.edges_info[action[0]]['schedule'][time_slot] -= self.background_traffic[stream_id]['length']
                time_slot += self.background_traffic[stream_id]['period']

            # Add action to the actions list of the stream info
            self.background_traffic[stream_id]['actions'].append(action)
            return True
        else:
            return False

    # Try to allocate resources for a real stream. If not possible, terminate episode. Called during step
    def schedule_stream(self, action):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources. Otherwise, (should never happen) terminate episode
        if self.get_position_availability(self.edges_info[action[0]]['schedule'],
                                          action[1],
                                          self.current_vnf['period']
                                          ) >= self.current_vnf['length']:
            time_slot = action[1]
            # Loop along all time slots of the position subtracting the requested resources
            while time_slot < self.hyperperiod * DIVISION_FACTOR:
                self.edges_info[action[0]]['schedule'][time_slot] -= self.current_vnf['length']
                time_slot += self.current_vnf['period']

            # Add action to the actions list of the stram info
            self.current_vnf['actions'].append(action)
        else:
            self.terminated = True
            self.logger.info('[I] Could not schedule the action!')

    # Returns an array which contains the availability (in bytes) of all positions of the specified edge
    # Called during reward_function
    def get_edge_positions_real(self, edge_id):
        position_availabilities = [0] * self.current_vnf['period'] * DIVISION_FACTOR
        for c in range(self.current_vnf['period'] * DIVISION_FACTOR):
            position_availabilities[c] = self.get_position_availability(self.edges_info[edge_id]['schedule'], c,
                                                                        self.current_vnf['period'])
        return position_availabilities

    # Finds the availability of a position, which is the minimum availability of the slots of that position
    # Called during generate_background_traffic, get_edge_positions, schedule_background_stream, schedule_stream,
    # get_edge_positions_real and get_observation
    # Returns the availability of a given position for a given period
    def get_position_availability(self, schedule, position, period):
        min_availability = SLOT_CAPACITY
        slot = position
        while slot < self.hyperperiod * DIVISION_FACTOR:
            if schedule[slot] < min_availability:
                min_availability = schedule[slot]
            slot += period
        return min_availability

    # Updates the reward of the current episode. Called during step
    def reward_function(self, action):
        if action[0] < len(self.edges_info):
            # Compute availabilities of all positions
            position_availabilities = self.get_edge_positions_real(action[0])

            # Not take into account the load of the current VNF
            position_availabilities[action[1]] += self.current_vnf['length']

            # Calculate optimal position to take and subtract difference with selected position.
            # Decrease reward by the calculated difference
            for i in range(self.current_vnf['period']):
                eval_position = (self.current_position + i) % self.current_vnf['period']
                if self.current_vnf['length'] <= position_availabilities[eval_position]:
                    if eval_position != action[1]:
                        if action[1] < eval_position:
                            position = action[1] + self.current_vnf['period']
                        else:
                            position = action[1]
                        self.reward -= (position - eval_position)
                        self.optimal_positions = False
                    break

            self.reward -= self.edges_info[action[0]]['delay']

        # If the episode is ended, check why
        if self.terminated:
            # If the stream has reached the destination, check how
            if self.current_node == self.current_vnf['destination']:
                # If routing and scheduling were optimal (shortest path and best positions) increase reward by 300
                if self.current_delay == nx.dijkstra_path_length(self.graph,
                                                                 self.current_vnf['source'],
                                                                 self.current_vnf['destination']) \
                        and self.optimal_positions is True \
                        and self.current_delay <= self.current_vnf['max_delay']:
                    self.reward += 300
                    return 0
                # If just routing was optimal increase reward by 50
                elif self.current_delay == nx.dijkstra_path_length(self.graph,
                                                                   self.current_vnf['source'],
                                                                   self.current_vnf['destination']) \
                        and self.optimal_positions is False \
                        and self.current_delay <= self.current_vnf['max_delay']:
                    self.reward += 50
                    return 2
                # If just scheduling was optimal increase reward by 150
                elif nx.dijkstra_path_length(self.graph,
                                             self.current_vnf['source'],
                                             self.current_vnf['destination']) < self.current_delay <= self.current_vnf[
                    'max_delay'] \
                        and self.optimal_positions is True:
                    self.reward += 150
                    return 1
                # If neither routing nor scheduling were optimal leave the reward as 0
                elif nx.dijkstra_path_length(self.graph,
                                             self.current_vnf['source'],
                                             self.current_vnf['destination']) < self.current_delay <= self.current_vnf[
                    'max_delay'] \
                        and self.optimal_positions is False:
                    self.reward = 0
                    return 3
            # If destination has not been reached, the TSN flow has left the TSN zone. Check how
            else:
                # If scheduling was optimal, increase reward by 150
                if nx.dijkstra_path_length(self.graph,
                                           self.current_vnf['source'],
                                           self.current_vnf['destination']) < self.current_delay <= self.current_vnf[
                    'max_delay'] \
                        and self.optimal_positions is True and self.current_vnf['destination'] < 8:
                    self.reward += 150
                # If scheduling was not optimal leave the reward as 0
                elif nx.dijkstra_path_length(self.graph,
                                             self.current_vnf['source'],
                                             self.current_vnf['destination']) < self.current_delay <= self.current_vnf[
                    'max_delay'] \
                        and self.optimal_positions is False and self.current_vnf['destination'] < 8:
                    self.reward = 0
                # If destination of the TSN flow is inside the TSN zone, decrease reward by 100
                elif self.current_vnf['destination'] > 7:
                    self.reward -= 100
                return 4
        # If delay has reached the maximum acceptable value, end the episode and decrease by 100 the reward
        if self.current_delay > self.current_vnf['max_delay']:
            self.logger.warning(f"[!] Maximum delay exceeded! ({self.current_delay} > {self.current_vnf['max_delay']})")
            self.terminated = True
            self.reward -= 100
            return -1
        return None

    # Returns the state vector that has to be returned to the agent as a numpy array. Called during reset and step
    def get_observation(self):
        # First build observation as a dictionary
        st = {}

        # Iterate along all edges
        for edge_id, edge in self.edges_info.items():
            # Extract just the state of the edges whose source is the current node
            if edge['source'] == self.current_node:
                # Compute position availabilities (vectors that show, for each edge, its positions' availabilities)
                # 1 --> Position available, 0 --> Position not available
                slot_loads = [0] * self.hyperperiod * DIVISION_FACTOR
                if len(self.route) < 2:
                    for c in range(self.current_vnf['period']):
                        free_bytes = self.get_position_availability(edge['schedule'], c, self.current_vnf['period'])
                        if self.current_vnf['length'] <= free_bytes:
                            slot_loads[c] = 1
                        else:
                            slot_loads[c] = 0
                elif edge['destination'] != self.route[-2]:
                    for c in range(self.current_vnf['period']):
                        free_bytes = self.get_position_availability(edge['schedule'], c, self.current_vnf['period'])
                        if self.current_vnf['length'] <= free_bytes:
                            slot_loads[c] = 1
                        else:
                            slot_loads[c] = 0
                st[edge_id] = slot_loads
        st_vec = list(chain.from_iterable(st.values()))

        # In case that the current node has less than 3 neighbors, fill the vector with padding zeros
        while len(st_vec) < (3 * self.hyperperiod * DIVISION_FACTOR):
            st_vec.append(0)

        # Calculate remaining delay, which is the difference between the maximum stated at the VNF and the current sum
        remaining_delay = self.current_vnf['max_delay'] - self.current_delay

        # Merge all data into a list
        obs = [self.current_vnf['source']] \
              + [self.current_vnf['destination']] \
              + [self.current_node] \
              + [self.current_position] \
              + [remaining_delay] \
              + st_vec

        # Convert state list to a numpy array
        obs = np.array(obs, dtype=np.int16)
        return obs

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reset terminated, route, schedule, reward, remaining time steps and current delay
        for e in self.edges_info.values():
            e['schedule'] = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR
        self.route = []
        self.terminated = False
        self.current_delay = 0
        self.current_position = 0
        self.optimal_positions = True
        self.reward = 0

        # Generate a random VNF
        if ALL_ROUTES is True:
            if self.evaluate is True:
                self.current_vnf = VNF(options['route']).get_request()
            else:
                self.current_vnf = VNF(list(self.graph.nodes)).get_request()
        else:
            self.current_vnf = VNF(None).get_request()

        self.logger.debug('[D] VNF: ' + str(self.current_vnf))

        # Set the current node as the source of the VNF and add it to the followed route
        self.current_node = self.current_vnf['source']
        self.route.append(self.current_node)

        # Generate background traffic in order to artificially load the network
        self.background_traffic = {}
        self.generate_background_traffic()
        self.logger.info('[I] Generated ' + str(len(self.background_traffic)) + ' background streams')

        # Acquire data to return to the agent
        obs = self.get_observation()
        info = {'previous_node': -1}
        if ENV_LOG_LEVEL == 10:
            self.logger.debug('[D] RESET. info = ' + str(info) + ', obs = ' + str(obs[0:5]))
            for i in range(3):
                self.logger.debug('[D] RESET. Edge ' + str(i) + ' |  position availabilities: ' +
                                  str(obs[5 + (i * (self.hyperperiod * DIVISION_FACTOR)):5 + (
                                          (i + 1) * (self.hyperperiod * DIVISION_FACTOR))]))
        return obs, info

    def step(self, action_int):
        # Convert numerical action to edge-position tuple
        action = (
            int(action_int / (self.hyperperiod * DIVISION_FACTOR)), action_int % (self.hyperperiod * DIVISION_FACTOR))
        self.logger.info('[I] STEP. Action: ' + str(action_int) + ' ' + str(action))

        if action[0] >= len(self.edges_info):
            self.terminated = True

            info = {'previous_node': -1, 'exit_code': -3}
        else:
            # Try to schedule the stream given the action to perform
            self.schedule_stream(action)

            # If scheduling is successful, update current node
            if self.terminated is False:
                self.current_node = self.edges_info[action[0]]['destination']
                self.route.append(self.current_node)
                self.current_delay += self.edges_info[action[0]]['delay']
                # If stream has reached its destination, end the episode
                if self.current_vnf['destination'] == self.current_node:
                    self.terminated = True
                    self.logger.info('[I] Reached destination!')
                if self.current_node < 8:
                    self.terminated = True
                    self.logger.info('[I] Exited from TSN zone')

            # Update reward of current episode
            exit_code = self.reward_function(action)
            if exit_code is None:
                info = {'previous_node': self.route[-2]}
            else:
                info = {'previous_node': self.route[-2], 'exit_code': exit_code}
            self.current_position = action[1]

        # Acquire data to return to the agent
        obs = self.get_observation()

        if ENV_LOG_LEVEL == 10:
            self.logger.debug('[D] STEP. info = ' + str(info) + ', terminated = ' + str(self.terminated) +
                              ', reward = ' + str(self.reward) +
                              ', obs = ' + str(obs[0:5]))
            for i in range(3):
                self.logger.debug('[D] STEP. Edge ' + str(i) + ' |  position availabilities: ' +
                                  str(obs[5 + (i * (self.hyperperiod * DIVISION_FACTOR)):5 + (
                                          (i + 1) * (self.hyperperiod * DIVISION_FACTOR))]))
        if self.terminated:
            self.logger.info('[I] Ending episode...\n')
        return obs, self.reward, self.terminated, False, info

# env = EnvironmentTSN()
# print('Hyperperiod:', env.hyperperiod)
# print('Edges info:', env.edges_info)
# observation, information = env.reset()
# print('VNF:', env.current_vnf)
# # print('Background traffic:', env.background_traffic)
# print('Edges info:', env.edges_info)
# observation2, reward, terminated, trunc, information2 = env.step((0, 1))
# print('Edges info:', env.edges_info)
# print('Terminated:', terminated)
