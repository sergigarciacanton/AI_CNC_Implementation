import random
import networkx as nx
import logging
import sys
from colorlog import ColoredFormatter
from gymnasium.spaces import Discrete
from gymnasium.core import ObsType
import gymnasium as gym
from scheduling_vnf_generator import VNF
import numpy as np
from typing import Tuple, Dict, Any
from scheduling_config import ENV_LOG_LEVEL, ENV_LOG_FILE_NAME, TIMESTEPS_LIMIT, SLOT_CAPACITY, DIVISION_FACTOR, \
    NODES, EDGES, BACKGROUND_STREAMS


class EnvironmentTSN(gym.Env):
    # Environment initialization
    def __init__(self, log_file_id):
        self.graph = nx.DiGraph()  # Graph containing the network topology and delays (given at config.py)
        self.vnf_list = {}  # Dict of current VNFs generated for real streams and their states
        self.background_traffic = {}  # Dict of current VNFs generated for background traffic and their states
        self.edges_slots = {}  # Dict containing all edges' schedule
        self.hyperperiod = None  # Time duration of a hyperperiod [ms]. Set as maximum VNF period (see config.py)
        self.remaining_timesteps = TIMESTEPS_LIMIT  # Number of time-steps available before truncating the episode
        self.reward = 0  # Cumulative scheduling reward of the episode
        self.optimal_positions = True  # False if agent choose non-optimal positions. Used in scheduling_reward_function
        self.terminated = False  # True when episode has ended
        self.current_vnf = None  # Current VNF that is being served. Iterates over vnf_list when not training
        self.current_node = None  # Current node where the current stream is located at
        self.next_node = None  # Next node where the current stream is about to be located at
        self.current_delay = None  # Current calculated delay for the current stream to follow the calculated path [ms]
        self.route = []  # Route that the current stream has been assigned to follow
        self.current_position = 0  # Current position where packets are being transmitted

        # Logging settings
        self.logger = logging.getLogger('env')
        self.logger.setLevel(ENV_LOG_LEVEL)
        self.logger.addHandler(
            logging.FileHandler(ENV_LOG_FILE_NAME + log_file_id + '.log', mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)

        # Init procedure: get network topology and VNFs list
        # If not training, RabbitMQ subscriber is set and env waits for topology and VNFs to be published
        # If training, Network topology is generated using config.py parameters and VNFs are randomly created
        self.logger.info('[I] Training enabled. Reading topology from config...')
        # Generate graph adding given topology (see config.py)
        self.graph.add_nodes_from(NODES)
        for edge, data in EDGES.items():
            source, target = edge
            self.graph.add_edge(source, target, weight=data['delay'])
        self.logger.info('[I] Received network topology: ' + str(self.graph.number_of_nodes()) + ' nodes and '
                         + str(self.graph.number_of_edges()) + ' edges')

        # Hyperperiod setting. Use maximum period of a given set (see config.py)
        self.hyperperiod = 16  # max(VNF_PERIOD)

        # Create edges schedule. Contains available bytes of each slot
        for edge, delay in EDGES.items():
            self.edges_slots[(edge[0], edge[1])] = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR

        self.logger.info('[I] Environment ready to operate')

        self.observation_space = Discrete(17)
        self.action_space = Discrete(17)

    # Background traffic generator. Called during reset
    def generate_background_traffic(self):
        # Iterate until having created all desired background streams (see config.py)
        num_abortions = 0
        for i in range(BACKGROUND_STREAMS):
            # Create random VNF (see vnf_generator.py) and get the route that will follow (shortest path)
            # VNF = {source, destination, length, period, max_delay, actions}
            self.background_traffic[i] = VNF(list(self.graph.nodes)).get_request()
            path = random.choice(list(nx.all_shortest_paths(self.graph,
                                                            source=self.background_traffic[i]['source'],
                                                            target=self.background_traffic[i]['destination'])))

            # Iterate along the path nodes until having assigned resources at all intermediate edges
            for j in range(len(path) - 1):
                self.current_node = path[j]
                # Search for the edge that has the desired source and destination nodes
                for edge_id, edge in self.edges_slots.items():
                    # If edge has the desired source and destination, schedule resources and go to next hop
                    if edge_id[0] == path[j] and edge_id[1] == path[j + 1]:
                        # Choose best position of the edge (the one with the highest minimum availability)
                        # action = (e, t)  e = edge ID  t = position ID
                        position_availabilities = self.get_edge_positions(edge_id)
                        position = position_availabilities.index(max(position_availabilities))
                        action = (edge_id[1], position)

                        # Schedule action
                        # If action is done (return True), pass to next hop
                        # If action is impossible, create new VNF and try to schedule it. If looping too much, stop
                        if self.schedule_background_stream(action, i):
                            break
                        else:
                            num_abortions += 1
                            i -= 1
                            if num_abortions >= 1000:
                                # If looping too much, network has collapsed. Stop execution
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
            free_bytes = self.get_position_availability(self.edges_slots[edge_id], c,
                                                        self.current_vnf['period'])
            position_availabilities[c] = int(100 * free_bytes / SLOT_CAPACITY)
        return position_availabilities

    # Allocate resources for a background stream. Called during generate_background_traffic
    # Returns True if scheduling is possible and False otherwise
    def schedule_background_stream(self, action, stream_id):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources and return True. Otherwise, return False
        if self.get_position_availability(self.edges_slots[(self.current_node, action[0])],
                                          action[1],
                                          self.background_traffic[stream_id]['period']
                                          ) >= self.background_traffic[stream_id]['length']:
            time_slot = action[1]
            # Loop along all time slots of the position subtracting the requested resources
            while time_slot < self.hyperperiod * DIVISION_FACTOR:
                self.edges_slots[(self.current_node, action[0])][time_slot] -= self.background_traffic[stream_id][
                    'length']
                time_slot += self.background_traffic[stream_id]['period']

            # Add action to the actions list of the stream
            self.background_traffic[stream_id]['actions'].append(action)
            return True
        else:
            return False

    # Returns the graph. Called during agent's initialization
    def get_graph(self):
        return self.graph

    # Returns the state vector that has to be returned to the scheduling agent as a numpy array.
    # Called during scheduling_reset and scheduling_step
    def get_observation(self):
        # First build position availabilities as an empty array
        position_availabilities = [0] * self.hyperperiod * DIVISION_FACTOR

        # Compute position availabilities. Minimum percentage of free bytes for all slots of each position
        for c in range(self.current_vnf['period']):
            if self.get_position_availability(self.edges_slots[(self.current_node, self.next_node)],
                                              c, self.current_vnf['period']) >= self.current_vnf['length']:
                position_availabilities[c] = 1

        # Merge all data into a list
        obs = [self.current_position] + position_availabilities

        # Convert state list to a numpy array
        obs = np.array(obs, dtype=np.int16)
        return obs

    # Try to allocate resources for a real stream. If not possible, terminate episode.
    # Called during step
    def schedule_stream(self, next_node, position_id):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources. Otherwise, terminate episode
        if self.get_position_availability(self.edges_slots[(self.current_node, next_node)],
                                          position_id,
                                          self.current_vnf['period']
                                          ) >= self.current_vnf['length']:
            time_slot = position_id
            # Loop along all time slots of the position subtracting the requested resources
            while time_slot < self.hyperperiod * DIVISION_FACTOR:
                self.edges_slots[(self.current_node, next_node)][time_slot] -= self.current_vnf['length']
                time_slot += self.current_vnf['period']

            # Add action to the actions list of the stream info
            self.current_vnf['actions'].append((next_node, position_id))
        else:
            self.terminated = True
            self.logger.info('[I] Could not schedule the action!')

    # Finds the availability of a position, which is the minimum availability of the slots of that position
    # Called during get_edge_positions, schedule_background_stream, schedule_stream, get_edge_positions_real,
    # get_routing_observation and get_scheduling_observation
    # Returns the availability (in bytes) of a given position for a given period
    def get_position_availability(self, schedule, position, period):
        min_availability = SLOT_CAPACITY
        slot = position
        while slot < self.hyperperiod * DIVISION_FACTOR:
            if schedule[slot] < min_availability:
                min_availability = schedule[slot]
            slot += period
        return min_availability

    # Updates the scheduling reward of the current episode
    # Called during scheduling_step
    def reward_function(self, position_id):
        well_done = 0
        if position_id < self.hyperperiod:
            if position_id == self.current_position:
                self.reward += 10
                well_done = 1
            else:
                i = position_id
                while i != self.current_position:
                    if self.get_position_availability(self.edges_slots[(self.current_node, self.next_node)],
                                                      i,
                                                      self.current_vnf['period']) >= self.current_vnf['length']:
                        self.optimal_positions = False
                        # self.reward -= 10
                        well_done = 0
                        break
                    i = (i + 1) % self.current_vnf['period']
                if i == self.current_position:
                    self.reward += 10
                    well_done = 1

        # If the episode is ended, check why
        if self.terminated:
            # If the stream has reached the destination, check how
            if self.next_node == self.current_vnf['destination']:
                # If scheduling was optimal (best positions) increase reward by 300
                if self.optimal_positions is True:
                    self.reward += 300
                    return {'exit_code': 0, 'well_done': well_done}
                elif self.optimal_positions is False:
                    return {'exit_code': 2, 'well_done': well_done}
            else:
                return {'exit_code': -3, 'well_done': well_done}
        else:
            return {'well_done': well_done}

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reset terminated, route, schedule, reward, remaining time steps and elapsed time
        for k in self.edges_slots.keys():
            self.edges_slots[k] = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR
        self.route = []
        self.terminated = False
        self.remaining_timesteps = TIMESTEPS_LIMIT
        self.current_delay = 0
        self.optimal_positions = True
        self.reward = 0
        self.current_position = 0

        # Generate random VNF
        self.current_vnf = VNF(list(self.graph.nodes)).get_request()

        # Generate background traffic in order to artificially load the network
        self.background_traffic = {}
        self.generate_background_traffic()
        self.logger.info('\n[I] Generated ' + str(len(self.background_traffic)) + ' background streams')

        # Set the current node as the source of the VNF and add it to the followed route
        self.current_node = self.current_vnf['source']
        self.route.append(self.current_node)

        path_to_dst = list(nx.dijkstra_path(self.graph, self.current_node, self.current_vnf['destination']))
        if len(path_to_dst) == 1:
            self.next_node = list(nx.neighbors(self.graph, self.current_node))[0]
        else:
            self.next_node = path_to_dst[1]

        # Acquire data to return to the agent
        obs = self.get_observation()
        info = {}

        self.logger.debug('[D] RESET. info = ' + str(info) +
                          ', current position = ' + str(obs[0]))
        self.logger.debug('[D] RESET. Edge ' + str(self.current_node) + ' --> ' + str(self.next_node) +
                          ' position availabilities: ' +
                          str(obs[1:]))
        return obs, info

    def step(self, position_id) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        self.logger.info('[I] Action: Position ID --> ' + str(position_id))

        # Subtract 1 to the remaining time steps
        self.remaining_timesteps -= 1

        # Truncate episode in case it is becoming too large (maybe due to network loops)
        truncated = self.remaining_timesteps < 0
        info = {}

        # If an invalid node has been received, end episode
        if position_id >= self.current_vnf['period']:
            self.terminated = True
            info = {'exit_code': -3, 'well_done': 0}
        elif truncated:
            info = {'exit_code': -4, 'well_done': 0}
        else:
            # Try to schedule the stream given the action to perform
            self.schedule_stream(self.next_node, position_id)
            self.current_delay += nx.get_edge_attributes(self.graph, "weight")[(self.current_node, self.next_node)]

            # If stream has reached its destination, end the episode
            if self.current_vnf['destination'] == self.next_node:
                self.terminated = True
                self.logger.info('[I] Reached destination!')

            # Update reward of current episode
            info = self.reward_function(position_id)

            # If scheduling is successful, update current node
            self.current_node = self.next_node
            self.route.append(self.current_node)
            self.current_position = position_id

            path_to_dst = list(nx.dijkstra_path(self.graph, self.current_node, self.current_vnf['destination']))
            if len(path_to_dst) == 1:
                self.next_node = list(nx.neighbors(self.graph, self.current_node))[0]
            else:
                self.next_node = path_to_dst[1]

        # Acquire data to return to the agent
        obs = self.get_observation()

        self.logger.debug('[D] STEP. info = ' + str(info) +
                          ', reward = ' + str(self.reward) +
                          ', current position = ' + str(obs[0]))
        self.logger.debug('[D] STEP. Edge ' + str(self.current_node) + ' --> ' + str(self.next_node) +
                          ' position availabilities: ' +
                          str(obs[1:]))
        if self.terminated:
            self.logger.info('[I] Ending episode...')
        return obs, self.reward, \
            self.terminated, truncated, info


# env = EnvironmentTSN('1')
# print('Hyperperiod:', env.hyperperiod)
# print('Edges info:', env.edges_slots)
# obse, information = env.reset()
# print('VNF:', env.current_vnf)
# print('Background traffic:', env.background_traffic)
# print('Edges info:', env.edges_slots)
# print(obse)
# print(information)
# observation2, reward, terminated, trunc, information2 = env.step(1)
# print('Edges info:', env.edges_slots)
# print('Terminated:', terminated)
# print('Truncated:', trunc)
# print(observation2)
# print(information2)
# print('Reward: ' + str(reward))
# observation2, reward, terminated, trunc, information2 = env.step(1)
# print('Edges info:', env.edges_slots)
# print('Terminated:', terminated)
# print('Truncated:', trunc)
# print(observation2)
# print(information2)
# print('Reward: ' + str(reward))
# observation2, reward, terminated, trunc, information2 = env.step(1)
# print('Edges info:', env.edges_slots)
# print('Terminated:', terminated)
# print('Truncated:', trunc)
# print(observation2)
# print(information2)
# print('Reward: ' + str(reward))
