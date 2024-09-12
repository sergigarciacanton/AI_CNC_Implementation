import random
import networkx as nx
import logging
import sys
from colorlog import ColoredFormatter
import time
from gymnasium.spaces import Discrete
from gymnasium.core import ObsType
from v_1.vnf_generator import VNF
import numpy as np
from typing import Tuple, Dict, Any
from v_2.configV2 import ENV_LOG_LEVEL, ENV_LOG_FILE_NAME, TIMESTEPS_LIMIT, SLOT_CAPACITY, DIVISION_FACTOR, TRAINING_IF, \
    NODES, EDGES, BACKGROUND_STREAMS


class EnvironmentTSN:
    # Environment initialization
    def __init__(self, log_file_id):
        self.graph = nx.DiGraph()  # Graph containing the network topology and delays (given at config.py)
        self.ip_addresses = {}  # Dict containing the IP addresses of all nodes in the network. Not used if training
        self.vnf_list = {}  # Dict of current VNFs generated for real streams and their states
        self.background_traffic = {}  # Dict of current VNFs generated for background traffic and their states
        self.edges_slots = {}  # Dict containing all edges' schedule
        self.hyperperiod = None  # Time duration of a hyperperiod [ms]. Set as maximum VNF period (see config.py)
        self.remaining_timesteps = TIMESTEPS_LIMIT  # Number of time-steps available before truncating the episode
        self.routing_reward = 0  # Cumulative routing reward of the episode
        self.scheduling_reward = 0  # Cumulative scheduling reward of the episode
        self.optimal_positions = True  # False if agent choose non-optimal positions. Used in scheduling_reward_function
        self.terminated = False  # True when episode has ended
        self.ready = False  # False until topology and VNFs have been received by RabbitMQ. True if training
        self.current_vnf = None  # Current VNF that is being served. Iterates over vnf_list when not training
        self.vnf_id = 0  # Current VNF ID that is being scheduled
        self.current_node = None  # Current node where the current stream is located at
        self.current_delay = None  # Current calculated delay for the current stream to follow the calculated path [ms]
        self.route = []  # Route that the current stream has been assigned to follow

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
        if not TRAINING_IF:
            self.logger.error('[!] Training not implemented! Stop program...')
        else:
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

            # Ready to work. Observation and action spaces can now be defined
            self.ready = True

        while self.ready is False:
            time.sleep(1)

        self.logger.info('[I] Environment ready to operate')

        self.routing_observation_space = Discrete(6)
        self.routing_action_space = Discrete(4)

        self.scheduling_observation_space = Discrete(17)
        self.scheduling_action_space = Discrete(16)

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

    # Returns the state vector that has to be returned to the routing agent as a numpy array.
    # Called during reset and step
    def get_routing_observation(self):
        # Maximum delay that can be introduced from the current node to the destination node
        remaining_delay = self.current_vnf['max_delay'] - self.current_delay

        # Compute whether adjacent edges are available to allocate VNFs resources or not
        edge_availabilities = []
        neighbors = list(nx.neighbors(self.graph, self.current_node))
        # For each adjacent edge, compute all positions' availability
        for n in neighbors:
            position_availabilities = [0] * self.current_vnf['period'] * DIVISION_FACTOR
            if len(self.route) < 2:
                for c in range(self.current_vnf['period']):
                    position_availabilities[c] = self.get_position_availability(
                        self.edges_slots[(self.current_node, n)], c, self.current_vnf['period'])
            elif n != self.route[-2]:
                for c in range(self.current_vnf['period']):
                    position_availabilities[c] = self.get_position_availability(
                        self.edges_slots[(self.current_node, n)], c, self.current_vnf['period'])

            # If some positions have enough capacity to carry VNFs packets, return the end-to-end delay
            # from the neighbor node to the destination following an optimal path.
            if max(position_availabilities) >= self.current_vnf['length']:
                edge_availabilities.append(nx.dijkstra_path_length(self.graph,
                                                                   n,
                                                                   self.current_vnf['destination']))
            # If edge is too loaded, return a big delay (meaning it cannot be routed to this edge)
            else:
                edge_availabilities.append(300)

        # Merge all data into a list  obs = [dst, curr_node, remaining_delay, e0, e1, e2]
        obs = [self.current_vnf['destination']] + [self.current_node] + [remaining_delay] + edge_availabilities

        # Convert state list to a numpy array
        obs = np.array(obs, dtype=np.int16)
        return obs

    # Returns the state vector that has to be returned to the scheduling agent as a numpy array.
    # Called during scheduling_reset and scheduling_step
    def get_scheduling_observation(self, next_node):
        # First build position availabilities as an empty array
        position_availabilities = [0] * self.hyperperiod * DIVISION_FACTOR

        # Iterate along all edges
        for edge_id, edge in self.edges_slots.items():
            # Extract just the state of the selected edge
            if edge_id[0] == self.current_node and edge_id[1] == next_node:
                # Compute position availabilities. Minimum percentage of free bytes for all slots of each position
                for c in range(self.current_vnf['period']):
                    free_bytes = self.get_position_availability(edge, c, self.current_vnf['period'])
                    position_availabilities[c] = int(100 * free_bytes / SLOT_CAPACITY)

        # Merge all data into a list
        obs = [int(100 * self.current_vnf['length'] / SLOT_CAPACITY)] + position_availabilities

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

    # Returns an array which contains the availability (in bytes) of all positions of the specified edge
    # Called during scheduling_reward_function
    def get_edge_positions_real(self, edge_id):
        position_availabilities = [0] * self.current_vnf['period'] * DIVISION_FACTOR
        for c in range(self.current_vnf['period'] * DIVISION_FACTOR):
            position_availabilities[c] = self.get_position_availability(self.edges_slots[edge_id], c,
                                                                        self.current_vnf['period'])
        return position_availabilities

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

    # Updates the routing reward of the current episode
    # Called during step
    def routing_reward_function(self, next_node):
        if next_node < len(self.edges_slots):
            # Decrease the reward by the distance from the next node to the target node (in delay terms)
            cur_len = nx.dijkstra_path_length(self.graph, source=next_node,
                                              target=self.current_vnf['destination'])
            self.routing_reward -= cur_len

        # If the episode is ended, check why
        if self.terminated:
            # If the stream has reached the destination, check how
            if next_node == self.current_vnf['destination']:
                # If routing was optimal (shortest path) increase routing reward by 300
                if self.current_delay == nx.dijkstra_path_length(self.graph,
                                                                 self.current_vnf['source'],
                                                                 self.current_vnf['destination']) \
                        and self.current_delay <= self.current_vnf['max_delay']:
                    self.routing_reward += 300
                    return 0
                # If routing was not optimal, increase by 100 the reward
                else:
                    self.routing_reward += 100
                    return 1
        # If delay has reached the maximum acceptable value, end the episode and decrease by 100 the reward
        if self.current_delay > self.current_vnf['max_delay']:
            self.logger.warning(f"Maximum delay exceeded! {self.current_delay} -- {self.current_vnf['max_delay']}")
            self.terminated = True
            self.routing_reward -= 100
            return -1
        return None

    # Updates the scheduling reward of the current episode
    # Called during scheduling_step
    def scheduling_reward_function(self, next_node, position_id, exit_code):
        if position_id < self.hyperperiod:
            # Compute availabilities of all positions of selected edge
            position_availabilities = self.get_edge_positions_real((self.current_node, next_node))

            # Not take into account the load of the current VNF
            position_availabilities[position_id] += self.current_vnf['length']

            # Decrease the reward if scheduling was not optimal (not selected the best position)
            self.scheduling_reward += 0.008 * (position_availabilities[position_id] - max(position_availabilities))

            # If the agent did not choose the position with more availability, mark as non-optimal
            if max(position_availabilities) != position_availabilities[position_id]:
                self.optimal_positions = False

        # If the episode is ended, check why
        if self.terminated:
            # If the stream has reached the destination, check how
            if next_node == self.current_vnf['destination']:
                # If scheduling was optimal (best positions) increase reward by 300
                if exit_code == 0 and self.optimal_positions is True:
                    self.scheduling_reward += 300
                    return 0
                # If scheduling was optimal (best positions) increase reward by 300
                elif exit_code == 1 and self.optimal_positions is True:
                    self.scheduling_reward += 300
                    return 1
                # If scheduling was not optimal increase reward by 100
                elif exit_code == 0 and self.optimal_positions is False:
                    self.scheduling_reward += 100
                    return 2
                # If scheduling was not optimal increase reward by 100
                elif exit_code == 1 and self.optimal_positions is False:
                    self.scheduling_reward += 100
                    return 3
            else:
                return exit_code

    def reset(self) -> tuple[ObsType, dict[str, Any]]:

        # Reset terminated, route, schedule, reward, remaining time steps and elapsed time
        for k in self.edges_slots.keys():
            self.edges_slots[k] = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR
        self.route = []
        self.terminated = False
        self.remaining_timesteps = TIMESTEPS_LIMIT
        self.current_delay = 0
        self.optimal_positions = True
        self.routing_reward = 0
        self.scheduling_reward = 0

        # Wait until being ready to work.
        # In training mode, this is skipped. Otherwise, topology and VNFs must be received before continuing
        while not self.ready:
            time.sleep(0.001)

        # If the agent is not training, the current VNF has to be selected from vnf_list
        # If it is training, a random VNF has to be generated
        if TRAINING_IF is True:
            # Generate a random VNF
            self.current_vnf = VNF(None).get_request()
        else:
            # Set the first VNF as the one to process
            self.current_vnf = self.vnf_list[self.vnf_id]

        # Generate background traffic in order to artificially load the network
        self.background_traffic = {}
        self.generate_background_traffic()
        self.logger.info('\n[I] Generated ' + str(len(self.background_traffic)) + ' background streams')

        # Set the current node as the source of the VNF and add it to the followed route
        self.current_node = self.current_vnf['source']
        self.route.append(self.current_node)

        # Acquire data to return to the agent
        routing_obs = self.get_routing_observation()
        info = {}
        if ENV_LOG_LEVEL == 10:
            self.logger.debug('[D] RESET. info = ' + str(info) +
                              ', routing obs = ' + str(routing_obs))
        return routing_obs, info

    def scheduling_reset(self, next_node) -> Tuple[np.ndarray, Dict]:
        # Convert neighbor ID to node ID
        try:
            next_node = list(nx.neighbors(self.graph, self.current_node))[next_node]
        except IndexError:
            next_node = len(self.graph.nodes)

        self.logger.info('[I] Action: Next node --> ' + str(next_node))

        scheduling_obs = self.get_scheduling_observation(next_node)

        info = {}

        self.logger.debug('[D] RESET. info = ' + str(info) +
                          ', scheduling length = ' + str(scheduling_obs[0]))
        self.logger.debug('[D] RESET. Edge ' + str(self.current_node) + ' --> ' + str(next_node) +
                          ' position availabilities: ' +
                          str(scheduling_obs[1:]))

        return scheduling_obs, info

    def step(self, next_node, position_id) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Convert neighbor ID to node ID
        try:
            next_node = list(nx.neighbors(self.graph, self.current_node))[next_node]
        except IndexError:
            next_node = len(self.graph.nodes)

        self.logger.info('[I] Action: Position ID --> ' + str(position_id))

        # Subtract 1 to the remaining time steps
        self.remaining_timesteps -= 1

        # Truncate episode in case it is becoming too large (maybe due to network loops)
        truncated = self.remaining_timesteps < 0
        info = {}

        # If an invalid node has been received, end episode
        if next_node >= len(self.graph.nodes):
            self.terminated = True

            info = {'exit_code': -3}
        elif truncated:
            info = {'exit_code': -4}
        else:
            # Try to schedule the stream given the action to perform
            self.schedule_stream(next_node, position_id)
            self.current_delay += nx.get_edge_attributes(self.graph, "weight")[(self.current_node, next_node)]

            # If stream has reached its destination, end the episode
            if self.current_vnf['destination'] == next_node:
                self.terminated = True
                self.logger.info('[I] Reached destination!')

            # Update reward of current episode
            exit_code = self.routing_reward_function(next_node)
            exit_code = self.scheduling_reward_function(next_node, position_id, exit_code)
            if exit_code is not None:
                info = {'exit_code': exit_code}

            # If scheduling is successful, update current node
            self.current_node = next_node
            self.route.append(self.current_node)

        # Acquire data to return to the agent
        routing_obs = self.get_routing_observation()

        # Skip to next VNF in case of ending episode (just needed when not training)
        if self.terminated is True and TRAINING_IF is False:
            self.vnf_id += 1

        if ENV_LOG_LEVEL == 10:
            self.logger.debug('[D] STEP. info = ' + str(info) + ', terminated = ' + str(self.terminated) +
                              ', truncated = ' + str(truncated) +
                              ', routing reward = ' + str(self.routing_reward) +
                              ', routing obs = ' + str(routing_obs))
        if self.terminated:
            self.logger.info('[I] Ending episode...')
        return routing_obs, self.routing_reward, \
            self.terminated, truncated, info

    def scheduling_step(self, next_node) -> Tuple[np.ndarray, float, Dict]:
        # Convert neighbor ID to node ID
        try:
            next_node = list(nx.neighbors(self.graph, self.current_node))[next_node]
        except IndexError:
            next_node = len(self.graph.nodes)

        self.logger.info('[I] Action: Next node --> ' + str(next_node))

        # Acquire data to return to the agent
        scheduling_obs = self.get_scheduling_observation(next_node)

        info = {}

        self.logger.debug('[D] STEP. info = ' + str(info) +
                          ', scheduling reward = ' + str(self.scheduling_reward) +
                          ', scheduling length = ' + str(scheduling_obs[0]))
        self.logger.debug('[D] STEP. Edge ' + str(self.current_node) + ' --> ' + str(next_node) +
                          ' position availabilities: ' +
                          str(scheduling_obs[1:]))

        return scheduling_obs, self.scheduling_reward, info


# env = EnvironmentTSN('1')
# print('Hyperperiod:', env.hyperperiod)
# print('Edges info:', env.edges_slots)
# routing_obse, information = env.reset()
# print('VNF:', env.current_vnf)
# print('Background traffic:', env.background_traffic)
# print('Edges info:', env.edges_slots)
# print(routing_obse)
# print(information)
# scheduling_obse, information = env.scheduling_reset(6)
# print(scheduling_obse)
# print(information)
# observation2, reward, terminated, trunc, information2 = env.step(6, 0)
# print('Edges info:', env.edges_slots)
# print('Terminated:', terminated)
# print('Truncated:', trunc)
# print(observation2)
# print(information2)
# print('Routing reward: ' + str(reward))
# scheduling_obse2, reward_scheduling, information2 = env.scheduling_step(7)
# print(scheduling_obse2)
# print('Scheduling reward: ' + str(reward_scheduling))
# print(information2)
# observation2, reward, terminated, trunc, information2 = env.step(7, 0)
# print('Edges info:', env.edges_slots)
# print('Terminated:', terminated)
# print('Truncated:', trunc)
# print(observation2)
# print(information2)
# print('Routing reward: ' + str(reward))
