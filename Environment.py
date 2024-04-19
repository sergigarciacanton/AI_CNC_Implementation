import random
import gymnasium as gym
import networkx as nx
import pika
import threading
import json
import logging
import sys
from colorlog import ColoredFormatter
import time
from gymnasium.spaces import Discrete
from vnf_generator import VNF
import numpy as np
from itertools import chain
from config import ENV_LOG_LEVEL, ENV_LOG_FILE_NAME, RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, \
    RABBITMQ_PASSWORD, RABBITMQ_EXCHANGE_NAME, TIMESTEPS_LIMIT, SLOT_CAPACITY, DIVISION_FACTOR, TRAINING_IF, \
    TRAINING_NODES, TRAINING_EDGES, BACKGROUND_STREAMS, VNF_PERIOD


class EnvironmentTSN(gym.Env):
    # Environment initialization
    def __init__(self):
        self.graph = nx.DiGraph()  # Graph containing the network topology. Training topology is given at config.py
        self.ip_addresses = {}  # Dict containing the IP addresses of all nodes in the network. Not used if training
        self.vnf_list = {}  # Dict of current VNFs generated for real streams and their states
        self.background_traffic = {}  # Dict of current VNFs generated for background traffic and their states
        self.edges_info = {}  # Dict containing all edges' information: source, destination, delay and schedule
        self.hyperperiod = None  # Time duration of a hyperperiod [ms]. Set as maximum VNF period (see config.py)
        self.remaining_timesteps = TIMESTEPS_LIMIT  # Number of time-steps available before truncating the episode
        self.reward = 0  # Cumulative reward of the episode
        self.optimal_positions = True  # Set to false if agent chooses non-optimal positions. Used in reward_function
        self.terminated = False  # Check whether episode has ended
        self.ready = False  # False until topology and VNFs have been received by RabbitMQ. True if training
        self.current_vnf = None  # Current VNF that is being served. Iterates over vnf_list when not training
        self.vnf_id = 0  # Current VNF ID that is being scheduled
        self.current_node = None  # Current node where the current stream is located at
        self.current_delay = None  # Current calculated delay for the current stream to follow the calculated path [ms]
        self.route = []  # Route that the current stream has been assigned to follow

        # Logging settings
        self.logger = logging.getLogger('env')
        self.logger.setLevel(ENV_LOG_LEVEL)
        self.logger.addHandler(logging.FileHandler(ENV_LOG_FILE_NAME, mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)
        logging.getLogger('pika').setLevel(logging.WARNING)

        # Init procedure: get network topology and VNFs list
        # If not training, RabbitMQ subscriber is set and env waits for topology and VNFs to be published
        # If training, Network topology is generated using config.py parameters and VNFs are randomly created
        if not TRAINING_IF:
            self.logger.info('[I] Training disabled. Waiting for topology and VNF list...')
            # If there are no auth settings, just try to connect to the given host (see config.py)
            # In case of having auth settings, use them to connect
            if RABBITMQ_PORT is None or RABBITMQ_USERNAME is None or RABBITMQ_PASSWORD is None:
                self.rabbitmq_conn = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
            else:
                self.rabbitmq_conn = pika.BlockingConnection(
                    pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT,
                                              credentials=pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)))

            # Create thread that subscribes to the RabbitMQ channels that publish the desired information
            self.subscribe_thread = threading.Thread(target=self.rabbitmq_subscribe,
                                                     args=(self.rabbitmq_conn, 'top-pre jet-pre'))
            self.subscribe_thread.daemon = True
            self.subscribe_thread.start()
        else:
            self.logger.info('[I] Training enabled. Reading topology from config...')
            # Generate graph adding given topology (see config.py)
            self.graph.add_nodes_from(TRAINING_NODES)
            for edge, data in TRAINING_EDGES.items():
                source, target = edge
                self.graph.add_edge(source, target, weight=data['delay'])
            self.logger.info('[I] Received network topology: ' + str(self.graph.number_of_nodes()) + ' nodes and '
                             + str(self.graph.number_of_edges()) + ' edges')

            # Hyperperiod setting. Use maximum period of a given set (see config.py)
            # self.hyperperiod = math.lcm(*[self.vnf_list[i]['period'] for i in range(len(self.vnf_list))])
            # self.hyperperiod = max(VNF_PERIOD)
            self.hyperperiod = 16

            # Create edges info. Contains source, destination, delay and schedule (available bytes of each slot)
            id_edge = 0
            for edge, delay in TRAINING_EDGES.items():
                self.edges_info[id_edge] = dict(source=edge[0], destination=edge[1],
                                                schedule=[SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR,
                                                delay=delay['delay'])
                id_edge += 1

            # Ready to work. Observation and action spaces can now be defined
            self.ready = True

        while self.ready is False:
            time.sleep(1)

        self.logger.info('[I] Environment ready to operate')

        # Observation space
        num_obs_features = 7 + ((self.hyperperiod * DIVISION_FACTOR) * 3)  # Num of obs features
        # self.observation_space = MultiDiscrete(np.array([1] * num_obs_features), dtype=np.int32)
        self.observation_space = Discrete(num_obs_features)

        # Action space
        self.action_space = Discrete(len(TRAINING_EDGES) * self.hyperperiod * DIVISION_FACTOR + 1)

    # Subscription function to RabbitMQ channels (executed at a thread). Called during init if not training
    def rabbitmq_subscribe(self, conn, key_string):
        # Initial settings: channel and queue creation
        channel = conn.channel()
        channel.exchange_declare(exchange=RABBITMQ_EXCHANGE_NAME, exchange_type='direct')
        queue = channel.queue_declare(queue='', durable=True).method.queue

        # Binding queues to the desired routing keys
        keys = key_string.split(' ')
        for key in keys:
            channel.queue_bind(exchange=RABBITMQ_EXCHANGE_NAME, queue=queue, routing_key=key)

        self.logger.info('[I] Waiting for published data...')

        # Callback function called each time subscriber receives new data
        def callback(ch, method, properties, body):
            self.logger.debug("[D] Received. Key: " + str(method.routing_key) + ". Message: " + body.decode("utf-8"))
            # If routing key is top-pre, the message contains the topology of the network
            # If routing key is jet-pre, the message contains the list of VNFs
            if str(method.routing_key) == 'top-pre':
                # Decode JSON message. Extract nodes, edges and IP addresses of nodes
                info = json.loads(body.decode('utf-8'))
                nodes = info["Network_nodes"]
                edges = info["Network_links"]
                self.ip_addresses = info["identificator"]

                # Add nodes and edges to nx graph. Use identifiers instead of IP addresses
                for node in nodes:
                    self.graph.add_node(node)
                for edge in edges:
                    self.graph.add_edge(edge[0], edge[1])
                    self.edges_info[edge]['source'] = edge[0]
                    self.edges_info[edge]['destination'] = edge[1]

                self.logger.info('[I] Received network topology: ' + str(self.graph.number_of_nodes()) + ' nodes and '
                                 + str(self.graph.number_of_edges()) + ' edges')
            elif str(method.routing_key) == 'jet-pre':
                # Decode message. Extract all necessary information of streams and order accordingly
                info = json.loads(body.decode('utf-8'))
                for i in range(info['Number_of_Streams']):
                    self.vnf_list[i] = dict(source=info['Stream_Source_Destination'][i][0],
                                            destination=info['Stream_Source_Destination'][i][1],
                                            length=info['Streams_size'][i],
                                            period=info['Streams_Period'][i],
                                            max_delay=info['Deathline_Stream'][i])
                self.logger.info('[I] Received list of VNF. Contains ' + str(len(self.vnf_list)) + ' requests')
                self.route.append(self.current_node)
            else:
                # Should never reach this point. Throw error message
                self.logger.error('[!] Received unexpected routing key ' + str(method.routing_key) + '!')

            # If both topology and VNFs have been received, some preprocessing has to be carried out before being ready
            if self.ip_addresses is not {} and self.vnf_list is not {}:
                # Hyperperiod setting. Use maximum period of a given set (see config.py)
                # self.hyperperiod = math.lcm(*[self.vnf_list[i]['period'] for i in range(len(self.vnf_list))])
                self.hyperperiod = max(VNF_PERIOD)

                # Create edges info. Contains source, destination, delay and schedule (available bytes of each slot)
                for edge in range(self.graph.number_of_edges()):
                    self.edges_info[edge]['schedule'] = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR
                    self.edges_info[edge]['delay'] = 10

                # Replace IP addresses stored in source and destination of VNFs by corresponding node identifiers
                for i in range(len(self.vnf_list)):
                    for j in range(len(self.ip_addresses)):
                        if self.vnf_list[i]['source'] == self.ip_addresses[j]:
                            self.vnf_list[i]['source'] = j
                        if self.vnf_list[i]['destination'] == self.ip_addresses[j]:
                            self.vnf_list[i]['destination'] = j

                # Ready to work. Reset can now be fully processed
                self.ready = True

            ch.basic_ack(delivery_tag=method.delivery_tag)

        # More config stuff outside the callback function definition
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
        channel.start_consuming()

    # Background traffic generator. Called during reset
    def generate_background_traffic(self):
        # Iterate until having created all desired background streams (see config.py)
        for i in range(BACKGROUND_STREAMS):
            # Create random VNF (see vnf_generator.py) and get the route that will follow (shortest path)
            # VNF = {source, destination, length, period, max_delay, actions}
            self.background_traffic[i] = VNF(list(self.graph.nodes)).get_request()
            path = random.choice(list(nx.all_shortest_paths(self.graph,
                                                            source=self.background_traffic[i]['source'],
                                                            target=self.background_traffic[i]['destination'])))

            # Iterate along the path nodes until having assigned resources at all intermediate edges
            for j in range(len(path) - 1):
                # Search for the edge that has the desired source and destination nodes
                for edge_id, edge in self.edges_info.items():
                    # If edge has the desired source and destination, schedule resources and go to next hop
                    if edge['source'] == path[j] and edge['destination'] == path[j + 1]:
                        # Infinite loop in order to make sure the stream is scheduled
                        num_retries = 0
                        while True:
                            # Choose random position with probability i/BACKGROUND_STREAMS
                            # Choose position with higher availability with probability 1 - (i/BACKGROUND_STREAMS)
                            if random.randrange(BACKGROUND_STREAMS) > int(100 * i / BACKGROUND_STREAMS):
                                # action = (e, t)  e = edge ID  t = position ID
                                action = (edge_id, random.randrange(int(self.hyperperiod * DIVISION_FACTOR /
                                                                        self.background_traffic[i]['period'])))
                            else:
                                # Search for the position with higher capacity
                                time_slot = 0
                                max_capacity = 0
                                for slot in range(self.background_traffic[i]['period']):
                                    capacity = self.get_position_availability(edge['schedule'],
                                                                              slot,
                                                                              self.background_traffic[i]['period'])
                                    if capacity > max_capacity:
                                        max_capacity = capacity
                                        time_slot = slot

                                # action = (e, t)  e = edge ID  t = position ID
                                action = (edge_id, time_slot)

                            # Schedule action
                            # If action is done (return True), pass to next hop
                            # If action is impossible, create new VNF and try to schedule it. If looping too much, stop
                            if self.schedule_background_stream(action, i):
                                break
                            else:
                                num_retries += 1
                                # If looping too much, network has collapsed. Stop execution
                                if num_retries >= 1000:
                                    self.logger.warning('[!] Background traffic could not be allocated! Ask for less '
                                                        'background streams (see config.py --> BACKGROUND_STREAMS)')
                                    return
                        break

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

            # Add action to the actions list of the stram info
            self.background_traffic[stream_id]['actions'].append(action)
            return True
        else:
            return False

    # Returns the graph. Called during agent's initialization
    def get_graph(self):
        return self.graph

    # Returns the state vector that has to be returned to the agent as a numpy array. Called during reset and step
    def get_observation(self):
        # First build observation as a dictionary
        st = {}

        # Iterate along all edges for extracting its state
        for edge_id, edge in self.edges_info.items():
            if edge['source'] == self.current_node:
                # st[edge_id] = [0, 0]  # [dist, cong]
                #
                # # Compute distance to destination [0]
                # # Distance is calculated as the length of the shortest path between
                # #     the destination node of the edge and the destination node of the stream
                # st[edge_id][0] = nx.shortest_path_length(self.graph, source=edge['destination'],
                #                                          target=self.current_vnf['destination'])
                #
                # # Compute edge traffic load [1]. Calculated as the average load of all slots
                # st[edge_id][1] = int(100 * sum(i for i in list(edge['schedule']))
                #                      / (SLOT_CAPACITY * self.hyperperiod * DIVISION_FACTOR))

                # Compute position traffic loads. Minimum percentage of free bytes for all slots of each position
                slot_loads = [0] * self.hyperperiod * DIVISION_FACTOR
                if len(self.route) < 2:
                    for c in range(self.current_vnf['period']):
                        free_bytes = self.get_position_availability(edge['schedule'], c, self.current_vnf['period'])
                        slot_loads[c] = int(100 * free_bytes / SLOT_CAPACITY)
                elif edge['destination'] != self.route[-2]:
                    for c in range(self.current_vnf['period']):
                        free_bytes = self.get_position_availability(edge['schedule'], c, self.current_vnf['period'])
                        slot_loads[c] = int(100 * free_bytes / SLOT_CAPACITY)
                # st[edge_id] += slot_loads
                st[edge_id] = slot_loads

        # Process VNF. Pop actions and convert length to percentage of slot capacity
        vnf = list(self.current_vnf.values())[:-1]
        vnf[2] = int(100 * vnf[2] / SLOT_CAPACITY)

        # Merge all data into a list
        obs = vnf + [self.current_node] + [self.current_delay] + list(chain.from_iterable(st.values()))

        # Convert state list to a numpy array
        obs = np.array(obs, dtype=np.int16)
        return obs

    # Try to allocate resources for a real stream. If not possible, terminate episode. Called during step
    def schedule_stream(self, action):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources. Otherwise, terminate episode
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

    # Finds the availability of a position, which is the minimum availability of the slots of that position
    # Called during generate_background_traffic, schedule_background_stream and schedule_stream
    # Returns the availability of a given position for a given period
    def get_position_availability(self, schedule, position, period):
        min_availability = SLOT_CAPACITY
        slot = position
        while slot < self.hyperperiod * DIVISION_FACTOR:
            if schedule[slot] < min_availability:
                min_availability = schedule[slot]
            slot += period
        return min_availability

    # Updates the reward of the current episode
    def reward_function(self, action):
        if action[0] < len(self.edges_info):
            # Compute availabilities of all positions
            position_availabilities = [0] * self.current_vnf['period'] * DIVISION_FACTOR
            for c in range(self.current_vnf['period'] * DIVISION_FACTOR):
                free_bytes = self.get_position_availability(self.edges_info[action[0]]['schedule'], c,
                                                            self.current_vnf['period'])
                position_availabilities[c] = int(100 * free_bytes / SLOT_CAPACITY)

            # Not take into account the load of the current VNF
            position_availabilities[action[1]] += int(100 * self.current_vnf['length'] / SLOT_CAPACITY)

            self.reward += 0.1 * (position_availabilities[action[1]] - max(position_availabilities))
            # print(position_availabilities)
            # print(max(position_availabilities))
            # print(position_availabilities[action[1]])
            # print(self.reward)
            # print()

            # pre_len = nx.shortest_path_length(self.graph, source=self.route[-2], target=self.current_vnf['destination'])
            cur_len = nx.dijkstra_path_length(self.graph, source=self.current_node,
                                              target=self.current_vnf['destination'])

            self.reward -= cur_len
            # if pre_len - 1 != cur_len:
            #     self.reward -= 10
            # print(self.reward)
            # print(pre_len)
            # print(cur_len)
            # print(self.reward)
            # print()

            # If the agent did not choose the position with more availability, mark as non-optimal
            if max(position_availabilities) != position_availabilities[action[1]]:
                self.optimal_positions = False

        # If the episode is ended, check why
        if self.terminated:
            # If the stream has reached the destination, increase by 50 the reward
            if self.current_node == self.current_vnf['destination']:
                # self.reward += 50
                # If scheduling was optimal (selected shortest path and best positions) increase reward by 300
                if self.current_delay == nx.dijkstra_path_length(self.graph,
                                                                 self.current_vnf['source'],
                                                                 self.current_vnf['destination']) \
                        and self.optimal_positions is True:
                    self.reward += 300
                elif self.current_delay == nx.dijkstra_path_length(self.graph,
                                                                   self.current_vnf['source'],
                                                                   self.current_vnf['destination']) \
                        and self.optimal_positions is False:
                    self.reward += 50
                elif self.current_delay > nx.dijkstra_path_length(self.graph,
                                                                  self.current_vnf['source'],
                                                                  self.current_vnf['destination']) \
                        and self.optimal_positions is True:
                    self.reward += 150
                elif self.current_delay > nx.dijkstra_path_length(self.graph,
                                                                  self.current_vnf['source'],
                                                                  self.current_vnf['destination']) \
                        and self.optimal_positions is False:
                    self.reward = 0
        # If delay has reached the maximum acceptable value, end the episode and decrease by 100 the reward
        if self.current_delay > self.current_vnf['max_delay']:
            self.logger.warning(f"Maximum delay exceeded! {self.current_delay} -- {self.current_vnf['max_delay']}")
            self.terminated = True
            self.reward -= 100
        return

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Reset terminated, route, schedule, reward, remaining time steps and elapsed time
        for e in self.edges_info.values():
            e['schedule'] = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR
        self.route = []
        self.terminated = False
        self.remaining_timesteps = TIMESTEPS_LIMIT
        self.current_delay = 0
        self.optimal_positions = True
        self.reward = 0

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
            self.logger.debug('[D] RESET. info = ' + str(info) + ', obs = ' + str(obs[0:7]))
            for i in range(3):
                self.logger.debug('[D] RESET. Edge ' + str(i) + ' |  position availabilities: ' +
                                  str(obs[7 + (i * (self.hyperperiod * DIVISION_FACTOR)):7 + (
                                              (i + 1) * (self.hyperperiod * DIVISION_FACTOR))]))
        return obs, info

    def step(self, action_int):
        # Convert numerical action to edge-position tuple
        action = (
            int(action_int / (self.hyperperiod * DIVISION_FACTOR)), action_int % (self.hyperperiod * DIVISION_FACTOR))
        self.logger.info('[I] STEP. Action: ' + str(action_int) + ' ' + str(action))

        # Subtract 1 to the remaining time steps
        self.remaining_timesteps -= 1

        if action[0] >= len(self.edges_info):
            self.terminated = True

            info = {'previous_node': -1}
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

            info = {'previous_node': self.route[-2]}

            # Update reward of current episode
            self.reward_function(action)

        # Truncate episode in case it is becoming too large (maybe due to network loops)
        truncated = self.remaining_timesteps <= 0

        # Acquire data to return to the agent
        obs = self.get_observation()

        # Skip to next VNF in case of ending episode (just needed when not training)
        if self.terminated is True and TRAINING_IF is False:
            self.vnf_id += 1

        if ENV_LOG_LEVEL == 10:
            self.logger.debug('[D] STEP. info = ' + str(info) + ', terminated = ' + str(self.terminated) +
                              ', truncated = ' + str(truncated) +
                              ', reward = ' + str(self.reward) +
                              ', obs = ' + str(obs[0:7]))
            for i in range(3):
                self.logger.debug('[D] STEP. Edge ' + str(i) + ' |  position availabilities: ' +
                                  str(obs[7 + (i * (self.hyperperiod * DIVISION_FACTOR)):7 + (
                                              (i + 1) * (self.hyperperiod * DIVISION_FACTOR))]))
        if self.terminated:
            self.logger.info('[I] Ending episode...\n')
        return obs, self.reward, self.terminated, truncated, info

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
