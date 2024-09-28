# Logging settings
ENV_LOG_LEVEL = 40  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
ENV_LOG_FILE_NAME = '../logs/env_'  # Name of file where to store all env generated logs. Make sure the directory exists
DQN_LOG_LEVEL = 20  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
DQN_LOG_FILE_NAME = '../logs/dqn_'  # Name of file where to store all DQN generated logs. Make sure the directory exists

# Environment settings
TRAINING_STEPS = 200000  # Number of time-steps to run for training an agent
CUSTOM_EVALUATION_EPISODES = 1000  # Number of episodes to run for a custom evaluation.
ROUTE_EVALUATION_EPISODES = 100  # Number of episodes to run for each route of a multi-route evaluation
MONITOR_TRAINING = 10000  # Number of time-steps to count for logging training results
SLOT_CAPACITY = 12500  # Number of bytes that can be transmitted at each time slot
DIVISION_FACTOR = 1  # Number of slots to define per millisecond
BACKGROUND_STREAMS = 300  # Number of streams to create as background traffic. Their scheduling is prefixed
CUSTOM_ROUTE = (0, 13)  # Custom route used for evaluating joint models A and B
ALL_ROUTES = False  # Flag that is set in case of wanting to train or evaluate for all routes of the graph.
#                     If set to False, use custom route defined at CUSTOM_ROUTE variables

# VNF generator settings
VNF_LENGTH = [128, 256, 512, 1024, 1500]  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [13]  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [2, 4, 8, 16]  # List of possible periods to generate in random VNFs
#                                          Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "../models/DQN/cent/"  # Path where models will be stored. Filenames are auto. Make sure that the directory exists!
SEED = 1976  # Seed used for randomization purposes
REPLAY_BUFFER_SIZE = 1000000  # Hyperparameter for DQN agent
BATCH_SIZE = 6  # Hyperparameter for DQN agent
TARGET_UPDATE = 400  # Hyperparameter for DQN agent
EPSILON_DECAY = (1 / 80000)  # Hyperparameter for DQN agent
MAX_EPSILON = 1.0  # Hyperparameter for DQN agent
MIN_EPSILON = 0.0  # Hyperparameter for DQN agent
GAMMA = 0.999  # Hyperparameter for DQN agent
LEARNING_RATE = 0.0001  # Hyperparameter for DQN agent
TAU = 0.85  # Hyperparameter for DQN agent

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = '../plots/DQN/cent/'

# Topology settings
TRAINING_NODES = [0, 1, 2, 3, 4, 5, 6, 7]  # List of nodes
TRAINING_EDGES = {
    (0, 1): {'delay': 1},
    (1, 0): {'delay': 1},
    (1, 3): {'delay': 2},
    (3, 1): {'delay': 2},
    (3, 2): {'delay': 3},
    (2, 3): {'delay': 3},
    (2, 0): {'delay': 4},
    (0, 2): {'delay': 4},
    (0, 4): {'delay': 3},
    (4, 0): {'delay': 3},
    (1, 5): {'delay': 5},
    (5, 1): {'delay': 5},
    (3, 7): {'delay': 7},
    (7, 3): {'delay': 7},
    (2, 6): {'delay': 1},
    (6, 2): {'delay': 1},
    (4, 5): {'delay': 4},
    (5, 4): {'delay': 4},
    (5, 7): {'delay': 1},
    (7, 5): {'delay': 1},
    (7, 6): {'delay': 2},
    (6, 7): {'delay': 2},
    (6, 4): {'delay': 3},
    (4, 6): {'delay': 3},
}  # List of edges
