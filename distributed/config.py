# Logging settings
ENV_LOG_LEVEL = 10  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
ENV_LOG_FILE_NAME = '../logs/env_'  # Name of file where to store all env generated logs. Make sure the directory exists
DQN_LOG_LEVEL = 20  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
DQN_LOG_FILE_NAME = '../logs/dqn_'  # Name of file where to store all DQN generated logs. Make sure the directory exists

# Environment settings
SLOT_CAPACITY = 12500  # Number of bytes that can be transmitted at each time slot
DIVISION_FACTOR = 10  # Number of slots to define per millisecond
BACKGROUND_STREAMS = 150  # Number of streams to create as background traffic. Their scheduling is prefixed
CUSTOM_ROUTE = (0, 13)  # Custom route used for evaluating joint models A and B
CUSTOM_ROUTE_A = (0, 8)  # Custom route used for training and evaluating models A
CUSTOM_ROUTE_B = (8, 13)  # Custom route used for training and evaluating models B
ALL_ROUTES = False  # Flag that is set in case of wanting to train or evaluate for all routes of the graph.
#                     If set to False, use custom route defined at CUSTOM_ROUTE variables

# VNF generator settings
VNF_LENGTH = [128, 256, 512, 1024, 1500]  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [20]  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [2, 4, 8, 16]  # List of possible periods to generate in random VNFs
#                                          Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "../models/DQN/dist/"  # Path where models will be stored. Filenames are auto. Make sure that the directory exists!
SEED = None  # 1976  # Seed used for randomization purposes
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 6
TARGET_UPDATE = 400
EPSILON_DECAY = 1 / 80000
MAX_EPSILON = 1.0
MIN_EPSILON = 0.0
GAMMA = 0.999
LEARNING_RATE = 0.0001
TAU = 0.85

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = '../plots/DQN/dist/'

# Topology settings
TRAINING_NODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # List of nodes in the training net
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
    (1, 8): {'delay': 2},
    (5, 9): {'delay': 1},
    (8, 10): {'delay': 4},
    (10, 8): {'delay': 4},
    (10, 11): {'delay': 3},
    (11, 10): {'delay': 3},
    (10, 12): {'delay': 1},
    (12, 10): {'delay': 1},
    (12, 13): {'delay': 5},
    (13, 12): {'delay': 5},
    (13, 11): {'delay': 4},
    (11, 13): {'delay': 4},
    (13, 14): {'delay': 1},
    (14, 13): {'delay': 1},
    (14, 9): {'delay': 2},
    (9, 14): {'delay': 2},
    (9, 8): {'delay': 1},
    (8, 9): {'delay': 1},
    (9, 5): {'delay': 1},
    (8, 1): {'delay': 2}
}
TRAINING_NODES_A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # List of nodes in the training net
TRAINING_EDGES_A = {
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
    (1, 8): {'delay': 2},
    (5, 9): {'delay': 1}
}
TRAINING_NODES_B = [1, 5, 8, 9, 10, 11, 12, 13, 14]  # List of nodes in the training net
TRAINING_EDGES_B = {
    (8, 10): {'delay': 4},
    (10, 8): {'delay': 4},
    (10, 11): {'delay': 3},
    (11, 10): {'delay': 3},
    (10, 12): {'delay': 1},
    (12, 10): {'delay': 1},
    (12, 13): {'delay': 5},
    (13, 12): {'delay': 5},
    (13, 11): {'delay': 4},
    (11, 13): {'delay': 4},
    (13, 14): {'delay': 1},
    (14, 13): {'delay': 1},
    (14, 9): {'delay': 2},
    (9, 14): {'delay': 2},
    (9, 8): {'delay': 1},
    (8, 9): {'delay': 1},
    (9, 5): {'delay': 1},
    (8, 1): {'delay': 2}
}
