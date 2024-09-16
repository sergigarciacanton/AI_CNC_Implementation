# Logging settings
ENV_LOG_LEVEL = 40  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
ENV_LOG_FILE_NAME = '../logs/env'  # Name of file where to store all env generated logs. Make sure the directory exists
DQN_LOG_LEVEL = 20  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
DQN_LOG_FILE_NAME = '../logs/dqn'  # Name of file where to store all DQN generated logs. Make sure the directory exists

# Rabbitmq settings
RABBITMQ_HOST = 'rabbitmq-microservice'  # Host of RabbitMQ server
RABBITMQ_PORT = None  # Port of RabbitMQ server. If None, the default port is used (5672) and auth data is ignored
RABBITMQ_USERNAME = None  # Username of RabbitMQ server. If None, auth data is ignored
RABBITMQ_PASSWORD = None  # Password of RabbitMQ server. If None, auth data is ignored
RABBITMQ_EXCHANGE_NAME = 'tsn'  # Name of the exchange where to publish data and subscribe

# Environment settings
TIMESTEPS_LIMIT = 10  # Maximum number of time-steps an episode can last
SLOT_CAPACITY = 12500  # Number of bytes that can be transmitted at each time slot
DIVISION_FACTOR = 1  # Number of slots to define per millisecond

# Background traffic settings
BACKGROUND_STREAMS = 300  # 300  # Number of streams to create as background traffic. Their scheduling is prefixed

# Training settings
TRAINING_IF = True  # Toggle training mode. When training, VNFs are random and topology is given in this section
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

# VNF generator settings
VNF_LENGTH = [128, 256, 512, 1024, 1500]  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [13]  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [2, 4, 8, 16]  # List of possible periods to generate in random VNFs
#                                          Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "../models/DQN/dist/"  # Path where models will be stored. Filenames are auto. Make sure that the directory exists!
SEED = 1976  # 1976  # Seed used for randomization purposes

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = '../plots/DQN/dist/'
