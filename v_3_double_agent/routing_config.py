# Logging settings
ENV_LOG_LEVEL = 40  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
ENV_LOG_FILE_NAME = 'logs/env'  # Name of file where to store all env generated logs. Make sure the directory exists
DQN_LOG_LEVEL = 20  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
DQN_LOG_FILE_NAME = 'logs/dqn'  # Name of file where to store all DQN generated logs. Make sure the directory exists

# Rabbitmq settings
RABBITMQ_HOST = 'rabbitmq-microservice'  # Host of RabbitMQ server
RABBITMQ_PORT = None  # Port of RabbitMQ server. If None, the default port is used (5672) and auth data is ignored
RABBITMQ_USERNAME = None  # Username of RabbitMQ server. If None, auth data is ignored
RABBITMQ_PASSWORD = None  # Password of RabbitMQ server. If None, auth data is ignored
RABBITMQ_EXCHANGE_NAME = 'tsn'  # Name of the exchange where to publish data and subscribe

# Environment settings
TIMESTEPS_LIMIT = 5  # Maximum number of time-steps an episode can last
SLOT_CAPACITY = 12500  # Number of bytes that can be transmitted at each time slot
DIVISION_FACTOR = 1  # Number of slots to define per millisecond

# Background traffic settings
BACKGROUND_STREAMS = 300  # 750  # Number of streams to create as background traffic. Their scheduling is prefixed

# Training settings
TRAINING_IF = True  # Toggle training mode. When training, VNFs are random and topology is given in this section
NODES = [0, 1, 2, 3, 4, 5, 6, 7]  # List of nodes in the training net
EDGES = {
    (0, 1): {'delay': 1},
    (0, 2): {'delay': 4},
    (0, 4): {'delay': 3},
    (1, 0): {'delay': 1},
    (1, 3): {'delay': 2},
    (1, 5): {'delay': 5},
    (2, 0): {'delay': 4},
    (2, 3): {'delay': 3},
    (2, 6): {'delay': 1},
    (3, 1): {'delay': 2},
    (3, 2): {'delay': 3},
    (3, 7): {'delay': 7},
    (4, 0): {'delay': 3},
    (4, 5): {'delay': 4},
    (4, 6): {'delay': 3},
    (5, 1): {'delay': 5},
    (5, 4): {'delay': 4},
    (5, 7): {'delay': 1},
    (6, 2): {'delay': 1},
    (6, 4): {'delay': 3},
    (6, 7): {'delay': 2},
    (7, 3): {'delay': 7},
    (7, 5): {'delay': 1},
    (7, 6): {'delay': 2},
}

# EDGES = {
#     (0, 1): {'delay': 1},
#     (0, 2): {'delay': 1},
#     (0, 4): {'delay': 1},
#     (1, 0): {'delay': 1},
#     (1, 3): {'delay': 1},
#     (1, 5): {'delay': 1},
#     (2, 0): {'delay': 1},
#     (2, 3): {'delay': 1},
#     (2, 6): {'delay': 1},
#     (3, 1): {'delay': 1},
#     (3, 2): {'delay': 1},
#     (3, 7): {'delay': 1},
#     (4, 0): {'delay': 1},
#     (4, 5): {'delay': 1},
#     (4, 6): {'delay': 1},
#     (5, 1): {'delay': 1},
#     (5, 4): {'delay': 1},
#     (5, 7): {'delay': 1},
#     (6, 2): {'delay': 1},
#     (6, 4): {'delay': 1},
#     (6, 7): {'delay': 1},
#     (7, 3): {'delay': 1},
#     (7, 5): {'delay': 1},
#     (7, 6): {'delay': 1},
# }

# VNF generator settings
VNF_LENGTH = [1500]  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [7, 10, 13, 16]  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [4]  # List of possible periods to generate in random VNFs
#                                          Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "../models/DQN/routing/"  # Path where models will be stored. Make sure that the directory exists!
SEED = None  # 1976  # Seed used for randomization purposes

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = '../plots/DQN/routing/'
