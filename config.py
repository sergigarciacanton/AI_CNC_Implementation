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
TRAINING_NODES = [0, 1, 2, 3, 4, 5, 6, 7]  # List of nodes in the training net
TRAINING_EDGES = {
    (0, 1): {'delay': 10},
    (1, 0): {'delay': 10},
    (1, 3): {'delay': 20},
    (3, 1): {'delay': 20},
    (3, 2): {'delay': 30},
    (2, 3): {'delay': 30},
    (2, 0): {'delay': 40},
    (0, 2): {'delay': 40},
    (0, 4): {'delay': 30},
    (4, 0): {'delay': 30},
    (1, 5): {'delay': 50},
    (5, 1): {'delay': 50},
    (3, 7): {'delay': 70},
    (7, 3): {'delay': 70},
    (2, 6): {'delay': 10},
    (6, 2): {'delay': 10},
    (4, 5): {'delay': 40},
    (5, 4): {'delay': 40},
    (5, 7): {'delay': 10},
    (7, 5): {'delay': 10},
    (7, 6): {'delay': 20},
    (6, 7): {'delay': 20},
    (6, 4): {'delay': 30},
    (4, 6): {'delay': 30},
}

# VNF generator settings
VNF_LENGTH = [1500]  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [160]  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [4]  # List of possible periods to generate in random VNFs
#                                          Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "models/DQN/"  # Path where models will be stored. Filenames are auto. Make sure that the directory exists!
SEED = None  # 1976  # Seed used for randomization purposes

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = 'plots/DQN/'
