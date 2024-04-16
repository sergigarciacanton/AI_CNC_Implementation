# Logging settings
ENV_LOG_LEVEL = 40  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
ENV_LOG_FILE_NAME = 'logs/env.log'  # Name of file where to store all env generated logs. Make sure the directory exists
DQN_LOG_LEVEL = 20  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
DQN_LOG_FILE_NAME = 'logs/dqn.log'  # Name of file where to store all DQN generated logs. Make sure the directory exists

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
BACKGROUND_STREAMS = 50  # Number of streams to create as background traffic. Their scheduling is prefixed

# Training settings
TRAINING_IF = True  # Toggle training mode. When training, VNFs are random and topology is given in this section
TRAINING_NODES = [0, 1, 2, 3, 4, 5, 6, 7]  # List of nodes in the training net
TRAINING_EDGES = [(0, 1), (1, 0), (1, 3), (3, 1), (3, 2), (2, 3), (2, 0), (0, 2),
                  (0, 4), (4, 0), (1, 5), (5, 1), (3, 7), (7, 3), (2, 6), (6, 2),
                  (4, 5), (5, 4), (5, 7), (7, 5), (7, 6), (6, 7), (6, 4), (4, 6)]  # List of edges in the training net

# VNF generator settings
VNF_LENGTH = [1500]  # 64, 128, 256, 512, 1024,  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [100]  # , 300, 500, 700, 1000  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [4]  # 1, 2, 8, 16, 32, 64  # List of possible periods to generate in random VNFs
#                                          Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "models/DQN/"  # Path where models will be stored. Filenames are auto. Make sure that the directory exists!
SEED = 1976  # Seed used for randomization purposes

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = 'plots/DQN/'
