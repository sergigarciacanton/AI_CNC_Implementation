import random
from config import VNF_LENGTH, VNF_PERIOD, VNF_DELAY


class VNF:
    def __init__(self, graph_nodes):
        self.source, self.target = random.sample(graph_nodes, k=2)  # Source and destination of stream
        # self.source, self.target = 0, 3  # Source and destination of stream
        self.length = random.choice(VNF_LENGTH)  # Length of packets to send [bytes]
        self.period = random.choice(VNF_PERIOD)  # Periodicity of sent packets [ms]
        self.max_delay = random.choice(VNF_DELAY)  # Maximum acceptable delay [ms]

    # Returns the generated VNF as a dictionary
    def get_request(self):
        return dict(source=self.source,
                    destination=self.target,
                    length=self.length,
                    period=self.period,
                    max_delay=self.max_delay,
                    actions=[])
