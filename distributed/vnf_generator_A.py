import random
from config import VNF_LENGTH, VNF_PERIOD, VNF_DELAY, SEED, CUSTOM_ROUTE_A

if SEED is not None:
    random.seed(SEED)


class VNF:
    def __init__(self, graph_nodes):
        if graph_nodes is not None and len(graph_nodes) > 2:
            self.source, self.target = random.sample(graph_nodes, k=2)  # Source and destination of stream
        elif graph_nodes is not None and len(graph_nodes) == 2:
            self.source, self.target = graph_nodes[0], graph_nodes[1]
        else:
            self.source, self.target = CUSTOM_ROUTE_A  # Source and destination of stream
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
