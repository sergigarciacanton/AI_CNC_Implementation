import random
from config_distributed import VNF_LENGTH, VNF_PERIOD, VNF_DELAY, SEED

if SEED is not None:
    random.seed(SEED)


class VNF:
    def __init__(self, graph_nodes, background_traffic):
        if graph_nodes is not None and len(graph_nodes) > 2:
            if background_traffic:
                self.source, self.target = random.sample(graph_nodes, k=2)  # Source and destination of stream
            else:
                while True:
                    self.source = random.choice(graph_nodes)
                    if self.source < 8:
                        break
                self.target = 9
        elif graph_nodes is not None and len(graph_nodes) == 2:
            self.source, self.target = graph_nodes[0], graph_nodes[1]
        else:
            self.source, self.target = 6, 9  # Source and destination of stream
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
