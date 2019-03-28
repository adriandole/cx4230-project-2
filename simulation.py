import numpy as np
from event_engine import EventEngine, Event


def generate_bimodal(mu: np.ndarray, sigma: np.ndarray, p: float):
    if np.random.uniform() < p:
        return np.random.normal(mu[0], sigma[0])
    else:
        return np.random.normal(mu[1], sigma[1])


class TrafficSimulation:

    def __init__(self):
        self.engine = EventEngine()
