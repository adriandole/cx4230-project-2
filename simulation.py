import numpy as np
from typing import List
from event_engine import EventEngine, Event


def generate_bimodal(mu: List[float], sigma: List[float], p: float):
    if np.random.uniform() < p:
        return np.random.normal(mu[0], sigma[0])
    else:
        return np.random.normal(mu[1], sigma[1])


def get_travel_time(section: int, time: str) -> float:
    """
    Get travel time for a specific section or road, either AM or PM conditions
    :param section: number of section. Renamed to [0, 1, 2] from [2, 3, 5] in the report
    :param time: AM or PM
    :return: travel time
    """
    mu_pm = [[14.51, 21.68], [11, 44.14], [12.92, 50.65]]
    sigma_pm = [[1.4767, 3.5132], [0.7638, 4.3315], [2.4884, 7.7452]]
    p_pm = [0.5918, 0.2715, 0.2104]

    mu_am = [[10.99, 66.88], [10.52, 45.88], [9.369, 52.79]]
    sigma_am = [[1.408, 2.541], [0.9083, 2.4368], [1.6507, 8.4215]]
    p_am = [0.8204, 0.4221, 0.6021]

    if not 0 <= section <= 2:
        raise ValueError('Section must be 0, 1, or 2')

    if time == 'AM':
        return generate_bimodal(mu_am[section], sigma_am[section], p_am[section])
    elif time == 'PM':
        return generate_bimodal(mu_pm[section], sigma_am[section], p_am[section])
    else:
        raise ValueError('Time must be AM or PM')


class Arrival(Event):

    def __init__(self, time, section):
        super().__init__(time)
        self.section = section


class Departure(Event):

    def __init__(self, time, section):
        super().__init__(time)
        self.section = section


class TrafficSimulation:

    def __init__(self):
        self.engine = EventEngine()
