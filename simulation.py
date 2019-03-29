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
        return generate_bimodal(mu_pm[section], sigma_pm[section], p_pm[section])
    else:
        raise ValueError('Time must be AM or PM')


class TrafficEvent(Event):

    def __init__(self, base: TrafficSimulation, time, section):
        super().__init__(time)
        self.base = base  # reference to the base simulation class, allowing the event to handle itself
        self.section = section


class Departure(TrafficEvent):

    def handle(self):
        if self.section < 2:
            next_arrival = Arrival(self.base, self.time, self.section + 1)
            self.base.engine.queue_event(next_arrival)


class Arrival(TrafficEvent):

    def handle(self):
        if self.section == 0:
            next_arrival_time = np.random.normal(self.base.inter_arrival_mu, self.base.inter_arrival_sigma) + self.time
            next_arrival = Arrival(self.base, next_arrival_time, self.section + 1)
            self.base.engine.queue_event(next_arrival)
        if self.section < 2:
            travel_time = get_travel_time(self.section, self.base.am_pm) + self.time
            departure = Departure(self.base, travel_time, self.section)
            self.base.engine.queue_event(departure)


class TrafficSimulation:

    def __init__(self, inter_arrival_mu, inter_arrival_sigma):
        self.engine = EventEngine()
        self.inter_arrival_mu = inter_arrival_mu  # all parameters will be in a config file later
        self.inter_arrival_sigma = inter_arrival_sigma
        self.am_pm = 'AM'

    def run_simulation(self, sim_time: int):
