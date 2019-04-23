import numpy as np
from typing import List
from random import random
from queue import Queue
from event_engine import EventEngine, Event


def get_light_status(section, time) -> float:
    """
    Evaluates the traffic light cycle for each section
    :param section: number of section being entered
    :param time: arrival at traffic light
    :return: time left to wait at light (0 if green)
    """
    section_cycle = [87.6, 100.1, 99.8, 83.9]
    section_red = [52.9, 58.6, 38.9, 49.3]

    cycle_time = time % section_cycle[section]
    if cycle_time > section_red[section]:
        return 0
    else:
        return section_red[section] - cycle_time


def get_red_wait(n_waiting) -> float:
    """
    Additional red light time as a function of number of cars waiting. See red_wait_data.py for how
    these coefficients were calculated
    :param n_waiting: number of cars in front
    :return: additional wait time (s)
    """
    return 2.878726 * n_waiting + 3.001445


def generate_bimodal(mu: List[float], sigma: List[float], p: float):
    """
    Sample from a mixture of two normal distribution
    :param mu: [mean of first dist., mean of second dist.]
    :param sigma: [st. dev. of first dist., st. dev. of second dist]
    :param p: probability of sampling from the first list
    :return: sample value
    """
    # if np.random.uniform() < p:
    #     return np.random.normal(mu[0], sigma[0])
    # else:
    #     return np.random.normal(mu[1], sigma[1])
    return np.random.normal(mu[0], sigma[0])


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


def get_interarrival_time(alpha=1.00) -> float:
    fname = 'interarrival_times/interarrival-{:.2f}-cdf.dat'.format(alpha)
    cdf = np.genfromtxt(fname, delimiter=',')

    r = random()
    cdf_row = np.argmax(cdf[:, 1] > r)
    return cdf[cdf_row, 0]


class TrafficEvent(Event):

    def __init__(self, base, time, section):
        super().__init__(time)
        self.base = base  # reference to the base simulation class, allowing the event to handle itself
        self.section = section


class Departure(TrafficEvent):

    def handle(self):
        delay = get_light_status(self.section, self.time)
        if delay > 0:
            delay += get_red_wait(self.base.section_queueing[self.section])

        self.base.section_queueing[self.section] += 1

        if self.section < 2:
            next_arrival = Arrival(self.base, self.time + delay, self.section + 1)
            self.base.engine.queue_event(next_arrival)
        elif self.section == 2:
            arrival_time = self.base.arrivals.get()
            self.base.travel_times.append(self.time - arrival_time)

        self.base.section_occupancy[self.section] -= 1
        self.base.section_data = np.append(self.base.section_data,
                                           np.array([self.time] + self.base.section_occupancy).reshape((1, 4)), axis=0)


class Arrival(TrafficEvent):

    def handle(self):
        if self.section == 0:
            next_arrival_time = get_interarrival_time(alpha=self.base.traffic_alpha) + self.time
            next_arrival = Arrival(self.base, next_arrival_time, self.section)
            self.base.engine.queue_event(next_arrival)
            self.base.arrivals.put(next_arrival_time)

        if self.section >= 1:
            self.base.section_queueing[self.section - 1] -= 1

        travel_time = get_travel_time(self.section, self.base.am_pm) + self.time
        departure = Departure(self.base, travel_time, self.section)
        self.base.engine.queue_event(departure)

        self.base.section_occupancy[self.section] += 1


class TrafficSimulation:

    def __init__(self, traffic_alpha):
        self.engine = EventEngine()
        # self.inter_arrival_mu = inter_arrival_mu  # all parameters will be in a config file later
        # self.inter_arrival_sigma = inter_arrival_sigma
        self.traffic_alpha = traffic_alpha
        self.am_pm = 'PM'

        self.section_occupancy = [0, 0, 0]
        self.section_queueing = [0, 0, 0, 0]
        self.section_data = np.empty((0, 4))

        self.arrivals = Queue()
        self.travel_times = []

    def run_simulation(self, sim_time: int):
        time = 0
        self.engine.queue_event(Arrival(self, time, 0))
        self.arrivals.put(0)
        while time <= sim_time:
            e = self.engine.get_next_event()
            time = e.time
            e.handle()

            if type(e).__name__ == 'Departure':
                print(self.section_occupancy, time)


if __name__ == '__main__':
    t = TrafficSimulation(1.00)
    t.run_simulation(10000)
    print(np.mean(t.travel_times))
