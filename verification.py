import simulation as s
import event_engine as e
import unittest
import numpy as np
from random import uniform


class TestFEL(unittest.TestCase):

    def test_heap(self):
        heap = e.EventEngine()
        for n in range(100):
            #  queue 100 events of random times
            heap.queue_event(e.Event(uniform(0, 100)))

        min_time = 0
        for n in range(100):
            event_time = heap.get_next_event().time
            self.assertTrue(event_time >= min_time)
            min_time = event_time


class TestTrafficSim(unittest.TestCase):

    def _gen_test_sim(self):
        sim = s.TrafficSimulation(traffic_alpha=1.00)
        for n in range(100):
            sim.engine.queue_event(s.Arrival(sim, n, 1, n))
        return sim

    def test_arrivals(self):
        sim = self._gen_test_sim()
        sim.run_simulation(500, queue_initial=False)
        self.assertEqual(len(sim.travel_times), 100)


if __name__ == '__main__':
    unittest.main()
