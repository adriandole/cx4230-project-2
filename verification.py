import simulation as s
import event_engine as e
import unittest
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
            assert(event_time >= min_time)
            min_time = event_time


if __name__ == '__main__':
    unittest.main()
