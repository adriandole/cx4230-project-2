import heapq


class Event:

    def __init__(self, time):
        self.heap_time = time  # Python heap returns the lowest value
        self.time = time

    def __eq__(self, other):
        return self.heap_time == other.heap_time

    def __lt__(self, other):
        return self.heap_time < other.heap_time


class EventEngine:

    def __init__(self):
        self.h = []

    def queue_event(self, event):
        heapq.heappush(self.h, event)

    def get_next_event(self):
        return heapq.heappop(self.h)
