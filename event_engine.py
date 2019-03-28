import heapq


class Event:

    def __init__(self, time):
        self.time = -time  # Python heap returns the lowest value

    def __eq__(self, other):
        return self.time == other.time

    def __lt__(self, other):
        return self.time < other.time


class EventEngine:

    def __init__(self):
        self.h = []

    def queue_event(self, event):
        heapq.heappush(self.h, event)

    def get_next_event(self):
        return heapq.heappop(self.h)
