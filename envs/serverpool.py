"""
The code is changed from the book named Feedback Control for Computer Systems by Philipp K. Janert (O'Reilly Media) - Chapter 13
link: https://github.com/oreillymedia/feedback_control_for_computer_systems
"""
import random
import numpy as np
import globalvars
from generators.read_data import get_workload
random.seed(globalvars.GLOBAL_SEED)
np.random.seed(globalvars.GLOBAL_SEED)


class AbstractServerPool(object):
    def __init__(self, **kwargs):
        if 'maximum_queue' in kwargs:
            self.maximum_queue = kwargs['maximum_queue']
        else:
            self.maximum_queue = 100000
        if 'maximum_instances' in kwargs:
            self.maximum_instances = kwargs['maximum_instances']
        else:
            self.maximum_instances = 5
        if 'maximum_workload_index' in kwargs:
            self.maximum_workload_index = kwargs['maximum_workload_index']
        else:
            self.maximum_workload_index = 24000
        if 'workload_type' in kwargs:
            self.workload_type = kwargs['workload_type']
        else:
            self.workload_type = globalvars.WORKLOAD_TYPE
        if 'time_pace' in kwargs:
            self.time_pace = kwargs['time_pace']
        else:
            self.time_pace = 1000
        if 'n' in kwargs:
            self.n = kwargs['n']
        else:
            self.n = 1
        self.queue = 0
        self.current_workload = 0
        self.current_index = 0
        if self.workload_type == 'file':
            print('Get Workload from file')
            self.workloads = get_workload()
            self.maximum_workload_index = len(self.workloads) - 10

    def reset(self):
        self.n = 1
        self.queue = 0
        self.current_index = random.randint(0, 10)
        print('start new eps with current index=', self.current_index)
        self.current_workload = self.workloads[self.current_index]
        queue_percent = 100*self.queue/self.maximum_queue
        return self.current_workload, queue_percent, self.n

    def work(self, u):
        """
        input u: number of servers
        """
        self.n = max(1, int(round(u)))  # server count: non-negative integer
        self.n = min(self.n, self.maximum_instances)  # server count: less than max
        completed = 0
        for _ in range(self.n):
            completed += self.consume_queue()  # each server does some amount of work
            if completed >= self.queue:
                completed = self.queue  # "trim" completed to queue length
                break  # stop if queue is empty
        self.queue -= completed  # reduce queue by work completed
        return completed

    # Monitoring part
    def monitoring(self):
        if self.is_done():
            return None
        return self.current_workload, 100*self.queue/self.maximum_queue, self.n

    # Get workload
    def load_queue(self):
        w = self.workloads[self.current_index]
        self.current_index += 1
        return w

    # Consuming workload
    def consume_queue(self):
        a, b = 20, 2
        return 50 * random.betavariate(a, b)  # mean: a/(a+b); var=~b/a^2

    def is_done(self):
        return self.current_index >= self.maximum_workload_index


class ServerPool(AbstractServerPool):
    def work(self, u):
        self.current_workload = self.load_queue()  # get the workload
        self.queue = min(self.current_workload,
                         self.maximum_queue)  # just reach to maximum queue
        if self.current_workload == 0.:
            return 1.  # no work: 100 percent completion rate
        completed = AbstractServerPool.work(self, u)
        return 100 * completed / self.current_workload


class QueuingServerPool(AbstractServerPool):
    def work(self, u):
        self.current_workload = self.load_queue()
        self.queue += self.current_workload
        self.queue = min(self.queue,
                         self.maximum_queue)
        completed = AbstractServerPool.work(self, u)
        return 100*self.queue/self.maximum_queue


class ServerPoolWithLatency(ServerPool):
    def __init__(self, n, latency=10, workload_type='constant'):
        ServerPool.__init__(self, n=n, workload_type=workload_type)
        self.latency = latency  # time steps before server active

    def work(self, u):
        u = max(1, int(round(u)))
        if u <= self.n:
            return ServerPool.work(self, u)
        for i in range(self.latency):
            ServerPool.work(self, self.n)
        return ServerPool.work(self, u)


class QueuingServerPoolWithLatency(QueuingServerPool):
    def __init__(self, n, latency, step_to_convergence=10,
                 workload_type='constant'):
        QueuingServerPool.__init__(self, n=n,
                                   workload_type=workload_type)
        self.latency = latency  # time steps before server active
        self.step_to_convergence = step_to_convergence  # Time step for stable

    def work(self, u):
        u = max(1, int(round(u)))
        if u <= self.n:
            return QueuingServerPool.work(self, u)
        for _ in range(self.latency):
            QueuingServerPool.work(self, self.n)
        return QueuingServerPool.work(self, u)
