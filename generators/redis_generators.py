#!/usr/bin/python
import time
import redis
import datetime
import random
import threading
from pytz import timezone

"""
Generate workload for redis
"""


def write_to_file(line_count=1000):
    f = open('access_log.log', 'w')
    # ips
    with open('ips.txt') as ips_file:
        ips = ips_file.read().splitlines()
    # referers
    with open('referers.txt') as referers_file:
        referers = referers_file.read().splitlines()
    # resources
    with open('resources.txt') as resources_file:
        resources = resources_file.read().splitlines()
    # user agents
    with open('user_agents.txt') as user_agents_file:
        useragents = user_agents_file.read().splitlines()
    # codes
    with open('codes.txt') as codes_file:
        codes = codes_file.read().splitlines()
    # requests
    with open('requests.txt') as requests_file:
        requests = requests_file.read().splitlines()
    event_time = datetime.datetime(2013, 10, 10).replace(tzinfo=timezone('UTC'))
    for i in range(0, line_count):
        increment = datetime.timedelta(seconds=random.randint(30, 300))
        event_time += increment
        uri = random.choice(resources)
        if uri.find("Store") > 0:
            uri += random.randint(1000, 1500)
        ip = random.choice(ips)
        useragent = random.choice(useragents)
        referer = random.choice(referers)
        code = random.choice(codes)
        request = random.choice(requests)
        string = '{} - - [{}] "{} {} HTTP/1.0" {} {} "{}" "{}"'.format(
            random.choice(ips), event_time.strftime('%d/%b/%Y:%H:%M:%S %z'),
            request, uri, code, random.randint(2000, 5000),
            referer, useragent)
        f.write(string)
        f.write('\n')
    f.close()
    print('Write file successfully')


class Sender(threading.Thread):
    def __init__(self, hostname='bufers', port=3389):
        threading.Thread.__init__(self)
        self.hostname = hostname
        self.port = port
        self.redis = redis.Redis(self.hostname, self.port,
                                 db=0, password=None)

    def initialized_workloads(self, filename='access_log.log'):
        with open(filename) as f:
            lines = f.read().splitlines()
        workloads = []
        for line in lines:
            json = {"message": line}
            workloads.append(str(json))
        return workloads

    def run(self):
        workloads = self.initialized_workloads()
        print(workloads[0])
        while(True):
            num = random.gauss(1800, 50)
            num = int(num)
            print('workload length:', num)
            self.redis.rpush('logstash', *workloads[:num])
            time.sleep(1)

# Send workload to redis
# if __name__ == '__main__':
#     write_to_file(line_count=4000)
#     t = Sender(hostname='10.211.56.99', port=6379)
#     t.start()
#     h = Sender(hostname='10.211.56.99', port=6379)
#     h.start()
