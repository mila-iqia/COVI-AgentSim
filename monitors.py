from config import TICK_MINUTE
from base import City
from simulator import Human
from matplotlib import pyplot as plt
import json
import pylab as pl
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import zipfile
from utils import _json_serialize


class BaseMonitor(object):

    def __init__(self, f=None, dest: str = None, chunk_size: int = None):
        self.data = []
        self.f = f or 60
        self.dest = dest
        self.chunk_size = chunk_size if self.dest and chunk_size else 0

    def run(self, env, city: City):
        raise NotImplementedError

    def dump(self):
        pass


class SEIRMonitor(BaseMonitor):

    def run(self, env, city: City):

        while True:
            S, E, I, R = 0, 0, 0, 0
            R0 = city.tracker.get_R()
            G = city.tracker.get_generation_time()

            for h in city.humans:
                S += h.is_susceptible
                E += h.is_exposed
                I += h.is_infectious
                R += h.is_removed

            print(env.timestamp, f"Ro: {R0:5.2f} G:{G:5.2f} S:{S} E:{E} I:{I} R:{R}")
            # print(city.tracker.recovered_stats)
            self.data.append({
                    'time': env.timestamp,
                    'susceptible': S,
                    'exposed': E,
                    'infectious':I,
                    'removed':R,
                    'R': R0
                    })
            yield env.timeout(self.f / TICK_MINUTE)

class EventMonitor(BaseMonitor):

    def __init__(self, f=None, dest: str = None, chunk_size: int = None):
        super().__init__(f, dest, chunk_size)
        self._iothread = threading.Thread()
        self._iothread.start()


    def run(self, env, city: City):
        while True:
            self.data = city.events

            if self.chunk_size and len(self.data) > self.chunk_size:
                self.data = city.pull_events()
                self.dump()

            yield env.timeout(self.f / TICK_MINUTE)

    def dump(self):
        if self.dest is None:
            print(json.dumps(self.data, indent=1, default=_json_serialize))
            return

        self._iothread.join()
        self._iothread = threading.Thread(target=EventMonitor.dump_chunk, args=(self.data, self.dest))
        self._iothread.start()

    def join_iothread(self):
        self._iothread.join()

    @staticmethod
    def dump_chunk(data, dest):
        timestamp = datetime.utcnow().timestamp()
        with zipfile.ZipFile(f"{dest}.zip", mode='a', compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(f"{timestamp}.pkl", pickle.dumps(data))

class TimeMonitor(BaseMonitor):

    def run(self, env, city: City):
        while True:
            # print(env.timestamp)
            yield env.timeout(self.f / TICK_MINUTE)


class PlotMonitor(BaseMonitor):

    def run(self, env, city: City):
        fig = plt.figure(figsize=(15, 12))
        while True:
            d = {
                'time': city.clock.time(),
                'htime': city.clock.time_of_day(),
                'sick': sum([int(h.is_sick) for h in city.humans]),
            }
            for k, v in Human.actions.items():
                d[k] = sum(int(h.action == v) for h in city.humans)

            self.data.append(d)
            yield env.timeout(self.f / TICK_MINUTE)
            self.plot()

    def plot(self):
        display.clear_output(wait=True)
        pl.clf()
        time_series = [d['time'] for d in self.data]
        sick_series = [d['sick'] for d in self.data]
        pl.plot(time_series, sick_series, label='sick')
        for k, v in Human.actions.items():
            action_series = [d[k] for d in self.data]
            pl.plot(time_series, action_series, label=k)

        pl.title(f"City at {self.data[-1]['htime']}")
        pl.legend()
        display.display(pl.gcf())


class LatLonMonitor(BaseMonitor):
    def __init__(self, f=None):
        super().__init__(f)
        self.city_data = {}

    def run(self, env, city: City):
        self.city_data['parks'] = [
            {'lat': l.lat,
             'lon': l.lon, } for l in city.parks
        ]
        self.city_data['stores'] = [
            {'lat': l.lat,
             'lon': l.lon, } for l in city.stores
        ]
        fig = plt.figure(figsize=(18, 16))
        while True:
            self.data.extend(
                {'time': city.clock.time_of_day(),
                 'is_sick': h.is_sick,
                 'lat': h.lat(),
                 'lon': h.lon(),
                 'human_id': h.name,
                 'household_id': h.household.name,
                 'location': h.location.name if h.location else None
                 } for h in city.humans
            )
            yield env.timeout(self.f / TICK_MINUTE)
            self.plot()

    def plot(self):
        display.clear_output(wait=True)
        pl.clf()
        # PLOT STORES AND PARKS
        lat_series = [d['lat'] for d in self.city_data['parks']]
        lon_series = [d['lon'] for d in self.city_data['parks']]
        s = 250
        pl.scatter(lat_series, lon_series, s=s, marker='o', color='green', label='parks')

        # PLOT STORES AND PARKS
        lat_series = [d['lat'] for d in self.city_data['stores']]
        lon_series = [d['lon'] for d in self.city_data['stores']]
        s = 50
        pl.scatter(lat_series, lon_series, s=s, marker='o', color='black', label='stores')

        lat_series = [d['lat'] for d in self.data]
        lon_series = [d['lon'] for d in self.data]
        c = ['red' if d['is_sick'] else 'blue' for d in self.data]
        s = 5
        pl.scatter(lat_series, lon_series, s=s, marker='^', color=c, label='human')
        sicks = sum([d['is_sick'] for d in self.data])
        pl.title(f"City at {self.data[-1]['time']} - sick:{sicks}")
        pl.legend()
        display.display(pl.gcf())

class StateMonitor(BaseMonitor):
    def run(self, env, city: City):
        while True:
            d = {
                'time': city.time_of_day(),
                'people': len(city.humans),
                'sick': sum([int(h.is_sick) for h in city.humans]),
            }
            self.data.append(d)
            print(city.clock.time_of_day())
            yield env.timeout(self.f / TICK_MINUTE)

    def dump(self):
        print(json.dumps(self.data, indent=1))
