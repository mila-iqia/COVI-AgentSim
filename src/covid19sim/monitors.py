"""
[summary]
"""
from matplotlib import pyplot as plt
import json
import pylab as pl
import pickle
from datetime import datetime, timedelta
import threading
import zipfile

from covid19sim.utils import _json_serialize
from covid19sim.configs.constants import *
from covid19sim.base import City
from covid19sim.simulator import Human


class BaseMonitor(object):
    """
    [summary]
    """

    def __init__(self, f=None, dest: str = None, chunk_size: int = None):
        """
        [summary]

        Args:
            f ([type], optional): [description]. Defaults to None.
            dest (str, optional): [description]. Defaults to None.
            chunk_size (int, optional): [description]. Defaults to None.
        """
        self.data = []
        self.f = f or SECONDS_PER_HOUR
        self.dest = dest
        self.chunk_size = chunk_size if self.dest and chunk_size else 0

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def dump(self):
        """
        [summary]
        """
        pass


class SEIRMonitor(BaseMonitor):
    """
    [summary]
    """

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Yields:
            [type]: [description]
        """
        n_days = 0
        while True:
            S, E, I, R = 0, 0, 0, 0
            R0 = city.tracker.get_R()
            G = city.tracker.get_generation_time()
            P = sum(city.tracker.cases_positive_per_day)
            H = sum(city.tracker.hospitalization_per_day)
            C = sum(city.tracker.critical_per_day)
            Projected3 = min(1.0*city.tracker.n_infected_init * 2 ** (n_days/3), len(city.humans))
            Projected5 = min(1.0*city.tracker.n_infected_init * 2 ** (n_days/5), len(city.humans))
            Projected10 = min(1.0*city.tracker.n_infected_init * 2 ** (n_days/10), len(city.humans))
            EM = city.tracker.expected_mobility[-1]
            F = city.tracker.feelings[-1]
            prec, _, _ = city.tracker.risk_precision_daily[-1]
            green, blue, orange, red = city.tracker.recommended_levels_daily[-1]

            S = city.tracker.s_per_day[-1]
            E = city.tracker.e_per_day[-1]
            I = city.tracker.i_per_day[-1]
            R = city.tracker.r_per_day[-1]
            T = E + I + R
            # print(np.mean([h.risk for h in city.humans]))
            # print(env.timestamp, f"Ro: {R0:5.2f} G:{G:5.2f} S:{S} E:{E} I:{I} R:{R} T:{T} P3:{Projected3:5.2f} M:{M:5.2f} +Test:{P} H:{H} C:{C} RiskP:{RiskP:3.2f}") RiskP:{RiskP:3.2f}
            print(env.timestamp, f"Ro: {R0:2.2f} S:{S} E:{E} I:{I} T:{T} P3:{Projected3:5.2f} RiskP:{prec[1][0]:3.2f} F:{F:3.2f} EM:{EM:3.2f} G:{green} B:{blue} O:{orange} R:{red} ")
            # print(city.tracker.recovered_stats)
            self.data.append({
                    'time': env.timestamp,
                    'susceptible': S,
                    'exposed': E,
                    'infectious':I,
                    'removed':R,
                    'R': R0
                    })
            yield env.timeout(self.f)
            n_days += 1

class EventMonitor(BaseMonitor):
    """
    [summary]
    """

    def __init__(self, f=None, dest: str = None, chunk_size: int = None):
        """
        [summary]

        Args:
            f ([type], optional): [description]. Defaults to None.
            dest (str, optional): [description]. Defaults to None.
            chunk_size (int, optional): [description]. Defaults to None.
        """
        super().__init__(f, dest, chunk_size)
        self._iothread = threading.Thread()
        self._iothread.start()

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Yields:
            [type]: [description]
        """
        while True:
            # Keep the last 2 days to make sure all events are sent to the
            # inference server before getting dumped
            self.data = city.events
            if self.chunk_size and len(city.events_slice(datetime.min,
                                                         env.timestamp - timedelta(days=2))) > self.chunk_size:
                self.data = city.pull_events_slice(env.timestamp - timedelta(days=2))
                self.dump()

            yield env.timeout(self.f)

    def dump(self):
        """
        [summary]
        """
        if self.dest is None:
            print(json.dumps(self.data, indent=1, default=_json_serialize))
            return

        self._iothread.join()
        self._iothread = threading.Thread(target=EventMonitor.dump_chunk, args=(self.data, self.dest))
        self._iothread.start()

    def join_iothread(self):
        """
        [summary]
        """
        self._iothread.join()

    @staticmethod
    def dump_chunk(data, dest):
        """
        [summary]

        Args:
            data ([type]): [description]
            dest ([type]): [description]
        """
        timestamp = datetime.utcnow().timestamp()
        with zipfile.ZipFile(f"{dest}.zip", mode='a', compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(f"{timestamp}.pkl", pickle.dumps(data))

class TimeMonitor(BaseMonitor):
    """
    [summary]
    """

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Yields:
            [type]: [description]
        """
        while True:
            # print(env.timestamp)
            yield env.timeout(self.f)


class PlotMonitor(BaseMonitor):
    """
    [summary]
    """

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Yields:
            [type]: [description]
        """
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
            yield env.timeout(self.f)
            self.plot()

    def plot(self):
        """
        [summary]
        """
        pl.clf()
        time_series = [d['time'] for d in self.data]
        sick_series = [d['sick'] for d in self.data]
        pl.plot(time_series, sick_series, label='sick')
        for k, v in Human.actions.items():
            action_series = [d[k] for d in self.data]
            pl.plot(time_series, action_series, label=k)

        pl.title(f"City at {self.data[-1]['htime']}")
        pl.legend()


class LatLonMonitor(BaseMonitor):
    """
    [summary]
    """

    def __init__(self, f=None):
        """
        [summary]

        Args:
            f ([type], optional): [description]. Defaults to None.
        """
        super().__init__(f)
        self.city_data = {}

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Yields:
            [type]: [description]
        """
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
            yield env.timeout(self.f)
            self.plot()

    def plot(self):
        """
        [summary]
        """
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

class StateMonitor(BaseMonitor):
    """
    [summary]
    """

    def run(self, env, city: City):
        """
        [summary]

        Args:
            env ([type]): [description]
            city (City): [description]

        Yields:
            [type]: [description]
        """
        while True:
            d = {
                'time': city.time_of_day(),
                'people': len(city.humans),
                'sick': sum([int(h.is_sick) for h in city.humans]),
            }
            self.data.append(d)
            print(city.clock.time_of_day())
            yield env.timeout(self.f)

    def dump(self):
        """
        [summary]
        """
        print(json.dumps(self.data, indent=1))
