"""
Contains classes for regular saving relevant logs to the disk or printing output to the console at regular intervals.
"""
import json
import pickle
import threading
import time
import zipfile
import numpy as np
from datetime import datetime, timedelta

from covid19sim.locations.city import City
from covid19sim.utils.utils import _json_serialize
from covid19sim.utils.constants import SECONDS_PER_HOUR, SECONDS_PER_DAY
from covid19sim.utils.utils import log

class SimulationMonitor(object):
    """
    Logs information at regular intervals to the console as well as to the disk.
    Saves intermediate state of tracker at regular intervals.

    Args:
        frequency (float): regular simulation-intervals at which the information needs to be printed. Defaults to 1 simulation day.
        logfile (str): filepath where the console output and final tracked metrics will be logged. Prints to the console only if None.
        conf (dict): yaml configuration of the experiment
    """

    def __init__(self, frequency=SECONDS_PER_DAY, logfile=None, conf={}):
        self.frequency = frequency
        self.logfile = logfile
        self.conf = conf
        self.legend = """
#################### SIMULATION PROGRESS ##################
Legend -
* [ +Test ]: Total positive test results observed this day (Note: test results are available after some delay from the test time) / total tests administered on this day
* [ P3 ]: Projected number of cases (E+I+R) if the cases were to grow with a doubling rate of 3 days.
* [ TestQueue ]: Total number of people present in the test queue at the time of this print out.
* [ H/C/D ]: Total number of people in hospital (H)/ ICU (C) at this point in simulation-time. Total died upto this day (D).
* [ MC ]: Mean number of known connections of a person in the population (average degree of the social network). The attributes for known connections are drawn from surveyed data on mean contacts.
* [ Q ]: Number of people quarantined as of midnight on that day.
* [ 2x ]: Number of days to double the initial infections to the current level.
        """
        if self.conf['INTERVENTION_DAY'] >= 0 and self.conf['RISK_MODEL'] is not None:
            self.legend += """
* [ G/B/O/R ]: Number of people in each of the 4 recommendation levels - Green, Blue, Orange, and Red.
* [ RiskP ]: Top 1% risk precision of the risk predictor computed for people with no test.
            """

        self.print_legend = True

    def run(self, env, city: City):
        """
        Infinite loop yields events at regular intervals. It logs and saves information to `self.logfile`.

        Args:
            env (): [description]
            city (City): [description]

        Yields:
            simpy.Environment.Timeout: Event that resumes after some specified duration.
        """
        process_start = time.time()
        n_days = 0
        while True:
            assert city.tracker.fully_initialized

            if self.print_legend:
                log(self.legend, self.logfile)
                self.print_legend = False

            # social network & mobility
            average_degree = np.mean([len(h.known_connections) for h in city.humans])
            mobility = f"| MC: {average_degree: 3.3f}"

            # simulation time related
            env_time = str(env.timestamp).split()[0]
            day = "Day {:2}:".format((city.env.timestamp - city.start_time).days)
            proc_time = "{:8}".format("({}s)".format(int(time.time() - process_start)))

            if not city.tracker.start_tracking:
                str_to_print = f"{proc_time} {day} {env_time} {mobility}"
                log(str_to_print, self.logfile)
                yield env.timeout(self.frequency)
                continue

            # test stats
            t_P = city.tracker.test_results_per_day[env.timestamp.date()]['positive']
            t_T = 0 if len(city.tracker.tested_per_day) < 2 else city.tracker.tested_per_day[-2]
            test_queue_length = len(city.covid_testing_facility.test_queue)

            # hospitalization stats
            H = sum(len(hospital.humans) for hospital in city.hospitals)
            C = sum(len(hospital.icu.humans) for hospital in city.hospitals)
            D = sum(city.tracker.deaths_per_day)

            # SEIR stats
            S = city.tracker.s_per_day[-1]
            E = city.tracker.e_per_day[-1]
            I = city.tracker.i_per_day[-1]
            R = city.tracker.r_per_day[-1]
            T = E + I + R

            # projections
            Projected3 = min(1.0*city.tracker.n_infected_init * 2 ** (n_days/3), len(city.humans))
            doubling_rate_days =  n_days / np.log2(1.0 * T / (city.tracker.n_infected_init + 1e-6))

            # other diseases
            cold = sum(h.has_cold for h in city.humans)
            allergies = sum(h.has_allergy_symptoms for h in city.humans)

            # intervention related
            n_quarantine = sum(h.intervened_behavior.quarantine_timestamp is not None for h in city.humans)

            # prepare string
            nd = str(len(str(city.n_people)))
            SEIR = f"| S:{S:<{nd}} E:{E:<{nd}} I:{I:<{nd}} E+I+R:{T:<{nd}} +Test:{t_P}/{t_T} TestQueue:{test_queue_length}"
            stats = f"| P3:{Projected3:5.2f}"
            stats += f" 2x:{doubling_rate_days: 2.2f}" if doubling_rate > 0 else ""
            other_diseases = f"| cold:{cold} allergies:{allergies}"
            hospitalizations = f"| H:{H} C:{C} D:{D}"
            quarantines = f"| Q: {n_quarantine}"

            str_to_print = f"{proc_time} {day} {env_time} {SEIR} {stats} {other_diseases} {hospitalizations} {mobility} {quarantines}"
            # conditional prints
            colors, risk = "", ""
            if self.conf['INTERVENTION_DAY'] >= 0 and self.conf['RISK_MODEL'] != "":
                # on day 1, if tracker is not informed about tracing, recommended levels daily are not appended.
                # this will throw an error about list out of index.
                green, blue, orange, red = 0, 0, 0, 0
                if len(city.tracker.recommended_levels_daily) > 0:
                    green, blue, orange, red = city.tracker.recommended_levels_daily[-1]
                colors = f"| G:{green} B:{blue} O:{orange} R:{red}"

                prec, _, _ = city.tracker.risk_precision_daily[-1]
                risk = f"| RiskP:{prec[1][0]:3.2f}"

                str_to_print = f"{str_to_print} {colors} {risk}"

            log(str_to_print, self.logfile)
            yield env.timeout(self.frequency)
            n_days += 1

class EventMonitor(object):
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
