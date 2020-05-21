"""
Contains utility classes for remote inference inside the simulation.
"""

import datetime
import functools
import numpy as np
import os
import pickle
import threading
import time
import typing
import warnings
import zmq

from ctt.inference.infer import InferenceEngine
from ctt.data_loading.loader import InvalidSetSize

import covid19sim.frozen.clustering.base
import covid19sim.frozen.clusters
import covid19sim.frozen.helper
import covid19sim.frozen.utils
import covid19sim.utils


expected_raw_packet_param_names = [
    "start", "current_day", "human", "log_path", "time_slot", "conf"
]
expected_processed_packet_param_names = [
    "current_day", "observed", "unobserved"
]

default_poll_delay_ms = 500
default_frontend_ipc_address = "ipc:///tmp/covid19sim-inference-frontend.ipc"
default_backend_ipc_address = "ipc:///tmp/covid19sim-inference-backend.ipc"


class AtomicCounter(object):
    """
    Implements an atomic & thread-safe counter.
    """

    def __init__(self, init=0):
        self._count = init
        self._lock = threading.Lock()

    def increment(self, delta=1):
        with self._lock:
            self._count += delta
            return self._count

    @property
    def count(self):
        return self._count


class InferenceWorker(threading.Thread):
    """
    Spawns a single inference worker instance.

    These workers are managed by the InferenceBroker class. They
    communicate with the broker using a backend connection.
    """

    def __init__(
            self,
            experiment_directory: typing.AnyStr,
            identifier: typing.Any,
            mp_backend: typing.AnyStr,
            mp_threads: int,
            weights_path: typing.Optional[typing.AnyStr] = None,
            context: typing.Optional[zmq.Context] = None,
    ):
        """
        Initializes the inference worker's attributes (counters, condvars, context).

        Args:
            experiment_directory: the path to the experiment directory to pass to the inference engine.
            identifier: identifier for this worker (name, used for debug purposes only).
            mp_backend: joblib parallel backend to use when processing humans in parallel.
            mp_threads: joblib parallel thread count to use when processing humans in parallel.
            weights_path: the path to the specific weight file to use. If not, will use the 'best
                checkpoint weights' inside the experiment directory.
            context: zmq context to create i/o objects from.
        """
        threading.Thread.__init__(self)
        self.experiment_directory = experiment_directory
        self.weights_path = weights_path
        self.identifier = identifier
        self.stop_flag = threading.Event()
        self.packet_counter = AtomicCounter(init=0)
        self.time_counter = AtomicCounter(init=0.)
        self.mp_backend = mp_backend
        self.mp_threads = mp_threads
        if context is None:
            context = zmq.Context()
        self.context = context
        self.init_time = None

    def run(self):
        """Main loop of the inference worker thread.

        Will receive brokered requests from the frontend, process them, and respond
        with the result through the broker.
        """
        engine = InferenceEngineWrapper(self.experiment_directory, self.weights_path)
        socket = self.context.socket(zmq.REQ)
        socket.identity = str(self.identifier).encode("ascii")
        socket.connect(default_backend_ipc_address)
        socket.send(b"READY")  # tell broker we're ready
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self.init_time = time.time()
        while not self.stop_flag.is_set():
            evts = dict(poller.poll(default_poll_delay_ms))
            if socket in evts and evts[socket] == zmq.POLLIN:
                proc_start_time = time.time()
                address, empty, buffer = socket.recv_multipart()
                sample = pickle.loads(buffer)
                response = proc_human_batch(sample, engine, self.mp_backend, self.mp_threads)
                response = pickle.dumps(response)
                socket.send_multipart([address, b"", response])
                self.time_counter.increment(time.time() - proc_start_time)
                self.packet_counter.increment()
        socket.close()

    def get_processed_count(self):
        """Returns the total number of processed requests by this inference server."""
        return self.packet_counter.count

    def get_averge_processing_delay(self):
        """Returns the average sample processing time between reception & response (in seconds)."""
        tot_delay, tot_packet_count = self.time_counter.count, self.packet_counter.count
        if not tot_packet_count:
            return float("nan")
        return tot_delay / self.packet_counter.count

    def get_processing_uptime(self):
        """Returns the fraction of total uptime that the server spends processing requests."""
        tot_process_time, tot_time = self.time_counter.count, time.time() - self.init_time
        return tot_process_time / tot_time

    def stop(self):
        """Stops the infinite data reception loop, allowing a clean shutdown."""
        self.stop_flag.set()


class InferenceBroker(threading.Thread):
    """Manages inference workers through a backend connection for load balancing."""

    def __init__(
            self,
            model_exp_path: typing.AnyStr,
            workers: int,
            mp_backend: typing.AnyStr,
            mp_threads: int,
            port: typing.Optional[int] = None,
            verbose: bool = False,
            verbose_print_delay: float = 5.,
            weights_path: typing.Optional[typing.AnyStr] = None,
            context: typing.Optional[zmq.Context] = None,
    ):
        """
        Initializes the inference broker's attributes (counters, condvars, context).

        Args:
            model_exp_path: the path to the experiment directory to pass to the inference engine.
            workers: the number of independent inference workers to spawn to process requests.
            mp_backend: joblib parallel backend to use when processing humans in parallel.
            mp_threads: joblib parallel thread count to use when processing humans in parallel.
            port: the port number to accept TCP requests on. If None, will accept IPC requests instead.
            verbose: toggles whether to print extra debug information while running.
            verbose_print_delay: specifies how often the extra debug info should be printed.
            context: zmq context to create i/o objects from.
        """
        threading.Thread.__init__(self)
        if context is None:
            context = zmq.Context()
        self.context = context
        self.workers = workers
        self.mp_backend = mp_backend
        self.mp_threads = mp_threads
        self.port = port
        self.model_exp_path = model_exp_path
        self.weights_path = weights_path
        self.stop_flag = threading.Event()
        self.verbose = verbose
        self.verbose_print_delay = verbose_print_delay

    def run(self):
        """Main loop of the inference broker thread.

        Will received requests from clients and dispatch them to available workers.
        """
        print(f"Initializing {self.workers} worker(s) from experiment: {self.model_exp_path}")
        if self.weights_path is not None:
            print(f"\t will use weights directly from: {self.weights_path}")
        frontend = self.context.socket(zmq.ROUTER)
        if self.port:
            frontend_address = f"tcp://*:{self.port}"
        else:
            frontend_address = default_frontend_ipc_address
        print(f"Will listen for inference requests at: {frontend_address}")
        frontend.bind(frontend_address)
        backend = self.context.socket(zmq.ROUTER)
        backend.bind(default_backend_ipc_address)
        worker_map = {}
        for worker_idx in range(self.workers):
            worker_id = f"worker:{worker_idx}"
            worker = InferenceWorker(
                self.model_exp_path,
                worker_id,
                self.mp_backend,
                self.mp_threads,
                weights_path=self.weights_path,
                context=self.context
            )
            worker_map[worker_id] = worker
            worker.start()
        available_worker_ids = []
        worker_poller = zmq.Poller()
        worker_poller.register(backend, zmq.POLLIN)
        worker_poller.register(frontend, zmq.POLLIN)
        last_update_timestamp = time.time()
        while not self.stop_flag.is_set():
            evts = dict(worker_poller.poll(default_poll_delay_ms))
            if backend in evts and evts[backend] == zmq.POLLIN:
                request = backend.recv_multipart()
                worker_id, empty, client = request[:3]
                available_worker_ids.append(worker_id)
                if client != b"READY" and len(request) > 3:
                    empty, reply = request[3:]
                    frontend.send_multipart([client, b"", reply])
            if available_worker_ids and frontend in evts and evts[frontend] == zmq.POLLIN:
                client, empty, request = frontend.recv_multipart()
                worker_id = available_worker_ids.pop(0)
                backend.send_multipart([worker_id, b"", client, b"", request])
            if self.verbose and time.time() - last_update_timestamp > self.verbose_print_delay:
                print(f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} stats:")
                for worker_id, worker in worker_map.items():
                    packets = worker.get_processed_count()
                    delay = worker.get_averge_processing_delay()
                    uptime = worker.get_processing_uptime()
                    print(f"  {worker_id}:  packets={packets}"
                          f"  avg_delay={delay:.6f}sec  proc_time_ratio={uptime:.1%}")
                last_update_timestamp = time.time()
        for w in worker_map.values():
            w.stop()
            w.join()

    def stop(self):
        """
        Stops the infinite data reception loop, allowing a clean shutdown.
        """
        self.stop_flag.set()


class InferenceClient:
    """
    Creates a client through which data samples can be sent for inference.

    This object will automatically be able to pick a proper remote inference
    engine. This object should be fairly lightweight and low-cost, so creating
    it once per day, per human *should* not create a significant overhead.
    """

    def __init__(
            self,
            server_address: typing.Optional[typing.AnyStr] = default_frontend_ipc_address,
            context: typing.Optional[zmq.Context] = None,
    ):
        """
        Initializes the client's attributes (socket, context).

        Args:
            server_address: address of the inference server frontend to send requests to.
            context: zmq context to create i/o objects from.
        """
        if context is None:
            context = zmq.Context()
        self.context = context
        self.socket = self.context.socket(zmq.REQ)
        if server_address is None:
            server_address = default_frontend_ipc_address
        self.socket.connect(server_address)

    def infer(self, sample):
        """Forwards a data sample for the inference engine using pickle."""
        self.socket.send_pyobj(sample)
        return self.socket.recv_pyobj()


class InferenceEngineWrapper(InferenceEngine):
    """Inference engine wrapper used to download & extract experiment data, if necessary."""

    def __init__(self, experiment_directory, *args, **kwargs):
        if experiment_directory.startswith("http"):
            assert os.path.isdir("/tmp"), "don't know where to download data to..."
            experiment_root_directory = \
                covid19sim.utils.download_exp_data_if_not_exist(experiment_directory, "/tmp")
            experiment_subdirectories = \
                [os.path.join(experiment_root_directory, p) for p in os.listdir(experiment_root_directory)
                 if os.path.isdir(os.path.join(experiment_root_directory, p))]
            assert len(experiment_subdirectories) == 1, "should only have one dir per experiment zip"
            experiment_directory = experiment_subdirectories[0]
        super().__init__(experiment_directory, *args, **kwargs)


def proc_human_batch(
        sample,
        engine,
        mp_backend,
        mp_threads,
):
    """
    Processes a chunk of human data, clustering messages and computing new risk levels.

    Args:
        sample: a dictionary of data necessary for clustering+inference.
        engine: the inference engine, pre-instantiated with the right experiment config.
        mp_backend: joblib parallel backend to use when processing humans in parallel.
        mp_threads: joblib parallel thread count to use when processing humans in parallel.

    Returns:
        The clustering + risk level update results.
    """
    if isinstance(sample, list):
        if mp_threads > 0:
            import joblib
            with joblib.Parallel(
                    n_jobs=mp_threads,
                    backend=mp_backend,
                    batch_size="auto",
                    prefer="threads") as parallel:
                results = parallel((joblib.delayed(_proc_human)(human, engine) for human in sample))
            return [(h.name, risk_history, h.clusters) for h, risk_history in results]
        else:
            return [proc_human_batch(human, engine, mp_backend, mp_threads) for human in sample]
    else:
        assert isinstance(sample, dict), "unexpected input data format"
        result_human, risk_history = _proc_human(sample, engine)
        return result_human.name, risk_history, result_human.clusters


def _proc_human(params, inference_engine):
    """Internal implementation of the `proc_human_batch` function."""
    assert isinstance(params, dict) and \
        all([p in params for p in expected_raw_packet_param_names]), \
        "unexpected/broken _proc_human input format between simulator and inference service"
    conf = params["conf"]

    # Cluster Messages
    cluster_algo_type = conf.get("CLUSTER_ALGO_TYPE")
    human = params["human"]
    if cluster_algo_type == "old":
        human.clusters.add_messages(human.messages)
        human.clusters.update_records(human.update_messages)
        human.clusters.purge(params["current_day"])
    else:
        if not isinstance(human.clusters, covid19sim.frozen.clustering.base.ClusterManagerBase):
            clustering_type = \
                covid19sim.frozen.clustering.base.get_cluster_manager_type(cluster_algo_type)
            if cluster_algo_type == "naive":
                clustering_type = functools.partial(clustering_type, ticks_per_uid_roll=1)
            # note: we create the manager to use day-level timestamps only
            human.clusters = clustering_type(
                max_history_ticks_offset=conf.get("TRACING_N_DAYS_HISTORY"),
                # note: the simulator should be able to match all update messages to encounters, but
                # since the time slot update voodoo, I (PLSC) have not been able to make no-adopt
                # version of the naive implementation work (both with and without batching)
                add_orphan_updates_as_clusters=True,
                generate_embeddings_by_timestamp=True,
                generate_backw_compat_embeddings=True,
            )
        # set the current day as the refresh timestamp to auto-purge outdated messages in advance
        human.clusters.set_current_timestamp(params["current_day"])
        encounter_messages = covid19sim.frozen.utils.convert_messages_to_batched_new_format(
            human.messages, human.exposure_message)
        update_messages = covid19sim.frozen.utils.convert_messages_to_batched_new_format(
            human.update_messages)
        earliest_new_encounter_message = encounter_messages[0][0] if len(encounter_messages) else None
        if earliest_new_encounter_message is not None:
            # quick verification: even with a 1-day buffer for timeslot craziness, we should not be
            # getting old encounters here; the simulator should not keep & transfer those every call
            assert earliest_new_encounter_message.encounter_time + 1 >= params["current_day"]
        human.clusters.add_messages(
            messages=[*encounter_messages, *update_messages],
            current_timestamp=params["current_day"],
        )

    # Format for supervised learning / transformer inference
    todays_date = params["start"] + datetime.timedelta(days=params["current_day"], hours=params["time_slot"])
    is_exposed, exposure_day = covid19sim.frozen.helper.exposure_array(human.infection_timestamp, todays_date, conf)
    is_recovered, recovery_day = covid19sim.frozen.helper.recovered_array(human.recovered_timestamp, todays_date, conf)
    candidate_encounters, exposure_encounter = covid19sim.frozen.helper.candidate_exposures(human, todays_date)
    reported_symptoms = human.rolling_all_reported_symptoms
    true_symptoms = human.rolling_all_symptoms

    daily_output = {
        "current_day": params["current_day"],
        "observed": {
            "reported_symptoms": reported_symptoms,
            "candidate_encounters": candidate_encounters,
            "test_results": covid19sim.frozen.helper.get_test_result_array(human.test_results, todays_date, conf),
            "preexisting_conditions": human.obs_preexisting_conditions,
            "age": human.obs_age,
            "sex": human.obs_sex,
            "risk_mapping": conf.get("RISK_MAPPING"),
        },
        "unobserved": {
            "true_symptoms": true_symptoms,
            "is_exposed": is_exposed,
            "exposure_encounter": exposure_encounter,
            "exposure_day": exposure_day,
            "is_recovered": is_recovered,
            "recovery_day": recovery_day,
            "infectiousness": np.array(human.infectiousnesses),
            "true_preexisting_conditions": human.preexisting_conditions,
            "true_age": human.age,
            "true_sex": human.sex
        }
    }

    if conf.get("COLLECT_TRAINING_DATA"):
        os.makedirs(params["log_path"], exist_ok=True)
        with open(os.path.join(params["log_path"], f"daily_human-{params['time_slot']}.pkl"), 'wb') as fd:
            pickle.dump(daily_output, fd)

    if conf.get("USE_ORACLE"):
        # return ground truth infectiousnesses
        human['risk_history'] = human['infectiousnesses']
    else:
        # estimate infectiousnesses using CTT inference engine
        inference_result = None
        if conf.get("RISK_MODEL") == "transformer":
            try:
                inference_result = inference_engine.infer(daily_output)
            except InvalidSetSize:
                pass  # return None for invalid samples
            except RuntimeError as error:
                # TODO: ctt.modules.HealthHistoryEmbedding can fail with :
                #  size mismatch, m1: [14 x 29], m2: [13 x 128]
                warnings.warn(str(error), RuntimeWarning)
        risk_history = None
        if inference_result is not None:
            risk_history = inference_result['infectiousness']

    # clear all messages for next time we update
    human.messages = []
    human.update_messages = []
    return human, risk_history
