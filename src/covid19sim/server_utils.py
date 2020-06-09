"""
Contains utility classes for remote inference inside the simulation.
"""

import datetime
import joblib
import multiprocessing
import multiprocessing.managers
import numpy as np
import os
import pickle
import sys
import time
import typing
import zmq
from pathlib import Path
from ctt.inference.infer import InferenceEngine

import covid19sim.frozen.clustering.blind
import covid19sim.frozen.clustering.gaen
import covid19sim.frozen.message_utils
import covid19sim.frozen.helper
import covid19sim.utils


expected_raw_packet_param_names = [
    "start", "current_day", "human", "log_path", "time_slot", "conf"
]
expected_processed_packet_param_names = [
    "current_day", "observed", "unobserved"
]

default_poll_delay_ms = 500

# if on slurm
if os.path.isdir("/Tmp"):
    frontend_path = Path("/Tmp/slurm.{}.0".format(os.environ.get("SLURM_JOB_ID")))
    backend_path = Path("/Tmp/slurm.{}.0".format(os.environ.get("SLURM_JOB_ID")))
else:
    frontend_path = "/tmp"
    backend_path = "/tmp"

default_frontend_ipc_address = "ipc://" + os.path.join(frontend_path, "covid19sim-inference-frontend.ipc")
default_backend_ipc_address = "ipc://" + os.path.join(backend_path, "covid19sim-inference-backend.ipc")


class InferenceWorker(multiprocessing.Process):
    """
    Spawns a single inference worker instance.

    These workers are managed by the InferenceBroker class. They
    communicate with the broker using a backend connection.
    """

    def __init__(
            self,
            experiment_directory: typing.AnyStr,
            backend_address: typing.AnyStr,
            identifier: typing.Any,
            cluster_mgr_map: typing.Dict,
            weights_path: typing.Optional[typing.AnyStr] = None,
    ):
        """
        Initializes the inference worker's attributes (counters, condvars, ...).

        Args:
            experiment_directory: the path to the experiment directory to pass to the inference engine.
            identifier: identifier for this worker (name, used for debug purposes only).
            n_parallel_procs: joblib parallel process count to use for clustering+inference.
            cluster_mgr_map: map of human-to-cluster-managers to use for clustering.
            weights_path: the path to the specific weight file to use. If not, will use the 'best
                checkpoint weights' inside the experiment directory.
        """
        super().__init__()
        self.experiment_directory = experiment_directory
        self.backend_address = backend_address
        self.weights_path = weights_path
        self.identifier = identifier
        self.stop_flag = multiprocessing.Event()
        self.reset_flag = multiprocessing.Event()
        self.running_flag = multiprocessing.Value("i", 0)
        self.packet_counter = multiprocessing.Value("i", 0)
        self.time_counter = multiprocessing.Value("f", 0.0)
        self.time_init = multiprocessing.Value("f", 0.0)
        self.cluster_mgr_map = cluster_mgr_map

    def run(self):
        """Main loop of the inference worker process.

        Will receive brokered requests from the frontend, process them, and respond
        with the result through the broker.
        """
        engine = InferenceEngineWrapper(self.experiment_directory, self.weights_path)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = self.identifier.encode()
        print(f"{self.identifier} contacting broker via: {self.backend_address}", flush=True)
        socket.connect(self.backend_address)
        socket.send(b"READY")  # tell broker we're ready
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self.time_init.value = time.time()
        self.time_counter.value = 0.0
        self.packet_counter.value = 0
        self.running_flag.value = 1
        while not self.stop_flag.is_set():
            if self.reset_flag.is_set():
                self.time_counter.value = 0.0
                self.packet_counter.value = 0
                self.time_init.value = 0.0
                self.reset_flag.clear()
            evts = dict(poller.poll(default_poll_delay_ms))
            if socket in evts and evts[socket] == zmq.POLLIN:
                proc_start_time = time.time()
                address, empty, buffer = socket.recv_multipart()
                sample = pickle.loads(buffer)
                response = proc_human_batch(
                    sample=sample,
                    engine=engine,
                    cluster_mgr_map=self.cluster_mgr_map,
                )
                response = pickle.dumps(response)
                socket.send_multipart([address, b"", response])
                with self.time_counter.get_lock():
                    self.time_counter.value += time.time() - proc_start_time
                with self.packet_counter.get_lock():
                    self.packet_counter.value += 1
        self.running_flag.value = 0
        socket.close()

    def get_processed_count(self):
        """Returns the total number of processed requests by this inference worker."""
        return int(self.packet_counter.value)

    def get_total_delay(self):
        """Returns the total time spent processing requests by this inference worker."""
        return float(self.time_counter.value)

    def get_uptime(self):
        """Returns the total uptime of this inference worker."""
        return time.time() - float(self.time_init.value)

    def is_running(self):
        """Returns whether this inference worker is running or not."""
        return bool(self.running_flag.value)

    def get_averge_processing_delay(self):
        """Returns the average sample processing time between reception & response (in seconds)."""
        tot_delay, tot_packet_count = self.get_total_delay(), self.get_processed_count()
        if not tot_packet_count:
            return float("nan")
        return tot_delay / tot_packet_count

    def get_processing_uptime(self):
        """Returns the fraction of total uptime that the server spends processing requests."""
        tot_process_time, tot_time = self.get_total_delay(), self.get_uptime()
        return tot_process_time / tot_time

    def stop(self):
        """Stops the infinite data reception loop, allowing a clean shutdown."""
        self.stop_flag.set()


class InferenceBroker:
    """Manages inference workers through a backend connection for load balancing."""

    def __init__(
            self,
            model_exp_path: typing.AnyStr,
            workers: int,
            frontend_address: typing.AnyStr = default_frontend_ipc_address,
            backend_address: typing.AnyStr = default_backend_ipc_address,
            verbose: bool = False,
            verbose_print_delay: float = 5.,
            weights_path: typing.Optional[typing.AnyStr] = None,
    ):
        """
        Initializes the inference broker's attributes (counters, condvars, ...).

        Args:
            model_exp_path: the path to the experiment directory to pass to the inference engine.
            workers: the number of independent inference workers to spawn to process requests.
            n_parallel_procs: joblib parallel process count to use for clustering+inference.
            port: the port number to accept TCP requests on. If None, will accept IPC requests instead.
            verbose: toggles whether to print extra debug information while running.
            verbose_print_delay: specifies how often the extra debug info should be printed.
        """
        self.workers = workers
        self.frontend_address = frontend_address
        self.backend_address = backend_address
        assert frontend_address != backend_address
        self.model_exp_path = model_exp_path
        self.weights_path = weights_path
        self.stop_flag = multiprocessing.Event()
        self.verbose = verbose
        self.verbose_print_delay = verbose_print_delay

    def run(self):
        """Main loop of the inference broker process.

        Will received requests from clients and dispatch them to available workers.
        """
        print(f"Initializing {self.workers} worker(s) from experiment: {self.model_exp_path}", flush=True)
        if self.weights_path is not None:
            print(f"\t will use weights directly from: {self.weights_path}", flush=True)
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        print(f"Will listen for inference requests at: {self.frontend_address}", flush=True)
        frontend.bind(self.frontend_address)
        backend = context.socket(zmq.ROUTER)
        print(f"Will dispatch inference work at: {self.backend_address}", flush=True)
        backend.bind(self.backend_address)
        worker_backend_address = self.backend_address.replace("*", "localhost")
        worker_poller = zmq.Poller()
        worker_poller.register(backend, zmq.POLLIN)
        worker_poller.register(frontend, zmq.POLLIN)
        with multiprocessing.Manager() as mem_manager:
            worker_map = {}
            cluster_mgr_map = mem_manager.dict()
            available_worker_ids = []
            for worker_idx in range(self.workers):
                worker_id = f"worker:{worker_idx}"
                print(f"Launching {worker_id}...", flush=True)
                worker = InferenceWorker(
                    experiment_directory=self.model_exp_path,
                    backend_address=worker_backend_address,
                    identifier=worker_id,
                    cluster_mgr_map=cluster_mgr_map,
                    weights_path=self.weights_path,
                )
                worker_map[worker_id] = worker
                worker.start()
                request = backend.recv_multipart()
                worker_id, empty, response = request[:3]
                assert worker_id == worker.identifier.encode() and response == b"READY"
                available_worker_ids.append(worker.identifier.encode())
            last_update_timestamp = time.time()
            print("Entering dispatch loop...", flush=True)
            while not self.stop_flag.is_set():
                evts = dict(worker_poller.poll(default_poll_delay_ms))
                if backend in evts and evts[backend] == zmq.POLLIN:
                    request = backend.recv_multipart()
                    worker_id, empty, client = request[:3]
                    assert worker_id not in available_worker_ids, \
                        f"got unexpected stuff from {worker_id}: {request}"
                    available_worker_ids.append(worker_id)
                    empty, reply = request[3:]
                    frontend.send_multipart([client, b"", reply])
                if available_worker_ids and frontend in evts and evts[frontend] == zmq.POLLIN:
                    client, empty, request = frontend.recv_multipart()
                    if request == b"RESET":
                        print("got reset request, will clear all clusters", flush=True)
                        assert len(available_worker_ids) == self.workers
                        for k in list(cluster_mgr_map.keys()):
                            del cluster_mgr_map[k]
                        for worker in worker_map.values():
                            worker.reset_flag.set()
                        frontend.send_multipart([client, b"", b"READY"])
                    else:
                        worker_id = available_worker_ids.pop(0)
                        backend.send_multipart([worker_id, b"", client, b"", request])
                if self.verbose and time.time() - last_update_timestamp > self.verbose_print_delay:
                    print(f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} stats:")
                    for worker_id, worker in worker_map.items():
                        packets = worker.get_processed_count()
                        delay = worker.get_averge_processing_delay()
                        uptime = worker.get_processing_uptime()
                        print(
                            f"  {worker_id}:"
                            f"  running={worker.is_running()}"
                            f"  packets={packets}"
                            f"  avg_delay={delay:.6f}sec"
                            f"  proc_time_ratio={uptime:.1%}"
                            f"  nb_clusters={len(worker.cluster_mgr_map)}"
                        )
                    sys.stdout.flush()
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

    def request_reset(self):
        self.socket.send(b"RESET")
        response = self.socket.recv()
        assert response == b"READY"


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
        cluster_mgr_map,
        clusters_dump_path: typing.Optional[typing.AnyStr] = None,
):
    """
    Processes a chunk of human data, clustering messages and computing new risk levels.

    Args:
        sample: a dictionary of data necessary for clustering+inference.
        engine: the inference engine, pre-instantiated with the right experiment config.
        cluster_mgr_map: map of human-to-cluster-managers to use for clustering.
        n_parallel_procs: internal joblib parallel process count for clustering+inference.
        clusters_dump_path: defines where to dump clusters (if required).

    Returns:
        The clustering + risk level update results.
    """
    assert isinstance(sample, list) and all([isinstance(p, dict) for p in sample])
    ref_timestamp = None
    for params in sample:
        human_name = params["human"].name
        timestamp = params["start"] + datetime.timedelta(days=params["current_day"], hours=params["time_slot"])
        if ref_timestamp is None:
            ref_timestamp = timestamp
        else:
            assert ref_timestamp == timestamp, "how can we possibly have different timestamps here"
        cluster_mgr_hash = str(params["city_hash"]) + ":" + human_name
        params["cluster_mgr_hash"] = cluster_mgr_hash
        if cluster_mgr_hash not in cluster_mgr_map:
            cluster_algo_type = params["conf"].get("CLUSTER_ALGO_TYPE", "gaen")
            assert cluster_algo_type in ["blind", "gaen"]
            if cluster_algo_type == "blind":
                cluster_algo_type = covid19sim.frozen.clustering.blind.BlindClusterManager
            else:
                cluster_algo_type = covid19sim.frozen.clustering.gaen.GAENClusterManager
            # cluster messages; as of the GAEN refactoring, only one algo can be used
            cluster_mgr = cluster_algo_type(
                max_history_offset=datetime.timedelta(days=params["conf"].get("TRACING_N_DAYS_HISTORY")),
                add_orphan_updates_as_clusters=True,
                generate_embeddings_by_timestamp=True,
                generate_backw_compat_embeddings=True,
            )
        else:
            cluster_mgr = cluster_mgr_map[cluster_mgr_hash]
        assert not cluster_mgr._is_being_used, "two processes should never try to access the same human"
        cluster_mgr._is_being_used = True
        params["cluster_mgr"] = cluster_mgr
    results = [_proc_human(params, engine) for params in sample]
    for params in sample:
        cluster_mgr = params["cluster_mgr"]
        assert cluster_mgr._is_being_used
        cluster_mgr._is_being_used = False
        cluster_mgr_map[params["cluster_mgr_hash"]] = cluster_mgr

    if clusters_dump_path and ref_timestamp:
        os.makedirs(clusters_dump_path, exist_ok=True)
        curr_date_str = ref_timestamp.strftime("%Y%m%d-%H%M%S")
        curr_dump_path = os.path.join(clusters_dump_path, curr_date_str + ".pkl")
        to_dump = {params["human"].name: params["cluster_mgr"] for params in sample}
        with open(curr_dump_path, "wb") as fd:
            pickle.dump(to_dump, fd)

    return results


def _proc_human(params, inference_engine):
    """Internal implementation of the `proc_human_batch` function."""
    assert isinstance(params, dict) and \
        all([p in params for p in expected_raw_packet_param_names]), \
        "unexpected/broken _proc_human input format between simulator and inference service"
    conf = params["conf"]
    todays_date = params["start"] + datetime.timedelta(days=params["current_day"], hours=params["time_slot"])
    human, cluster_mgr = params["human"], params["cluster_mgr"]

    # set the current day as the refresh timestamp to auto-purge outdated messages in advance
    cluster_mgr.set_current_timestamp(todays_date)
    update_messages = covid19sim.frozen.message_utils.batch_messages(human.update_messages)
    cluster_mgr.add_messages(messages=update_messages, current_timestamp=todays_date)
    # FIXME: there are messages getting duplicated somewhere, this is pretty bad @@@@@
    # uids = []
    # for c in cluster_mgr.clusters:
    #     uids.extend(c.get_encounter_uids())
    # assert len(np.unique(uids)) == len(uids), "found collision"

    # Format for supervised learning / transformer inference
    is_exposed, exposure_day = covid19sim.frozen.helper.exposure_array(human.infection_timestamp, todays_date, conf)
    is_recovered, recovery_day = covid19sim.frozen.helper.recovered_array(human.recovered_timestamp, todays_date, conf)
    candidate_encounters, exposure_encounter = covid19sim.frozen.helper.candidate_exposures(cluster_mgr)
    reported_symptoms = human.rolling_all_reported_symptoms
    true_symptoms = human.rolling_all_symptoms

    # FIXME: DIRTY DIRTY HACK; Nasim's DataLoader expects that the embeddings contain an absolute
    #  day index instead of a relative offset (i.e. the exact simulation day instead of [0,14])...
    if len(candidate_encounters):
        candidate_encounters[:, 3] = params["current_day"] - candidate_encounters[:, 3]
        # Nasim also does some masking with a hard-coded 14-day history length, let's do the same...
        valid_encounter_mask = candidate_encounters[:, 3] > (params["current_day"] - 14)
        candidate_encounters = candidate_encounters[valid_encounter_mask]
        exposure_encounter = exposure_encounter[valid_encounter_mask]

    daily_output = {
        "current_day": params["current_day"],
        "observed": {
            "reported_symptoms": reported_symptoms,
            "candidate_encounters": candidate_encounters,
            "test_results": human.test_results,
            "preexisting_conditions": human.obs_preexisting_conditions,
            "age": human.obs_age,
            "sex": human.obs_sex,
            "risk_mapping": conf.get("RISK_MAPPING"),
        },
        "unobserved": {
            "incubation_days": human.incubation_days,
            "recovery_days": human.recovery_days,
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

    inference_result, risk_history = None, None
    if conf.get("USE_ORACLE"):
        # return ground truth infectiousnesses
        risk_history = human.infectiousnesses
    elif conf.get("RISK_MODEL") == "transformer":
        # no need to do actual inference if the cluster count is zero
        inference_result = inference_engine.infer(daily_output)
        if inference_result is not None:
            risk_history = inference_result['infectiousness']
    return human.name, risk_history
