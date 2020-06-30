"""
Entrypoint that can be used to start many inference workers on a single node.

The client-server architecture should only be used for large-scale simulations
that want to share compute resources for clustering/inference. Local tests, CI,
and debugging sessions should avoid using the server altogether and turn it off
via the `USE_INFERENCE_SERVER = False` setting in the experimental config.

By default, if no port is specified, the server will only expect IPC-based
requests through a file handle created in the working directory. If a port
is specified, the server will listen for TCP connections. All internal
communication between the broker and the workers is handled through IPC sockets.
"""

import argparse
import functools
import signal
import sys

import covid19sim.inference.server_utils

default_workers = 6
# TWILIGHT-RAIN-696
default_model_exp_path = "https://drive.google.com/file/d/1kXA-0juviQOL0R08YlQpaS5gKumQ8zrT"
default_data_buffer_size = ((10 * 1024) * 1024)  # 1MB


def parse_args(args=None):
    """
    Parses command line arguments for the server bootstrap main entrypoint.

    Args:
        args: overridable arguments forwarded to argparse.

    Returns:
        The parsed arguments.
    """
    argparser = argparse.ArgumentParser(
        description="COVID19-P2P-Transformer Inference Server Spawner",
    )
    frontend_doc = "Frontend port or address to accept TCP/IPC connections on."
    argparser.add_argument("--frontend", default=None, type=str, help=frontend_doc)
    backend_doc = "Backend port or address to dispatch work on."
    argparser.add_argument("--backend", default=None, type=str, help=backend_doc)
    verbosity_doc = "Toggles program verbosity on/off. Default is OFF (0). Variable expects 0 or 1."
    argparser.add_argument("-v", "--verbose", default=0, type=int, help=verbosity_doc)
    subparsers = argparser.add_subparsers(title="Server type", dest="type")
    inference_argparser = subparsers.add_parser("inference", help="Create an inference server")
    exp_path_doc = "Path to the experiment directory that should be used to instantiate the " \
                   "inference engine(s). Will use Google Drive reference exp if not provided. " \
                   "See `infer.py` for more information."
    inference_argparser.add_argument("-e", "--exp-path", default=None, type=str, help=exp_path_doc)
    workers_doc = f"Number of inference workers to spawn. Will use {default_workers} by default."
    inference_argparser.add_argument("-w", "--workers", default=None, type=int, help=workers_doc)
    weights_path_doc = "Path to the specific weights to reload inside the inference engine(s). " \
                       "Will use the 'best checkpoint' weights if not specified."
    inference_argparser.add_argument("--weights-path", default=None, type=str, help=weights_path_doc)
    datacollect_argparser = subparsers.add_parser("datacollect", help="Create a data collection server")
    data_output_path_doc = "Path to the HDF5 file that will contain all collected data samples."
    datacollect_argparser.add_argument("-o", "--out-path", type=str, help=data_output_path_doc)
    data_buffer_size_doc = "Size of the data buffer used to queue collected samples for writing " \
                           f"(in bytes). Will use {default_data_buffer_size // (1024 * 1024)}MB " \
                           "by default."
    datacollect_argparser.add_argument("--buffer-size", default=default_data_buffer_size,
                                       type=int, help=data_buffer_size_doc)
    args = argparser.parse_args(args)
    if args.frontend is not None:
        if args.frontend.isdigit():
            args.frontend = int(args.frontend)
    if args.backend is not None:
        if args.backend.isdigit():
            args.backend = int(args.backend)
    if args.type == "inference":
        if args.exp_path is None:
            args.exp_path = default_model_exp_path
        if args.workers is None:
            args.workers = default_workers
        assert args.workers > 0, f"invalid worker count: {args.workers}"
    return args


def interrupt_handler(signal, frame, broker):
    """Signal callback used to gently stop workers (releasing sockets) & exit."""
    print("Received SIGINT; shutting down inference worker(s) gracefully...", flush=True)
    broker.stop()
    print("All done.", flush=True)
    sys.exit(0)


def main(args=None):
    """Main entrypoint; see parse_args for information on the arguments."""
    args = parse_args(args)
    frontend_address, backend_address = None, None
    if args.frontend is not None:
        if isinstance(args.frontend, int):
            frontend_address = f"tcp://*:{args.frontend}"
        else:
            frontend_address = args.frontend
    if args.backend is not None:
        if isinstance(args.backend, int):
            backend_address = f"tcp://*:{args.backend}"
        else:
            backend_address = args.backend
    if args.type == "inference":
        if frontend_address is None:
            frontend_address = covid19sim.inference.server_utils.default_inference_frontend_address
        if backend_address is None:
            backend_address = covid19sim.inference.server_utils.default_inference_backend_address
        broker = covid19sim.inference.server_utils.InferenceBroker(
            model_exp_path=args.exp_path,
            workers=args.workers,
            frontend_address=frontend_address,
            backend_address=backend_address,
            weights_path=args.weights_path,
            verbose=args.verbose,
        )
    elif args.type == "datacollect":
        if frontend_address is None:
            frontend_address = covid19sim.inference.server_utils.default_datacollect_frontend_address
        if backend_address is None:
            backend_address = covid19sim.inference.server_utils.default_datacollect_backend_address
        broker = covid19sim.inference.server_utils.DataCollectionBroker(
            data_output_path=args.out_path,
            data_buffer_size=args.buffer_size,
            frontend_address=frontend_address,
            backend_address=backend_address,
            verbose=args.verbose,
        )
    else:
        raise AssertionError(f"unknown server type: {args.type}")
    handler = functools.partial(interrupt_handler, broker=broker)
    signal.signal(signal.SIGINT, handler)
    broker.run()


if __name__ == "__main__":
    main()
