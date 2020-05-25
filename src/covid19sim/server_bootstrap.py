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

import covid19sim.server_utils

default_workers = 4
default_nproc = 4
default_model_exp_path = "https://drive.google.com/file/d/1Z7g3gKh2kWFSmK2Yr19MQq0blOWS5st0"


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
    port_doc = f"Input port to accept TCP connections on; will use IPC if not provided. "
    argparser.add_argument("-p", "--port", default=None, type=str, help=port_doc)
    exp_path_doc = f"Path to the experiment directory that should be used to instantiate the " \
                   f"inference engine(s). Will use Google Drive reference exp if not provided. " \
                   f"See `infer.py` for more information."
    argparser.add_argument("-e", "--exp-path", default=None, type=str, help=exp_path_doc)
    workers_doc = f"Number of inference workers to spawn. Will use {default_workers} by default."
    argparser.add_argument("-w", "--workers", default=None, type=int, help=workers_doc)
    verbosity_doc = "Toggles program verbosity on/off. Default is OFF (0). Variable expects 0 or 1."
    argparser.add_argument("-v", "--verbose", default=0, type=int, help=verbosity_doc)
    mp_nproc_doc = f"Number of processes to spawn in each worker. Will use {default_nproc} by default."
    argparser.add_argument("--nproc", default=None, type=int, help=mp_nproc_doc)
    weights_path_doc = "Path to the specific weights to reload inside the inference engine(s). " \
                       "Will use the 'best checkpoint' weights if not specified."
    argparser.add_argument("--weights-path", default=None, type=str, help=weights_path_doc)
    args = argparser.parse_args(args)
    if args.port is not None:
        assert args.port.isdigit(), f"unexpected port number format ({args.port})"
        args.port = int(args.port)
    if args.exp_path is None:
        args.exp_path = default_model_exp_path
    if args.workers is None:
        args.workers = default_workers
    assert args.workers > 0, f"invalid worker count: {args.workers}"
    if args.nproc is None:
        args.nproc = default_nproc
    assert args.nproc >= 0, f"invalid process count: {args.nproc}"
    return args


def interrupt_handler(signal, frame, broker):
    """Signal callback used to gently stop workers (releasing sockets) & exit."""
    print("Received SIGINT; shutting down inference worker(s) gracefully...")
    broker.stop()
    print("All done.")
    sys.exit(0)


def main(args=None):
    """Main entrypoint; see parse_args for information on the arguments."""
    args = parse_args(args)
    broker = covid19sim.server_utils.InferenceBroker(
        model_exp_path=args.exp_path,
        workers=args.workers,
        n_parallel_procs=args.nproc,
        port=args.port,
        weights_path=args.weights_path,
        verbose=args.verbose,
    )
    handler = functools.partial(interrupt_handler, broker=broker)
    signal.signal(signal.SIGINT, handler)
    broker.run()


if __name__ == "__main__":
    main()
