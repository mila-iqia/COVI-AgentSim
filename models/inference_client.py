import pickle
import typing
import zmq


class InferenceClient:
    """Creates a client through which data samples can be sent for inference.

    This object will automatically be able to pick a proper remote inference
    engine. This object should be fairly lightweight and low-cost, so creating
    it once per day, per human *should* not create a significant overhead.
    """

    def __init__(
            self,
            target_port: typing.Union[int, typing.List[int]],
            target_addr: typing.Union[str, typing.List[str]] = "localhost",
            context: typing.Optional[zmq.Context] = None,
    ):
        self.target_ports = [target_port] if isinstance(target_port, int) else target_port
        self.target_addrs = [target_addr] if isinstance(target_addr, str) else target_addr
        if len(self.target_ports) != len(self.target_addrs):
            assert len(self.target_addrs) == 1 and len(self.target_ports) > 1, \
                "must either match all ports to one address or provide full port/addr combos"
            self.target_addrs = self.target_addrs * len(self.target_ports)
        if context is None:
            context = zmq.Context()
        self.context = context
        self.socket = self.context.socket(zmq.REQ)
        for addr, port in zip(self.target_addrs, self.target_ports):
            self.socket.connect(f"tcp://{addr}:{port}")

    def infer(self, sample):
        """Forwards a data sample for the inference engine using pickle."""
        self.socket.send(pickle.dumps(sample))
        return pickle.loads(self.socket.recv())
