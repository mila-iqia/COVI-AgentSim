import pickle
import typing
import zmq

class InferenceClient:
    def __init__(
            self,
            target_port: typing.Union[int, typing.List[int]],
            target_addr: typing.AnyStr = "localhost",
    ):
        if isinstance(target_port, int):
            self.target_ports = [target_port]
        else:
            self.target_ports = target_port
        self.target_addr = target_addr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        for port in self.target_ports:
            self.socket.connect(f"tcp://{target_addr}:{port}")

    def infer(self, sample):
        self.socket.send(pickle.dumps(sample))
        return pickle.loads(self.socket.recv())
