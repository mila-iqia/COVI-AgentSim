import os
import time
import unittest.mock
from tempfile import TemporaryDirectory

import covid19sim.inference.server_utils
import covid19sim.inference.server_bootstrap


def fake_proc_human_batch(sample, *args, **kwargs):
    return sample


class InferenceServerWrapper(covid19sim.inference.server_utils.InferenceServer):
    def __init__(self, **kwargs):
        covid19sim.inference.server_utils.InferenceServer.__init__(self, **kwargs)

    def run(self):
        with unittest.mock.patch("covid19sim.inference.server_utils.proc_human_batch") as mock:
            mock.side_effect = fake_proc_human_batch
            return covid19sim.inference.server_utils.InferenceServer.run(self)


class ServerTests(unittest.TestCase):
    def test_inference(self):
        with TemporaryDirectory() as d:
            frontend_address = "ipc://" + os.path.join(d, "frontend.ipc")
            backend_address = "ipc://" + os.path.join(d, "backend.ipc")
            inference_server = InferenceServerWrapper(
                model_exp_path=covid19sim.inference.server_bootstrap.default_model_exp_path,
                workers=2,
                frontend_address=frontend_address,
                backend_address=backend_address,
                verbose=True,
            )
            inference_server.start()
            time.sleep(10)
            remote_engine = covid19sim.inference.server_utils.InferenceClient(
                server_address=frontend_address,
            )
            for test_idx in range(100):
                remote_output = remote_engine.infer(test_idx)
                assert remote_output == test_idx
            inference_server.stop_gracefully()
            inference_server.join()


if __name__ == "__main__":
    unittest.main()
