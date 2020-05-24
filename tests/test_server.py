import multiprocessing
import time
import unittest.mock

import covid19sim.server_utils
import covid19sim.server_bootstrap


class ServerTests(unittest.TestCase):

    @staticmethod
    def fake_proc_human_batch(sample, *args, **kwargs):
        return sample

    @staticmethod
    def start_server():
        manager = covid19sim.server_utils.InferenceBroker(
            model_exp_path=covid19sim.server_bootstrap.default_model_exp_path,
            workers=2,
            n_parallel_procs=0,
            verbose=True,
        )
        with unittest.mock.patch("covid19sim.server_utils.proc_human_batch") as mock:
            mock.side_effect = ServerTests.fake_proc_human_batch
            manager.run()

    def test_inference(self):
        proc = multiprocessing.Process(target=ServerTests.start_server)
        proc.start()
        # make sure the server is up & running, and we're not throwing stuff against a dead handle
        time.sleep(5)
        remote_engine = covid19sim.server_utils.InferenceClient()
        for test_idx in range(100):
            remote_output = remote_engine.infer(test_idx)
            self.assertEqual(remote_output, test_idx)
        proc.terminate()


if __name__ == "__main__":
    unittest.main()
