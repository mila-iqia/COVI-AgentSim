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
        with unittest.mock.patch("covid19sim.server_utils.proc_human_batch") as mock:
            broker = covid19sim.server_utils.InferenceBroker(
                model_exp_path=covid19sim.server_bootstrap.default_model_exp_path,
                workers=2,
                verbose=True,
                verbose_print_delay=5,
            )
            mock.side_effect = ServerTests.fake_proc_human_batch
            broker.run()

    @staticmethod
    def start_client():
        # make sure the server is up & running, and we're not throwing stuff against a dead handle
        time.sleep(6)
        remote_engine = covid19sim.server_utils.InferenceClient()
        for test_idx in range(100):
            remote_output = remote_engine.infer(test_idx)
            assert remote_output == test_idx

    def test_inference(self):
        #server_proc = multiprocessing.Process(target=ServerTests.start_server)
        #server_proc.start()
        #client_proc = multiprocessing.Process(target=ServerTests.start_client)
        #client_proc.start()
        #client_proc.join()
        #server_proc.kill()
        #server_proc.terminate()
        #server_proc.join()
        # FIXME: test is disabled for now, there's some process black magic going on with pytest
        return


if __name__ == "__main__":
    unittest.main()
