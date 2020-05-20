import unittest.mock

import covid19sim.server_utils
import covid19sim.server_bootstrap


class ServerTests(unittest.TestCase):

    @staticmethod
    def fake_proc_human_batch(sample, *args, **kwargs):
        return sample

    def test_inference(self):
        manager = covid19sim.server_utils.InferenceBroker(
            model_exp_path=covid19sim.server_bootstrap.default_model_exp_path,
            workers=2,
            mp_backend="loky",
            mp_threads=4,
            verbose=False,
        )
        manager.start()
        remote_engine = covid19sim.server_utils.InferenceClient()
        with unittest.mock.patch("covid19sim.server_utils.proc_human_batch") as mock:
            mock.side_effect = self.fake_proc_human_batch
            for test_idx in range(100):
                remote_output = remote_engine.infer(test_idx)
                self.assertEqual(remote_output, test_idx)
        manager.stop()
        manager.join()


if __name__ == "__main__":
    unittest.main()
