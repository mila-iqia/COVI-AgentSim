import unittest.mock

import covid19sim.server_utils


class ServerTests(unittest.TestCase):

    EXPERIMENT_DATA_URL = "https://drive.google.com/file/d/1Z7g3gKh2kWFSmK2Yr19MQq0blOWS5st0"

    @staticmethod
    def fake_proc_human_batch(sample, *args, **kwargs):
        return sample

    def test_inference(self):
        manager = covid19sim.server_utils.InferenceBroker(
            model_exp_path=self.EXPERIMENT_DATA_URL,
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
