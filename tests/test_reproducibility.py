import datetime
import hashlib
import os
import unittest
import zipfile
import pytest
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.run import simulate

TEST_CONF_NAME = "base.yaml"


class ReproducibilityTests(unittest.TestCase):
    config = None

    def setUp(self):
        self.config = get_test_conf(TEST_CONF_NAME)

        self.test_seed = 136
        self.n_people = 100
        self.location_start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 20

    def test_reproducibility(self):
        """
        Run three simulations to have a pair of same seed simulation and ensure we get the same output.
        """

        events_logs = []

        for seed in (self.test_seed, self.test_seed, self.test_seed+1):
            with self.subTest(seed=seed):
                with TemporaryDirectory() as d:
                    md5 = hashlib.md5()
                    outfile = os.path.join(d, "data")
                    city, monitors, tracker = simulate(
                        n_people=self.n_people,
                        start_time=self.location_start_time,
                        simulation_days=self.simulation_days,
                        outfile=outfile,
                        out_chunk_size=0,
                        init_fraction_sick=0.1,
                        seed=seed,
                        conf=self.config
                    )
                    monitors[0].dump()
                    monitors[0].join_iothread()
                    import time
                    time.sleep(10)
                    with zipfile.ZipFile(f"{outfile}.zip", 'r') as zf:
                        for pkl in zf.namelist():
                            pkl_bytes = zf.read(pkl)
                            md5.update(pkl_bytes)

                events_logs.append(md5.hexdigest())

        md5sum, md5sum_same_seed, md5sum_diff_seed = events_logs

        self.assertEqual(md5sum, md5sum_same_seed,
                         msg=f"Two simulations run with the same seed "
                         f"{self.test_seed} yielded different results")

        self.assertNotEqual(md5sum, md5sum_diff_seed,
                            msg=f"Two simulations run with different seeds "
                            f"{self.test_seed}, {self.test_seed+1} yielded "
                            f"different results")
