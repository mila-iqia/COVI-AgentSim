import datetime
import hashlib
import os
import unittest
import zipfile
from tempfile import TemporaryDirectory

from covid19sim.run import run_simu
from covid19sim.configs.exp_config import ExpConfig
class GoogleExpDeterminismTest(unittest.TestCase):

    def setUp(self):
        self.test_seed = 136
        self.n_people = 100
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 5

    def test_google_exp_determinism(self):
        """
        Run two simulations with the same seed and ensure we get the same output
        Note: If this test is failing, it is a good idea to load the data of both files and use DeepDiff to compare
        """
        with TemporaryDirectory() as d1, TemporaryDirectory() as d2, TemporaryDirectory() as d3:
            of1 = os.path.join(d1, "data")
            of2 = os.path.join(d2, "data")
            of3 = os.path.join(d3, "data")

            # TODO: @nasim when we refactor the configs, we'll want to replace these possibly with test configs that inherit from the ones hardcoded below
            ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "../src/covid19sim/configs/transformer_config.yml"))
            ExpConfig.set("COLLECT_LOGS", "True")

            monitors1, _ = run_simu(
                n_people=self.n_people,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=of1,
                out_chunk_size=0,
                init_percent_sick=0.1,
                seed=self.test_seed
            )
            monitors1[0].dump()
            monitors1[0].join_iothread()
            ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "../src/covid19sim/configs/lockdown.yml"))
            ExpConfig.set("COLLECT_LOGS", "True")
            monitors2, _ = run_simu(
                n_people=self.n_people,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=of2,
                out_chunk_size=0,
                init_percent_sick=0.1,
                seed=self.test_seed
            )
            monitors2[0].dump()
            monitors2[0].join_iothread()

            ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "../src/covid19sim/configs/binary_digital_tracing.yml"))
            ExpConfig.set("COLLECT_LOGS", "True")
            monitors3, _ = run_simu(
                n_people=self.n_people,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=of3,
                out_chunk_size=0,
                init_percent_sick=0.1,
                seed=self.test_seed
            )

            monitors3[0].dump()
            monitors3[0].join_iothread()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{of1}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.read(pkl))
            md5sum1 = md5.hexdigest()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{of2}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.read(pkl))
            md5sum2 = md5.hexdigest()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{of3}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.read(pkl))
            md5sum3 = md5.hexdigest()

            self.assertTrue(md5sum1 == md5sum2,
                            msg=f"the first and second simulations yield the same results before intervention day")
            self.assertTrue(md5sum2 == md5sum3,
                            msg=f"the second and third simulations yield the same results before intervention day")

            tracker.infection_monitor
            #serialize, hash, and assert trackers are the same