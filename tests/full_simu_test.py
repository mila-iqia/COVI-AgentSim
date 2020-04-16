import datetime
import filecmp
import hashlib
import pickle
import unittest
import zipfile
from tempfile import NamedTemporaryFile

from run import run_simu


# Force COLLECT_LOGS=True
import config
config.COLLECT_LOGS = True
from base import Event
import simulator
simulator.Event = Event


class FullUnitTest(unittest.TestCase):

    def test_simu_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        with NamedTemporaryFile() as f:
            n_people = 100
            monitors, _ = run_simu(
                n_people=n_people,
                init_percent_sick=0.1,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=30,
                outfile=f.name,
                out_chunk_size=500
            )
            monitors[0].dump()
            monitors[0].join_iothread()
            f.seek(0)

            # Ensure
            data = []
            with zipfile.ZipFile(f"{f.name}.zip", 'r') as zf:
                data.extend([pickle.load(zf.open(pkl, 'r')) for pkl in zf.namelist()])

            self.assertTrue(len(data) > 0)

            self.assertTrue(Event.encounter in {d['event_type'] for d in data})
            self.assertTrue(Event.test in {d['event_type'] for d in data})

            self.assertTrue(len({d['human_id'] for d in data}) > n_people / 2)


class SeedUnitTest(unittest.TestCase):

    def setUp(self):
        self.test_seed = 136
        self.n_people = 100
        self.init_percent_sick = 0.1
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 10

    def test_sim_same_seed(self):
        """
        Run two simulations with the same seed and ensure we get the same output
        Note: If this test is failing, it is a good idea to load the data of both files and use DeepDiff to compare
        """
        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2:
            monitors1, _ = run_simu(
                n_people=self.n_people,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f1.name,
                out_chunk_size=0,
                seed=self.test_seed
            )
            monitors1[0].dump()
            monitors1[0].join_iothread()
            f1.seek(0)

            monitors2, _ = run_simu(
                n_people=self.n_people,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f2.name,
                out_chunk_size=0,
                seed=self.test_seed
            )
            monitors2[0].dump()
            monitors2[0].join_iothread()
            f2.seek(0)

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f1.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.open(pkl, 'r'))
            md5sum1 = md5.hexdigest()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f2.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.open(pkl, 'r'))
            md5sum2 = md5.hexdigest()

            self.assertTrue(md5sum1 == md5sum2,
                            msg=f"Two simulations run with the same seed "
                            f"({self.test_seed}) yielded different results")

    def test_sim_diff_seed(self):
        """
        Using different seeds should yield different output
        """

        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2:
            monitors1, _ = run_simu(
                n_people=self.n_people,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f1.name,
                out_chunk_size=0,
                seed=self.test_seed
            )
            monitors1[0].dump()
            monitors1[0].join_iothread()
            f1.seek(0)

            monitors2, _ = run_simu(
                n_people=self.n_people,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f2.name,
                out_chunk_size=0,
                seed=self.test_seed+1
            )
            monitors2[0].dump()
            monitors2[0].join_iothread()
            f2.seek(0)

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f1.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.open(pkl, 'r'))
            md5sum1 = md5.hexdigest()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f2.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    md5.update(zf.open(pkl, 'r'))
            md5sum2 = md5.hexdigest()

            self.assertFalse(md5sum1 == md5sum2,
                             msg=f"Two simulations run with different seeds "
                             f"({self.test_seed},{self.test_seed+1}) yielded "
                             f"the same result")
