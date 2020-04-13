import datetime
import filecmp
import pickle
import unittest
from tempfile import NamedTemporaryFile

from run import run_simu
from simulator import Event


class FullUnitTest(unittest.TestCase):

    def test_simu_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        with NamedTemporaryFile() as f:
            n_people = 100
            monitors = run_simu(
                n_stores=2,
                n_people=n_people,
                n_parks=1,
                n_hospitals=1,
                n_misc=2,
                init_percent_sick=0.1,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=30,
                outfile=f.name
            )
            monitors[0].dump(f.name)
            f.seek(0)

            # Ensure
            with open(f"{f.name}.pkl", 'rb') as f_output:
                data = pickle.load(f_output)

                self.assertTrue(len(data) > 0)

                self.assertTrue(Event.encounter in {d['event_type'] for d in data})
                self.assertTrue(Event.test in {d['event_type'] for d in data})

                self.assertTrue(len({d['human_id'] for d in data}) > n_people / 2)


class SeedUnitTest(unittest.TestCase):

    def setUp(self):
        self.test_seed = 136
        self.n_stores = 2
        self.n_people = 100
        self.n_parks = 1
        self.n_misc = 2
        self.init_percent_sick = 0.1
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 30

    def test_sim_same_seed(self):
        """
        Run two simulations with the same seed and ensure we get the same output
        Note: If this test is failing, it is a good idea to load the data of both files and use DeepDiff to compare
        """
        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2:
            monitors1 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f1.name,
                seed=self.test_seed
            )
            monitors1[0].dump(f1.name)
            f1.seek(0)

            monitors2 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f2.name,
                seed=self.test_seed
            )
            monitors2[0].dump(f2.name)
            f2.seek(0)

            self.assertTrue(filecmp.cmp(f"{f1.name}.pkl", f"{f2.name}.pkl"),
                            msg=f"Two simulations run with the same seed\
                            ({self.test_seed}) yielded different results")

    def test_sim_diff_seed(self):
        """
        Using different seeds should yield different output
        """

        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2:
            monitors1 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f1.name,
                seed=self.test_seed
            )
            monitors1[0].dump(f1.name)
            f1.seek(0)

            monitors2 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f2.name,
                seed=self.test_seed+1
            )
            monitors2[0].dump(f2.name)
            f2.seek(0)

            self.assertFalse(filecmp.cmp(f"{f1.name}.pkl", f"{f2.name}.pkl"),
                             msg=f"Two simulations run with different seeds\
                             ({self.test_seed},{self.test_seed+1}) yielded \
                                the same result")
