import datetime
import unittest
from tempfile import NamedTemporaryFile

from run import run_simu
import pickle

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
