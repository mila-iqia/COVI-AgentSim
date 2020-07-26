import datetime
import os
import time
import unittest
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.inference.server_utils import DataCollectionServer
from covid19sim.plotting import debug
from covid19sim.run import simulate
from covid19sim.utils.utils import dump_tracker_data, extract_tracker_data


class PlotTest(unittest.TestCase):

    def test_baseball_cards(self):
        """ Run a single simulation and ensure that baseball cards plots can be generated from the outputs
        """

        # Load the experimental configuration
        conf_name = "test_heuristic.yaml"
        conf = get_test_conf(conf_name)
        conf['KEEP_FULL_OBJ_COPIES'] = True
        conf['COLLECT_TRAINING_DATA'] = False
        conf['tune'] = True
        conf['INTERVENTION_DAY'] = 2

        with TemporaryDirectory() as d:

            # Run the simulation
            start_time = datetime.datetime(2020, 2, 28, 0, 0)
            n_people = 10
            n_days = 14

            outfile=os.path.join(d, "output")
            plotdir=os.path.join(d, "plots")
            os.mkdir(outfile)
            os.mkdir(plotdir)
            conf["outdir"] = outfile
            hdf5_path = os.path.join(outfile, "human_backups.hdf5")

            city, monitors, tracker = simulate(
                n_people=n_people,
                start_time=start_time,
                simulation_days=n_days,
                init_percent_sick=0.5,
                outfile=outfile,
                out_chunk_size=1,
                seed=0,
                return_city=True,
                conf=conf,
            )

            # with the 'KEEP_FULL_OBJ_COPIES' set, the tracker should spawn its own collection server
            assert hasattr(city, "tracker") and \
                hasattr(city.tracker, "collection_server") and \
                isinstance(city.tracker.collection_server, DataCollectionServer) and \
                city.tracker.collection_server is not None
            city.tracker.collection_server.stop_gracefully()
            city.tracker.collection_server.join()
            assert os.path.exists(hdf5_path)

            filename = f"tracker_data.pkl"
            data = extract_tracker_data(tracker, conf)
            dump_tracker_data(data, conf["outdir"], filename)

            # Ensure that baseball plots can be produced from the simulation outputs
            debug.main(outfile, os.path.join(d, "plots"), num_chains=1)
