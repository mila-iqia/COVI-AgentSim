import datetime
import os
import time
import unittest
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.inference.server_utils import DataCollectionServer
from covid19sim.plotting import debug_plots
from covid19sim.run import simulate


class PlotTest(unittest.TestCase):

    def test_baseball_cards(self):
        """ Run a single simulation and ensure that baseball cards plots can be generated from the outputs
        """

        # Load the experimental configuration
        conf_name = "test_heuristic.yaml"
        conf = get_test_conf(conf_name)
        conf['KEEP_FULL_OBJ_COPIES'] = True
        conf['COLLECT_TRAINING_DATA'] = False

        with TemporaryDirectory() as d:

            # Run the simulation
            start_time = datetime.datetime(2020, 2, 28, 0, 0)
            n_people = 20
            n_days = 10

            outfile=os.path.join(d, "output")
            os.mkdir(outfile)
            conf["outdir"] = outfile

            hdf5_path = os.path.join(outfile, "human_backups.hdf5")
            collection_server = DataCollectionServer(
                data_output_path=hdf5_path,
                config_backup=conf,
                human_count=n_people,
                simulation_days=n_days,
            )
            collection_server.start()

            city, monitors, tracker = simulate(
                n_people=n_people,
                start_time=start_time,
                simulation_days=n_days,
                init_percent_sick=0.25,
                outfile=outfile,
                out_chunk_size=1,
                seed=0,
                return_city=True,
                conf=conf,
            )

            collection_server.stop_gracefully()
            collection_server.join()
            assert os.path.exists(hdf5_path)

            # This sleep is needed to ensure everything is clean up and the write lock
            # on the human_backups.hdf5 file is properly released
            time.sleep(30)

            # Ensure that baseball plots can be produced from the simulation outputs
            debug_plots.main(debug_data_path=hdf5_path,
                             output_folder=os.path.join(d, "plots"))
