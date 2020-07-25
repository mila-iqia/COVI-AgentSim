import datetime
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from omegaconf import OmegaConf

from covid19sim.utils.utils import dump_conf, parse_configuration
from tests.utils import HYDRA_SIM_PATH, get_test_conf


class HydraTests(unittest.TestCase):
    def test_config_yaml_exists(self):
        """
        There should be a config.yaml file in hydra's conf dir
        """
        self.assertTrue((HYDRA_SIM_PATH / "config.yaml").exists())

    def test_only_yaml_no_yml(self):
        """
        All files in hydra conf dir and test_configs should be yaml, not yml or other
        """
        all_files = [
            f
            for f in HYDRA_SIM_PATH.glob("**/*")
            if f.is_file() and not f.name.startswith(".")
        ] + [
            f
            for f in (Path(__file__).parent / "utils" / "test_configs").glob("**/*")
            if f.is_file() and not f.name.startswith(".")
        ]
        for f in all_files:
            self.assertTrue(f.suffix == ".yaml")

    def test_all_defaults_exist(self):
        with (HYDRA_SIM_PATH / "config.yaml").open("r") as f:
            config = yaml.safe_load(f)

        self.assertTrue("defaults" in config)

        for key in config["defaults"]:
            self.assertTrue((HYDRA_SIM_PATH / (key + ".yaml")).exists())

    def test_no_nested_dirs(self):
        """
        There can be a maximum of 1 level of sub-directories in HYDRA_SIM_PATH
        """
        all_dirs = [d for d in HYDRA_SIM_PATH.iterdir() if d.is_dir()]
        for d in all_dirs:
            self.assertListEqual([n for n in d.iterdir() if n.is_dir()], [])

    def no_hyphen_dirs(self):
        """
        No sub-folder can have a "-" in their name
        """
        all_dirs = [d for d in HYDRA_SIM_PATH.iterdir() if d.is_dir()]
        for d in all_dirs:
            self.assertFalse("-" in d.name)

    def test_all_are_valid_yamls(self):
        """
        All files in hydra conf dir and test_configs should be valid yaml files
        """
        for fname in HYDRA_SIM_PATH.glob("**/*.yaml"):
            with fname.open("r") as f:
                self.assertIsInstance(yaml.safe_load(f), dict)

        for fname in (Path(__file__).parent / "utils" / "test_configs").glob("**/*.yaml"):
            with fname.open("r") as f:
                self.assertIsInstance(yaml.safe_load(f), dict)

    def test_get_test_conf(self):
        """
        Check tests/utils.py/get_test_conf
        """
        conf_name = "naive_local.yaml"
        self.assertIsInstance(get_test_conf(conf_name), dict)
        test_conf = get_test_conf(conf_name)

        path = Path(__file__).parent / "utils" / "test_configs" / conf_name
        with path.open("r") as f:
            reference_conf = yaml.safe_load(f)

        for k in reference_conf:
            self.assertEqual(test_conf[k], reference_conf[k])

    def test_parse_configuration(self):
        """
        Check covid19sim/utils.py/parse_configuration
        """

        toy_conf = OmegaConf.create(
            {
                "SMARTPHONE_OWNER_FRACTION_BY_AGE": {"1-15": "something"},
                "HUMAN_DISTRIBUTION": {"0-20": "something_else"},
                "start_time": "2020-02-28 00:00:00",
            }
        )

        with self.assertRaises(AssertionError):
            parse_configuration(toy_conf)

        toy_conf["RISK_MODEL"] = ""
        parsed_conf = parse_configuration(toy_conf)

        self.assertDictEqual(
            parsed_conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"], {(1, 15): "something"}
        )
        self.assertDictEqual(
            parsed_conf["HUMAN_DISTRIBUTION"], {(0, 20): "something_else"}
        )
        self.assertEqual(
            parsed_conf["start_time"], datetime.datetime(2020, 2, 28, 0, 0, 0)
        )

    def test_dump_conf(self):
        """
        asserts that a conf that is dumped and parsed again yields identical results
        """

        conf = get_test_conf("naive_local.yaml")

        with TemporaryDirectory() as d:
            dump_conf(conf, Path(d) / "dumped.yaml")
            with (Path(d) / "dumped.yaml").open("r") as f:
                loaded_conf = yaml.safe_load(f)
            parsed_conf = parse_configuration(loaded_conf)

            # assertDictEqual cannot handle equality with np.array, e.g. by using np.all,
            # so we need to do it manually.
            age_group_contact_avg1 = conf.pop('AGE_GROUP_CONTACT_AVG')
            age_group_contact_avg2 = parsed_conf.pop('AGE_GROUP_CONTACT_AVG')
            self.assertListEqual(age_group_contact_avg1['age_groups'], age_group_contact_avg2['age_groups'])
            np.testing.assert_almost_equal(age_group_contact_avg1['contact_avg'], age_group_contact_avg2['contact_avg'])

            self.assertDictEqual(conf, parsed_conf)
