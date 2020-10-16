import os
from covid19sim.utils.utils import download_exp_data_if_not_exist
from ctt.inference.infer import InferenceEngine

class InferenceEngineWrapper(InferenceEngine):
    """Inference engine wrapper used to download & extract experiment data, if necessary."""

    def __init__(self, experiment_directory, *args, **kwargs):
        if experiment_directory.startswith("http"):
            assert os.path.isdir("/tmp"), "don't know where to download data to..."
            experiment_root_directory = download_exp_data_if_not_exist(experiment_directory, "/tmp")
            experiment_subdirectories = \
                [os.path.join(experiment_root_directory, p) for p in os.listdir(experiment_root_directory)
                 if os.path.isdir(os.path.join(experiment_root_directory, p))]
            assert len(experiment_subdirectories) == 1, "should only have one dir per experiment zip"
            experiment_directory = experiment_subdirectories[0]
        super().__init__(experiment_directory, *args, **kwargs)
