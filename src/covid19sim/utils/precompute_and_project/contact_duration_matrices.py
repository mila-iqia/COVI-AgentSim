import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from covid19sim.utils.constants import SECONDS_PER_MINUTE

def _get_mean_and_sigma_of_product_of_two_gaussians(mean1, mean2, var1, var2):
    """
    product of two gaussian pdfs is a gaussian with mean = (var1 * mean2 + var2 * mean1) / (var1 + var2) and var = (1/var1 + 1/var2)^ -1
    """
    return (mean2 * var1 + mean1 * var2) / (var1 + var2), ((1 / var1) + (1 / var2)) ** -1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, help="path to the country's yaml file", required=True)
    parser.add_argument("--region", type=str, help="path to the region's yaml file", required=True)
    parser.add_argument("--no_comments", action="store_true", default=False, help="do not write comments associated to computation procedure")
    args = parser.parse_args()

    country_path = Path(args.country)
    assert country_path.exists(), f"{country_path} does not exist"

    region_path = Path(args.region)
    assert region_path.exists(), f"{region_path} does not exist"

    with country_path.open("r") as f:
        country = yaml.safe_load(f)

    with region_path.open("r") as f:
        region = yaml.safe_load(f)

    # check for existence in region
    duration_keys = ["CONTACT_DURATION_GAMMA_SHAPE_MATRIX", "CONTACT_DURATION_GAMMA_SCALE_MATRIX"]
    assert all(k not in region for k in duration_keys), "duration matrices already exist. Delete it if you want to overwrite it."

    # check for existence in country
    assert "MEAN_DURATION_MATRIX" in country, "MEAN_DURATION_MATRIX not found in the source file"
    assert "MEAN_MINUS_95_CI_MATRIX" in country, "MEAN_MINUS_95_CI_MATRIX not found in the source file"

    # source
    MEAN_DURATION_MATRIX = country["MEAN_DURATION_MATRIX"]
    MEAN_MINUS_95_CI_MATRIX = country["MEAN_MINUS_95_CI_MATRIX"]

    ########## 1: time duration matrices
    mu = pd.DataFrame(MEAN_DURATION_MATRIX)
    ci = pd.DataFrame(MEAN_MINUS_95_CI_MATRIX)

    # 95% CI = mu - 1.96 * sigma
    sigma = (mu - ci)/1.96

    # Gamma(k, theta) has a mean of k * theta and a variance of k * theta^2
    CONTACT_DURATION_GAMMA_SCALE_MATRIX = sigma ** 2 / mu
    CONTACT_DURATION_GAMMA_SHAPE_MATRIX = mu ** 2 / (sigma ** 2)

    CONTACT_DURATION_NORMAL_MEAN_MATRIX = mu * SECONDS_PER_MINUTE
    CONTACT_DURATION_NORMAL_SIGMA_MATRIX = sigma * SECONDS_PER_MINUTE

    # location based duration matrices obtained by multiplying two gaussian distributions
    # product of two gaussian pdfs is a gaussian with mean = (var1 * mean2 + var2 * mean1) / (var1 + var2) and var = (1/var1 + 1/var2)^ -1
    mean1 = CONTACT_DURATION_NORMAL_MEAN_MATRIX
    var1 = CONTACT_DURATION_NORMAL_SIGMA_MATRIX ** 2

    # household
    MEAN_HOUSEHOLD_CONTACT_SECONDS = country['MEAN_HOUSEHOLD_CONTACT_MINUTES'] * SECONDS_PER_MINUTE
    STDDEV_HOUSEHOLD_CONTACT_SECONDS = country['STDDEV_HOUSEHOLD_CONTACT_MINUTES'] * SECONDS_PER_MINUTE

    mean2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * MEAN_HOUSEHOLD_CONTACT_SECONDS
    var2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * (STDDEV_HOUSEHOLD_CONTACT_SECONDS ** 2)

    HOUSEHOLD_CONTACT_DURATION_NORMAL_MEAN_MATRIX, HOUSEHOLD_CONTACT_DURATION_NORMAL_SIGMA_MATRIX = _get_mean_and_sigma_of_product_of_two_gaussians(mean1, mean2, var1, var2)

    # school
    MEAN_SCHOOL_CONTACT_SECONDS = country['MEAN_SCHOOL_CONTACT_MINUTES']
    STDDEV_SCHOOL_CONTACT_SECONDS = country['STDDEV_SCHOOL_CONTACT_MINUTES']

    mean2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * MEAN_SCHOOL_CONTACT_SECONDS
    var2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * (STDDEV_SCHOOL_CONTACT_SECONDS ** 2)

    SCHOOL_CONTACT_DURATION_NORMAL_MEAN_MATRIX, SCHOOL_CONTACT_DURATION_NORMAL_SIGMA_MATRIX = _get_mean_and_sigma_of_product_of_two_gaussians(mean1, mean2, var1, var2)

    # workplace
    MEAN_WORKPLACE_CONTACT_SECONDS = country['MEAN_WORKPLACE_CONTACT_MINUTES']
    STDDEV_WORKPLACE_CONTACT_SECONDS = country['STDDEV_WORKPLACE_CONTACT_MINUTES']

    mean2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * MEAN_WORKPLACE_CONTACT_SECONDS
    var2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * (STDDEV_WORKPLACE_CONTACT_SECONDS ** 2)

    WORKPLACE_CONTACT_DURATION_NORMAL_MEAN_MATRIX, WORKPLACE_CONTACT_DURATION_NORMAL_SIGMA_MATRIX = _get_mean_and_sigma_of_product_of_two_gaussians(mean1, mean2, var1, var2)

    # other locations
    MEAN_OTHER_CONTACT_SECONDS = country['MEAN_OTHER_CONTACT_MINUTES']
    STDDEV_OTHER_CONTACT_SECONDS = country['STDDEV_OTHER_CONTACT_MINUTES']

    mean2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * MEAN_OTHER_CONTACT_SECONDS
    var2 = np.ones_like(CONTACT_DURATION_NORMAL_MEAN_MATRIX) * (STDDEV_OTHER_CONTACT_SECONDS ** 2)

    OTHER_CONTACT_DURATION_NORMAL_MEAN_MATRIX, OTHER_CONTACT_DURATION_NORMAL_SIGMA_MATRIX = _get_mean_and_sigma_of_product_of_two_gaussians(mean1, mean2, var1, var2)


    comment = """
################################################################################################
## projected and precomputed using utils/precompute_and_project/contact_duration_matrices.py  ##
################################################################################################
# contact duration matrices (procedure)
# We use the survey data provdided in us.yaml without density corrections (until we find some study doing it)
# Survey data has mean and 95%CI. We parameterize the duration as a gamma distribution with shape and scale inferred from
# mean and 95% CI.
# Since gamma distribution is not assumed in the survey, we use the survey data as gaussian distribution with sigma inferred from 95% CI.
# We combine this duration data with location based duration distribution which is assumed to be gaussian.
# We obtain location and age based duration matrices via product of two gaussian pdfs as -
# product of two gaussian pdfs is a gaussian with mean = (var1 * mean2 + var2 * mean1) / (var1 + var2) and var = (1/var1 + 1/var2)^ -1

    """
    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
            + "CONTACT_DURATION_GAMMA_SCALE_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, CONTACT_DURATION_GAMMA_SCALE_MATRIX.values.tolist())))
            + "\n\n"
            + "CONTACT_DURATION_GAMMA_SHAPE_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, CONTACT_DURATION_GAMMA_SHAPE_MATRIX.values.tolist())))
            + "\n\n"
            + "CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, CONTACT_DURATION_NORMAL_MEAN_MATRIX.values.tolist())))
            + "\n\n"
            + "CONTACT_DURATION_NORMAL_SIGMA_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, CONTACT_DURATION_NORMAL_SIGMA_MATRIX.values.tolist())))
            + "\n\n"
            + "HOUSEHOLD_CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, HOUSEHOLD_CONTACT_DURATION_NORMAL_MEAN_MATRIX.values.tolist())))
            + "\n\n"
            + "HOUSEHOLD_CONTACT_DURATION_NORMAL_SIGMA_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, HOUSEHOLD_CONTACT_DURATION_NORMAL_SIGMA_MATRIX.values.tolist())))
            + "\n\n"
            + "SCHOOL_CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, SCHOOL_CONTACT_DURATION_NORMAL_MEAN_MATRIX.values.tolist())))
            + "\n\n"
            + "SCHOOL_CONTACT_DURATION_NORMAL_SIGMA_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, SCHOOL_CONTACT_DURATION_NORMAL_SIGMA_MATRIX.values.tolist())))
            + "\n\n"
            + "WORKPLACE_CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, WORKPLACE_CONTACT_DURATION_NORMAL_MEAN_MATRIX.values.tolist())))
            + "\n\n"
            + "WORKPLACE_CONTACT_DURATION_NORMAL_SIGMA_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, WORKPLACE_CONTACT_DURATION_NORMAL_SIGMA_MATRIX.values.tolist())))
            + "\n\n"
            + "OTHER_CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, OTHER_CONTACT_DURATION_NORMAL_MEAN_MATRIX.values.tolist())))
            + "\n\n"
            + "OTHER_CONTACT_DURATION_NORMAL_SIGMA_SECONDS_MATRIX: "
            + "[\n{}\n]".format(",\n    ".join(map(str, OTHER_CONTACT_DURATION_NORMAL_SIGMA_MATRIX.values.tolist())))
            + "\n\n"
        )
