import argparse
import yaml
from pathlib import Path
import pandas as pd

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

    comment = """
################################################################################################
## projected and precomputed using utils/precompute_and_project/contact_duration_matrices.py  ##
################################################################################################
# contact duration matrices (procedure)
# We use the survey data provdided in us.yaml without density corrections (until we find some study doing it)
# Survey data has mean and 95%CI. We parameterize the duration as a gamma distribution with shape and scale inferred from
# mean and 95% CI.
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
        )