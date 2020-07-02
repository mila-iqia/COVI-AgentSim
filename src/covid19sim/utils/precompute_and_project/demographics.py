import argparse
import yaml
from pathlib import Path


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

    assert "P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1" not in region, "P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1 already exist. Delete it if you want to overwrite it."

    POPULATION_SIZE_COUNTRY = country["POPULATION_SIZE_COUNTRY"]
    P_AGE_COUNTRY = country["P_AGE_COUNTRY"]
    N_AGE_SOLO_DWELLERS_COUNTRY = country["N_AGE_SOLO_DWELLERS_COUNTRY"]

    P_AGE_REGION = region['P_AGE_REGION']
    POPULATION_SIZE_REGION = region["POPULATION_SIZE_REGION"]
    N_HOUSESIZE_1 = region["N_HOUSESIZE_1"]

    # precompute 1
    # computes age distribution of solo dwellers across all solo dwellers
    # it projects the country level statistics to regional statistics
    # a correction factor is used to correct for unequal number of houses of size 1
    age_bins = [(x[0], x[1]) for x in P_AGE_REGION]
    N_AGEBINS_SOLO_DWELLERS_COUNTRY = [[x[0], x[1], 0] for x in age_bins]
    for row in N_AGE_SOLO_DWELLERS_COUNTRY:
        age = row[0]
        total = row[1] + row[2]
        for i,x in enumerate(age_bins):
            if x[0] <= age <= x[1]:
                break
        N_AGEBINS_SOLO_DWELLERS_COUNTRY[i][-1] += total

    N_AGE_SOLO_DWELLERS_REGION = [[x[0], x[1], 0] for x in age_bins]
    for idx in range(len(age_bins)):
        rgn = P_AGE_REGION[idx]
        cntry = P_AGE_COUNTRY[idx]
        solo = N_AGEBINS_SOLO_DWELLERS_COUNTRY[idx]
        N_AGE_SOLO_DWELLERS_REGION[idx][-1] = (solo[2] / (cntry[2] * POPULATION_SIZE_COUNTRY)) * (rgn[2] * POPULATION_SIZE_REGION)

    total_solo_dwellers = sum(x[2] for x in N_AGE_SOLO_DWELLERS_REGION)

    correction_factor  = N_HOUSESIZE_1 / total_solo_dwellers
    N_AGE_SOLO_DWELLERS_CORRECTED = [[x[0], x[1], x[2] * correction_factor] for x in N_AGE_SOLO_DWELLERS_REGION]
    P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1 = [[x[0], x[1], x[2]/N_HOUSESIZE_1] for x in N_AGE_SOLO_DWELLERS_CORRECTED]

    comment = """
######################################################################
## Pre-computed using utils/precompute_and_project/demographics.py  ##
######################################################################

# Solo dwellers
# procedure -
# Data for solo dwellers for country is from country.yaml
# project it back per age group as per region's demographics
# Multiply by a correction term to make number of solo houses equal to A.1 * TOTAL_HOUSES
# Normalize it to get probabilities given housesize of 1
# Keys to read the list - lower age limit, upper age limit, proportion of population in this age bin that live in household of size 1
    """
    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
            + "P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1: "
            + "[{}]".format(",\n    ".join(map(str, P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1)))
        )


    # precompute 2
    # P_FAMILIY_TYPE_ are proportion of **all** households that are of particular type
    # normalize them here to proportion of households of size that are of some type
    P_FAMILY_TYPE_SIZE_2 = region['P_FAMILY_TYPE_SIZE_2']
    NORMALIZED_P_FAMILY_TYPE_SIZE_2 = [x/sum(P_FAMILY_TYPE_SIZE_2) for x in P_FAMILY_TYPE_SIZE_2]

    P_FAMILY_TYPE_SIZE_3 = region['P_FAMILY_TYPE_SIZE_3']
    NORMALIZED_P_FAMILY_TYPE_SIZE_3 = [x/sum(P_FAMILY_TYPE_SIZE_3) for x in P_FAMILY_TYPE_SIZE_3]

    P_FAMILY_TYPE_SIZE_4 = region['P_FAMILY_TYPE_SIZE_4']
    NORMALIZED_P_FAMILY_TYPE_SIZE_4 = [x/sum(P_FAMILY_TYPE_SIZE_4) for x in P_FAMILY_TYPE_SIZE_4]

    P_FAMILY_TYPE_SIZE_MORE_THAN_5 = region['P_FAMILY_TYPE_SIZE_MORE_THAN_5']
    NORMALIZED_P_FAMILY_TYPE_SIZE_MORE_THAN_5 = [x/sum(P_FAMILY_TYPE_SIZE_MORE_THAN_5) for x in P_FAMILY_TYPE_SIZE_MORE_THAN_5]

    comment = """
# Normalized P_FAMILY_TYPE_
# P_FAMILIY_TYPE_ are proportion of **all** households that are of particular type
# normalized values - proportion of households of size that are of some type
    """
    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
            + "NORMALIZED_P_FAMILY_TYPE_SIZE_2: "
            + str(NORMALIZED_P_FAMILY_TYPE_SIZE_2)
            + "\n"
            + "NORMALIZED_P_FAMILY_TYPE_SIZE_3: "
            + str(NORMALIZED_P_FAMILY_TYPE_SIZE_3)
            + "\n"
            + "NORMALIZED_P_FAMILY_TYPE_SIZE_4: "
            + str(NORMALIZED_P_FAMILY_TYPE_SIZE_4)
            + "\n"
            + "NORMALIZED_P_FAMILY_TYPE_SIZE_MORE_THAN_5: "
            + str(NORMALIZED_P_FAMILY_TYPE_SIZE_MORE_THAN_5)
            + "\n"
        )
