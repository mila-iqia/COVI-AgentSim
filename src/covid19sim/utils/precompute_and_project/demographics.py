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

    # precompute 3
    # P_COLLECTIVE_60_64, P_COLLECTIVE_65_69, P_COLLECTIVE_70_74, P_COLLECTIVE_75_79, P_COLLECTIVE_80_above
    # probability of living in a collective given age ranges
    N_COLLECTIVE = region['N_COLLECTIVE']
    N_COLLECTIVE_60_79 = region['N_COLLECTIVE_60_79']
    N_COLLECTIVE_80_above = region['N_COLLECTIVE_80_above']

    assert P_AGE_REGION[-4][0] == 60 and P_AGE_REGION[-4][1] == 64, "misalignment in age groupings detected"
    N_60_64 = P_AGE_REGION[-4][2] * POPULATION_SIZE_REGION

    assert P_AGE_REGION[-3][0] == 65 and P_AGE_REGION[-3][1] == 69, "misalignment in age groupings detected"
    N_65_69 = P_AGE_REGION[-3][2] * POPULATION_SIZE_REGION

    assert P_AGE_REGION[-2][0] == 70 and P_AGE_REGION[-2][1] == 74, "misalignment in age groupings detected"
    N_70_74 = P_AGE_REGION[-2][2] * POPULATION_SIZE_REGION

    assert P_AGE_REGION[-1][0] == 75, "misalignment in age groupings detected"
    N_75_above = P_AGE_REGION[-1][2] * POPULATION_SIZE_REGION

    # (assumption) distribution of people aged 60-79 living in collective is assumed to be skewed towards larger range [1/16 ,3/16, 5/16, 7/16]
    N_COLLECTIVE_60_64 = N_COLLECTIVE_60_79 // 16
    N_COLLECTIVE_65_69 = N_COLLECTIVE_60_79 * 3 // 16
    N_COLLECTIVE_70_74 = N_COLLECTIVE_60_79 * 5 // 16
    N_COLLECTIVE_75_79 = N_COLLECTIVE_60_79 - N_COLLECTIVE_60_64 - N_COLLECTIVE_65_69 - N_COLLECTIVE_70_74

    P_COLLECTIVE = N_COLLECTIVE / POPULATION_SIZE_REGION
    P_COLLECTIVE_60_64 = N_COLLECTIVE_60_64 / N_60_64
    P_COLLECTIVE_65_69 = N_COLLECTIVE_65_69 / N_65_69
    P_COLLECTIVE_70_74 = N_COLLECTIVE_70_74 / N_70_74
    P_COLLECTIVE_75_above = (N_COLLECTIVE_75_79 + N_COLLECTIVE_80_above) / N_75_above

    comment = """
# P_COLLECTIVE_
# Probability of living in a collective given an age range.
# (procedure) use N_COLLECTIVE_, P_AGE_REGION, and POPULATION_SIZE_REGION to compute the ratios.
# (assumption) distribution of people aged 60-79 living in collective is assumed to be skewed towards larger range [1/16 ,3/16, 5/16, 7/16]
"""
    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
            + "P_COLLECTIVE: "
            + str(P_COLLECTIVE)
            + "\n"
            + "P_COLLECTIVE_60_64: "
            + str(P_COLLECTIVE_60_64)
            + "\n"
            + "P_COLLECTIVE_65_69: "
            + str(P_COLLECTIVE_65_69)
            + "\n"
            + "P_COLLECTIVE_70_74: "
            + str(P_COLLECTIVE_70_74)
            + "\n"
            + "P_COLLECTIVE_75_above: "
            + str(P_COLLECTIVE_75_above)
            + "\n"
        )
