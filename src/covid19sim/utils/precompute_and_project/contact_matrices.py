import argparse
import yaml
from pathlib import Path
import pandas as pd

def _get_demographics_adjusted_to_contact_matrix_agebins(P_AGE, POPULATION_SIZE):
    """
    Given that we are aggregating data from various sources, it is required that we adjust for the underlying assumptions.
    One of these structural assumption is of age groups. Contact matrices available consider only 16 classes while we assume 17 classes.
    This function aggregates numbers in the last two groups.

    Args:
        P_AGE (list): each element is a list of 3 elements - min age in the group, max age in the group, proportion of population that falls in this age group
            expects a total of 17 elements (age groups) in this list
        POPULATION_SIZE (int): total size of the population

    Returns:
        (list): each element is a list of 3 elements - min age in the group, max age in the group, number of people in this age group
            The list has a total of 16 elements (age groups) in this list.
    """
    assert len(P_AGE) == 17, "Unknown age breakdown"
    contact_matrix_age_bins = [[0,4], [5,9], [10,14], [15,19], [20,24], [25,29], [30,34], [35,39], [40,44], \
                         [45,49], [50,54], [55,59], [60,64], [65,69], [70,74], [75, 110]]
    N = []
    for i, x in enumerate(P_AGE):
        if x[1] <= 74:
            N += [[x[0], x[1], x[2] * POPULATION_SIZE]]
        elif x[0] == 75:
            # aggregate the age groups above 75
            total_proportion = sum(x[2] for x in P_AGE[i+1:]) + x[2]
            N += [[x[0], 110, total_proportion * POPULATION_SIZE]] # 75+ yo
            break
        else:
            raise

    assert len(N) == len(contact_matrix_age_bins), "age bins of contact matrix do not align with internal age bins"
    assert abs(sum(x[2] for x in N) - POPULATION_SIZE) < 1e-02, f"populations do not sum up, {sum(x[2] for x in N)} != {POPULATION_SIZE}"

    return N

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
    adjusted_matrix_keys = ["ADJUSTED_CONTACT_MATRIX_HOUSEHOLD", "ADJUSTED_CONTACT_MATRIX_WORK", "ADJUSTED_CONTACT_MATRIX_SCHOOL", "ADJUSTED_CONTACT_MATRIX_OTHER", "ADJUSTED_CONTACT_MATRIX_ALL"]
    assert all(k not in region for k in adjusted_matrix_keys), "Adjusted matrices already exist. Delete it if you want to overwrite it."

    p_matrix_keys = ["P_CONTACT_MATRIX_HOUSEHOLD", "P_CONTACT_MATRIX_WORK", "P_CONTACT_MATRIX_SCHOOL", "P_CONTACT_MATRIX_OTHER", "P_CONTACT_MATRIX_ALL"]
    assert all(k not in region for k in p_matrix_keys), "Adjusted matrices already exist. Delete it if you want to overwrite it."

    # check for existence in country
    country_keys = ["COUNTRY_CONTACT_MATRIX_HOUSEHOLD", "COUNTRY_CONTACT_MATRIX_WORK", "COUNTRY_CONTACT_MATRIX_ALL", "COUNTRY_CONTACT_MATRIX_OTHER", "COUNTRY_CONTACT_MATRIX_SCHOOL"]
    for key in country_keys:
        assert key in country, f"{key} not found in the source file"

    # source
    POPULATION_SIZE_COUNTRY = country["POPULATION_SIZE_COUNTRY"]
    P_AGE_COUNTRY = country["P_AGE_COUNTRY"]
    COUNTRY_CONTACT_MATRIX_HOUSE = country["COUNTRY_CONTACT_MATRIX_HOUSEHOLD"]
    COUNTRY_CONTACT_MATRIX_WORK = country["COUNTRY_CONTACT_MATRIX_WORK"]
    COUNTRY_CONTACT_MATRIX_ALL = country["COUNTRY_CONTACT_MATRIX_ALL"]
    COUNTRY_CONTACT_MATRIX_OTHER = country["COUNTRY_CONTACT_MATRIX_OTHER"]
    COUNTRY_CONTACT_MATRIX_SCHOOL = country["COUNTRY_CONTACT_MATRIX_SCHOOL"]

    CONTACT_MATRICES = [COUNTRY_CONTACT_MATRIX_HOUSE, COUNTRY_CONTACT_MATRIX_WORK, COUNTRY_CONTACT_MATRIX_SCHOOL, COUNTRY_CONTACT_MATRIX_OTHER, COUNTRY_CONTACT_MATRIX_ALL]

    # target
    P_AGE_REGION = region['P_AGE_REGION']
    POPULATION_SIZE_REGION = region["POPULATION_SIZE_REGION"]

    N_COUNTRY_adjusted = _get_demographics_adjusted_to_contact_matrix_agebins(P_AGE_COUNTRY, POPULATION_SIZE_COUNTRY)
    N_REGION_adjusted = _get_demographics_adjusted_to_contact_matrix_agebins(P_AGE_REGION, POPULATION_SIZE_REGION)

    ######### 1: contact matrices
    # until empirical estimates are available, we use location specific contact matrices derived in Prem et al.
    # We project the matrix for Canada to Montreal to account for the differences in demographical structures.
    # NOTE: Data from Prem et al. has 16 classes while original POLYMOD study has 15 classes with last class as 70+ yo
    # (assumption until verified) Prem et al. has broken down the last class as 70-74 and 75+ yo resulting in an extra class

    # for the definition of M_ij look at "Epidemiological Modelling: Simulating the Initial Phase of an Epidemic" of the original POLYMOD study
    # ref: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074#s2
    # M_ij - average number of daily contacts reported by an individual in age group j with an individual in estimated age group i

    # deriving population contact matrix C from raw contact matrix M
    ADJUSTED_CONTACT_MATRICES = {}
    P_CONTACT_MATRICES = {}
    for adjusted_key, p_key, CM in zip(adjusted_matrix_keys, p_matrix_keys, CONTACT_MATRICES):
        # adjust for local population structure using "Projecting social contact matrices to different demographic structures"
        # ref: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006638#sec018
        # method M2 Density correction
        # N_j = Number of individuals in age group j in the reference country
        # N = population size of the reference country
        # N` =  population size of the concerned region
        # N_j` = Number of individuals in age group j in the concerned region
        # M_ij` = M_ij * (N / N_j) * (N_j` / N`)
        M = pd.DataFrame(CM)
        M_density_corrected = pd.DataFrame(index=M.index, columns=M.columns)
        for j in range(len(M.columns)):
            N_j = N_COUNTRY_adjusted[j][2]
            N_j_dash = N_REGION_adjusted[j][2]
            N = POPULATION_SIZE_COUNTRY
            N_dash = POPULATION_SIZE_REGION
            M_density_corrected.iloc[:,j] = M.iloc[:,j] * (N / N_j) * (N_j_dash / N_dash)

        # adjust for reciprocity
        # following the procedure described in BBC pandemic study
        # ref: https://www.medrxiv.org/content/10.1101/2020.02.16.20023754v2.full.pdf
        # C_ij = (M_ij + M_ji *(w_i / w_j)) * 0.5
        C = pd.DataFrame(index=M.index, columns=M.columns)
        for j in range(len(M.columns)):
            for i in range(len(M.columns)):
                w_i = N_REGION_adjusted[i][2]
                w_j = N_REGION_adjusted[j][2]
                C.iloc[i,j] = 0.5 * (M_density_corrected.iloc[i, j] + M_density_corrected.iloc[j, i] * w_i / w_j)

        ADJUSTED_CONTACT_MATRICES[adjusted_key] = C
        # NOTE: C is not symmetric matrix. Reciprocity translates into number of contacts from i to j to be equal to number of contacts
        # from j to i. However, C is an average matrix so the matrix will not be symmetric. refer the above cited studies for more info.

        # convert the contact matrix into likelihood of interaction between different age groups
        P = pd.DataFrame(index=C.index, columns=C.columns)
        for j in range(len(C.columns)):
            P.iloc[:, j] = C.iloc[:, j] / C.iloc[:, j].sum()

        P_CONTACT_MATRICES[p_key] = P
    assert all(abs(P.sum(axis=0) - 1) < 1e-2)

    # write it to the yaml file
    comment = """
#######################################################################################
## projected and precomputed using utils/precompute_and_project/contact_matrices.py  ##
#######################################################################################
# ---- Adjusted Contact Matrix (procedure) ----
# Raw contact matrix at the level of country is read from configs/simulation/country/country_name.yaml
# Density correction is performed using the method M2 in "Projecting social contact matrices to different demographic structures"
# ref: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006638#sec018
# Reciprocity is ensured via correction used in the BBC Pandemic study
# ref: https://www.medrxiv.org/content/10.1101/2020.02.16.20023754v2.full.pdf
# ---- probability of contact between two age groups (procedure) ----
# Resulting contact matrix is converted to likelihood via column normalization.
# Thus, P_ij = probability of someone from age group j contacting someone in age group i
    """

    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
        )

    comment = """
####### Adjusted contact matrices
    """
    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
        )

    for key in adjusted_matrix_keys:
        with region_path.open("a") as f:
            f.write(
                f"{key}: "
                + "[\n{}\n]".format(",\n    ".join(map(str, ADJUSTED_CONTACT_MATRICES[key].values.tolist())))
                + "\n\n"
            )

    comment = """
####### Probability of contact  matrices
    """
    with region_path.open("a") as f:
        f.write(
            "\n"
            + (comment if not args.no_comments else "")
            + "\n"
        )

    for key in p_matrix_keys:
        with region_path.open("a") as f:
            f.write(
                f"{key}: "
                + "[\n{}\n]".format(",\n    ".join(map(str, P_CONTACT_MATRICES[key].values.tolist())))
                + "\n\n"
            )

    ########## 2: time duration matrices
    comment = """
# Following matrix is copied manually from "Using Time-Use Data to Parameterize Models for the Spread of Close-Contact Infectious Diseases"
# ref: https://pubmed.ncbi.nlm.nih.gov/18801889/
# the participants in the survey were from California, U.S. The survey itself several was conducted during the years 1988-2003.
# We use these matrices as it is.
# (assumption) duration of contacts are similar.
# No correction is applied.
    """
