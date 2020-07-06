"""
Functions to intialize a synthetic population using constants in configuration file.
"""
import numpy as np
from collections import defaultdict
from covid19sim.utils.utils import log

MAX_TRIES=100

def get_humans_with_age(city, age_histogram, conf, rng, chosen_infected, human_type):
    """
    Creats human_type objects corresponding to the numbers in `age_histogram`.

    Args:
        city (covid19sim.location.City): simulator's city object
        age_histogram (dict): a dictionary with keys as age bins (a tuple) and values as number of humans in that bin (int)
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator
        chosen_infected (set): human ids that are initialized to be infected
        human_type (covid19.simulator.Human): Class for the city's human instances

    Returns:
        dict: keys are age bins (tuple) and values are a list of human_type objects
    """
    humans = defaultdict(list)
    human_id = -1
    for age_bin, n in age_histogram.items():
        # sample age of humans before initialization
        ages = city.rng.randint(low=age_bin[0], high=age_bin[1]+1, size=n)

        for i in range(n):
            human_id += 1
            humans[age_bin].append(human_type(
                env=city.env,
                city=city,
                rng=np.random.RandomState(rng.randint(2 ** 16)),
                has_app=False, # to be initialized at the start of app-based intervention
                name=human_id,
                age=ages[i],
                household=None, # to be initialized separately
                workplace=None, # to be initialized separately
                profession="", # to be initialized separately
                rho=conf.get("RHO"),
                gamma=conf.get("GAMMA"),
                infection_timestamp=city.start_time if human_id in chosen_infected else None,
                conf=conf
                ))

    return humans

def assign_profession_to_humans(humans, city, conf):
    p_profession = conf['PROFESSION_PROFILE']
    professions = conf['PROFESSIONS']
    for age_bin, specs in p_profession:
        p = [specs[x] for x in professions]
        assigned_profession = city.rng.choice(professions, p=p, size=len(humans[age_bin]))
        for i, profession in enumerate(assigned_profession):
            humans[age_bin][i].profession = profession

    return humans

def assign_workplace_to_humans(humans, city, conf):
    """
    Considered workplaces are - "hospital", "senior_residency", "school" for children,
    "others" for rest that includes workplace, stores, and miscs, "senior_residency_social_activities"

    """
    for age_bin, humans in humans:
        for human in humans:
            if human.profession == "healthcare":
                workplace = city.rng.choice(city.hospitals + city.senior_residencys)
            elif profession == "school":
                workplace = city.rng.choice(city.schools)
            elif profession == "others":
                workplace = city.rng.choice(city.workplace + city.stores + city.miscs)
            elif profession == "retired":
                sr = city.rng.choice(city.senior_residencys)
                workplace = sr.social_common_room
            human.workplace = workplace
    return humans

def assign_households_to_humans(humans, city, conf, logfile=None):
    """
    Finds a best grouping of humans to assign to households based on regional configuration.
    Algorithm adopted from - "An Iterative Approach for Generating Statistically Realistic Populations of Households"
    ref: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0008828

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        logfile (str): filepath where the console output and final tracked metrics will be logged.

    Returns:
        list: a list of humans with a residence
    """

    MAX_FAILED_ATTEMPTS_ALLOWED = 10000
    AVG_HOUSEHOLD_SIZE = conf['AVG_HOUSEHOLD_SIZE']
    P_HOUSEHOLD_SIZE= conf['P_HOUSEHOLD_SIZE']
    P_COLLECTIVE_60_64 = conf['P_COLLECTIVE_60_64']
    P_COLLECTIVE_65_69 = conf['P_COLLECTIVE_65_69']
    P_COLLECTIVE_70_74 = conf['P_COLLECTIVE_70_74']
    P_COLLECTIVE_75_79 = conf['P_COLLECTIVE_75_79']
    P_COLLECTIVE_80_above = conf['P_COLLECTIVE_80_above']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']

    P_FAMILY_TYPE_SIZE_2 = conf['P_FAMILY_TYPE_SIZE_2']
    P_FAMILY_TYPE_SIZE_3 = conf['P_FAMILY_TYPE_SIZE_3']
    P_FAMILY_TYPE_SIZE_4 = conf['P_FAMILY_TYPE_SIZE_4']
    P_FAMILY_TYPE_SIZE_MORE_THAN_5 = conf['P_FAMILY_TYPE_SIZE_MORE_THAN_5']

    FAMILY_TYPES = [("couple", 0), ("single_parent", 1) , ("other", 2), ("couple", 1), \
                    ("single_parent", 2), ("other", 3), ("couple", 2), \
                    ("single_parent", 3), ("other", 4), ("couple", 3), \
                    ("single_parent", 4), ("other", 5), ("solo", 0)]

    P_TYPES = P_FAMILY_TYPE_SIZE_2 + P_FAMILY_TYPE_SIZE_3
    P_TYPES += P_FAMILY_TYPE_SIZE_4 + P_FAMILY_TYPE_SIZE_MORE_THAN_5
    P_TYPES += [P_HOUSEHOLD_SIZE[0]]

    assert len(P_TYPES) == len(FAMILY_TYPES), "not a valid mapping of probability and family types"
    assert abs(sum(P_TYPES) - 1) < 1e-2, "Probabilities do not sum to 1."

    # re-normalize to ensure that the probabilities sum to 1
    P_TYPES = [x/sum(P_TYPES) for x in P_TYPES]

    n_people = city.n_people
    unassigned_humans = humans
    age_bins = sorted(humans.keys(), key=lambda x:x[0])
    allocated_humans = []

    log("Allocating houses ... ", logfile)

    # allocate senior residencies
    collectives = [
        [(60,64), P_COLLECTIVE_60_64],
        [(65,69), P_COLLECTIVE_65_69],
        [(70,74), P_COLLECTIVE_70_74],
        [(75,79), P_COLLECTIVE_75_79],
        [(80,110), P_COLLECTIVE_80_above]
        ]
    assigned_to_collectives = []
    for bin, P in collectives:
        for human in unassigned_humans[bin]:
            if city.rng.random() < P:
                assigned_to_collectives.append((bin, human))

    for bin, human in assigned_to_collectives:
        res = city.rng.choice(city.senior_residencys, size=1).item()
        allocated_humans = _assign_household(human, res, allocated_humans)
        unassigned_humans[bin].remove(human)

    # allocate households
    # presample houses equal to the number of houses as per census
    n_houses_approx = int(n_people / AVG_HOUSEHOLD_SIZE)
    housetypes = _random_choice_tuples(FAMILY_TYPES, city.rng, P=P_TYPES, size=n_houses_approx, replace=True)

    # ordering -
    presampled_housetypes = housetypes

    n_failed_attempts = 0
    while len(allocated_humans) < city.n_people and n_failed_attempts < MAX_FAILED_ATTEMPTS_ALLOWED:
        if len(presampled_housetypes) > 0:
            housetype = presampled_housetypes.pop()
        else:
            housetype = _random_choice_tuples(FAMILY_TYPES, city.rng, P=P_TYPES, size=1)[0]

        humans_with_same_house, unassigned_humans = find_best_fit_humans(housetype, conf, city, unassigned_humans)

        # allocate if succesful
        if humans_with_same_house:
            if  housetype[0] == "other":
                typestr = f"other_{housetype[1]}"
            elif housetype[0] == "solo":
                typestr = "solo"
            else:
                typestr = f"{housetype[0]}_with_{housetype[1]}_kids"
            type = (typestr, P_TYPES[FAMILY_TYPES.index(housetype)])

            allocated_humans = create_and_assign_household(humans_with_same_house, type, conf, city, allocated_humans)
        else:
            n_failed_attempts += 1
            if n_failed_attempts % 100 == 0:
                log(f"Failed attempt - {n_failed_attempts}. Total allocated:{len(allocated_humans)}", logfile)

    # when number of attempts exceed the limit randomly allocate remaining humans
    if n_failed_attempts >= MAX_FAILED_ATTEMPTS_ALLOWED:
        log(f"Failed attempt - {n_failed_attempts}. Exceeded the maximum number of failed attempts allowed to allocate houses... trying random allocation!", logfile)

        while len(allocated_humans) < city.n_people:
            housetype = city.rng.choice(FAMILY_TYPES, p=P_TYPES, size=n_houses_approx).item()

            humans_with_same_house = _sample_random_humans(unassigned_humans.keys(), unassigned_humans, city.rng, size=housesize)
            if humans_with_same_house:
                allocated_humans = create_and_assign_household(humans_with_same_house, ("random", -1), conf, allocated_humans)
            else:
                log(f"(Random attempt) Could not find humans for house size {housesize}... trying other house size!! Total allocated:{len(allocated_humans)}", logfile)

    assert len(allocated_humans) == city.n_people, "assigned humans and total population do not add up"
    assert sum(len(val) for x,val in unassigned_humans.items()) == 0, "there are unassigned humans in the list"
    assert len(city.households) > 0, "no house generated"
    assert all(not all(human.age < MAX_AGE_CHILDREN for human in house.residents) for house in city.households), "house with only children allocated"
    # shuffle the list of humans so that the simulation is not dependent on the order of house allocation
    city.rng.shuffle(allocated_humans)
    log(f"Housing allocated with failed attempts: {n_failed_attempts} ", logfile)
    return allocated_humans

def find_best_fit_humans(housetype, conf, city, unassigned_humans):
    """
    Finds humans to be allocated to a house of type `housetype`.

    Args:
        housetype (tuple): type of house to sample
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        humans (list): humans sampled to live together
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
    """
    humans = []
    valid_age_bins = [x for x, val in unassigned_humans.items() if len(val) >= 1]
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']

    if housetype[0] == "couple":
        n_kids = housetype[1]
        humans =  _sample_couple_with_n_kids(valid_age_bins, conf, unassigned_humans, city.rng, n=n_kids)
        if humans:
            assert len(humans) == n_kids + 2, "not a valid allocation"
            assert sum(h.age < MAX_AGE_CHILDREN for h in humans) == n_kids, "not a valid allocation"

    elif housetype[0] == "single_parent":
        n_kids = housetype[1]
        humans = _sample_single_parent_n_kids(valid_age_bins, conf, unassigned_humans, city.rng, n=n_kids)
        if humans:
            assert len(humans) == n_kids + 1, "not a valid allocation"
            assert sum(h.age < MAX_AGE_CHILDREN for h in humans) == n_kids, "not a valid allocation"

    elif housetype[0] == "other":
        n_humans = housetype[1]
        humans = _sample_random_humans(valid_age_bins, conf, unassigned_humans, city.rng, size=n_humans)
        if humans:
            assert len(humans) == n_humans, "not a valid allocation"
            assert sum(h.age < MAX_AGE_CHILDREN for h in humans) == 0, "not a valid allocation"

    elif housetype[0] == "solo":
        humans = _sample_solo_dweller(valid_age_bins, conf, unassigned_humans, city.rng)
        if humans:
            assert len(humans) == 1, "not a valid allocation"
            assert sum(h.age < MAX_AGE_CHILDREN for h in humans) == 0, "not a valid allocation"

    else:
        raise ValueError

    return humans, unassigned_humans

def _assign_household(human, res, allocated_humans):
    """
    Allocates human to the residence `res`.

    Args:
        human (covid19sim.human.Human): `Human` to be assigned to the residence `res`
        res (covid19sim.locations.location.Household): house to which human is assigned to
        allocated_humans (list): a list of humans that have been allocated a household

    Returns:
        allocated_humans (list): a list of humans that have been allocated a household
    """
    assert human not in allocated_humans, f"reassigning household to human:{human}"
    human.assign_household(res)
    res.residents.append(human)
    allocated_humans.append(human)
    return allocated_humans

def create_and_assign_household(humans_with_same_house, type, conf, city, allocated_humans):
    """
    Creates a residence and allocates humans in `humans_with_same_house` to the same.

    Args:
        humans_with_same_house (list): a list of `Human` objects which are to be allocated to the same residence of type `type`.
        type (tuple): type of allocation (str) and census probability of this allocation type (float)
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        allocated_humans (list): a list of humans that have been allocated a household

    Returns:
        allocated_humans (list): a list of humans that have been allocated a household
    """
    assert all(human not in allocated_humans for human in humans_with_same_house), f"reassigning household to human:{human}"
    res = city.create_location(
        specs = conf.get("LOCATION_DISTRIBUTION")["household"],
        type = "household",
        name = len(city.households)
    )

    for human in humans_with_same_house:
        allocated_humans = _assign_household(human, res, allocated_humans)

    res.allocation_type = type
    city.households.add(res)
    return allocated_humans

def _sample_solo_dweller(valid_age_bins, conf, unassigned_humans, rng):
    """
    Finds one human to be allocated to a household of size 1.
    NOTE: Maximum number of tries are MAX_TRIES failing which an empty list is returned which triggers next sampling

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator

    Returns:
        human (list): human sampled to live alone
    """

    P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1 = conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']
    P_AGE_SOLO = [x[2] for x in P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1]

    n_iters, human = 0, []
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    while n_iters < MAX_TRIES:
        n_iters += 1
        valid_idx = [i for i,bin in enumerate(age_bins) if len(unassigned_humans[bin]) >= 1 and P_AGE_SOLO[i] > 0]
        valid_age_bins =  [age_bins[i] for i in valid_idx]
        if valid_age_bins:
            P = [P_AGE_SOLO[i] for i in valid_idx]
            P = [i / sum(P) for i in P]
            age_bin = _random_choice_tuples(valid_age_bins, rng, 1, P)[0]
            human = rng.choice(unassigned_humans[age_bin], size=1).tolist()
            unassigned_humans[age_bin].remove(human[0])
            break

    return human

def _sample_couple_with_n_kids(valid_age_bins, conf, unassigned_humans, rng, n=0):
    """
    Samples a couple and n kids to be allocated to a same house
    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        n (int): number of kids to sample. Defaults to 0.

    Returns:
        list: humans belonging to a same household (length = n + 2 if succesful else 0)
    """
    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MAX_AGE_COUPLE_WITH_CHILDREN = conf['MAX_AGE_COUPLE_WITH_CHILDREN']

    valid_couple_bins = _get_valid_bins(valid_age_bins, min_age=MIN_AGE_COUPLE)
    valid_younger_bins = []
    if n > 0:
        valid_couple_bins = _get_valid_bins(valid_couple_bins, max_age=MAX_AGE_COUPLE_WITH_CHILDREN)
        valid_younger_bins = _get_valid_bins(valid_age_bins, max_age=MAX_AGE_CHILDREN)

    # filter candidates for couple
    min_couple_single_bins = [(x,x) for x in valid_couple_bins if len(unassigned_humans[x]) >= 2]

    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    sequential_couple_bins = [
                            (x,y) for x,y in zip(age_bins, age_bins[1:]) \
                            if (len(unassigned_humans[x]) >= 1
                                and len(unassigned_humans[y]) >= 1
                                and x in valid_couple_bins
                                and y in valid_couple_bins)
                            ]

    valid_couple_bins = min_couple_single_bins + sequential_couple_bins

    # filter candidates for kids
    all_kids = [(y,x) for x in valid_younger_bins for y in unassigned_humans[x]]
    if len(all_kids) < n or len(valid_couple_bins) == 0:
        return []

    # sample
    sampled_humans = []
    if len(all_kids) >= n and valid_couple_bins:
        two_bins = _random_choice_tuples(valid_couple_bins, rng, size=1)[0]

        human1 = rng.choice(unassigned_humans[two_bins[0]], size=1).item()
        unassigned_humans[two_bins[0]].remove(human1)

        human2 = rng.choice(unassigned_humans[two_bins[1]], size=1).item()
        unassigned_humans[two_bins[1]].remove(human2)

        sampled_humans = [human1, human2]
        if n > 0:
            kids = _random_choice_tuples(all_kids, rng, size=n)
            for kid, bin in kids:
                unassigned_humans[bin].remove(kid)
                sampled_humans.append(kid)

        assert len(sampled_humans) == n + 2, f"improper sampling for couples with {n} kids. Sampled {len(sampled_humans)} humans..."

    return sampled_humans

def _sample_single_parent_n_kids(valid_age_bins, conf, unassigned_humans, rng, n=1):
    """
    Samples a single parent and `n` kids randomly to be put in the same house

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        n (int): number of kids to sample

    Returns:
        list: humans belonging to a same household (length = n + 1 if succesful else 0)
    """
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']

    valid_older_bins = _get_valid_bins(valid_age_bins, min_age=MIN_AGE_SINGLE_PARENT, max_age=MAX_AGE_SINGLE_PARENT)
    valid_younger_bins = _get_valid_bins(valid_age_bins, max_age=MAX_AGE_CHILDREN)

    all_kids = [(y,x) for x in valid_younger_bins for y in unassigned_humans[x]]
    if len(all_kids) < n or len(valid_older_bins) == 0:
        return []

    # sample single parent
    older_bin = _random_choice_tuples(valid_older_bins, rng, size=1)[0]
    older_human = rng.choice(unassigned_humans[older_bin], size=1).item()
    unassigned_humans[older_bin].remove(older_human)

    # pick n kids
    sampled = _random_choice_tuples(all_kids, rng, size=n)
    younger_humans = []
    for human, bin in sampled:
        unassigned_humans[bin].remove(human)
        younger_humans.append(human)

    return [older_human] + younger_humans

def _sample_random_humans(valid_age_bins, conf, unassigned_humans, rng, size):
    """
    Samples humans randomly to be put in the same house

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        size (int): number of humans to sample

    Returns:
        list: humans belonging to a same household (length = size if succesful else 0)
    """
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    valid_other_bins = _get_valid_bins(valid_age_bins, min_age=MAX_AGE_CHILDREN)

    all_humans = [(y,x) for x in valid_other_bins for y in unassigned_humans[x]]
    if len(all_humans) < size:
        return []

    sampled = _random_choice_tuples(all_humans, rng, size=size)
    humans = []
    for human, bin in sampled:
        unassigned_humans[bin].remove(human)
        humans.append(human)

    assert len(humans) == size, "number of humans sampled doesn't equal the expected size"
    return humans

def _get_valid_bins(valid_age_bins, min_age=-1, max_age=200):
    """
    Filters out age bins according to minium and maximum age specified.

    Args:
        valid_age_bins (list): a list of potential age bins
        min_age (int): age above which all bins are considered valid
        max_age (int): age below which all bins are considered valid
    Returns:
        (list): valid age bins
    """
    filtered = [x for x in valid_age_bins if x[0] >= min_age]
    filtered = [x for x in filtered if x[1] <= max_age]
    return filtered

def _random_choice_tuples(tuples, rng, size, P=None, replace=False):
    """
    samples `size` random elements from `tuples` with probability `P`.
    NOTE: This function work arounds the internal conversion of the elements
            in `tuples` to np.ndarray.
    Args:
        tuples (list): a list of tuples
        rng (np.random.RandomState): Random number generator
        size (int): number of elements to sample from `tuples`
        P (list): probability with which to sample. Defaults to None.
        replace (bool): True if sampling is to be done with replace.

    Returns:
        list: sampled elements from tuples
    """
    total = len(tuples)
    idxs = rng.choice(range(total), size=size, p=P, replace=replace)
    return [tuples[x] for x in idxs]
