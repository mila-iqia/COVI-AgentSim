"""
Functions to intialize a synthetic population using constants in configuration file.
"""
import numpy as np
from collections import defaultdict
from covid19sim.utils.utils import log

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

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        logfile (str): filepath where the console output and final tracked metrics will be logged.

    Returns:
        list: a list of humans with a residence
    """
    def _assign_household(human, res, allocated_humans):
        assert human not in allocated_humans, f"reassigning household to human:{human}"
        human.assign_household(res)
        res.residents.append(human)
        allocated_humans.append(human)
        return allocated_humans

    MAX_FAILED_ATTEMPTS_ALLOWED = 10000
    P_HOUSEHOLD_SIZE= conf['P_HOUSEHOLD_SIZE']
    P_COLLECTIVE_65_69 = conf['P_COLLECTIVE_65_69']
    P_COLLECTIVE_70_74 = conf['P_COLLECTIVE_70_74']
    P_COLLECTIVE_75_79 = conf['P_COLLECTIVE_75_79']
    P_COLLECTIVE_80_above = conf['P_COLLECTIVE_80_above']

    n_people = city.n_people
    unassigned_humans = humans
    age_bins = sorted(humans.keys(), key=lambda x:x[0])
    allocated_humans = []

    log("Allocating houses ... ", logfile)
    # allocate senior residencies
    for bin, P in [[(65,69), P_COLLECTIVE_65_69], [(70,74), P_COLLECTIVE_70_74], [(75,79), P_COLLECTIVE_75_79], [(80,110), P_COLLECTIVE_80_above]]:
        for human in unassigned_humans[bin]:
            if city.rng.random() < P:
                res = city.rng.choice(city.senior_residencys, size=1).item()
                allocated_humans = _assign_household(human, res, allocated_humans)
                unassigned_humans[bin].remove(human)

    # allocate households
    n_failed_attempts, deviation = 0, 0
    while len(allocated_humans) < city.n_people and n_failed_attempts < MAX_FAILED_ATTEMPTS_ALLOWED:
        housesize = city.rng.choice(range(1,6), p=P_HOUSEHOLD_SIZE, size=1).item()
        res = city.create_location(
            specs = conf.get("LOCATION_DISTRIBUTION")["household"],
            type = "household",
            name = len(city.households)
        )

        if housesize == 1:
            humans_with_same_house, unassigned_humans, n_iters, type = find_one_human_for_solo_house(conf, city, unassigned_humans)

        elif housesize == 2:
            humans_with_same_house, unassigned_humans, n_iters, type = find_two_humans_for_house(conf, city, unassigned_humans)

        elif housesize == 3:
            humans_with_same_house, unassigned_humans, n_iters, type = find_three_humans_for_house(conf, city, unassigned_humans)

        elif housesize == 4:
            humans_with_same_house, unassigned_humans, n_iters, type = find_four_humans_for_house(conf, city, unassigned_humans)

        elif housesize == 5:
            humans_with_same_house, unassigned_humans, n_iters, type = find_five_humans_for_house(conf, city, unassigned_humans)

        # allocate if succesful
        if humans_with_same_house:
            for human in humans_with_same_house:
                allocated_humans = _assign_household(human, res, allocated_humans)
            res.allocation_type = type
            city.households.add(res)
            deviation += n_iters - 1
        else:
            n_failed_attempts += 1
            if n_failed_attempts % 100 == 0:
                log(f"Failed attempt - {n_failed_attempts}. Deviation: {deviation}. Total allocated:{len(allocated_humans)}", logfile)

    # when number of attempts exceed the limit randomly allocate remaining humans
    if n_failed_attempts >= MAX_FAILED_ATTEMPTS_ALLOWED:
        log("Deviation:{deviation}. Exceeded the maximum number of failed attempts allowed to allocate houses... trying random allocation!", logfile)

        while len(allocated_humans) < city.n_people:
            housesize = city.rng.choice(range(1,6), p=P_HOUSEHOLD_SIZE, size=1).item()
            res = city.create_location(
                specs = conf.get("LOCATION_DISTRIBUTION")["household"],
                type = "household",
                name = len(city.households)
            )

            humans_with_same_house = _sample_random_humans(unassigned_humans.keys(), unassigned_humans, city.rng, size=housesize)
            if humans_with_same_house:
                for human in humans_with_same_house:
                    allocated_humans = _assign_household(human, res, allocated_humans)
                city.households.add(res)
            else:
                log(f"(Random attempt) Could not find humans for house size {housesize}... trying other house size!! Total allocated:{len(allocated_humans)}", logfile)


    assert len(allocated_humans) == city.n_people, "assigned humans and total population do not add up"
    assert sum(len(val) for x,val in unassigned_humans.items()) == 0, "there are unassigned humans in the list"
    assert len(city.households) > 0
    # shuffle the list of humans so that the simulation is not dependent on the order of house allocation
    city.rng.shuffle(allocated_humans)
    log(f"Housing allocated with deviation:{deviation}", logfile)
    return allocated_humans

def _random_choice_tuples(tuples, rng, size, P=None):
    """
    samples `size` random elements from `tuples` with probability `P`.
    NOTE: This function work arounds the internal conversion of the elements
            in `tuples` to np.ndarray.
    Args:
        tuples (list): a list of tuples
        rng (np.random.RandomState): Random number generator
        size (int): number of elements to sample from `tuples`
        P (list): probability with which to sample. Defaults to None.

    Returns:
        list: sampled elements from tuples
    """
    total = len(tuples)
    idxs = rng.choice(range(total), size=size, p=P, replace=False)
    return [tuples[x] for x in idxs]

def _sample_couple_with_n_kids(couple_bins, unassigned_humans, rng, n=0, younger_bins=[]):
    """
    Samples a couple and n kids to be allocated to a same house
    Args:
        couple_bins (list): age bins that qualify to sample a couple
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        n (int): number of kids to sample
        younger_bins (list): age bins that qualify to sample a kid

    Returns:
        list: humans belonging to a same household (length = n + 2 if succesful else 0)
    """
    # filter candidates for couple
    min_couple_single_bins = [(x,x) for x in couple_bins if len(unassigned_humans[x]) >= 2]

    age_bins = sorted(couple_bins, key=lambda x:x[0])
    sequential_couple_bins = [(x,y) for x,y in zip(age_bins, age_bins[1:]) \
                        if len(unassigned_humans[x]) >= 1 and len(unassigned_humans[y]) >= 1]

    valid_couple_bins = min_couple_single_bins + sequential_couple_bins

    # filter candidates for kids
    valid_younger_bins =[x for x in younger_bins if len(unassigned_humans[x]) >= 1]
    all_kids = [(y,x) for x in valid_younger_bins for y in unassigned_humans[x]]

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

def _sample_random_humans(age_bins, unassigned_humans, rng, size):
    """
    Samples humans randomly to be put in the same house

    Args:
        age_bins (list): age bins that qualify to sample humans
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        size (int): number of humans to sample

    Returns:
        list: humans belonging to a same household (length = size if succesful else 0)
    """
    all_humans = [(y,x) for x in age_bins for y in unassigned_humans[x]]

    if len(all_humans) < size:
        return []

    sampled = _random_choice_tuples(all_humans, rng, size=size)
    humans = []
    for human, bin in sampled:
        unassigned_humans[bin].remove(human)
        humans.append(human)

    assert len(humans) == size, "number of humans sampled doesn't equal the expected size"
    return humans

def _sample_single_parent_n_kids(valid_older_bins, valid_younger_bins, unassigned_humans, rng, n=1):
    """
    Samples a single parent and `n` kids randomly to be put in the same house

    Args:
        valid_older_bins (list): age bins that qualify to sample a single parent
        valid_younger_bins (list): age bins that qualify to sample children
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        n (int): number of kids to sample

    Returns:
        list: humans belonging to a same household (length = n + 1 if succesful else 0)
    """

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

def find_one_human_for_solo_house(conf, city, unassigned_humans):
    """
    Finds one human to be allocated to a household of size 1.
    NOTE: Maximum number of tries are 10 failing which an empty list is returned which triggers next sampling

    Args:
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        human (list): human sampled to live alone
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        n_iters (int): number of iterations it took to find this allocation
        type (tuple): type of allocation (str) and census probability of this allocation type (float)
    """
    P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1 = conf['P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1']
    P_AGE_SOLO = [x[2] for x in P_AGE_SOLO_DWELLERS_GIVEN_HOUSESIZE_1]

    n_iters, human = 0, []
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    while n_iters < 10:
        n_iters += 1
        valid_idx = [i for i,bin in enumerate(age_bins) if len(unassigned_humans[bin]) >= 1 and P_AGE_SOLO[i] > 0]
        valid_age_bins =  [age_bins[i] for i in valid_idx]
        if valid_age_bins:
            P = [P_AGE_SOLO[i] for i in valid_idx]
            P = [i / sum(P) for i in P]
            age_bin = _random_choice_tuples(valid_age_bins, city.rng, 1, P)[0]
            human = city.rng.choice(unassigned_humans[age_bin], size=1).tolist()
            unassigned_humans[age_bin].remove(human[0])
            break

    return human, unassigned_humans, n_iters, ("solo", conf['P_HOUSEHOLD_SIZE'][0])

def find_two_humans_for_house(conf, city, unassigned_humans):
    """
    Finds two human to be allocated to a household of size 2
    NOTE: Maximum number of tries are 10 failing which an empty list is returned which triggers next sampling

    Args:
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        humans (list): humans sampled to live together
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        n_iters (int): number of iterations it took to find this allocation
        type (tuple): type of allocation (str) and census probability of this allocation type (float)
    """
    P_FAMILY_TYPE_SIZE_2 = conf['P_FAMILY_TYPE_SIZE_2']
    NORMALIZED_P_FAMILY_TYPE_SIZE_2 = conf['NORMALIZED_P_FAMILY_TYPE_SIZE_2']

    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']

    types = ["couple", "single_parent", "other-2"]
    n_iters, two_humans = 0, []
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=NORMALIZED_P_FAMILY_TYPE_SIZE_2, size=1).item()

        if type == "couple":
            valid_age_bins = [x for x in age_bins if x[0] >= MIN_AGE_COUPLE]
            two_humans = _sample_couple_with_n_kids(valid_age_bins, unassigned_humans, city.rng, n=0)

        elif type == "single_parent":
            valid_age_bins = [x for x, val in unassigned_humans.items() if len(val) >= 1]
            valid_older_bins = [x for x in valid_age_bins if MIN_AGE_SINGLE_PARENT < x[1] < MAX_AGE_SINGLE_PARENT]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            two_humans = _sample_single_parent_n_kids(valid_older_bins, valid_younger_bins, unassigned_humans, city.rng, n=1)

        elif type == "other-2":
            # (no-source) all other type of housing is resided by adults only
            valid_age_bins = [x for x in age_bins if len(unassigned_humans[x]) >= 1 and x[0] > MAX_AGE_CHILDREN]
            two_humans = _sample_random_humans(valid_age_bins, unassigned_humans, city.rng, size=2)

        if two_humans:
            break

    return two_humans, unassigned_humans, n_iters, (type, P_FAMILY_TYPE_SIZE_2[types.index(type)])

def find_three_humans_for_house(conf, city, unassigned_humans):
    """
    Finds three human to be allocated to a household of size 3
    NOTE: Maximum number of tries are 10 failing which an empty list is returned which triggers next sampling

    Args:
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        humans (list): humans sampled to live together
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        n_iters (int): number of iterations it took to find this allocation
        type (tuple): type of allocation (str) and census probability of this allocation type (float)
    """
    P_FAMILY_TYPE_SIZE_3 = conf['P_FAMILY_TYPE_SIZE_3']
    NORMALIZED_P_FAMILY_TYPE_SIZE_3 = conf['NORMALIZED_P_FAMILY_TYPE_SIZE_3']

    P_MULTIGENERATIONAL_FAMILY = conf['P_MULTIGENERATIONAL_FAMILY']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_COUPLE_WITH_CHILDREN = conf['MAX_AGE_COUPLE_WITH_CHILDREN']

    types = ["couple_with_kid", "single_parent_with_2_kids", "other-3"]
    n_iters, three_humans = 0, []
    valid_age_bins = [x for x, val in unassigned_humans.items() if len(val) >= 1]
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=NORMALIZED_P_FAMILY_TYPE_SIZE_3, size=1).item()

        if type == "couple_with_kid":
            valid_couple_bins = [x for x in valid_age_bins if MIN_AGE_COUPLE < x[0] < MAX_AGE_COUPLE_WITH_CHILDREN]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            three_humans = _sample_couple_with_n_kids(valid_couple_bins, unassigned_humans, city.rng, n=1, younger_bins=valid_younger_bins)

        elif type == "single_parent_with_2_kids":
            valid_older_bins = [x for x in valid_age_bins if MIN_AGE_SINGLE_PARENT < x[1] < MAX_AGE_SINGLE_PARENT]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            three_humans = _sample_single_parent_n_kids(valid_older_bins, valid_younger_bins, unassigned_humans, city.rng, n=2)

        elif type == "other-3":
            # (no-source) all other type of housing is resided by adults only
            valid_age_bins = [x for x in age_bins if len(unassigned_humans[x]) >= 1 and x[0] > MAX_AGE_CHILDREN]
            three_humans = _sample_random_humans(valid_age_bins, unassigned_humans, city.rng, size=3)

        if three_humans:
            break

    return three_humans, unassigned_humans, n_iters, (type, P_FAMILY_TYPE_SIZE_3[types.index(type)])

def find_four_humans_for_house(conf, city, unassigned_humans):
    """
    Finds four human to be allocated to a household of size 4
    NOTE: Maximum number of tries are 10 failing which an empty list is returned which triggers next sampling

    Args:
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        humans (list): humans sampled to live together
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        n_iters (int): number of iterations it took to find this allocation
        type (tuple): type of allocation (str) and census probability of this allocation type (float)
    """
    P_FAMILY_TYPE_SIZE_4 = conf['P_FAMILY_TYPE_SIZE_4']
    NORMALIZED_P_FAMILY_TYPE_SIZE_4 = conf['NORMALIZED_P_FAMILY_TYPE_SIZE_4']

    P_MULTIGENERATIONAL_FAMILY = conf['P_MULTIGENERATIONAL_FAMILY']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_COUPLE_WITH_CHILDREN = conf['MAX_AGE_COUPLE_WITH_CHILDREN']

    types = ["couple_with_two_kids", "single_parent_with_three_kids", "other-4"]
    n_iters, four_humans = 0, []
    valid_age_bins = [x for x, val in unassigned_humans.items() if len(val) >= 1]
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=NORMALIZED_P_FAMILY_TYPE_SIZE_4, size=1).item()
        if type == "couple_with_two_kids":
            valid_couple_bins = [x for x in valid_age_bins if MIN_AGE_COUPLE < x[0] < MAX_AGE_COUPLE_WITH_CHILDREN]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            four_humans = _sample_couple_with_n_kids(valid_couple_bins, unassigned_humans, city.rng, n=2, younger_bins=valid_younger_bins)

        elif type == "single_parent_with_three_kids":
            valid_older_bins = [x for x in valid_age_bins if MIN_AGE_SINGLE_PARENT < x[1] < MAX_AGE_SINGLE_PARENT]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            four_humans = _sample_single_parent_n_kids(valid_older_bins, valid_younger_bins, unassigned_humans, city.rng, n=3)

        elif type == "other-4":
            # (no-source) all other type of housing is resided by adults only
            valid_age_bins = [x for x in age_bins if len(unassigned_humans[x]) >= 1 and x[0] > MAX_AGE_CHILDREN]
            four_humans = _sample_random_humans(valid_age_bins, unassigned_humans, city.rng, size=4)

        if four_humans:
            break

    return four_humans, unassigned_humans, n_iters, (type, P_FAMILY_TYPE_SIZE_4[types.index(type)])

def find_five_humans_for_house(conf, city, unassigned_humans):
    """
    Finds five human to be allocated to a household of size 5
    NOTE: Maximum number of tries are 10 failing which an empty list is returned which triggers next sampling

    Args:
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        humans (list): humans sampled to live together
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        n_iters (int): number of iterations it took to find this allocation
        type (tuple): type of allocation (str) and census probability of this allocation type (float)
    """
    P_FAMILY_TYPE_SIZE_MORE_THAN_5 = conf['P_FAMILY_TYPE_SIZE_MORE_THAN_5']
    NORMALIZED_P_FAMILY_TYPE_SIZE_MORE_THAN_5 = conf['NORMALIZED_P_FAMILY_TYPE_SIZE_MORE_THAN_5']

    P_MULTIGENERATIONAL_FAMILY = conf['P_MULTIGENERATIONAL_FAMILY']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_COUPLE_WITH_CHILDREN = conf['MAX_AGE_COUPLE_WITH_CHILDREN']

    types = ["couple_with_three_kids", "single_parent_with_four_or_more_kids", "other-5"]
    n_iters, more_than_four_humans = 0, []
    valid_age_bins = [x for x, val in unassigned_humans.items() if len(val) >= 1]
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=NORMALIZED_P_FAMILY_TYPE_SIZE_MORE_THAN_5, size=1).item()
        if type == "couple_with_three_kids":
            valid_couple_bins = [x for x in valid_age_bins if MIN_AGE_COUPLE < x[0] < MAX_AGE_COUPLE_WITH_CHILDREN]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            five_humans = _sample_couple_with_n_kids(valid_couple_bins, unassigned_humans, city.rng, n=3, younger_bins=valid_younger_bins)

        elif type == "single_parent_with_four_or_more_kids":
            valid_older_bins = [x for x in valid_age_bins if MIN_AGE_SINGLE_PARENT < x[1] < MAX_AGE_SINGLE_PARENT]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            five_humans = _sample_single_parent_n_kids(valid_older_bins, valid_younger_bins, unassigned_humans, city.rng, n=4)

        elif type == "other-5":
            # (no-source) all other type of housing is resided by adults only;
            # (no-source) we consider the max size to be 5
            valid_age_bins = [x for x in age_bins if len(unassigned_humans[x]) >= 1 and x[0] > MAX_AGE_CHILDREN]
            five_humans = _sample_random_humans(valid_age_bins, unassigned_humans, city.rng, size=5)

        if five_humans:
            break

    return five_humans, unassigned_humans, n_iters, (type, P_FAMILY_TYPE_SIZE_MORE_THAN_5[types.index(type)])
