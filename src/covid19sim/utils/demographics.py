import numpy as np

from collections import defaultdict
from copy import deepcopy

def get_humans_with_age(city, age_histogram, conf, rng, chosen_infected, human_type):
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
                has_app=False,
                name=human_id,
                age=ages[i],
                household=None,
                workplace=None,
                profession="",
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
            huamns[age_bin][i].profession = profession

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


def assign_households_to_humans(humans, city, conf):


    def _assign_household(human, res, allocated_humans):
        human.household = res
        res.residents.append(human)
        allocated_humans.append(human)
        return allocated_humans

    n_people = city.n_people
    unassigned_humans = deepcopy(humans)

    P_HOUSEHOLD_SIZE= conf['P_HOUSEHOLD_SIZE']
    P_COLLECTIVE_75_79 = conf['P_COLLECTIVE_75_79']
    P_COLLECTIVE_70_74 = conf['P_COLLECTIVE_70_74']
    P_COLLECTIVE_80_above = conf['P_COLLECTIVE_80_above']
    P_AGE_SOLO_DWELLLERS_GIVEN_HOUSESIZE_1 = conf['P_AGE_SOLO_DWELLLERS_GIVEN_HOUSESIZE_1']
    P_AGE_SOLO = [x[2] for x in P_AGE_SOLO_DWELLLERS_GIVEN_HOUSESIZE_1]

    age_bins = sorted(humans.keys(), key=lambda x:x[0])
    remaining_humans_per_agebin = {x:len(humans[x]) for x in age_bins}

    allocated_humans = []
    # allocate senior residencies
    for bin, P in [[(70,74), P_COLLECTIVE_70_74], [(75,79), P_COLLECTIVE_75_79], [(80,110), P_COLLECTIVE_80_above]]:
        for human in unassigned_humans[bin]:
            if city.rng.random() < P:
                res = city.rng.choice(city.senior_residencys, size=1).item()
                allocated_humans = _assign_household(human, res, allocated_humans)
                remaining_humans_per_agebin[bin] -= 1
                unassigned_humans[bin].remove(human)

    # allocate households
    while True:
        housesize = city.rng.choice(range(1,6), p=P_HOUSEHOLD_SIZE, size=1)
        res = city.create_location(
            specs = conf.get("LOCATION_DISTRIBUTION")["household"],
            type = "household",
            name = len(city.households)
        )

        if housesize == 1:
            human, remaining_humans_per_agebin, unassigned_humans, n_iters = find_one_human_for_solo_house(conf, city, P_AGE_SOLO, remaining_humans_per_agebin, unassigned_humans)
            if human:
                allocated_humans = _assign_household(human, res, allocated_humans)
            else:
                print("Could not find a human for house size 1... trying other house size!!")

        elif housesize == 2:
            two_humans, remaining_humans_per_agebin, unassigned_humans, n_iters = find_two_humans_for_house(conf, city, remaining_humans_per_agebin, unassigned_humans)
            if two_humans:
                for h in two_humans:
                    allocated_humans = _assign_household(h, res, allocated_humans)
            else:
                print("Could not find two humans for house size 2... trying other house size!!")

        elif housesize == 3:
            three_humans, remaining_humans_per_agebin, unassigned_humans, n_iters = find_three_humans_for_house(conf, city, remaining_humans_per_agebin, unassigned_humans)
            if three_humans:
                for h in three_humans:
                    allocated_humans = _assign_household(h, res, allocated_humans)
            else:
                print("Could not find three humans for house size 3... trying other house size!!")

        elif housesize == 4:
            continue
            four_humans, remaining_humans_per_agebin, unassigned_humans, n_iters = find_four_humans_for_house(conf, city, remaining_humans_per_agebin, unassigned_humans)
            if four_humans:
                for h in four_humans:
                    allocated_humans = _assign_household(h, res, allocated_humans)
            else:
                print("Could not find four humans for house size 4... trying other house size!!")

        elif housesize == 5:
            continue
            humans, remaining_humans_per_agebin, unassigned_humans, n_iters = find_more_than_four_humans_for_house(conf, city, remaining_humans_per_agebin, unassigned_humans)
            if humans:
                for h in humans:
                    allocated_humans = _assign_household(h, res, allocated_humans)
            else:
                print("Could not find more than 4 humans for house size more than 4... trying other house size!!")

    assert sum(remaining_humans_per_agebin.values()) == 0, "there are humans who do not have a house..."
    assert len(allocated_humans) == city.n_people, "assigned humans and total population do not add up"
    assert sum(len(val) for x,val in unassigned_humans.items()) == 0, "there are unassigned humans in the list"
    city.rng.shuffle(allocated_humans)
    return allocated_humans, city.households


def _random_choice_tuples(tuples, rng, size, P=None):
    total = len(tuples)
    idxs = rng.choice(range(total), size=size, p=P)
    return [tuples[x] for x in idxs]


def find_one_human_for_solo_house(conf, city, P_AGE_SOLO, remaining_humans_per_agebin, unassigned_humans):
    n_iters = 0
    human = []
    while n_iters < 10:
        n_iters += 1
        valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val >= 1]
        age_bin = _random_choice_tuples(valid_age_bins, city.rng, 1, P_AGE_SOLO)[0]
        human = city.rng.choice(unassigned_humans[age_bin], size=1).item()
        remaining_humans_per_agebin[age_bin] -= 1
        unassigned_humans[age_bin].remove(human)
        break
    return human, remaining_humans_per_agebin, unassigned_humans, n_iters

def find_two_humans_for_house(conf, city, remaining_humans_per_agebin, unassigned_humans):
    P_FAMILY_TYPE_SIZE_2 = conf['P_FAMILY_TYPE_SIZE_2']
    NORMALIZED_P_FAMILY_TYPE_SIZE_2 = [x/sum(P_FAMILY_TYPE_SIZE_2) for x in P_FAMILY_TYPE_SIZE_2]
    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']

    types = ["couple", "single_parent", "other"]
    n_iters = 0
    two_humans = []
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=NORMALIZED_P_FAMILY_TYPE_SIZE_2, size=1)
        if type == "couple":
            valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val >= 2]
            valid_age_bins = [x for x in valid_age_bins if x[0] >= MIN_AGE_COUPLE]
            if valid_age_bins:
                age_bin = _random_choice_tuples(valid_age_bins, city.rng, size=1)[0]
                two_humans = city.rng.choice(unassigned_humans[age_bin], size=2, replace=False).tolist()
                for h in two_humans:
                    remaining_humans_per_agebin[age_bin] -= 1
                    unassigned_humans[age_bin].remove(h)
                break

        elif type == "single_parent":
            valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val >= 1]
            # pick a responsible adult for the kid
            valid_older_bins = [x for x in valid_age_bins if MIN_AGE_SINGLE_PARENT < x[1] < MAX_AGE_SINGLE_PARENT]
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]

            if len(valid_older_bins) > 0 and len(valid_younger_bins) > 0:
                older_bin = _random_choice_tuples(valid_older_bins, city.rng, size=1)[0]
                older_human = city.rng.choice(unassigned_humans[older_bin], size=1).item()
                remaining_humans_per_agebin[older_bin] -= 1
                unassigned_humans[older_bin].remove(older_human)

                # pick a kid
                valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
                younger_bin = _random_choice_tuples(valid_younger_bins, city.rng, size=1)[0]
                younger_human = city.rng.choice(unassigned_humans[younger_bin], size=1).item()
                remaining_humans_per_agebin[younger_bin] -= 1
                unassigned_humans[younger_bin].remove(younger_human)

                two_humans = [older_human, younger_human]
                break

        else:
            valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val >= 1]
            if len(valid_age_bins) >= 2:
                # randomly pick two age bins
                chosen_bins = _random_choice_tuples(valid_age_bins, city.rng, size=2)

                human1 = city.rng.choice(unassigned_humans[chosen_bins[0]], size=1).item()
                remaining_humans_per_agebin[chosen_bins[0]] -= 1
                unassigned_humans[chosen_bins[0]].remove(human1)

                human2 = city.rng.choice(unassigned_humans[chosen_bins[1]], size=1).item()
                remaining_humans_per_agebin[chosen_bins[1]] -= 1
                unassigned_humans[chosen_bins[1]].remove(human2)

                two_humans = [human1, human2]
                break

    return two_humans, remaining_humans_per_agebin, unassigned_humans, n_iters

def find_three_humans_for_house(conf, city, remaining_humans_per_agebin, unassigned_humans):
    P_FAMILY_TYPE_SIZE_3 = conf['P_FAMILY_TYPE_SIZE_3']
    NORMALIZED_P_FAMILY_TYPE_SIZE_3 = [x/sum(P_FAMILY_TYPE_SIZE_3) for x in P_FAMILY_TYPE_SIZE_3]
    P_MULTIGENERATIONAL_FAMILY = conf['P_MULTIGENERATIONAL_FAMILY']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MIN_AGE_COUPLE = conf['MIN_AGE_COUPLE']
    MIN_AGE_SINGLE_PARENT = conf['MIN_AGE_SINGLE_PARENT']
    MAX_AGE_SINGLE_PARENT = conf['MAX_AGE_SINGLE_PARENT']
    MAX_AGE_COUPLE_WITH_CHILDREN = conf['MAX_AGE_COUPLE_WITH_CHILDREN']

    types = ["couple_with_kid", "single_parent_with_2_kids", "other"]
    n_iters = 0
    three_humans = []
    valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val >= 1]
    age_bins = sorted(remaining_humans_per_agebin.keys(), key=lambda x:x[0])
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=NORMALIZED_P_FAMILY_TYPE_SIZE_3, size=1)
        if type == "couple_with_kid":
            valid_couple_bins = [x for x in valid_age_bins if MIN_AGE_COUPLE < x[0] < MAX_AGE_COUPLE_WITH_CHILDREN ]
            min_two_couple_bins = [x for x in valid_couple_bins if remaining_humans_per_agebin[x] >= 2]
            sequential_couple_bins = [(x,y) for x,y in zip(age_bins, age_bins[1:]) \
                                if remaining_humans_per_agebin[x] >= 1 and remaining_humans_per_agebin[y] >= 1]

            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            if (min_two_couple_bins or sequential_couple_bins) and valid_younger_bins:
                pass
                break

        elif type == "single_parent_with_2_kids":
            valid_younger_bins = [x for x in valid_age_bins if x[1] < MAX_AGE_CHILDREN]
            valid_older_bins = [x for x in valid_age_bins if MIN_AGE_SINGLE_PARENT < x[1] < MAX_AGE_SINGLE_PARENT]
            if valid_older_bins and valid_younger_bins:
                for i  in range(2):
                    bin = _random_choice_tuples(valid_younger_bins, city.rng, size=1)[0]
                    three_humans.append(city.rng.choice(unassigned_humans[bin], size=1).item())

                older_bin = _random_choice_tuples(valid_older_bins, city.rng, size=1)[0]
                older_human = city.rng.choice(unassigned_humans[older_bin], size=1).item()
                three_humans.append(older_human)

                remaining_humans_per_agebin[older_bin] -= 1
                unassigned_humans[older_bin].remove(older_human)
                break

        elif type == "other":
            if sum(remaining_humans_per_agebin[x] for x in valid_age_bins) >= 3:
                _all_humans = [y for x in valid_age_bins for y in unassigned_humans[x]]
                three_humans = city.rng.choice(_all_humans, size=3).tolist()
                break

    return three_humans, remaining_humans_per_agebin, unassigned_humans, n_iters


def find_four_humans_for_house(conf, remaining_humans_per_agebin, unassigned_humans):
    P_FAMILY_TYPE_SIZE_4 = [0.096, 0.008, 0.008]
    P_MULTIGENERATIONAL_FAMILY = 0.015
    MAX_AGE_CHILDREN = 15
    MIN_AGE_COUPLE = 20

    types = ["couple_with_two_kids", "single_parent_with_three_kids", "other"]
    n_iters = 0
    two_humans = []
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=P_FAMILY_TYPE_SIZE_4, size=1)
        if type == "couple_with_two_kids":
            pass

        elif type == "single_parent_with_three_kids":
            pass

        elif type == "other":
            pass


def find_more_than_four_humans_for_house(conf, remaining_humans_per_agebin, unassigned_humans):
    P_FAMILY_TYPE_SIZE_MORE_THAN_5 = [0.044, 0.003, 0.014]
    P_MULTIGENERATIONAL_FAMILY = 0.015
    MAX_AGE_CHILDREN = 15
    MIN_AGE_COUPLE = 20

    types = ["couple_with_three_kids", "single_parent_with_four_or_more_kids", "other"]
    n_iters = 0
    two_humans = []
    while n_iters < 10:
        n_iters += 1
        type = city.rng.choice(types, p=P_FAMILY_TYPE_SIZE_MORE_THAN_5, size=1)
        if type == "couple_with_three_kids":
            pass

        elif type == "single_parent_with_four_or_more_kids":
            pass

        elif type == "other":
            pass
