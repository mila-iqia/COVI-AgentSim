def get_humans_with_age(city, age_histogram, conf, rng, chosen_infected):
    humans = defaultdict(list)
    human_id = -1
    for age_bin, n in age_histogram.keys():
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
                age=-1,
                household=None,
                workplace=None,
                profession="",
                rho=conf.get("RHO"),
                gamma=conf.get("GAMMA"),
                infection_timestamp=city.start_time if human_id in chosen_infected else None,
                conf=self.conf
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

    n_people = city.n_people
    p_senior_residencies = conf['AGE_SENIOR_RESIDENCY_DISTRIBUTION']
    p_size = conf['HOUSE_SIZE_DISTRIBUTION']
    p_couple_and_size2 = conf["PROPORTION_COUPLE_HOUSE_SIZE_2"]
    p_singeparent_and_size2 = conf["PROPORTION_SINGLE_PARENT_HOUSE_SIZE_2"]
    p_rest_and_size2 = p_size[2] - p_couple_and_size2 - p_singeparent_and_size2
    p_type_size2 = [p_couple_and_size2, p_singeparent_and_size2, p_rest_and_size2]
    p_type_size2 = [x/sum(p_type_size2) for x in p_type_size2]
    p_age_solo_dwellers = conf['AGE_BIN_PROPORTION_SOLO_DWELLERS']
    p_type_size3 = conf['TYPE_']

    age_bins = sorted(humans.keys(), lambda x:x[0])
    remaining_humans_per_agebin = {x:len(humans[x]) for x in age_bins}

    # allocate senior residencies

    # allocate households
    while True:
        house_size = city.rng.choice(range(1,6), p=p_size, size=1)
        res = city.create_location(
            specs = conf.get("LOCATION_DISTRIBUTION")["household"],
            type = "household",
            id = len(city.households)
        )

        if house_size == 1:
            n_iters = 0
            while n_iters < 10
                age_bin = city.rng.choice(age_bins, p_age_solo_dwellers)
                if remaining_humans_per_agebin[age_bin] > 0:
                    human = city.rng.choice(humans[age_bin], size=1)
                    human.household = res
                    res.residents.append(human)
                    remaining_humans_per_agebin[age_bin] -= 1
                    break

        elif house_size == 2:
            type = city.rng.choice(["couple", "single_parent", "other"], p=p_type, size=1)
            if type == "couple":
                valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val > 2]
                valid_age_bins = [x for x in valid_age_bins if x[1] > 20]
                age_bin = city.rng.choice(valid_age_bins, size=1)
                two_humans = city.rng.choice(humans[age_bin], size=2)
                remaining_humans_per_agebin[age_bin] -= 2

            elif type == "single_parent":
                valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val > 1]

                # pick a responsible adult for the kid
                valid_older_bins = [x for x in valid_age_bins if 30 < x[1] < 60]
                older_bin = city.rng.choice(valid_older_bins, size=1)
                older_human = city.rng.choice(humans[older_bin], size=1)
                remaining_humans_per_agebin[older_bin] -= 1

                # pick a kid
                valid_younger_bins = [x for x in valid_age_bins if x[1] < 15]
                younger_bin = city.rng.choice(valid_younger_bins, size=1)
                younger_human = city.rng.choice(humans[younger_bin], size=2)
                remaining_humans_per_agebin[younger_bin] -= 1

                two_humans = [older_human, younger_human]

            else:
                valid_age_bins = [x for x, val in remaining_humans_per_agebin.items() if val > 1]
                # randomly pick two age bins
                chosen_bins = city.rng.choice(valid_age_bins, size=2)

                human1 = city.rng.choice(humans[chosen_bins[0]], size=1)
                remaining_humans_per_agebin[chosen_bin[0]] -= 1

                human2 = city.rng.choice(humans[chosen_bins[1]], size=1)
                remaining_humans_per_agebin[chosen_bin[1]] -= 1

                two_humans = [human1, human2]

            for h in two_humans:
                h.household = res
                res.residents.append(h)

        elif house_size == 3:
            type = city.rng.choice(["couple_with_kid", "single_parent_with_2_kids", "other"], p=p_type, size=1)
            if type == "couple_with_kid":
