"""
Functions to intialize a synthetic population using constants in configuration file.
"""
import numpy as np
import math
from copy import deepcopy

from collections import defaultdict
from covid19sim.utils.utils import log, relativefreq2absolutefreq, _get_random_area, _random_choice
from covid19sim.utils.constants import AGE_BIN_WIDTH_5
from covid19sim.locations.location import Location, Household, School, WorkplaceA, WorkplaceB
from covid19sim.locations.hospital import Hospital
from covid19sim.human import Human

MAX_FAILED_ATTEMPTS_ALLOWED = 10000

class HouseType(object):
    """
    Class to hold the attributes of a house.
    Args:
        type (str): type of living arrangement - "couple", "single_parent", "other"
        n_kids (int): number of kids in the house
        n_humans (int): total number of humans in the house
        probability (float): census probability of sampling this type of type
        multigenerational (bool): whether grandparents live with grandchildren
    """
    def __init__(self, living_arrangement, n_kids, n_humans, probability, multigenerational=False):

        assert living_arrangement in ["couple", "single_parent", "other", "solo"], "not a valid living arrangement"
        assert n_humans > n_kids, "kids can't live alone"

        self.random = False # only set when random allocation is required at the end
        self.living_arrangement = living_arrangement
        self.n_kids = n_kids
        self.n_humans = n_humans
        self.probability = probability

        if  self.living_arrangement == "other":
            basestr = f"other_{self.n_humans}"
        elif self.living_arrangement == "solo":
            basestr = "solo"
        else:
            basestr = f"{self.living_arrangement}_with_{self.n_kids}_kids"
        self.basestr = basestr

        self.n_people_generation = [0, 0, 0]
        if self.n_kids:
            self.n_people_generation = [0, n_humans - n_kids, n_kids]

    def __repr__(self):
        if self.multigenerational:
            n_grandparents, n_parents, n_kids = self.n_people_generation
            return f"{self.basestr} Grandparents:{n_grandparents} Parents:{n_parents} Kids:{n_kids}"
        return self.basestr

    @property
    def multigenerational(self):
        return self.n_people_generation[0] > 0 and self.n_people_generation[2] > 0

    def set_generations(self, n_grandparents, n_parents, n_kids):
        """
        """
        assert self.living_arrangement == "other", "setting generation in non-other living arrangement is not valid"
        self.n_kids = n_kids
        self.n_people_generation = [n_grandparents, n_parents, n_kids]
        assert sum(self.n_people_generation) == self.n_humans, "size does not match"

def get_humans_with_age(city, age_histogram, conf, rng, chosen_infected):
    """
    Creats human objects corresponding to the numbers in `age_histogram`.

    Args:
        city (covid19sim.location.City): simulator's city object
        age_histogram (dict): a dictionary with keys as age bins (a tuple) and values as number of humans in that bin (int)
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator
        chosen_infected (set): human ids that are initialized to be infected

    Returns:
        dict: keys are age bins (tuple) and values are a list of human_type objects
    """
    # TODO - parallelize this
    humans = defaultdict(list)
    human_id = -1
    for age_bin, n in age_histogram.items():
        # sample age of humans before initialization
        ages = city.rng.randint(low=age_bin[0], high=age_bin[1]+1, size=n)

        for i in range(n):
            human_id += 1
            humans[age_bin].append(Human(
                env=city.env,
                city=city,
                rng=np.random.RandomState(rng.randint(2 ** 16)),
                name=human_id,
                age=ages[i],
                infection_timestamp=city.start_time if human_id in chosen_infected else None,
                conf=conf
                ))

    return humans

def create_locations_and_assign_workplace_to_humans(humans, city, conf, logfile=None):
    """
    Sets up locations in the `city` and assigns `humans` to their workplace (a type of location).

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        logfile (str): filepath where the console output and final tracked metrics will be logged.

    Returns:
        humans (list): a list of `human`s with assigned workplaces
        city (covid19sim.locations.city.City): city object containing the locations and humans in it.
    """
    MIN_WORKING_AGE = conf['MIN_WORKING_AGE']
    # P_EMPLOYED_BY_AGEGROUP = conf['P_EMPLOYED_BY_AGE']

    # create schools
    humans, city = _build_and_allocate_schools(humans, city, conf, city.rng)

    # allocate seniors to social common room as their workplace and allocate some nurses to these senior residences
    humans, city = _assign_senior_residences(humans, city, conf, city.rng)

    # allocate doctors and nurses to hospitals
    humans, city = _build_and_allocate_hospitals(humans, city, conf, city.rng)

    # create stores and miscs and allocate some people as workers there
    humans, city = _build_and_allocate_workplace_type_A("STORE", humans, city, conf, city.rng)
    humans, city = _build_and_allocate_workplace_type_A("MISC", humans, city, conf, city.rng)

    # allocate rest of the work force to workplaces
    humans, city = _build_and_allocate_workplace_type_B(humans, city, conf, city.rng)

    # workplace - a typical workplace comprises of employed people coming together for some duration
    # during this duration they interact with their colleagues with `PREFERENTIAL_ATTACHEMENT_FACTOR` as the probability of
    # interacting with previously known colleagues
    # In order to obtain the contact pattern, allocate people as per their age groups.
    unassigned_humans = [human for human in humans if human.workplace is None]
    for human in unassigned_humans:
        human.does_not_work = True

    # create parks
    PARK_PROPORTION_AREA = conf['PARK_PROPORTION_AREA']
    park = Location(
                    env=city.env,
                    rng=np.random.RandomState(city.rng.randint(2 ** 16)),
                    conf=conf,
                    name=f"PARK:0",
                    location_type="PARK",
                    lat=city.rng.randint(*city.x_range),
                    lon=city.rng.randint(*city.y_range),
                    area=city.total_area*PARK_PROPORTION_AREA,
                    capacity=None
                )
    city.parks = [park]
    return humans, city

def _build_and_allocate_workplace_type_B(humans, city, conf, rng, logfile=None):
    """
    Initializes the locations of type `type` that falls in type B.
    Refer the detailed explanation of docs/workplace_modeling.md

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator
        logfile (str): filepath where the console output and final tracked metrics will be logged.

    Returns:
        humans (list): a list of `human`s. some of them are assigned workplace of type A
        city (covid19sim.locations.city.City): city object containing the locations and humans in it.
    """

    AVERAGE_N_EMPLOYEES_PER_WORKPLACE = conf['AVERAGE_N_EMPLOYEES_PER_WORKPLACE']
    P_EMPLOYEES_1_4_PER_WORKPLACE = conf['P_EMPLOYEES_1_4_PER_WORKPLACE']
    P_EMPLOYEES_5_99_PER_WORKPLACE = conf['P_EMPLOYEES_5_99_PER_WORKPLACE']
    P_EMPLOYEES_100_499_PER_WORKPLACE = conf['P_EMPLOYEES_100_499_PER_WORKPLACE']
    P_EMPLOYEES_500_above_PER_WORKPLACE = conf['P_EMPLOYEES_500_above_PER_WORKPLACE']

    MIN_WORKING_AGE = conf['MIN_WORKING_AGE']
    MAX_WORKING_AGE = conf['MAX_WORKING_AGE']

    ASSORTATIVITY_STRENGTH = conf['WORKPLACE_ASSORTATIVITY_STRENGTH']

    # If unemployment need to be modeled, filter out unemployed population here
    valid_worker_bins = _get_valid_bins(AGE_BIN_WIDTH_5, min_age=MIN_WORKING_AGE, max_age=MAX_WORKING_AGE)
    potential_workers = defaultdict(list)
    for human in humans:
        if human.workplace is None and human.age_bin_width_5.bin in valid_worker_bins:
            potential_workers[human.age_bin_width_5.bin].append(human)


    CAPACITIES = [(1, 4), (5, 99), (100, 499), (500, 750)]
    P_WORKPLACES = [
        P_EMPLOYEES_1_4_PER_WORKPLACE,
        P_EMPLOYEES_5_99_PER_WORKPLACE,
        P_EMPLOYEES_100_499_PER_WORKPLACE,
        P_EMPLOYEES_500_above_PER_WORKPLACE
    ]
    P_WORKPLACES = np.array(P_WORKPLACES)
    P_WORKPLACES /= P_WORKPLACES.sum()

    # create workplaces
    idx, workplaces = -1, []
    while sum(len(val) for val in potential_workers.values()) > 0:
        idx += 1
        _capacity = _random_choice(CAPACITIES, size=1, P=P_WORKPLACES, rng=rng)[0]
        capacity = rng.randint(*_capacity)
        workplace = WorkplaceB(
                    env=city.env,
                    rng=np.random.RandomState(city.rng.randint(2 ** 16)),
                    conf=conf,
                    name=f"WORKPLACE{_capacity}:{idx}",
                    location_type="WORKPLACE",
                    lat=rng.randint(*city.x_range),
                    lon=rng.randint(*city.y_range),
                    area=city.total_area,
                    capacity=capacity
                    )
        workplaces.append(workplace)

        # sample workers for this location
        p_bin = np.array([len(potential_workers[x]) for x in valid_worker_bins])
        while workplace.n_workers < capacity and p_bin.sum() > 0:
            p = p_bin / p_bin.sum()
            bin = _random_choice(valid_worker_bins, rng=rng, size=1, P=p)[0]
            worker = rng.choice(potential_workers[bin], size=1).item()
            #
            worker.assign_workplace(workplace)
            workplace.assign_worker(worker)
            potential_workers[bin].remove(worker)

            # reassign probabilities
            idx = valid_worker_bins.index(bin)
            p_bin[idx] += ASSORTATIVITY_STRENGTH

            if idx < len(valid_worker_bins) - 1:
                p_bin[idx+1] += ASSORTATIVITY_STRENGTH

            if idx > 0:
                p_bin[idx-1] += ASSORTATIVITY_STRENGTH

            p_bin[[i for i, bin in enumerate(valid_worker_bins) if len(potential_workers[bin]) == 0]] = 0

    city.workplaces = workplaces
    return humans, city

def _build_and_allocate_workplace_type_A(type, humans, city, conf, rng, logfile=None):
    """
    Initializes the locations of type `type` that falls in type A.
    Refer the detailed explanation of docs/workplace_modeling.md

    Args:
        type (str): type of location to initialize. either of "STORE" or "MISC"
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator
        logfile (str): filepath where the console output and final tracked metrics will be logged.

    Returns:
        humans (list): a list of `human`s. some of them are assigned workplace of type A
        city (covid19sim.locations.city.City): city object containing the locations and humans in it.
    """
    assert type in ["STORE", "MISC"], "Unknown type A workplace"

    AVERAGE_N_EMPLOYEES_PER_TYPE = conf[f'AVERAGE_N_EMPLOYEES_PER_{type}']
    N_TYPES_PER_1K_PEOPLE = conf[f'N_{type}_PER_1K_PEOPLE']
    TYPE_PROPORTION_AREA = conf[f'{type}_PROPORTION_AREA']

    MIN_WORKING_AGE = conf['MIN_WORKING_AGE']
    MAX_WORKING_AGE = conf['MAX_WORKING_AGE']

    # number of locations needed
    n_locations = math.ceil(city.n_people *  N_TYPES_PER_1K_PEOPLE / 1000)

    # calculate area for each location
    area = _get_random_area(n_locations, TYPE_PROPORTION_AREA * city.total_area, rng)

    ## create them with some capacity
    WORKING_BINS = _get_valid_bins(AGE_BIN_WIDTH_5, min_age=MIN_WORKING_AGE, max_age=MAX_WORKING_AGE)
    potential_workers = [human for human in humans if human.workplace is None and human.age_bin_width_5.bin in WORKING_BINS]
    idx = -1
    locations = []
    for i in range(n_locations):
        idx += 1
        location = WorkplaceA(
                        env=city.env,
                        rng=np.random.RandomState(rng.randint(2 ** 16)),
                        conf=conf,
                        name=f"{type}:{idx}",
                        location_type=type,
                        lat=rng.randint(*city.x_range),
                        lon=rng.randint(*city.y_range),
                        area=area[idx],
                        capacity=AVERAGE_N_EMPLOYEES_PER_TYPE
                    )
        locations.append(location)

        # sample workers for this location
        while location.n_workers < AVERAGE_N_EMPLOYEES_PER_TYPE and len(potential_workers) > 0:
            worker = rng.choice(potential_workers, size=1).item()
            potential_workers.remove(worker)
            worker.assign_workplace(location)
            location.assign_worker(worker)

    if type == "STORE":
        city.stores = locations
    else:
        city.miscs = locations

    return humans, city

def _build_and_allocate_hospitals(humans, city, conf, rng):
    """
    Builds hospitals as per the configuration file.
    Assigns doctors and nurses to it.

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator

    Returns:
        humans (list): a list of `human`s. some of them are assigned workplace as one of the created hospitals
        city (covid19sim.locations.city.City): city object containing the hospitals and humans
    """

    HOSPITAL_PROPORTION_AREA = conf['HOSPITAL_PROPORTION_AREA']

    N_DOCTOR_PER_100K_PEOPLE = conf['N_DOCTOR_PER_100K_PEOPLE']
    N_HOSPITALS_PER_100K_PEOPLE = conf['N_HOSPITALS_PER_100K_PEOPLE']
    HOSPITAL_BEDS_PER_1K_PEOPLE = conf['HOSPITAL_BEDS_PER_1K_PEOPLE']
    ICU_BEDS_PER_1K_PEOPLE = conf['ICU_BEDS_PER_1K_PEOPLE']

    NURSE_TO_DOCTOR_RATIO = conf['NURSE_TO_DOCTOR_RATIO']

    ICU_BEDS_OCCUPANCY = conf['ICU_BEDS_OCCUPANCY']
    HOSPITAL_BEDS_OCCUPANCY = conf['HOSPITAL_BEDS_OCCUPANCY']

    MIN_AGE_HEALTHCARE_WORKER = conf['MIN_AGE_HEALTHCARE_WORKER']
    MAX_AGE_HEALTHCARE_WORKER = conf['MAX_AGE_HEALTHCARE_WORKER']

    #
    n_doctors = math.ceil(city.n_people * N_DOCTOR_PER_100K_PEOPLE / 100000)
    n_nurses = math.ceil(NURSE_TO_DOCTOR_RATIO * n_doctors)
    n_icu_beds = math.ceil(city.n_people * ICU_BEDS_PER_1K_PEOPLE / 1000)
    n_hospital_beds = math.ceil(city.n_people * HOSPITAL_BEDS_PER_1K_PEOPLE / 1000)

    # create one big hospital and control interactions there
    area = _get_random_area(1, HOSPITAL_PROPORTION_AREA * city.total_area, rng)
    hospital = Hospital(
                    env=city.env,
                    rng=np.random.RandomState(city.rng.randint(2 ** 16)),
                    conf=conf,
                    name=f"HOSPITAL:0",
                    lat=rng.randint(*city.x_range),
                    lon=rng.randint(*city.y_range),
                    area=area[0],
                    hospital_bed_capacity=n_hospital_beds,
                    icu_bed_capacity=n_icu_beds,
                )


    # select and allocate doctors and nurses to this hospital
    WORKING_BINS = _get_valid_bins(AGE_BIN_WIDTH_5, min_age=MIN_AGE_HEALTHCARE_WORKER, max_age=MAX_AGE_HEALTHCARE_WORKER)
    doctors_and_nurses = [human for human in humans if human.workplace is None and human.age_bin_width_5.bin in WORKING_BINS]
    doctors = _random_choice(doctors_and_nurses, size=n_doctors, replace=False, rng=rng, catch_size_error=True)
    for doctor in doctors:
        doctor.assign_workplace(hospital)
        hospital.assign_worker(doctor, doctor=True)
        doctors_and_nurses.remove(doctor)

    nurses = _random_choice(doctors_and_nurses, size=n_nurses, replace=False, rng=rng, catch_size_error=True)
    for nurse in nurses:
        nurse.assign_workplace(hospital)
        hospital.assign_worker(nurse, doctor=False)
        doctors_and_nurses.remove(nurse)

    city.hospitals = [hospital]
    return humans, city

def _assign_senior_residences(humans, city, conf, rng):
    """
    Assigns social common rooms as a workplace for seniors to socialize.
    Assigns nurses to senior residences.

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator

    Returns:
        humans (list): a list of `human`s. some of them are assigned social common room as a workplace
        city (covid19sim.locations.city.City): city object containing the locations and humans in it.
    """
    RESIDENT_TO_STAFF_RATIO = conf['RESIDENT_TO_STAFF_RATIO']
    MIN_AGE_HEALTHCARE_WORKER = conf['MIN_AGE_HEALTHCARE_WORKER']
    MAX_AGE_HEALTHCARE_WORKER = conf['MAX_AGE_HEALTHCARE_WORKER']

    senior_residents = [human for human in humans if human.household.location_type == "SENIOR_RESIDENCE"]
    n_senior_residents = len(senior_residents)

    # assign social common room as workplace for seniors
    for senior in senior_residents:
        assert senior.workplace is None, "senior already has a common room"
        senior.assign_workplace(senior.household.social_common_room)

    # assign nurses for senior residences to look after seniors
    WORKING_BINS = _get_valid_bins(AGE_BIN_WIDTH_5, min_age=MIN_AGE_HEALTHCARE_WORKER, max_age=MAX_AGE_HEALTHCARE_WORKER)
    potential_nurses = [human for human in humans if human.workplace is None]
    for senior_residence in city.senior_residences:
        senior_residence.n_nurses = math.ceil(len(senior_residence.residents) / RESIDENT_TO_STAFF_RATIO)
        nurses = _random_choice(potential_nurses, size=senior_residence.n_nurses, replace=False, rng=rng, catch_size_error=True)
        for nurse in nurses:
            nurse.assign_workplace(senior_residence.social_common_room)

    return humans, city

def _build_and_allocate_schools(humans, city, conf, rng):
    """
    Builds schools and allocates kids to them as their workplace.

    Args:
        humans (dict): keys are age bins (tuple) and values are a list of covid19sim.Human object of that age
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator

    Returns:
        humans (list): a list of `human`s with proper allocation of schools to those in relevant age groups
        city (covid19sim.locations.city.City): city object containing the locations and humans in it.
    """
    N_STUDENTS_PER_SCHOOL_2_4 = conf['N_STUDENTS_PER_SCHOOL_2_4']
    N_STUDENTS_PER_SCHOOL_4_5 = conf['N_STUDENTS_PER_SCHOOL_4_5']
    N_STUDENTS_PER_SCHOOL_5_12 = conf['N_STUDENTS_PER_SCHOOL_5_12']
    N_STUDENTS_PER_SCHOOL_12_17 = conf['N_STUDENTS_PER_SCHOOL_12_17']
    N_STUDENTS_PER_SCHOOL_17_29 = conf['N_STUDENTS_PER_SCHOOL_17_29']

    P_SCHOOL_FOR_AGE_17_19 = conf['P_SCHOOL_FOR_AGE_17_19']
    P_SCHOOL_FOR_AGE_19_24 = conf['P_SCHOOL_FOR_AGE_19_24']
    P_SCHOOL_FOR_AGE_25_29 = conf['P_SCHOOL_FOR_AGE_25_29']

    P_STUDENT_4_5 = conf['P_STUDENT_4_5']
    P_STUDENT_2_4 = conf['P_STUDENT_2_4']

    STUDENT_TEACHER_RATIO_SCHOOL_2_4 = conf['STUDENT_TEACHER_RATIO_SCHOOL_2_4']
    STUDENT_TEACHER_RATIO_SCHOOL_4_5 = conf['STUDENT_TEACHER_RATIO_SCHOOL_4_5']
    STUDENT_TEACHER_RATIO_SCHOOL_5_12 = conf['STUDENT_TEACHER_RATIO_SCHOOL_5_12']
    STUDENT_TEACHER_RATIO_SCHOOL_12_17 = conf['STUDENT_TEACHER_RATIO_SCHOOL_12_17']
    STUDENT_TEACHER_RATIO_SCHOOL_17_29 = conf['STUDENT_TEACHER_RATIO_SCHOOL_17_29']


    SCHOOL_PROPORTION_AREA = conf['SCHOOL_PROPORTION_AREA']

    #
    kids_2_4 = [human for human in humans if 2 <= human.age < 4 and rng.uniform() < P_STUDENT_2_4]
    kids_4_5 = [human for human in humans if 4 <= human.age < 5 and rng.uniform() < P_STUDENT_4_5]
    kids_5_12 = [human for human in humans if 5 <= human.age <= 12]
    kids_12_17 = [human for human in humans if 12 < human.age <= 17]
    kids_17_29 = [human for human in humans if 17 < human.age <= 29]

    #
    n_schools_2_4 = math.ceil(len(kids_2_4) / N_STUDENTS_PER_SCHOOL_2_4)
    n_schools_4_5 = math.ceil(len(kids_4_5) / N_STUDENTS_PER_SCHOOL_4_5 )
    n_schools_5_12 = math.ceil(len(kids_5_12) / N_STUDENTS_PER_SCHOOL_5_12)
    n_schools_12_17 = math.ceil(len(kids_12_17) / N_STUDENTS_PER_SCHOOL_5_12)
    n_schools_17_29 = math.ceil(len(kids_17_29) / N_STUDENTS_PER_SCHOOL_17_29)
    n_schools = n_schools_2_4 + n_schools_4_5 + n_schools_5_12 + n_schools_12_17 + n_schools_17_29

    area = _get_random_area(n_schools, SCHOOL_PROPORTION_AREA * city.total_area, rng)
    area_idx = -1

    n_schools = 0

    # 2-4 yo (daycares)
    schools_for_2_4 = _build_schools_and_enroll_students(city, conf, kids_2_4, n_schools_2_4, "(2-4)", area[:n_schools_2_4], rng, N_STUDENTS_PER_SCHOOL_2_4)
    n_schools += n_schools_2_4

    # 4-5 yo
    schools_for_4_5 = _build_schools_and_enroll_students(city, conf, kids_4_5, n_schools_4_5, "(4-5)", area[n_schools:n_schools + n_schools_4_5], rng, N_STUDENTS_PER_SCHOOL_4_5)
    n_schools += n_schools_4_5

    # 5-12
    schools_for_5_12 = _build_schools_and_enroll_students(city, conf, kids_5_12, n_schools_5_12, "(5-12)", area[n_schools:n_schools + n_schools_5_12], rng, N_STUDENTS_PER_SCHOOL_5_12)
    n_schools += n_schools_5_12

    # 12-17
    schools_for_12_17 = _build_schools_and_enroll_students(city, conf, kids_12_17, n_schools_12_17, "(12-17)", area[n_schools:n_schools + n_schools_12_17], rng, N_STUDENTS_PER_SCHOOL_12_17)
    n_schools += n_schools_12_17

    # 17-29
    rng.shuffle(kids_17_29)
    schools_for_17_29 = []
    for i in range(n_schools_17_29):
        area_idx += 1
        school = _build_school(city, conf, f"SCHOOL(17-29):{i}", area[area_idx], rng)
        schools_for_17_29.append(school)
        enrolled_17_29 = 0
        while school.n_students < N_STUDENTS_PER_SCHOOL_17_29 and len(kids_17_29) > 0:
            kid = kids_17_29.pop()
            if 17 < kid.age <= 19 and rng.uniform() < P_SCHOOL_FOR_AGE_17_19:
                kid.assign_workplace(school)
                school.n_students += 1
            elif 20 <= kid.age <= 24 and rng.uniform() < P_SCHOOL_FOR_AGE_19_24:
                kid.assign_workplace(school)
                school.n_students += 1
            elif 25 <= kid.age <= 29 and rng.uniform() < P_SCHOOL_FOR_AGE_25_29:
                kid.assign_workplace(school)
                school.n_students += 1

    # teachers
    potential_teachers = [human for human in humans if 30 <= human.age <= 50]
    all_schools = [
        [STUDENT_TEACHER_RATIO_SCHOOL_2_4, schools_for_2_4],
        [STUDENT_TEACHER_RATIO_SCHOOL_4_5, schools_for_4_5],
        [STUDENT_TEACHER_RATIO_SCHOOL_5_12, schools_for_5_12],
        [STUDENT_TEACHER_RATIO_SCHOOL_12_17, schools_for_12_17],
        [STUDENT_TEACHER_RATIO_SCHOOL_17_29, schools_for_17_29],
    ]
    for STUDENT_TEACHER_RATIO, schools_x_x in all_schools:
        for school in schools_x_x:
            n_teachers = math.ceil(school.n_students / STUDENT_TEACHER_RATIO)
            teachers = _random_choice(potential_teachers, size=n_teachers, replace=False, rng=rng, catch_size_error=True)
            for teacher in teachers:
                teacher.assign_workplace(school)
                potential_teachers.remove(teacher)
                school.n_teachers += 1

    # add schools to city
    city.schools = schools_for_2_4 + schools_for_4_5 + schools_for_5_12 + schools_for_12_17 + schools_for_17_29

    return humans, city

def _build_schools_and_enroll_students(city, conf, kids, n_schools, name, area, rng, N_STUDENTS_PER_SCHOOL):
    """
    Builds `n_schools` and allocates all of the `kids` to them in a batch of `N_STUDENTS_PER_SCHOOL`.

    Args:
        city (covid19sim.location.City): simulator's city object
        conf (dict): yaml configuration of the experiment
        kids (list): list of `human`s that are to be assigned to schools
        n_schools (int): number of schools to build
        name (str): an identifier of the school
        area (np.array): ith value in this array is the area to be alloted to ith school
        rng (np.random.RandomState): Random number generator
        N_STUDENTS_PER_SCHOOL (int): maximum limit to the number of enrolled students.

    Returns:
        (list): list of `School`s with enrolled students.
    """
    rng.shuffle(kids)
    schools = []
    idx = -1
    for i in range(n_schools):
        idx += 1
        school = _build_school(city, conf, f"SCHOOL{name}:{i}", area[idx], rng)
        schools.append(school)
        while school.n_students < N_STUDENTS_PER_SCHOOL and len(kids) > 0:
            kid = kids.pop()
            assert kid.workplace is None, "Already assigned a workplace"
            kid.assign_workplace(school)
            school.n_students += 1

    return schools

def _build_school(city, conf, name, area, rng):
    """
    Initializes a School object.
    """
    return School(
                env=city.env,
                rng=np.random.RandomState(rng.randint(2 ** 16)),
                conf=conf,
                name=name,
                location_type="SCHOOL",
                lat=rng.randint(*city.x_range),
                lon=rng.randint(*city.y_range),
                area=city.total_area,
                capacity=None
            )

def _create_senior_residences(n_senior_residents, city, rng, conf):
    """
    Creates senior residences for `n_senior_residents` according to the configuration.

    Args:
        n_senior_residents (int): number of senior residents
        city (covid19sim.location.City): simulator's city object
        rng (np.random.RandomState): Random number generator
        conf (dict): yaml configuration of the experiment

    Returns:
        (list): list of initialized locations of type "SENIOR_RESDENCY"
    """
    N_RESIDENTS_PER_COLLECTIVE = conf['N_RESIDENTS_PER_COLLECTIVE']
    SENIOR_RESIDENCE_PROPORTION_AREA = conf['SENIOR_RESIDENCE_PROPORTION_AREA']

    n_senior_residences = math.ceil(n_senior_residents/N_RESIDENTS_PER_COLLECTIVE)

    # calculate area for each location
    area = _get_random_area(n_senior_residences, SENIOR_RESIDENCE_PROPORTION_AREA * city.total_area, rng)

    senior_residences = []
    for i in range(n_senior_residences):
        senior_residences.append(
            Household(
                    env=city.env,
                    rng=np.random.RandomState(rng.randint(2 ** 16)),
                    conf=conf,
                    name=f"SENIOR_RESIDENCE:{i}",
                    location_type="SENIOR_RESIDENCE",
                    lat=rng.randint(*city.x_range),
                    lon=rng.randint(*city.y_range),
                    area=area[i],
                    capacity=None,
            )
        )

    return senior_residences

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

    P_HOUSEHOLD_SIZE = conf['P_HOUSEHOLD_SIZE']
    AVG_HOUSEHOLD_SIZE = conf['AVG_HOUSEHOLD_SIZE']

    P_COLLECTIVE_65_69 = conf['P_COLLECTIVE_65_69']
    P_COLLECTIVE_70_74 = conf['P_COLLECTIVE_70_74']
    P_COLLECTIVE_75_above = conf['P_COLLECTIVE_75_above']

    P_MULTIGENERATIONAL_FAMILY = conf['P_MULTIGENERATIONAL_FAMILY']
    P_MULTIGENERTIONAL_FAMILY_GIVEN_OTHER_HOUSEHOLDS = conf['P_MULTIGENERTIONAL_FAMILY_GIVEN_OTHER_HOUSEHOLDS']

    AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID = conf['AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID']
    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MAX_AGE_WITH_PARENT = conf['MAX_AGE_WITH_PARENT']

    FAMILY_TYPES, P_TYPES = _get_family_types(conf)

    assert len(P_TYPES) == len(FAMILY_TYPES), "not a valid mapping of probability and family types"
    assert abs(sum(P_TYPES) - 1) < 1e-2, "Probabilities do not sum to 1."

    n_people = city.n_people
    unassigned_humans = humans
    age_bins = sorted(humans.keys(), key=lambda x:x[0])
    allocated_humans = []

    log("Allocating houses ... ", logfile)

    # allocate senior residencies
    collectives = [
        [(65,69), P_COLLECTIVE_65_69],
        [(70,74), P_COLLECTIVE_70_74],
        [(75,110), P_COLLECTIVE_75_above]
        ]

    assigned_to_collectives = []
    for bin, P in collectives:
        for human in unassigned_humans[bin]:
            if city.rng.random() < P:
                assigned_to_collectives.append((bin, human))

    n_senior_residents = len(assigned_to_collectives)
    city.senior_residences = _create_senior_residences(n_senior_residents, city, city.rng, conf)

    for bin, human in assigned_to_collectives:
        res = city.rng.choice(city.senior_residences, size=1).item()
        allocated_humans = _assign_household(human, res, allocated_humans)
        unassigned_humans[bin].remove(human)

    # presample houses equal to the number of houses as per census
    n_houses_approx = math.ceil(n_people / AVG_HOUSEHOLD_SIZE)
    housetypes = relativefreq2absolutefreq(
                bins_fractions={x: x.probability for x in FAMILY_TYPES},
                n_elements=n_houses_approx,
                rng=city.rng
                )

    housetypes = [deepcopy(x) for x, i in housetypes.items() for _ in range(i)]

    # initialize with unallocated houses
    unallocated_houses = defaultdict(list)
    for housetype in housetypes:
        # determine if multiple generations live together or not
        # all possiblities are listed here - [number of grandparents, parents, kids] and sampled uniformly given a size
        if housetype.living_arrangement == "other" and city.rng.random() < P_MULTIGENERTIONAL_FAMILY_GIVEN_OTHER_HOUSEHOLDS:
            size = housetype.n_humans
            if size == 2:
                n_people_generation = [[1, 0, 1]]
            elif size == 3:
                n_people_generation = [[2, 0 ,1], [1, 1 ,1]]
            elif size == 4:
                n_people_generation = [[2, 1, 1], [1, 2 ,1], [1, 1, 2]]
            elif size == 5:
                n_people_generation = [[2, 1, 2], [2, 2, 1], [1, 2, 2], [1, 1, 3]]
            n_grandparents, n_parents, n_kids = _random_choice(n_people_generation, rng=city.rng, size=1)[0]
            housetype.set_generations(n_grandparents, n_parents, n_kids)
            unallocated_houses['multigenerational'].append(housetype) # needed to prioritize these houses to fill first

        unallocated_houses[housetype.n_kids].append(housetype)

    # start sampling
    n_failed_attempts = 0

    # Step 1: start by allocating kids to house so that age difference between parents and kids is valid
    OTHER_MID_BINS = _get_valid_bins(AGE_BIN_WIDTH_5, min_age=MAX_AGE_CHILDREN, max_age=MAX_AGE_WITH_PARENT)
    KID_BINS = _get_valid_bins(AGE_BIN_WIDTH_5, max_age=MAX_AGE_CHILDREN)
    SEARCH_BINS = KID_BINS
    kid_keys = [n for n, houses in unallocated_houses.items() if type(n) == int and n > 0 and len(houses) > 0]

    # failure mode
    start_search_for_kids = sum(len(unassigned_humans[bin]) for bin in KID_BINS + OTHER_MID_BINS) > 0
    random_allocation_of_kids = False

    while start_search_for_kids and n_failed_attempts < MAX_FAILED_ATTEMPTS_ALLOWED:
        n_kids_needed = sum(len(unallocated_houses[kid_key]) for kid_key in kid_keys)
        valid_kid_bins = [bin for bin in SEARCH_BINS if len(unassigned_humans[bin]) > 0]

        # expand the search to middle generation
        if n_kids_needed != 0 and len(valid_kid_bins) == 0:
            if SEARCH_BINS != OTHER_MID_BINS:
                log("expanding the search for kids to middle generation for census family households", logfile)
                SEARCH_BINS = OTHER_MID_BINS
            else:
                n_failed_attempts += 1

            continue

        # allocation succesful in regards to parent-kid constraints
        _valid_bins = [bin for bin in KID_BINS if len(unassigned_humans[bin]) > 0]
        if n_kids_needed == 0 and len(_valid_bins) == 0:
            break

        # sample humans
        # Note: We sample from broader set of bins while sampling for other kids in _sample_other_residents
        kid = _sample_n_kids(valid_kid_bins, conf, unassigned_humans, city.rng, size=1, with_kid=None)[0]
        # find a house and other residents
        housetype = _sample_house_type(unallocated_houses, city.rng, kid=True)
        # find other residents
        humans_with_same_house, unassigned_humans = _sample_other_residents(housetype, unassigned_humans, city.rng, conf, with_kid=kid)
        if humans_with_same_house:
            allocated_humans = create_and_assign_household(humans_with_same_house, housetype, conf, city, allocated_humans)
            unallocated_houses[housetype.n_kids].remove(housetype)
        else:
            n_failed_attempts += 1
            unassigned_humans = _revert_allocation([kid], unassigned_humans)

    if start_search_for_kids and n_failed_attempts < MAX_FAILED_ATTEMPTS_ALLOWED:
        assert sum(len(houses) for n_kids, houses in unallocated_houses.items() if n_kids > 0) == 0, "kids remain to be allocated"
        assert sum(len(unassigned_humans[kid_bin]) for kid_bin in KID_BINS) == 0, "kids remain to be allocated"
    elif not start_search_for_kids:
        log("Not searching for kids because there are none", logfile)
    else:
        random_allocation_of_kids = True
        log(f"kids allocation failed. not satisfying the parent-kid constraints anymore. Allocating randomly...", logfile)

    # Step 2: remaining living arrangements are - "couple" (with 0 kids), "other" (not multigenerational), "solo"
    while len(unallocated_houses[0]) > 0:
        housetype = unallocated_houses[0].pop()
        humans_with_same_house, unassigned_humans = _sample_other_residents(housetype, unassigned_humans, city.rng, conf, with_kid=None)
        if humans_with_same_house:
            allocated_humans = create_and_assign_household(humans_with_same_house, housetype, conf, city, allocated_humans)
        else:
            n_failed_attempts += 1

    # Step 3: if there are more humans than the presampled housetypes;
    # create more households with the distribution only towards solo, couple and others because there are no more kids
    FAMILY_TYPES, P_TYPES = _get_family_types(conf, without_kids=True)
    while len(allocated_humans) < city.n_people and n_failed_attempts < 2 * MAX_FAILED_ATTEMPTS_ALLOWED:
        # sample house
        housetype = _random_choice(FAMILY_TYPES, city.rng, 1, P=P_TYPES, replace=True)[0]
        # sample humans (without kids)
        humans_with_same_house, unassigned_humans = _sample_other_residents(housetype, unassigned_humans, city.rng, conf, with_kid=None)

        # allocate if succesful
        if humans_with_same_house:
            housetype = deepcopy(housetype)
            housetype.random = True
            allocated_humans = create_and_assign_household(humans_with_same_house, housetype, conf, city, allocated_humans)
        else:
            n_failed_attempts += 1
            if n_failed_attempts % 1000 == 0:
                log(f"Failed attempt - {n_failed_attempts}. Total allocated:{len(allocated_humans)}", logfile)

    if not start_search_for_kids or n_failed_attempts == 2 * MAX_FAILED_ATTEMPTS_ALLOWED or random_allocation_of_kids:
        # something is wrong with the algo.
        log(f"Could not find suitable housing for the population. Allocating solo residences.\nFailed attempt - {n_failed_attempts}. Total allocated:{len(allocated_humans)}", logfile)
        for bin, humans in unassigned_humans.items():
            for human in humans:
                unassigned_humans[bin].remove(human)
                housetype = HouseType("solo", 0, 1, P_HOUSEHOLD_SIZE[0])
                housetype.random = True
                allocated_humans = create_and_assign_household([human], housetype, conf, city, allocated_humans)
    else:
        assert len(allocated_humans) == city.n_people, "assigned humans and total population do not add up"
        assert sum(len(val) for x,val in unassigned_humans.items()) == 0, "there are unassigned humans in the list"
        assert len(city.households) > 0, "no house generated"
        assert all(not all(human.age < MAX_AGE_CHILDREN for human in house.residents) for house in city.households), "house with only children allocated"

    # assign area to houses
    HOUSEHOLD_PROPORTION_AREA = conf['HOUSEHOLD_PROPORTION_AREA']
    area = _get_random_area(len(city.households), HOUSEHOLD_PROPORTION_AREA * city.total_area, city.rng)
    for i,house in enumerate(city.households):
        house.area = area[i]

    # shuffle the list of humans so that the simulation is not dependent on the order of house allocation
    city.rng.shuffle(allocated_humans)
    log(f"Housing allocated with failed attempts: {n_failed_attempts} ", logfile)
    return allocated_humans

def _sample_house_type(unallocated_houses, rng, kid=True):
    """
    Samples house type from `unallocated_houses`.

    Args:
        unallocated_houses (dict): keys are number of kids in house and values are a list of `HouseType` objects
        rng (np.random.RandomState): Random number generator
        kid (bool): whether to sample of the houses with kids. Defaults to True.

    Returns:
        (HouseType): sampled house from `unallocated_houses`
    """
    if kid:
        # send multigenerational households first as they have more constraints to satisfy in regards to age
        if unallocated_houses['multigenerational']:
            housetype = unallocated_houses['multigenerational'].pop()
            return housetype
        else:
            unallocated_houses.pop('multigenerational')

        # failure mode
        remaining_kids_houses = [(n,len(houses)) for n, houses in unallocated_houses.items() if n > 0 and len(houses) > 0]
        if not remaining_kids_houses:
            return None

        # rest afterwards
        n_kids, count = list(zip(*remaining_kids_houses))
        count = np.array(count)
        p = count / count.sum()
        n_kids = rng.choice(n_kids, size=1, p=p).item()
        return unallocated_houses[n_kids][0]
    else:
        return all_housetypes['unallocated'][0][0]

def _sample_other_residents(housetype, unassigned_humans, rng, conf, with_kid=None):
    """
    Samples residents for `housetype`.

    Args:
        housetype (HouseType): type of house to assign humans to.
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        conf (dict): yaml configuration of the experiment
        with_kid (covid19sim.human.Human): a presampled kid. Default to None.

    Returns:
        (list): list of sampled humans. Returns an empty list if unsuccessful.
    """
    # failure mode
    if housetype is None:
        return [], unassigned_humans

    MAX_AGE_CHILDREN = conf['MAX_AGE_CHILDREN']
    MAX_AGE_WITH_PARENT = conf['MAX_AGE_WITH_PARENT']
    AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID = conf['AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID']
    P_CONTACT_HOUSE = np.array(conf['P_CONTACT_MATRIX_HOUSEHOLD'])

    valid_age_bins = [x for x, val in unassigned_humans.items() if len(val) >= 1]
    if with_kid is not None:
        n_grandparents, n_parents, n_kids = housetype.n_people_generation

        # NOTE: ASSORTATIVITY handles samples not being far away
        if with_kid.age > MAX_AGE_CHILDREN:
            valid_younger_bins = _get_valid_bins(valid_age_bins, min_age=MAX_AGE_CHILDREN, max_age=MAX_AGE_WITH_PARENT)
        else:
            valid_younger_bins = _get_valid_bins(valid_age_bins, max_age=MAX_AGE_CHILDREN + 5, inclusive=True)
        # check if there are enough kids
        if sum(len(unassigned_humans[bin]) for bin in valid_younger_bins) < n_kids-1:
            return [], unassigned_humans

        kids = _sample_n_kids(valid_younger_bins, conf, unassigned_humans, rng, size=n_kids-1, with_kid=with_kid)
        assert with_kid in kids, "kid not present in sampled kids"

        # set constraints on other residents
        max_age_kids = max(k.age for k in kids)
        min_age_parent = max_age_kids + AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID
        min_age_grandparent = min_age_parent + AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID

        # Algo gets here when sampling for "multigenerational", "couple with kids", "single parent with kids"
        parents, grandparents = [], []
        if n_parents:
            # Note: we allow for some discrepancy in max age by taking the bin in which min age lies.
            # valid_parent_bins = [x for x in valid_parent_bins if x[1] >=  min_age_parent and x[0] <= max_age] $
            valid_parent_bins = _get_valid_bins(valid_age_bins, min_age=min_age_parent, inclusive=True)
            if n_grandparents > 0:
                valid_parent_bins = _get_valid_bins(valid_parent_bins, max_age=MAX_AGE_WITH_PARENT, inclusive=True)

            ref_age = np.min([k.age for k in kids])
            p_bin = _get_probability_of_drawing_bins(P_CONTACT_HOUSE, valid_parent_bins, age=ref_age)

            parents = _sample_n_parents(valid_parent_bins, conf, unassigned_humans, rng, n=n_parents, p_bin=p_bin)
            if len(parents) == 0:
                unassigned_humans = _revert_allocation(kids, unassigned_humans)
                return [], unassigned_humans

            max_age_parent = max(p.age for p in parents)
            min_age_grandparent = max_age_parent + AGE_DIFFERENCE_BETWEEN_PARENT_AND_KID

        # Algorithm should get here only in the begining i.e. through prioritizing multigenerational houses first
        if n_grandparents:
            # Note: we allow for some discrepancy in max age by taking the bin in which min age lies.
            # valid_grandparent_bins = [x for x in valid_grandparent_bins if x[1] >=  min_age_grandparent] $
            valid_grandparent_bins = _get_valid_bins(valid_age_bins, min_age=min_age_grandparent, inclusive=True)

            ref_age = np.min([p.age for p in parents]) if parents else min_age_parent
            p_bin = _get_probability_of_drawing_bins(P_CONTACT_HOUSE, valid_grandparent_bins, age=ref_age)
            grandparents = _sample_n_parents(valid_grandparent_bins, conf, unassigned_humans, rng, n=n_grandparents, p_bin=p_bin)

            if len(grandparents) == 0:
                unassigned_humans = _revert_allocation(kids + parents, unassigned_humans)
                return [], unassigned_humans

        sampled_humans = kids + parents + grandparents
        assert len(sampled_humans) == housetype.n_humans

    else:
        assert housetype.n_kids == 0, f"can not sample {housetype.n_kids} without other kid"

        if housetype.living_arrangement == "solo":
            sampled_humans = _sample_solo_dweller(valid_age_bins, conf, unassigned_humans, rng)

        elif housetype.living_arrangement == "couple":
            sampled_humans = _sample_couple(valid_age_bins, conf, unassigned_humans, rng)

        elif housetype.living_arrangement == "other":
            # (no-source) all other type of housing is resided by adults only
            valid_other_bins = _get_valid_bins(valid_age_bins, min_age=MAX_AGE_CHILDREN)
            sampled_humans = _sample_random_humans(valid_other_bins, conf, unassigned_humans, rng, size=housetype.n_humans)

        else:
            raise ValueError

    return sampled_humans, unassigned_humans

def _sample_n_parents(valid_age_bins, conf, unassigned_humans, rng, n, p_bin):
    """
    Samples parents in  unassigned humans of `valid_age_bins`.

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        n (int): number of kids to sample. Defaults to 0.
        p_bin (list): probability to sample a bin in `valid_age_bins`. Defaults to None.

    Returns:
        list: humans belonging to a same household (length = n + 2 if succesful else 0)
    """
    # failure mode
    if len(valid_age_bins) == 0:
        return []

    if p_bin is None:
        p_bin = np.ones_like(valid_age_bins)
        p_bin /= p_bin.sum()

    assert n in [1, 2], f"can not sample {n} parents"
    assert abs(p_bin.sum() - 1) < 1e-2, "probabilities do not sum to 1"

    sampled_parents = []
    # single parent
    if n == 1:
        older_bin = _random_choice(valid_age_bins, rng, size=1, P=p_bin)[0]
        parent = rng.choice(unassigned_humans[older_bin], size=1).item()
        unassigned_humans[older_bin].remove(parent)
        sampled_parents = [parent]

    # n=2 couple parent
    if n==2:
        sampled_parents = _sample_couple(valid_age_bins, conf, unassigned_humans, rng, p_bin=p_bin)

    if sampled_parents:
        assert len(sampled_parents) == n, "not a valid allocation"

    return sampled_parents

def _sample_couple(valid_age_bins, conf, unassigned_humans, rng, p_bin=None, return_bins=False):
    """
    Samples two humans to live together.

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        n (int): number of kids to sample. Defaults to 0.
        p_bin (list): probability to sample a bin in `valid_age_bins`. Defaults to None.
        return_bins (bool): return potential bins to draw couples from if True.

    Returns:
        list: humans belonging to a same household (length = n + 2 if succesful else 0). Returns potential bins to draw couples from if return_bins is True.
    """
    ASSORTATIVITY_STRENGTH = conf['HOUSEHOLD_ASSORTATIVITY_STRENGTH']

    # couple can be in two consecutive bins
    min_couple_single_bins = [(x,x) for x in valid_age_bins if len(unassigned_humans[x]) >= 2]

    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    sequential_couple_bins = [
                            (x,y) for x,y in zip(age_bins, age_bins[1:]) \
                            if (len(unassigned_humans[x]) >= 1
                                and len(unassigned_humans[y]) >= 1
                                and x in valid_age_bins
                                and y in valid_age_bins)
                            ]

    valid_couple_bins = min_couple_single_bins + sequential_couple_bins
    if return_bins:
        return  valid_couple_bins

    if len(valid_couple_bins) == 0:
        return []

    sampled_humans = []
    if p_bin is not None:
        p_couple = np.zeros(len(valid_couple_bins))
        # sample couple from same bins according to the contact probability.
        for j, (x,y) in enumerate(valid_couple_bins):
            idxs = [i for i, bin in enumerate(valid_age_bins) if bin in [x,y]]
            p_couple[j] += np.max(p_bin[idxs])

    else:
        # couples are more likley to be in the same age bin
        p_couple = np.ones(len(valid_couple_bins))

    p_couple[:len(min_couple_single_bins)] += ASSORTATIVITY_STRENGTH
    p_couple /= p_couple.sum()

    two_bins = _random_choice(valid_couple_bins, rng, size=1, P=p_couple)[0]

    human1 = rng.choice(unassigned_humans[two_bins[0]], size=1).item()
    unassigned_humans[two_bins[0]].remove(human1)

    human2 = rng.choice(unassigned_humans[two_bins[1]], size=1).item()
    unassigned_humans[two_bins[1]].remove(human2)

    sampled_humans += [human1, human2]

    return sampled_humans

def _sample_solo_dweller(valid_age_bins, conf, unassigned_humans, rng):
    """
    Finds one human to be allocated to a household of size 1.

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

    human = []
    age_bins = sorted(unassigned_humans.keys(), key=lambda x:x[0])
    valid_idx = [i for i,bin in enumerate(age_bins) if len(unassigned_humans[bin]) >= 1 and P_AGE_SOLO[i] > 0]
    valid_age_bins =  [age_bins[i] for i in valid_idx]
    if valid_age_bins:
        P = [P_AGE_SOLO[i] for i in valid_idx]
        P = [i / sum(P) for i in P]
        age_bin = _random_choice(valid_age_bins, rng, 1, P)[0]
        human = rng.choice(unassigned_humans[age_bin], size=1).tolist()
        unassigned_humans[age_bin].remove(human[0])

    return human

def _sample_random_humans(valid_other_bins, conf, unassigned_humans, rng, size):
    """
    Samples humans randomly to be put in the same house while ensuring assortativity in age.

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        size (int): number of humans to sample

    Returns:
        humans (list): humans belonging to a same household (length = size if succesful else 0)
    """
    ASSORTATIVITY_STRENGTH = conf['HOUSEHOLD_ASSORTATIVITY_STRENGTH']

    all_humans = [(y,x) for x in valid_other_bins for y in unassigned_humans[x]]
    if len(all_humans) < size:
        return []

    valid_other_bins = sorted(valid_other_bins, key=lambda x:x[0])
    # NOTE: we assign more probability to the bin which has more humans to achieve balance in sampling
    p_bin = np.array([len(unassigned_humans[x]) for x in valid_other_bins])
    humans = []
    while len(humans) < size:
        p = p_bin / p_bin.sum()
        bin = _random_choice(valid_other_bins, rng=rng, size=1, P=p)[0]
        human = rng.choice(unassigned_humans[bin], size=1).item()
        humans.append(human)
        unassigned_humans[bin].remove(human)

        # reassign probabilities
        idx = valid_other_bins.index(bin)
        p_bin[idx] += ASSORTATIVITY_STRENGTH

        if idx < len(valid_other_bins) - 1:
            p_bin[idx+1] += ASSORTATIVITY_STRENGTH

        if idx > 0:
            p_bin[idx-1] += ASSORTATIVITY_STRENGTH

        p_bin[[i for i, bin in enumerate(valid_other_bins) if len(unassigned_humans[bin]) == 0]] = 0

    assert len(humans) == size, "number of humans sampled doesn't equal the expected size"
    return humans

def _sample_n_kids(valid_age_bins, conf, unassigned_humans, rng, size, with_kid=None):
    """
    Samples kids with priority given to younger kids.

    Args:
        valid_age_bins (list): age bins that qualify to sample humans
        conf (dict): yaml configuration of the experiment
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
        rng (np.random.RandomState): Random number generator
        size (int): number of humans to sample

    Returns:
        (list): list of `Human`s sampled
    """
    ASSORTATIVITY_STRENGTH = conf['HOUSEHOLD_ASSORTATIVITY_STRENGTH']
    P_CONTACT_HOUSE = np.array(conf['P_CONTACT_MATRIX_HOUSEHOLD'])

    kids, total_kids = [], size
    valid_younger_bins = sorted(valid_age_bins, key=lambda x: x[0])

    # to balance unsampled bins (only when with_kid is None)
    p_bin = np.array([len(unassigned_humans[bin]) for bin in valid_younger_bins])
    if with_kid is not None:
        kid_bin = with_kid.age_bin_width_5.bin
        assert with_kid not in unassigned_humans[kid_bin],  "kid has been sampled but not removed from unassigned_humans"
        #
        total_kids += 1
        kids += [with_kid]
        p_bin = _get_probability_of_drawing_bins(P_CONTACT_HOUSE, valid_younger_bins, with_kid.age)

    while len(kids) < total_kids:
        p_bin = p_bin / p_bin.sum()
        bin = _random_choice(valid_younger_bins, rng=rng, size=1, P=p_bin)[0]
        kid = rng.choice(unassigned_humans[bin], size=1).item()
        kids.append(kid)
        unassigned_humans[bin].remove(kid)

        # reassign probabilities
        idx = valid_younger_bins.index(bin)
        if len(unassigned_humans[bin]) > 0:
            p_bin[idx] += ASSORTATIVITY_STRENGTH
        else:
            p_bin[idx] = 0.0

    assert len(kids) == total_kids,  "not a valid size"
    return kids

def _get_family_types(conf, without_kids=False, only_kids=False):
    """
    Creates a list of all possible housetypes with their probabilities.
    Note 1: P_FAMILY_TYPE_ is assumed to carry probabilities for types = "couple", "single_parent", "other" in this sequence.
    Note 2: We consider a max house size of 5.

    Args:
        conf (dict): yaml configuration of the experiment
        without_kids (bool): Whether to consider housing with kids. Returns non-zero kid requirement housing if True.
        only_kids (bool): Whether to return housing only with kids.

    Returns:
        (list): list of `HouseType`s
    """
    P_HOUSEHOLD_SIZE= conf['P_HOUSEHOLD_SIZE']
    P_FAMILY_TYPE_SIZE_2 = conf['P_FAMILY_TYPE_SIZE_2']
    P_FAMILY_TYPE_SIZE_3 = conf['P_FAMILY_TYPE_SIZE_3']
    P_FAMILY_TYPE_SIZE_4 = conf['P_FAMILY_TYPE_SIZE_4']
    P_FAMILY_TYPE_SIZE_MORE_THAN_5 = conf['P_FAMILY_TYPE_SIZE_MORE_THAN_5']

    FAMILY_TYPES = [
                    # size=2
                    HouseType("couple", 0, 2, P_FAMILY_TYPE_SIZE_2[0]),
                    HouseType("single_parent", 1, 2, P_FAMILY_TYPE_SIZE_2[1]) ,
                    HouseType("other", 0, 2, P_FAMILY_TYPE_SIZE_2[2]),

                    # size=3
                    HouseType("couple", 1, 3, P_FAMILY_TYPE_SIZE_3[0]),
                    HouseType("single_parent", 2, 3, P_FAMILY_TYPE_SIZE_3[1]),
                    HouseType("other", 0, 3, P_FAMILY_TYPE_SIZE_3[2]),

                    # size=4
                    HouseType("couple", 2, 4, P_FAMILY_TYPE_SIZE_4[0]),
                    HouseType("single_parent", 3, 4, P_FAMILY_TYPE_SIZE_4[1]),
                    HouseType("other", 0, 4, P_FAMILY_TYPE_SIZE_4[2]),

                    # size=5
                    HouseType("couple", 3, 5, P_FAMILY_TYPE_SIZE_MORE_THAN_5[0]),
                    HouseType("single_parent", 4, 5, P_FAMILY_TYPE_SIZE_MORE_THAN_5[1]),
                    HouseType("other", 0, 5, P_FAMILY_TYPE_SIZE_MORE_THAN_5[2]),

                    # size=1
                    HouseType("solo", 0, 1, P_HOUSEHOLD_SIZE[0])
                ]

    if without_kids:
        FAMILY_TYPES = [x for x in FAMILY_TYPES if x.n_kids == 0]

    if only_kids:
        FAMILY_TYPES = [x for x in FAMILY_TYPES if x.n_kids > 0]

    P_TYPES = [x.probability for x in FAMILY_TYPES]
    # re-normalize to ensure that the probabilities sum to 1
    P_TYPES = [x/sum(P_TYPES) for x in P_TYPES]

    return FAMILY_TYPES, P_TYPES

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

def create_and_assign_household(humans_with_same_house, housetype, conf, city, allocated_humans):
    """
    Creates a residence and allocates humans in `humans_with_same_house` to the same.

    Args:
        humans_with_same_house (list): a list of `Human` objects which are to be allocated to the same residence of type `type`.
        housetype (HouseType): type of allocation
        conf (dict): yaml configuration of the experiment
        city (covid19sim.location.City): simulator's city object
        allocated_humans (list): a list of humans that have been allocated a household

    Returns:
        allocated_humans (list): a list of humans that have been allocated a household
    """
    assert all(human not in allocated_humans for human in humans_with_same_house), f"reassigning household to human"
    res =  Household(
            env=city.env,
            rng=np.random.RandomState(city.rng.randint(2 ** 16)),
            conf=conf,
            name=f"HOUSEHOLD:{len(city.households)}",
            location_type="HOUSEHOLD",
            lat=city.rng.randint(*city.x_range),
            lon=city.rng.randint(*city.y_range),
            area=None,
            capacity=None,
        )

    for human in humans_with_same_house:
        allocated_humans = _assign_household(human, res, allocated_humans)

    res.allocation_type = housetype
    city.households.add(res)
    return allocated_humans

def _get_valid_bins(valid_age_bins, min_age=-1, max_age=200, inclusive=False):
    """
    Filters out age bins according to minium and maximum age specified.

    Args:
        valid_age_bins (list): a list of potential age bins
        min_age (int): age above which all bins are considered valid
        max_age (int): age below which all bins are considered valid
        inclusive (bool): includes the bins which have min or max values within the limit. (used with adjusted ranges)
    Returns:
        (list): valid age bins
    """
    if inclusive:
        filtered = [x for x in valid_age_bins if x[1] >= min_age]
        filtered = [x for x in filtered if x[0] <= max_age]
    else:
        filtered = [x for x in valid_age_bins if x[0] >= min_age]
        filtered = [x for x in filtered if x[1] <= max_age]
    return filtered

def _get_probability_of_drawing_bins(P_CONTACT, valid_bins, age):
    """
    Returns the probability corresponding to valid_bins as indexed by age.

    Args:
        P_CONTACT (np.array): 2D square matrix
        valid_bins (list): age bins to consider
        age (int): age of the concerned person. corresp. to column in the matrix (refer to the definition of contact matrices).

    Returns:
        (np.array): probabilities corresp. to valid_bins
    """
    idx = [i for i, x in enumerate(AGE_BIN_WIDTH_5) if x[0] <= math.floor(age) <= x[1]][0]
    valid_bins_idx = [i for i, x in enumerate(AGE_BIN_WIDTH_5) if x in valid_bins]
    p_bin = P_CONTACT[valid_bins_idx, idx]
    return p_bin / p_bin.sum()

def _revert_allocation(humans, unassigned_humans):
    """
    Puts the humans back in their respective bins.

    Args:
        humans (list): list of humans to be put back in their respective bins
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated

    Returns:
        unassigned_humans (dict): keys are age bin (tuple) and values are humans that do not have a household allocated
    """
    for human in humans:
        if human not in unassigned_humans[human.age_bin_width_5.bin]:
            unassigned_humans[human.age_bin_width_5.bin].append(human)
    return unassigned_humans
