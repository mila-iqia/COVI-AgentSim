class BehaviorInterventions(object):
    """
    A base class to modify behavior based on the type of intervention.
    """
    def __init__(self):
        pass

    def modify_behavior(self, human):
        """
        Changes the behavior attributes of `Human`.
        This function can add new attributes to `Human`.
        If the name of the attribute being changed is `attr`, a new attribute
        is `_attr`.
        `_attr` stores the `attribute` value of `Human` before the change will be made.
        `attr` will store new value.

        Args:
            human (Human): `Human` object.
        """
        pass

    def revert_behavior(self, human):
        """
        Resets the behavior attributes of `Human`.
        It changes `attr` back to what it was before modifying the `attribute`.
        deletes `_attr` from `Human`.

        Args:
            human (Human): `Human` object.
        """
        pass

    def __repr__(self):
        return "BehaviorInterventions"


class CityInterventions(object):
    """
    Implements city based interventions such as opening or closing of stores/parks/miscs etc.
    """
    def __init__(self):
        pass

    def modify_city(self, city):
        """
        Modify attributes of city.

        Args:
            city (City): `City` object
        """
        pass

    def revert_city(self, city):
        """
        resets attributes of the city.

        Args:
            city (City): `City` object
        """
        pass


class StayHome(BehaviorInterventions):
    """
    TODO.
    Not currently being used.
    """
    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

    def __repr__(self):
        return "Stay Home"


class LimitContact (BehaviorInterventions):
    """
    TODO.
    Not currently being used.
    """
    def __init__(self):
        pass

    def modify_behavior(self, human):
        human._maintain_distance = human.maintain_distance
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.maintain_distance = human.conf.get("DEFAULT_DISTANCE") + 100 * (human.carefulness - 0.5)
        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert_behavior(self, human):
        human.maintain_distance = human._maintain_distance
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_maintain_distance")
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

    def __repr__(self):
        return "Limit Contact"


class StandApart(BehaviorInterventions):
    """
    `Human` should maintain an extra distance with other people.
    It adds `_maintain_extra_distance_2m` because of the conflict with a same named attribute in
    `SocialDistancing`
    """
    def __init__(self, stand_apart_distance, sampling_method="use_carefulness"):
        self.DEFAULT_SOCIAL_DISTANCE = stand_apart_distance
        self.sampling_method = sampling_method

    def modify_behavior(self, human):
        human._maintain_extra_distance = human.maintain_extra_distance
        if self.sampling_method == "use_carefulness":
            human.maintain_extra_distance = self.DEFAULT_SOCIAL_DISTANCE + 100 * (human.carefulness - 0.5)
        elif self.sampling_method == "random":
            human.maintain_extra_distance = human.rng.random(0, self.DEFAULT_SOCIAL_DISTANCE)
        elif self.sampling_method == "deterministic":
            human.maintain_extra_distance = self.DEFAULT_SOCIAL_DISTANCE
        else:
            raise

    def revert_behavior(self, human):
        human.maintain_extra_distance = human._maintain_extra_distance
        delattr(human, "_maintain_extra_distance")

    def __repr__(self):
        return f"StandApart {self.DEFAULT_SOCIAL_DISTANCE}cms"


class ReducedSocialTime(BehaviorInterventions):
    """
    Reduces time duration of any contact by a fraction.
    """
    def __init__(self, time_reduction_factor, sampling_method = "deterministic"):
        self.TIME_ENCOUNTER_REDUCTION_FACTOR = time_reduction_factor
        self.sampling_method = sampling_method

    def modify_behavior(self, human):
        human._time_encounter_reduction_factor = human.time_encounter_reduction_factor
        if self.sampling_method == "use_carefulness":
            human.time_encounter_reduction_factor = human.rng.uniform(min(human.carefulness, 1) , 1)
        elif self.sampling_method == "random":
            human.time_encounter_reduction_factor = human.rng.random(0, self.TIME_ENCOUNTER_REDUCTION_FACTOR)
        elif self.sampling_method == "deterministic":
            human.time_encounter_reduction_factor = self.TIME_ENCOUNTER_REDUCTION_FACTOR
        else:
            raise
        human.time_encounter_reduction_factor = self.TIME_ENCOUNTER_REDUCTION_FACTOR

    def revert_behavior(self, human):
        human.time_encounter_reduction_factor = human._time_encounter_reduction_factor
        delattr(human, "_time_encounter_reduction_factor")

    def __repr__(self):
        return f"Reduced social time by {self.TIME_ENCOUNTER_REDUCTION_FACTOR}"


class WashHands(BehaviorInterventions):
    """
    Increases `Human.hygeine`.
    This factor is used to decay likelihood of getting infected/infecting others exponentially.
    """

    def __init__(self, sampling_method="use_carefulness"):
        self.sampling_method = sampling_method

    def modify_behavior(self, human):
        human._hygiene = human.hygiene
        if self.sampling_method == "use_carefulness":
            human.hygiene = human.rng.uniform(min(human.carefulness, 1) , 1)
        elif self.sampling_method == "random":
            human.hygiene = human.rng.uniform(0 , 1)
        else:
            raise

    def revert_behavior(self, human):
        human.hygiene = human._hygiene
        delattr(human, "_hygiene")

    def __repr__(self):
        return "Wash Hands"


class WearMask(BehaviorInterventions):
    """
    `Human` wears a mask according to `Human.wear_mask()`.
    Sets `Human.WEAR_MASK` to True.
    """

    def __init__(self, available=None):
        super(WearMask, self).__init__()
        self.available = available

    def modify_behavior(self, human):
        if self.available is None:
            human.WEAR_MASK = True
            return

        elif self.available > 0:
            human.WEAR_MASK = True
            self.available -= 1

    def revert_behavior(self, human):
        human.WEAR_MASK = False

    def __repr__(self):
        return f"Wear Mask"


class TestCapacity(CityInterventions):
    """
    Change the test capacity of the city.
    """

    def modify_city(self, city):
        raise NotImplementedError

    def revert_city(self, city):
        raise NotImplementedError
