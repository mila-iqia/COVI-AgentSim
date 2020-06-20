"""
Implements human behavior/government policy changes.

- * Behaviors * change the behavior of a human, specifically they can modify the human's propensity to take actions which
reduce their liklihood of infecting others (wearing a mask, washing hands); they can choose to go out less
(reducing gamma and rho); or they can take more distance / reduce time spent with others while out.

- * RecommendationGetter * A list of behaviors which are recommended by a Tracing Method? Model? Your mom?

- * Your mom * the person who raised you AKA the Manager

- * NonMLRiskComputer * Handle computing the risk for non-ml tracing methods like heuristic and binarytracing

"""
import typing
if typing.TYPE_CHECKING:
    from covid19sim.human import Human

class Behavior(object):
    """
    A base class to modify behavior based on the type of intervention.
    """

    def modify(self, human):
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
        raise NotImplementedError

    def revert(self, human):
        """
        Resets the behavior attributes of `Human`.
        It changes `attr` back to what it was before modifying the `attribute`.
        deletes `_attr` from `Human`.

        Args:
            human (Human): `Human` object.
        """
        raise NotImplementedError


class Unmitigated(Behavior):
    def modify(self, human):
        pass

    def revert(self, human):
        pass

    def __repr__(self):
        return "Unmitigated"


class StayHome(Behavior):
    """
    TODO.
    Not currently being used.
    """
    def __init__(self):
        pass

    def modify(self, human):
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert(self, human):
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

    def __repr__(self):
        return "Stay Home"

class LimitContact(Behavior):
    """
    TODO.
    Not currently being used.
    """
    def __init__(self):
        pass

    def modify(self, human):
        human._maintain_distance = human.maintain_distance
        human._max_misc_per_week = human.max_misc_per_week
        human._max_shop_per_week = human.max_shop_per_week

        human.maintain_distance = human.conf.get("DEFAULT_DISTANCE") + 100 * (human.carefulness - 0.5)
        human.max_misc_per_week = 1
        human.max_shop_per_week = 1

    def revert(self, human):
        human.maintain_distance = human._maintain_distance
        human.max_misc_per_week = human._max_misc_per_week
        human.max_shop_per_week = human._max_shop_per_week
        delattr(human, "_maintain_distance")
        delattr(human, "_max_misc_per_week")
        delattr(human, "_max_shop_per_week")

    def __repr__(self):
        return "Limit Contact"


class StandApart(Behavior):
    """
    `Human` should maintain an extra distance with other people.
    It adds `_maintain_extra_distance_2m` because of the conflict with a same named attribute in
    `SocialDistancing`
    """
    def __init__(self, default_distance=25):
        self.DEFAULT_SOCIAL_DISTANCE = default_distance

    def modify(self, human):
        distance = self.DEFAULT_SOCIAL_DISTANCE + 100 * (human.carefulness - 0.5)
        human.set_temporary_maintain_extra_distance(distance)

    def revert(self, human):
        human.revert_maintain_extra_distance()

    def __repr__(self):
        return f"Stand {self.DEFAULT_SOCIAL_DISTANCE} cms apart"


class WashHands(Behavior):
    """
    Increases `Human.hygeine`.
    This factor is used to decay likelihood of getting infected/infecting others exponentially.
    """

    def __init__(self):
        pass

    def modify(self, human):
        human._hygiene = human.hygiene
        human.hygiene = human.rng.uniform(min(human.carefulness, 1) , 1)

    def revert(self, human):
        human.hygiene = human._hygiene
        delattr(human, "_hygiene")

    def __repr__(self):
        return "Wash Hands"

class Quarantine(Behavior):
    """
    Implements quarantining for `Human`. Following is included -
        1. work from home (changes `Human.workplace` to `Human.household`)
        2. rest at home (not go out unless)
        3. stay at home unless hospitalized (so there can still be household infections)
        4. go out with a reduce probability of 0.10 to stores/parks/miscs, but every time `Human` goes out
            they do not explore i.e. do not go to more than one location. (reduce RHO and GAMMA)

    Adds an attribute `_quarantine` to be used as a flag.
    """
    _RHO = 0.1
    _GAMMA = 1

    def __init__(self):
        pass

    def modify(self, human):
        human.set_temporary_workplace(human.household)
        human.rho = self._RHO
        human.gamma = self._GAMMA
        human.rest_at_home = True
        human._quarantine = True
        # print(f"{human} quarantined {human.tracing_method}")

    def revert(self, human):
        human.revert_workplace()
        human.rho = human.conf.get("RHO")
        human.gamma = human.conf.get("GAMMA")
        human.rest_at_home = False
        human._quarantine = False

    def __repr__(self):
        return f"Quarantine"

# FIXME: Lockdown should be a mix of CityBasedIntervention and RecommendationGetter.
class Lockdown(Behavior):
    """
    Implements lockdown. Needs some more work.
    It only implements behvior modification for `Human`. Ideally, it should close down stores/parks/etc.

    Following behavior modifications are included -
        1. reducde mobility through RHO and GAMMA. Enables minimal exploration if going out.
            i.e. `Human` revisits the previously visited location with increased probability.
            If `Human` is on a leisure trip, it visits only a few location.
        2. work from home (changes `Human.workplace` to `Human.household`)
    """
    _RHO = 0.1
    _GAMMA = 1

    def modify(self, human):
        human.set_temporary_workplace(human.household)
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert(self, human):
        human.revert_workplace()
        human.rho = human.conf.get("RHO")
        human.gamma = human.conf.get("GAMMA")

    def __repr__(self):
        return f"Lockdown"


class SocialDistancing(Behavior):
    """
    Implements social distancing. Following is included -
        1. maintain a distance of 200 cms with other people.
        2. Reduce the time of encounter by 0.5 than what one would do without this intervention.
        3. Reduced mobility (using RHO and GAMMA)

    """
    def __init__(self, default_distance=100, time_encounter_reduction_factor=0.5):
        self.DEFAULT_SOCIAL_DISTANCE = default_distance # cm
        self.TIME_ENCOUNTER_REDUCTION_FACTOR = time_encounter_reduction_factor
        self._RHO = 0.2
        self._GAMMA = 0.5

    def modify(self, human):
        maintain_extra_distance = self.DEFAULT_SOCIAL_DISTANCE + 100 * (human.carefulness - 0.5)
        time_encounter_reduction_factor = self.TIME_ENCOUNTER_REDUCTION_FACTOR
        human.set_temporary_maintain_extra_distance(maintain_extra_distance)
        human.set_temporary_time_encounter_reduction_factor(time_encounter_reduction_factor)
        human.rho = self._RHO
        human.gamma = self._GAMMA

    def revert(self, human):
        human.revert_maintain_extra_distance()
        human.revert_time_encounter_reduction_factor()
        human.rho = human.conf.get("RHO")
        human.gamma = human.conf.get("GAMMA")

    def __repr__(self):
        """
        [summary]
        """
        return f"Social Distancing"


class WearMask(Behavior):
    """
    `Human` wears a mask according to `Human.wear_mask()`.
    Sets `Human.WEAR_MASK` to True.
    """

    def __init__(self, available=None):
        self.available = available

    def modify(self, human):
        if self.available is None:
            human.WEAR_MASK = True
            return

        elif self.available > 0:
            human.WEAR_MASK = True
            self.available -= 1

    def revert(self, human):
        human.WEAR_MASK = False

    def __repr__(self):
        return f"Wear Mask"


class GetTested(Behavior):
    """
    `Human` should get tested.
    """
    # FIXME: can't be called as a stand alone class. Needs human.recommendations_to_follow to work
    # FIXME: test_recommended should be _test_recommended. Make it a convention that any attribute added here,
    # starts with _
    def __init__(self, source):
        """
        Args:
            source (str): reason behind getting tested e.g. recommendation, diagnosis, etc.
        """
        self.source = source

    def modify(self, human):
        human.test_recommended  = True
        # send human to the testing center
        human.check_covid_testing_needs()

    def revert(self, human):
        human.test_recommended  = False

    def __repr__(self):
        return "Get Tested"

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


class TestCapacity(CityInterventions):
    """
    Change the test capacity of the city.
    """

    def modify_city(self, city):
        raise NotImplementedError

    def revert_city(self, city):
        raise NotImplementedError

