import dataclasses
import math
import typing
from collections import namedtuple, OrderedDict
import numpy as np


"""
---------------------------------------
-------------SYMPTOMS -----------------
---------------------------------------
"""

CONDITIONS_CAUSING_MODERATE = ['smoker','diabetes','heart_disease','cancer','COPD','asthma','stroke','immuno-suppressed','lung_disease']

# Utility dict to avoid storing the string symptom's name in its instance
_INT_TO_SYMPTOMS_NAME = {
    1: 'mild',
    0: 'moderate',
    2: 'severe',
    3: 'extremely-severe',
    4: 'fever',
    5: 'chills',
    6: 'gastro',
    7: 'diarrhea',
    8: 'nausea_vomiting',
    9: 'fatigue',
    10: 'unusual',
    11: 'hard_time_waking_up',
    12: 'headache',
    13: 'confused',
    14: 'lost_consciousness',
    15: 'trouble_breathing',
    16: 'sneezing',
    17: 'cough',
    18: 'runny_nose',
    20: 'sore_throat',
    21: 'severe_chest_pain',
    24: 'light_trouble_breathing',
    23: 'mild_trouble_breathing',
    25: 'moderate_trouble_breathing',
    26: 'heavy_trouble_breathing',
    22: 'loss_of_taste',
    19: 'aches'
    # commented out because these are not used elsewhere for now
    # __: 'hives',
    # __: 'swelling'
}


@dataclasses.dataclass(frozen=True)
class Symptom:
    """A symptom

    Attributes
        ----------
        name : str
            name of the symptom
        id : positive int
            id of the symptom
            This attribute should never change once set. It is used to define the
            position of the symptom in a multi-hot encoding
    """
    id: int

    @property
    def name(self):
        return _INT_TO_SYMPTOMS_NAME[self.id]

    def __repr__(self):
        return f"{self.id}:{self.name}"

    def __str__(self):
        return self.name

    def __int__(self):
        return self.id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Symptom):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        elif isinstance(other, str):
            return self.name == other
        else:
            raise ValueError(f"Could not compare {self} with {other}")


STR_TO_SYMPTOMS: typing.Dict[str, Symptom] = {
    name: Symptom(id) for id, name in _INT_TO_SYMPTOMS_NAME.items()
}

MILD = STR_TO_SYMPTOMS['mild']
MODERATE = STR_TO_SYMPTOMS['moderate']
SEVERE = STR_TO_SYMPTOMS['severe']
EXTREMELY_SEVERE = STR_TO_SYMPTOMS['extremely-severe']
FEVER = STR_TO_SYMPTOMS['fever']
CHILLS = STR_TO_SYMPTOMS['chills']
GASTRO = STR_TO_SYMPTOMS['gastro']
DIARRHEA = STR_TO_SYMPTOMS['diarrhea']
NAUSEA_VOMITING = STR_TO_SYMPTOMS['nausea_vomiting']
FATIGUE = STR_TO_SYMPTOMS['fatigue']
UNUSUAL = STR_TO_SYMPTOMS['unusual']
HARD_TIME_WAKING_UP = STR_TO_SYMPTOMS['hard_time_waking_up']
HEADACHE = STR_TO_SYMPTOMS['headache']
CONFUSED = STR_TO_SYMPTOMS['confused']
LOST_CONSCIOUSNESS = STR_TO_SYMPTOMS['lost_consciousness']
TROUBLE_BREATHING = STR_TO_SYMPTOMS['trouble_breathing']
SNEEZING = STR_TO_SYMPTOMS['sneezing']
COUGH = STR_TO_SYMPTOMS['cough']
RUNNY_NOSE = STR_TO_SYMPTOMS['runny_nose']
SORE_THROAT = STR_TO_SYMPTOMS['sore_throat']
SEVERE_CHEST_PAIN = STR_TO_SYMPTOMS['severe_chest_pain']
LIGHT_TROUBLE_BREATHING = STR_TO_SYMPTOMS['light_trouble_breathing']
MILD_TROUBLE_BREATHING = STR_TO_SYMPTOMS['mild_trouble_breathing']
MODERATE_TROUBLE_BREATHING = STR_TO_SYMPTOMS['moderate_trouble_breathing']
HEAVY_TROUBLE_BREATHING = STR_TO_SYMPTOMS['heavy_trouble_breathing']
LOSS_OF_TASTE = STR_TO_SYMPTOMS['loss_of_taste']
ACHES = STR_TO_SYMPTOMS['aches']
# commented out because these are not used elsewhere for now
# HIVES = STR_TO_SYMPTOMS['hives']
# SWELLING = STR_TO_SYMPTOMS['swelling']


class SymptomGroups:
    DROP_IN_GROUPS = [
        [MILD],
        [MODERATE],
        [SEVERE],
        [EXTREMELY_SEVERE],
        [FEVER],
        [FEVER, MODERATE],
        [FEVER, CHILLS],
        [GASTRO],
        [GASTRO, DIARRHEA],
        [GASTRO, DIARRHEA, NAUSEA_VOMITING],
        [GASTRO, NAUSEA_VOMITING],
        [FATIGUE],
        [FATIGUE, UNUSUAL],
        [FATIGUE, LOST_CONSCIOUSNESS],
        [FATIGUE, HARD_TIME_WAKING_UP],
        [FATIGUE, HEADACHE],
        [FATIGUE, CONFUSED],
        [TROUBLE_BREATHING],
        [TROUBLE_BREATHING, SEVERE_CHEST_PAIN],
        [TROUBLE_BREATHING, SNEEZING],
        [TROUBLE_BREATHING, COUGH],
        [TROUBLE_BREATHING, RUNNY_NOSE],
        [TROUBLE_BREATHING, SORE_THROAT],
        [LOSS_OF_TASTE],
        [GASTRO, MODERATE],
        [FATIGUE, GASTRO],
        [MILD, MILD_TROUBLE_BREATHING, MILD_TROUBLE_BREATHING, LIGHT_TROUBLE_BREATHING],
        [MODERATE, TROUBLE_BREATHING, MODERATE_TROUBLE_BREATHING],
        [SEVERE, TROUBLE_BREATHING, HEAVY_TROUBLE_BREATHING],
        [EXTREMELY_SEVERE, TROUBLE_BREATHING, HEAVY_TROUBLE_BREATHING],
    ]

    @classmethod
    def sample(cls, rng: np.random.RandomState, p_num_drops: typing.List[int]):
        assert len(cls.DROP_IN_GROUPS) >= len(p_num_drops) > 0
        p_num_drops = np.array(p_num_drops) / sum(p_num_drops)
        # Sample the number of symptom groups to drop-in
        num_drops = rng.choice(list(range(1, len(p_num_drops) + 1)),
                               p=p_num_drops)
        # Sample that many symptom groups
        dropin_groups = rng.choice(cls.DROP_IN_GROUPS,
                                   size=num_drops, replace=False).tolist()
        return dropin_groups


#
# DISEASES PHASES
#


COVID_FIRST_PHASE = 10  # Used to delimit COVID from other diseases
COVID_INCUBATION = 10 + 0
COVID_ONSET = 10 + 1
COVID_PLATEAU = 10 + 2
COVID_POST_PLATEAU_1 = 10 + 3
COVID_POST_PLATEAU_2 = 10 + 4
ALLERGY_FIRST_PHASE = 20  # Used to delimit allergies from other diseases
ALLERGY_MAIN = 20 + 0
COLD_FIRST_PHASE = 30  # Used to delimit cold from other diseases
COLD_MAIN = 30 + 0
COLD_LAST_DAY = 30 + 1
FLU_FIRST_PHASE = 40  # Used to delimit flu from other diseases
FLU_FIRST_DAY = 40 + 0
FLU_MAIN = 40 + 1
FLU_LAST_DAY = 40 + 2

DISEASES_PHASES = {'covid': {0: COVID_INCUBATION, 1: COVID_ONSET, 2: COVID_PLATEAU,
                             3: COVID_POST_PLATEAU_1, 4: COVID_POST_PLATEAU_2},
                   'allergy': {0: ALLERGY_MAIN},
                   'cold': {0: COLD_MAIN, 1: COLD_LAST_DAY},
                   'flu': {0: FLU_FIRST_DAY, 1: FLU_MAIN, 2: FLU_LAST_DAY}}


def _disease_phase_idx_to_id(disease: str, phase_idx: int):
    if disease == 'covid':
        return phase_idx + COVID_FIRST_PHASE
    elif disease == 'allergy':
        return phase_idx + ALLERGY_FIRST_PHASE
    elif disease == 'cold':
        return phase_idx + COLD_FIRST_PHASE
    elif disease == 'flu':
        return phase_idx + FLU_FIRST_PHASE
    else:
        raise ValueError(f"Invalid disease: {disease}")


def _disease_phase_id_to_idx(disease: str, phase_id: int):
    if disease == 'covid':
        return phase_id - COVID_FIRST_PHASE
    elif disease == 'allergy':
        return phase_id - ALLERGY_FIRST_PHASE
    elif disease == 'cold':
        return phase_id - COLD_FIRST_PHASE
    elif disease == 'flu':
        return phase_id - FLU_FIRST_PHASE
    else:
        raise ValueError(f"Invalid disease: {disease}")


#
# DISEASES SYMPTOMS PROBABILITIES
#


SymptomProbability = namedtuple('SymptomProbability', ['name', 'id', 'probabilities'])
SymptomProbability.__doc__ = """A symptom probabilities collection given a disease phase

Attributes
    ----------
    name : str
        name of the symptom
    id : positive int
        id of the symptom
        This attribute should never change once set. It is used to define the
        position of the symptom in a multi-hot encoding
    probabilities : dict
        probabilities of the symptom per disease phase
        A probability of `-1` is assigned when it is heavily variable given
        multiple factors and is handled entirely in the code
        A probability of `None` is assigned when the symptom can be skipped
        entirely in the disease phase. The disease phase can also be removed 
        from the dict
"""


def _get_covid_fever_probability(phase_id: int, really_sick: bool, extremely_sick: bool,
                                 preexisting_conditions: list, initial_viral_load: float):
    p_fever = SYMPTOMS[FEVER].probabilities[phase_id]
    # covid_onset phase
    if phase_id == COVID_ONSET and \
            (really_sick or extremely_sick or
             len(preexisting_conditions) > 2 or initial_viral_load > 0.6):
        p_fever *= 2.
    # covid_plateau phase
    elif phase_id == COVID_PLATEAU and initial_viral_load > 0.6:
        p_fever = 1.
    return p_fever


def _get_covid_gastro_probability(phase_id: int, initial_viral_load: float):
    # gastro symptoms are more likely to be earlier
    p_gastro = initial_viral_load - 0.15
    # covid_onset phase
    if phase_id == COVID_ONSET:
        pass
    # covid_plateau phase
    elif phase_id == COVID_PLATEAU:
        p_gastro *= 0.25
    # covid_post_plateau_1 phase
    elif phase_id == COVID_POST_PLATEAU_1:
        p_gastro *= 0.1
    # covid_post_plateau_2 phase
    elif phase_id == COVID_POST_PLATEAU_2:
        p_gastro *= 0.1
    else:
        p_gastro = 0.
    return p_gastro


def _get_covid_fatigue_probability(phase_id: int, age: int, initial_viral_load: float,
                                   carefulness: float):
    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    p_lethargy = age / 200 + initial_viral_load * 0.6 - carefulness / 2
    # covid_onset phase
    if phase_id == COVID_ONSET:
        pass
    # covid_plateau phase
    elif phase_id == COVID_PLATEAU:
        # if you had gastro symptoms before you are more likely to be lethargic now
        # initial_viral_load - .15 is the same probaility than p_gastro
        # (previous code version was using p_gastro)
        p_lethargy = p_lethargy + initial_viral_load - 0.15
    # covid_post_plateau_1 phase
    elif phase_id == COVID_POST_PLATEAU_1:
        # if you had gastro symptoms before you are more likely to be lethargic now
        # initial_viral_load - .15 is the same probaility than p_gastro
        # (previous code version was using p_gastro)
        p_lethargy = p_lethargy * 1.5 + initial_viral_load - 0.15
    # covid_post_plateau_2 phase
    elif phase_id == COVID_POST_PLATEAU_2:
        # if you had gastro symptoms before you are more likely to be lethargic now
        # initial_viral_load - .15 is the same probaility than p_gastro
        # (previous code version was using p_gastro)
        p_lethargy = p_lethargy * 2. + initial_viral_load - 0.15
    else:
        p_lethargy = 0.

    # TODO: Make sure that it ok to have a p_lethargy >= to 1.
    return min(p_lethargy, 1.0)


def _get_covid_trouble_breathing_probability(phase_id: int, age: int, initial_viral_load: float,
                                             carefulness: float, preexisting_conditions: list):
    # covid_onset phase
    if phase_id == COVID_ONSET:
        # respiratory symptoms not so common at this stage
        # e.g. 0.5*0.5 - 0.7*0.25 = 0.25-0.17
        p_respiratory = 0.5 * initial_viral_load - carefulness * 0.25
    # covid_plateau phase
    elif phase_id == COVID_PLATEAU:
        # respiratory symptoms more common at this stage
        # e.g. 2* (0.5 - 0.7*0.25) = 2*(0.5-0.17)
        p_respiratory = 2 * (initial_viral_load - carefulness * 0.25)
    # covid_post_plateau_1 phase
    elif phase_id == COVID_POST_PLATEAU_1:
        # respiratory symptoms more common at this stage but less than plateau
        # The comment was modified to be consistent with the code
        # e.g. (0.5 - 0.7*0.25) = (0.5-0.17)
        p_respiratory = initial_viral_load - carefulness * 0.25
    # covid_post_plateau_2 phase
    elif phase_id == COVID_POST_PLATEAU_2:
        # respiratory symptoms getting less common
        # The comment was modified to be consistent with the code
        # e.g. 0.5* (0.5 - 0.7*0.25) = 0.5*(0.5-0.17)
        p_respiratory = 0.5 * (initial_viral_load - carefulness * 0.25)
    else:
        p_respiratory = 0.

    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
        p_respiratory = (p_respiratory * 4.) + age/200

    # TODO: Make sure that it ok to have a p_respiratory >= to 1.
    return min(p_respiratory, 1.0)


SYMPTOMS: typing.Dict[Symptom, SymptomProbability] = OrderedDict([
    # Sickness severity
    # A lot of symptoms are dependent on the sickness severity so severity
    # level needs to be first
    (
        MILD,
        SymptomProbability('mild', 1, {COVID_INCUBATION: 0.0,
                                       COVID_ONSET: -1,
                                       COVID_PLATEAU: -1,
                                       COVID_POST_PLATEAU_1: -1,
                                       COVID_POST_PLATEAU_2: -1,
                                       COLD_MAIN: -1,
                                       COLD_LAST_DAY: 1.0,
                                       FLU_FIRST_DAY: 1.0,
                                       FLU_MAIN: -1,
                                       FLU_LAST_DAY: 1.0})
    ),
    (
        MODERATE,
        SymptomProbability('moderate', 0, {COVID_INCUBATION: 0.0,
                                           COVID_ONSET: -1,
                                           COVID_PLATEAU: -1,
                                           COVID_POST_PLATEAU_1: -1,
                                           COVID_POST_PLATEAU_2: -1,
                                           COLD_MAIN: -1,
                                           COLD_LAST_DAY: 0.0,
                                           FLU_FIRST_DAY: 0.0,
                                           FLU_MAIN: -1,
                                           FLU_LAST_DAY: 0.0})
    ),
    (
        SEVERE,
        SymptomProbability('severe', 2, {COVID_INCUBATION: 0.0,
                                         COVID_ONSET: 0.0,
                                         COVID_PLATEAU: -1,
                                         COVID_POST_PLATEAU_1: -1,
                                         COVID_POST_PLATEAU_2: 0.0})
    ),
    (
        EXTREMELY_SEVERE,
        SymptomProbability('extremely-severe', 3, {COVID_INCUBATION: 0.0,
                                                   COVID_ONSET: 0.0,
                                                   COVID_PLATEAU: -1,
                                                   COVID_POST_PLATEAU_1: 0.0,
                                                   COVID_POST_PLATEAU_2: 0.0})
    ),

    # Symptoms

    (
        FEVER,
        SymptomProbability('fever', 4, {COVID_INCUBATION: 0.0,
                                        COVID_ONSET: 0.2,
                                        COVID_PLATEAU: 0.8,
                                        COVID_POST_PLATEAU_1: 0.0,
                                        COVID_POST_PLATEAU_2: 0.0,
                                        FLU_FIRST_DAY: 0.7,
                                        FLU_MAIN: 0.7,
                                        FLU_LAST_DAY: 0.3})
    ),
    # 'fever' is a dependency of 'chills' so it needs to be inserted before
    # this position
    (
        CHILLS,
        SymptomProbability('chills', 5, {COVID_INCUBATION: 0.0,
                                         COVID_ONSET: 0.8,
                                         COVID_PLATEAU: 0.5,
                                         COVID_POST_PLATEAU_1: 0.0,
                                         COVID_POST_PLATEAU_2: 0.0})
    ),

    (
        GASTRO,
        SymptomProbability('gastro', 6, {COVID_INCUBATION: 0.0,
                                         COVID_ONSET: -1,
                                         COVID_PLATEAU: -1,
                                         COVID_POST_PLATEAU_1: -1,
                                         COVID_POST_PLATEAU_2: -1,
                                         FLU_FIRST_DAY: 0.7,
                                         FLU_MAIN: 0.7,
                                         FLU_LAST_DAY: 0.2})
    ),
    # 'gastro' is a dependency of 'diarrhea' so it needs to be inserted before
    # this position
    (
        DIARRHEA,
        SymptomProbability('diarrhea', 7, {COVID_INCUBATION: 0.0,
                                           COVID_ONSET: 0.9,
                                           COVID_PLATEAU: 0.9,
                                           COVID_POST_PLATEAU_1: 0.9,
                                           COVID_POST_PLATEAU_2: 0.9,
                                           FLU_FIRST_DAY: 0.5,
                                           FLU_MAIN: 0.5,
                                           FLU_LAST_DAY: 0.5})
    ),
    # 'gastro' is a dependency of 'nausea_vomiting' so it needs to be inserted
    # before this position
    (
        NAUSEA_VOMITING,
        SymptomProbability('nausea_vomiting', 8, {COVID_INCUBATION: 0.0,
                                                  COVID_ONSET: 0.7,
                                                  COVID_PLATEAU: 0.7,
                                                  COVID_POST_PLATEAU_1: 0.7,
                                                  COVID_POST_PLATEAU_2: 0.7,
                                                  FLU_FIRST_DAY: 0.5,
                                                  FLU_MAIN: 0.5,
                                                  FLU_LAST_DAY: 0.25})
    ),

    # Age based lethargies
    # 'gastro' is a dependency of so it needs to be inserted before this
    # position
    (
        FATIGUE,
        SymptomProbability('fatigue', 9, {COVID_INCUBATION: 0.0,
                                          COVID_ONSET: -1,
                                          COVID_PLATEAU: -1,
                                          COVID_POST_PLATEAU_1: -1,
                                          COVID_POST_PLATEAU_2: -1,
                                          ALLERGY_MAIN: 0.2,
                                          COLD_MAIN: 0.8,
                                          COLD_LAST_DAY: 0.8,
                                          FLU_FIRST_DAY: 0.4,
                                          FLU_MAIN: 0.8,
                                          FLU_LAST_DAY: 0.8})
    ),
    (
        UNUSUAL,
        SymptomProbability('unusual', 10, {COVID_INCUBATION: 0.0,
                                           COVID_ONSET: 0.2,
                                           COVID_PLATEAU: 0.5,
                                           COVID_POST_PLATEAU_1: 0.5,
                                           COVID_POST_PLATEAU_2: 0.5})
    ),
    (
        HARD_TIME_WAKING_UP,
        SymptomProbability('hard_time_waking_up', 11, {COVID_INCUBATION: 0.0,
                                                       COVID_ONSET: 0.6,
                                                       COVID_PLATEAU: 0.6,
                                                       COVID_POST_PLATEAU_1: 0.6,
                                                       COVID_POST_PLATEAU_2: 0.6,
                                                       ALLERGY_MAIN: 0.3,
                                                       FLU_FIRST_DAY: 0.3,
                                                       FLU_MAIN: 0.5,
                                                       FLU_LAST_DAY: 0.4})
    ),
    (
        HEADACHE,
        SymptomProbability('headache', 12, {COVID_INCUBATION: 0.0,
                                            COVID_ONSET: 0.5,
                                            COVID_PLATEAU: 0.5,
                                            COVID_POST_PLATEAU_1: 0.5,
                                            COVID_POST_PLATEAU_2: 0.5,
                                            ALLERGY_MAIN: 0.6})
    ),
    (
        CONFUSED,
        SymptomProbability('confused', 13, {COVID_INCUBATION: 0.0,
                                            COVID_ONSET: 0.1,
                                            COVID_PLATEAU: 0.1,
                                            COVID_POST_PLATEAU_1: 0.1,
                                            COVID_POST_PLATEAU_2: 0.1})
    ),
    (
        LOST_CONSCIOUSNESS,
        SymptomProbability('lost_consciousness', 14, {COVID_INCUBATION: 0.0,
                                                      COVID_ONSET: 0.1,
                                                      COVID_PLATEAU: 0.1,
                                                      COVID_POST_PLATEAU_1: 0.1,
                                                      COVID_POST_PLATEAU_2: 0.1})
    ),

    # Respiratory symptoms
    # 'trouble_breathing' is a dependency of all this category so it should be
    # inserted before them
    (
        TROUBLE_BREATHING,
        SymptomProbability('trouble_breathing', 15, {COVID_INCUBATION: 0.0,
                                                     COVID_ONSET: -1,
                                                     COVID_PLATEAU: -1,
                                                     COVID_POST_PLATEAU_1: -1,
                                                     COVID_POST_PLATEAU_2: -1})
    ),
    (
        SNEEZING,
        SymptomProbability('sneezing', 16, {COVID_INCUBATION: 0.0,
                                            COVID_ONSET: 0.2,
                                            COVID_PLATEAU: 0.3,
                                            COVID_POST_PLATEAU_1: 0.3,
                                            COVID_POST_PLATEAU_2: 0.3,
                                            ALLERGY_MAIN: 1.0,
                                            COLD_MAIN: 0.4,
                                            COLD_LAST_DAY: 0.0})
    ),
    (
        COUGH,
        SymptomProbability('cough', 17, {COVID_INCUBATION: 0.0,
                                         COVID_ONSET: 0.6,
                                         COVID_PLATEAU: 0.9,
                                         COVID_POST_PLATEAU_1: 0.9,
                                         COVID_POST_PLATEAU_2: 0.9,
                                         COLD_MAIN: 0.8,
                                         COLD_LAST_DAY: 0.8})
    ),
    (
        RUNNY_NOSE,
        SymptomProbability('runny_nose', 18, {COVID_INCUBATION: 0.0,
                                              COVID_ONSET: 0.1,
                                              COVID_PLATEAU: 0.2,
                                              COVID_POST_PLATEAU_1: 0.2,
                                              COVID_POST_PLATEAU_2: 0.2,
                                              COLD_MAIN: 0.8,
                                              COLD_LAST_DAY: 0.8})
    ),
    (
        SORE_THROAT,
        SymptomProbability('sore_throat', 20, {COVID_INCUBATION: 0.0,
                                               COVID_ONSET: 0.5,
                                               COVID_PLATEAU: 0.8,
                                               COVID_POST_PLATEAU_1: 0.8,
                                               COVID_POST_PLATEAU_2: 0.8,
                                               ALLERGY_MAIN: 0.3,
                                               COLD_MAIN: 0.0,
                                               COLD_LAST_DAY: 0.6})
    ),
    (
        SEVERE_CHEST_PAIN,
        SymptomProbability('severe_chest_pain', 21, {COVID_INCUBATION: 0.0,
                                                     COVID_ONSET: 0.4,
                                                     COVID_PLATEAU: 0.5,
                                                     COVID_POST_PLATEAU_1: 0.15,
                                                     COVID_POST_PLATEAU_2: 0.15})
    ),

    # 'trouble_breathing' is a dependency of any '*_trouble_breathing' so it
    # needs to be inserted before this position
    (
        LIGHT_TROUBLE_BREATHING,
        SymptomProbability('light_trouble_breathing', 24, {COVID_INCUBATION: 0.0,
                                                           COVID_ONSET: -1,
                                                           COVID_PLATEAU: -1,
                                                           COVID_POST_PLATEAU_1: -1,
                                                           COVID_POST_PLATEAU_2: -1,
                                                           ALLERGY_MAIN: 0.02})
    ),
    # This symptoms was in fact a mislabeled light_trouble_breathing
    (
        MILD_TROUBLE_BREATHING,
        SymptomProbability('mild_trouble_breathing', 23, {})
    ),
    (
        MODERATE_TROUBLE_BREATHING,
        SymptomProbability('moderate_trouble_breathing', 25, {COVID_INCUBATION: 0.0,
                                                              COVID_ONSET: -1,
                                                              COVID_PLATEAU: -1,
                                                              COVID_POST_PLATEAU_1: -1,
                                                              COVID_POST_PLATEAU_2: -1})
    ),
    (
        HEAVY_TROUBLE_BREATHING,
        SymptomProbability('heavy_trouble_breathing', 26, {COVID_INCUBATION: 0.0,
                                                           COVID_ONSET: 0,
                                                           COVID_PLATEAU: -1,
                                                           COVID_POST_PLATEAU_1: -1,
                                                           COVID_POST_PLATEAU_2: -1})
    ),

    (
        LOSS_OF_TASTE,
        SymptomProbability('loss_of_taste', 22, {COVID_INCUBATION: 0.0,
                                                 COVID_ONSET: 0.25,
                                                 COVID_PLATEAU: 0.35,
                                                 COVID_POST_PLATEAU_1: 0.0,
                                                 COVID_POST_PLATEAU_2: 0.0})
    ),

    (
        ACHES,
        SymptomProbability('aches', 19, {FLU_FIRST_DAY: 0.3,
                                         FLU_MAIN: 0.5,
                                         FLU_LAST_DAY: 0.8})
    )

    # commented out because these are not used elsewhere for now
    # (
    #     HIVES,
    #     SymptomProbability('hives', __, {ALLERGY_MAIN: 0.4})
    # ),
    # (
    #     SWELLING,
    #     SymptomProbability('swelling', __, {ALLERGY_MAIN: 0.3})
    # )
])


def _get_covid_sickness_severity(rng, phase_id: int, really_sick: bool, extremely_sick: bool,
                                 preexisting_conditions: list, initial_viral_load: float):
    # covid_incubation
    if phase_id == COVID_INCUBATION:
        return None
    # covid_onset phase
    elif phase_id == COVID_ONSET:
        if really_sick or extremely_sick or len([i for i in CONDITIONS_CAUSING_MODERATE if
                                                 i in preexisting_conditions]) > 2 or initial_viral_load > 0.6:
            return MODERATE
        else:
            return MILD
    # covid_plateau phase
    elif phase_id == COVID_PLATEAU:
        if extremely_sick:
            return EXTREMELY_SEVERE
        elif really_sick or len(preexisting_conditions) > 2 or initial_viral_load > 0.6:
            return SEVERE
        elif rng.rand() < initial_viral_load - .15:
            return MODERATE
        else:
            return MILD
    # covid_post_plateau_1 phase
    elif phase_id == COVID_POST_PLATEAU_1:
        if extremely_sick:
            return SEVERE
        elif really_sick:
            return MODERATE
        else:
            return MILD
    # covid_post_plateau_2 phase
    elif phase_id == COVID_POST_PLATEAU_2:
        if extremely_sick:
            return MODERATE
        else:
            return MILD
    else:
        raise ValueError(f"Invalid phase_id [{phase_id}]")


def _get_covid_trouble_breathing_severity(sickness_severity: str, symptoms: list):
    if TROUBLE_BREATHING not in symptoms:
        return None

    if sickness_severity == MILD:
        return LIGHT_TROUBLE_BREATHING
    elif sickness_severity == MODERATE:
        return MODERATE_TROUBLE_BREATHING
    elif sickness_severity in (SEVERE, EXTREMELY_SEVERE):
        return HEAVY_TROUBLE_BREATHING
    else:
        raise ValueError(f"Invalid sickness_severity [{sickness_severity}]")


def _get_covid_symptoms(symptoms_progression: list, phase_id: int, rng, really_sick: bool,
                        extremely_sick: bool, age: int, initial_viral_load: float,
                        carefulness: float, preexisting_conditions: list):
    symptoms = []

    # covid_incubation phase is symptoms-free
    if phase_id == COVID_INCUBATION:
        return symptoms

    sickness_severity = _get_covid_sickness_severity(
        rng, phase_id, really_sick, extremely_sick,
        preexisting_conditions, initial_viral_load)

    symptoms.append(sickness_severity)

    # fever related computations
    # covid_onset phase
    if phase_id == COVID_ONSET:
        p_fever = _get_covid_fever_probability(phase_id,
                                               really_sick, extremely_sick,
                                               preexisting_conditions,
                                               initial_viral_load)

        if rng.rand() < p_fever:
            symptoms.append(FEVER)

            if extremely_sick and rng.rand() < SYMPTOMS[CHILLS].probabilities[phase_id]:
                symptoms.append(CHILLS)

    # covid_plateau phase
    elif phase_id == COVID_PLATEAU:
        if FEVER in symptoms_progression[-1]:
            p_fever = 1.
        else:
            p_fever = _get_covid_fever_probability(phase_id,
                                                   really_sick, extremely_sick,
                                                   preexisting_conditions,
                                                   initial_viral_load)

        if rng.rand() < p_fever:
            symptoms.append(FEVER)

            if rng.rand() < SYMPTOMS[CHILLS].probabilities[phase_id]:
                symptoms.append(CHILLS)

    # gastro related computations
    if GASTRO in symptoms_progression[-1]:
        p_gastro = 1.
    else:
        p_gastro = _get_covid_gastro_probability(phase_id,
                                                 initial_viral_load)

    # gastro symptoms are more likely to show extreme symptoms later
    if rng.rand() < p_gastro:
        symptoms.append(GASTRO)

        for symptom in (DIARRHEA, NAUSEA_VOMITING):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase_id]:
                symptoms.append(symptom)

    # fatigue related computations
    p_lethargy = _get_covid_fatigue_probability(phase_id,
                                                age,
                                                initial_viral_load,
                                                carefulness)

    if rng.rand() < p_lethargy:
        symptoms.append(FATIGUE)

        if age > 75 and rng.rand() < SYMPTOMS[UNUSUAL].probabilities[phase_id]:
            symptoms.append(UNUSUAL)
        if (really_sick or extremely_sick or len(preexisting_conditions) > 2) and \
                rng.rand() < SYMPTOMS[LOST_CONSCIOUSNESS].probabilities[phase_id]:
            symptoms.append(LOST_CONSCIOUSNESS)

        for symptom in (HARD_TIME_WAKING_UP, HEADACHE, CONFUSED):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase_id]:
                symptoms.append(symptom)

    # trouble_breathing related computations
    p_respiratory = _get_covid_trouble_breathing_probability(phase_id,
                                                             age,
                                                             initial_viral_load,
                                                             carefulness,
                                                             preexisting_conditions)

    if rng.rand() < p_respiratory:
        symptoms.append(TROUBLE_BREATHING)

        if extremely_sick and rng.rand() < SYMPTOMS[SEVERE_CHEST_PAIN].probabilities[phase_id]:
            symptoms.append(SEVERE_CHEST_PAIN)

        for symptom in (SNEEZING, COUGH, RUNNY_NOSE, SORE_THROAT):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase_id]:
                symptoms.append(symptom)

    trouble_breathing_severity = _get_covid_trouble_breathing_severity(sickness_severity, symptoms)
    if trouble_breathing_severity is not None:
        symptoms.append(trouble_breathing_severity)

    # loss_of_taste related computations
    if phase_id in (COVID_ONSET, COVID_PLATEAU) and \
            LOSS_OF_TASTE in symptoms_progression[-1]:
        p_loss_of_taste = 1.
    else:
        p_loss_of_taste = SYMPTOMS[LOSS_OF_TASTE].probabilities[phase_id]

    if rng.rand() < p_loss_of_taste:
        symptoms.append(LOSS_OF_TASTE)

    return symptoms


# 2D Array of symptoms; first axis is days after exposure (infection), second is an array of symptoms
def _get_covid_progression(initial_viral_load, viral_load_plateau_start, viral_load_plateau_end,
                           recovery_days, age, incubation_days, infectiousness_onset_days,
                           really_sick, extremely_sick, rng, preexisting_conditions, carefulness):
    """
    [summary]

    Args:
        initial_viral_load ([type]): [description]
        viral_load_plateau_start ([type]): [description]
        viral_load_plateau_end ([type]): [description]
        recovery_days (float): time to recover
        age ([type]): [description]
        incubation_days ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]
        rng ([type]): [description]
        preexisting_conditions ([type]): [description]
        carefulness ([type]): [description]

    Returns:
        [type]: [description]
    """
    progression = []
    symptoms_per_phase = []

    # Phase 0 - Before onset of symptoms (incubation)
    # ====================================================
    symptoms = _get_covid_symptoms(
        symptoms_per_phase, COVID_INCUBATION, rng, really_sick,
        extremely_sick, age, initial_viral_load,
        carefulness, preexisting_conditions)
    symptoms_per_phase.append(symptoms)

    # Phase 1 - Onset of symptoms (including plateau Part 1)
    # ====================================================
    symptoms = _get_covid_symptoms(
        symptoms_per_phase, COVID_ONSET, rng, really_sick,
        extremely_sick, age, initial_viral_load,
        carefulness, preexisting_conditions)
    symptoms_per_phase.append(symptoms)

    # During the symptoms plateau Part 2 (worst part of the disease)
    # ====================================================
    symptoms = _get_covid_symptoms(
        symptoms_per_phase, COVID_PLATEAU, rng, really_sick,
        extremely_sick, age, initial_viral_load,
        carefulness, preexisting_conditions)
    symptoms_per_phase.append(symptoms)

    # After the plateau (recovery part 1)
    # ====================================================
    symptoms = _get_covid_symptoms(
        symptoms_per_phase, COVID_POST_PLATEAU_1, rng, really_sick,
        extremely_sick, age, initial_viral_load,
        carefulness, preexisting_conditions)
    symptoms_per_phase.append(symptoms)

    # After the plateau (recovery part 2)
    # ====================================================
    symptoms = _get_covid_symptoms(
        symptoms_per_phase, COVID_POST_PLATEAU_2, rng, really_sick,
        extremely_sick, age, initial_viral_load,
        carefulness, preexisting_conditions)
    symptoms_per_phase.append(symptoms)

    viral_load_plateau_duration = math.ceil(viral_load_plateau_end - viral_load_plateau_start)
    recovery_duration = math.ceil(recovery_days - viral_load_plateau_end)
    incubation_days_wrt_infectiousness_onset_days = incubation_days - infectiousness_onset_days

    # same delay in symptom plateau as there was in symptom onset
    incubation_duration = math.ceil(incubation_days)
    covid_onset_duration = math.ceil(viral_load_plateau_start -
                                     incubation_days_wrt_infectiousness_onset_days) + \
                           viral_load_plateau_duration // 3
    plateau_duration = math.ceil(viral_load_plateau_duration * 2/3)
    post_plateau_1_duration = recovery_duration // 2
    post_plateau_2_duration = recovery_duration - post_plateau_1_duration

    assert viral_load_plateau_start >= incubation_days_wrt_infectiousness_onset_days

    for duration, symptoms in zip((incubation_duration, covid_onset_duration, plateau_duration,
                                   post_plateau_1_duration, post_plateau_2_duration),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _get_allergy_progression(rng):
    """
    [summary]

    Args:
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    disease_phases = DISEASES_PHASES['allergy']
    phase_i = 0
    phase = disease_phases[phase_i]

    symptoms = []
    for symptom in (SNEEZING, LIGHT_TROUBLE_BREATHING, SORE_THROAT, FATIGUE,
                    HARD_TIME_WAKING_UP, HEADACHE):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms.append(symptom)

            # commented out because these are not used elsewhere for now
            # if symptom == 'light_trouble_breathing':
            #     for symptom in ('hives', 'swelling'):
            #         rand = rng.rand()
            #         if rand < SYMPTOMS[symptom].probabilities[phase]:
            #             symptoms.append(symptom)
    progression = [symptoms]
    return progression


def _get_flu_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick, AVG_FLU_DURATION):
    """
    [summary]

    Args:
        age ([type]): [description]
        rng ([type]): [description]
        carefulness ([type]): [description]
        preexisting_conditions ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]

    Returns:
        [type]: [description]
    """
    disease_phases = DISEASES_PHASES['flu']
    symptoms_per_phase = [[] for _ in range(len(disease_phases))]

    # Day 1 symptoms:
    phase_i = 0
    phase = disease_phases[phase_i]

    symptoms_per_phase[phase_i].append(MILD)

    for symptom in (FATIGUE, FEVER, ACHES, HARD_TIME_WAKING_UP, GASTRO):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == GASTRO:
                for symptom in (DIARRHEA, NAUSEA_VOMITING):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    # Day 2-4ish if it's a longer flu, if 2 days long this doesn't get added
    phase_i = 1
    phase = disease_phases[phase_i]

    if really_sick or extremely_sick or any([i for i in CONDITIONS_CAUSING_MODERATE if i in preexisting_conditions]):
        symptoms_per_phase[phase_i].append(MODERATE)
    else:
        symptoms_per_phase[phase_i].append(MILD)

    for symptom in (FATIGUE, FEVER, ACHES, HARD_TIME_WAKING_UP, GASTRO):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == GASTRO:
                for symptom in (DIARRHEA, NAUSEA_VOMITING):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    # Last day
    phase_i = 2
    phase = disease_phases[phase_i]

    symptoms_per_phase[phase_i].append(MILD)

    for symptom in (FATIGUE, FEVER, ACHES, HARD_TIME_WAKING_UP, GASTRO):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == GASTRO:
                for symptom in (DIARRHEA, NAUSEA_VOMITING):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = AVG_FLU_DURATION + 2 - 2 * carefulness
    else:
        mean = AVG_FLU_DURATION - 2 * carefulness

    len_flu = rng.normal(mean,3)

    if len_flu < 2:
        len_flu = 3
    else:
        len_flu = round(len_flu)

    progression = []
    for duration, symptoms in zip((1, len_flu - 2, 1), symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _get_cold_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick):
    """
    [summary]

    Args:
        age ([type]): [description]
        rng ([type]): [description]
        carefulness ([type]): [description]
        preexisting_conditions ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]

    Returns:
        [type]: [description]
    """
    disease_phases = DISEASES_PHASES['cold']

    symptoms_per_phase = [[] for _ in range(len(disease_phases))]

    # Day 2-4ish if it's a longer cold, if 2 days long this doesn't get added
    phase_i = 0
    phase = disease_phases[phase_i]

    if really_sick or extremely_sick or any([i for i in CONDITIONS_CAUSING_MODERATE if i in preexisting_conditions]):
        symptoms_per_phase[phase_i].append(MODERATE)
    else:
        symptoms_per_phase[phase_i].append(MILD)

    for symptom in (RUNNY_NOSE, COUGH, FATIGUE, SNEEZING):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

    # Last day
    phase_i = 1
    phase = disease_phases[phase_i]

    symptoms_per_phase[phase_i].append(MILD)

    for symptom in (RUNNY_NOSE, COUGH, FATIGUE, SORE_THROAT):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 4 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_cold = rng.normal(mean,3)
    if len_cold < 1:
        len_cold = 1
    else:
        len_cold = math.ceil(len_cold)

    progression = [[]]
    for duration, symptoms in zip((len_cold - 1, 1),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _reported_symptoms(all_symptoms, rng, carefulness):
    all_reported_symptoms = []
    for symptoms in all_symptoms:
        reported_symptoms = []
        # miss a day of symptoms
        if rng.rand() < carefulness:
            continue
        for symptom in symptoms:
            if rng.rand() < carefulness:
                continue
            reported_symptoms.append(symptom)
        all_reported_symptoms.append(reported_symptoms)
    return all_reported_symptoms
