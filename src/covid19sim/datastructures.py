"""
[summary]
"""
import covid19sim.constants as constants


class HumanBase:
    """
    Human Base Class

    Includes several "slots" for extremely commonly-used attributes which need
    not indirect via the __dict__.
    """

    __slots__ = [
        #
        # App-Recorded Information
        #
        'uuid',                    # Unique User ID, 128 bits.
        'ts_born',                 # [UI-1] POSIX timestamp of birthday. (Age can be computed in-sim)
        'ts_symptomatic',          # [SY-1/DIG-3] POSIX timestamp of first appearance of symptoms
        'weight',                  # [UI-19], float, kg
        'height',                  # [UI-20], float, m
        'flags_basicinfo',         # [UI-2, UI-3, UI-4, UI-5, UI-6, UI-7, UI-8, UI-18]
        'flags_condition',         # [UI-9 (implicit), UI-10 to UI-17 (explicit)]
        'flags_symptoms',          # [SY-2 to SY-18]
        'flags_psych',             # [PS-1, PS-2, PS-3]
        'ts_test'                  # [DIG-4] POSIX timestamp of test.

        #
        # Latent Simulation Information
        # This includes the user's mobility, disease progression and "fate".
        #
        'ts_exposed',              # Timestamp of exposure
        'ts_infectious',           # Timestamp of infectious phase start
        'ts_viral_plateau_start',  # Timestamp of viral plateau start
        'ts_viral_plateau_end',    # Timestamp of viral plateau end
        'ts_recovered',            # Timestamp of recovery
        'ts_dead',                 # Timestamp of death
        'viral_plateau_height',    # Viral plateau height
        'flags_status',            # SEIR Status + Immune + Dead flags.

        # Python Attributes
        '__dict__',                # Delete to prevent new attributes being added.
        '__weakref__',
    ]


    @property
    def symptom_severity(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return (self.flags_symptoms & constants.SYMPTOMS_SEVERITY_MASK) >> 0

    @symptom_severity.setter
    def symptom_severity(self, v):
        """
        [summary]

        Args:
            v ([type]): [description]
        """
        v <<=  0
        v  &=  constants.SYMPTOMS_SEVERITY_MASK
        v  |= ~constants.SYMPTOMS_SEVERITY_MASK & self.flags_symptoms
        self.flags_symptoms = v


