"""
[summary]
"""

#
# Time.
#
# Time is a complicated business.
#
# - Everyone agrees what a second, a minute and an hour mean.
#
# - People will define a day variously:
#
#     - "The time it takes for the Earth to rotate 360 degrees" (sidereal day)
#     - "The time between consecutive local noons of the Sun"   (mean solar day)
#         NB: This is not the same as the sidereal day because the Earth,
#             having advanced along its orbit of the Sun, must rotate a tiny
#             bit more for the Sun to achieve local noon again.
#     - "24 hours"                                              (ephemeris day)
#
#   However, if pressed to give an exact duration, most people will agree that
#   a day is exactly 24 hours, or 86400 seconds (ephemeris day). The definition
#   of the second makes the mean solar day currently 86400.002 seconds long,
#   while the sidereal day is only 86164.0905 seconds long (~4 minutes short).
#
# - Everyone agrees that a week lasts 7 days, but don't agree on the first day
#   of the week. Python thinks the zeroth day of the week is Monday; C
#   specifies the zeroth day of the week as Sunday.
#
# - The definition of a year is most confusing. Some people will say:
#
#     - "The time taken to orbit the Sun once"          (sidereal year)
#     - "The time between consecutive summer solstices" (tropical year)
#         NB: This is not the same as the sidereal year because of the
#             precession of the equinoxes.
#     - "365 days except on leap years, when there are 366 days."
#
#   The Gregorian calendar leap-year rule (365 + 1/4 - 3/400 ephemeris days)
#   attempts to approximate the tropical year (because the Paschal computus
#   is based on the vernal equinox). Most people use the Gregorian calendar.
#
#   Therefore, for very-many-year durations (e.g. a person's age), most people
#   mean tropical years, but for one year people generally "mean" 365 or 366
#   ephemeris days.
#
SECONDS_PER_MINUTE        = 60
SECONDS_PER_HOUR          = 60*60                                    # 3600
SECONDS_PER_DAY           = 24*60*60                                 # 86400
SECONDS_PER_EPHEMERIS_DAY = SECONDS_PER_DAY                          # Most people, if asked, would agree that one day is
                                                                     # exactly 24 hours, which is exactly the "ephemeris day".
SECONDS_PER_WEEK          = 7*24*60*60                               # 604800
SECONDS_PER_YEAR          = int(365.00000*SECONDS_PER_EPHEMERIS_DAY) # 365 ephemeris days
SECONDS_PER_TROPICAL_YEAR = int(365.24219*SECONDS_PER_EPHEMERIS_DAY) # ~365 1/4 ephemeris days
SECONDS_PER_LEAP_YEAR     = int(366.00000*SECONDS_PER_EPHEMERIS_DAY) # 366 ephemeris days


# Gender
GENDER_MASK   = 3 <<  0
GENDER_FEMALE = 1 <<  0
GENDER_MALE   = 2 <<  0
GENDER_OTHER  = 3 <<  0


# Symptoms
## Severity
SYMPTOMS_SEVERITY_MASK                    = 7 <<  0
SYMPTOMS_SEVERITY_NONE                    = 0 <<  0
SYMPTOMS_SEVERITY_MILD                    = 1 <<  0
SYMPTOMS_SEVERITY_MODERATE                = 2 <<  0
SYMPTOMS_SEVERITY_SEVERE                  = 3 <<  0
SYMPTOMS_SEVERITY_EXTREMELYSEVERE         = 4 <<  0
## Type
SYMPTOMS_COUGH                            = 1 <<  3       # SY-2
SYMPTOMS_TROUBLE_BREATHING_MASK           = 3 <<  4       # SY-3
SYMPTOMS_TROUBLE_BREATHING_NONE           = 0 <<  4
SYMPTOMS_TROUBLE_BREATHING_LIGHT          = 1 <<  4
SYMPTOMS_TROUBLE_BREATHING_MODERATE       = 2 <<  4
SYMPTOMS_TROUBLE_BREATHING_HEAVY          = 3 <<  4
SYMPTOMS_FEVER                            = 1 <<  6       # SY-4
SYMPTOMS_ANOSMIA                          = 1 <<  7       # SY-5
SYMPTOMS_SNEEZING                         = 1 <<  8       # SY-6
SYMPTOMS_DIARRHEA                         = 1 <<  9       # SY-7
SYMPTOMS_NAUSEA                           = 1 << 10       # SY-8
SYMPTOMS_HEADACHE                         = 1 << 11       # SY-9
SYMPTOMS_FATIGUE                          = 1 << 12       # SY-10
SYMPTOMS_DIFFICULTY_WAKING_UP             = 1 << 13       # SY-11
SYMPTOMS_SORE_THROAT                      = 1 << 14       # SY-12
SYMPTOMS_MUSCLE_ACHES                     = 1 << 15       # SY-13
SYMPTOMS_RUNNY_NOSE                       = 1 << 16       # SY-14
SYMPTOMS_CHILLS                           = 1 << 17       # SY-15
SYMPTOMS_SEVERE_CHEST_PAINS               = 1 << 18       # SY-16
SYMPTOMS_CONFUSION                        = 1 << 19       # SY-17
SYMPTOMS_LOST_CONSCIOUSNESS               = 1 << 20       # SY-18


# Psychological Symptoms
PSY_SYMPTOMS_ANXIETY                      = 1 <<  0       # PS-1
PSY_SYMPTOMS_LONELINESS                   = 1 <<  1       # PS-2
PSY_SYMPTOMS_DEPRESSION                   = 1 <<  2       # PS-3


# Conditions
CONDITION_SMOKER                          = 1 <<  0       # UI-10
CONDITION_IMMUNOSUPPRESSED                = 1 <<  1       # UI-11
CONDITION_CANCER                          = 1 <<  2       # UI-12
CONDITION_DIABETES                        = 1 <<  3       # UI-13
CONDITION_HEART_DISEASE                   = 1 <<  4       # UI-14
CONDITION_HYPERTENSION                    = 1 <<  5       # UI-15
CONDITION_ASTHMA                          = 1 <<  6       # UI-16
CONDITION_NEUROLOGICAL_DISORDER           = 1 <<  7       # UI-17

"""STUFF THAT WAS IN CONFIG BUT IS ACTUALLY A CONSTANT"""
BIG_NUMBER = 10000000
