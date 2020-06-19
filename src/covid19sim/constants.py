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

BIG_NUMBER = 10000000
