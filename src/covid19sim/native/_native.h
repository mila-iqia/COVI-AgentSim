/**
 * Includes
 */

#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>



/* Defines */

/**
 *  Time.
 * 
 *  Time is a complicated business.
 * 
 *  - Everyone agrees what a second, a minute and an hour mean.
 * 
 *  - People will define a day variously:
 * 
 *      - "The time it takes for the Earth to rotate 360 degrees" (sidereal day)
 *      - "The time between consecutive local noons of the Sun"   (mean solar day)
 *          NB: This is not the same as the sidereal day because the Earth,
 *              having advanced along its orbit of the Sun, must rotate a tiny
 *              bit more for the Sun to achieve local noon again.
 *      - "24 hours"                                              (ephemeris day)
 * 
 *    However, if pressed to give an exact duration, most people will agree that
 *    a day is exactly 24 hours, or 86400 seconds (ephemeris day). The definition
 *    of the second makes the mean solar day currently 86400.002 seconds long,
 *    while the sidereal day is only 86164.0905 seconds long (~4 minutes short).
 * 
 *  - Everyone agrees that a week lasts 7 days, but don't agree on the first day
 *    of the week. Python thinks the zeroth day of the week is Monday; C
 *    specifies the zeroth day of the week as Sunday.
 * 
 *  - The definition of a year is most confusing. Some people will say:
 * 
 *      - "The time taken to orbit the Sun once"          (sidereal year)
 *      - "The time between consecutive summer solstices" (tropical year)
 *          NB: This is not the same as the sidereal year because of the
 *              precession of the equinoxes.
 *      - "365 days except on leap years, when there are 366 days."
 * 
 *    The Gregorian calendar leap-year rule (365 + 1/4 - 3/400 ephemeris days)
 *    attempts to approximate the tropical year (because the Paschal computus
 *    is based on the vernal equinox). Most people use the Gregorian calendar.
 * 
 *    Therefore, for very-many-year durations (e.g. a person's age), most people
 *    mean tropical years, but for one year people generally "mean" 365 or 366
 *    ephemeris days.
 */

#define SECONDS_PER_MINUTE        (60)
#define SECONDS_PER_HOUR          (60*60)                                    /* 3600 */
#define SECONDS_PER_DAY           (24*60*60)                                 /* 86400 */
#define SECONDS_PER_EPHEMERIS_DAY SECONDS_PER_DAY                            /* Most people, if asked, would agree that one day is
                                                                                exactly 24 hours, which is exactly the "ephemeris day". */
#define SECONDS_PER_WEEK          (7*24*60*60)                               /* 604800 */
#define SECONDS_PER_YEAR          (365*SECONDS_PER_EPHEMERIS_DAY)            /* 365 ephemeris days */
#define SECONDS_PER_TROPICAL_YEAR (31556925)                                 /* 365.24219 ephemeris days */
#define SECONDS_PER_LEAP_YEAR     (366*SECONDS_PER_EPHEMERIS_DAY)            /* 366 ephemeris days */

#define HUMAN_SYMPTOM_SNEEZING 1<<0 /* Example */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declaration & Typedef */
typedef struct BaseEnvironmentObject BaseEnvironmentObject;
// typedef struct BaseHumanObject       BaseHumanObject;

extern PyTypeObject BaseEnvironmentType;
// extern PyTypeObject BaseHumanType;


/* Data Structure and Constant Definitions */

/**
 * @brief The BaseEnvironmentObject struct
 */

struct BaseEnvironmentObject{
    PyObject_HEAD
    PyObject* __dict__;
    PyObject* __weaklist__;
    double    ts_now;
    double    ts_initial;
    double    ts_initial_midnight;/* Midnight preceding ts_initial. */
    double    ts_initial_monday;  /* Monday midnight preceding ts_initial. */
    PyObject* active_process;
    PyObject* queue;
};


// /**
//  * @brief The BaseHumanObject struct
//  */

// struct BaseHumanObject{
//     PyObject_HEAD
//     PyObject*              __dict__;
//     PyObject*              __weaklist__;
//     BaseEnvironmentObject* env;
//     PyObject*              name;
//     double                 ts_birth;
//     double                 ts_death;
//     double                 ts_cold_symptomatic,  ts_flu_symptomatic,    ts_allergy_symptomatic;
//     double                 ts_covid19_infection, ts_covid19_infectious, ts_covid19_symptomatic,
//                            ts_covid19_recovery,  ts_covid19_immunity;
//     double                 infectiousness_onset_days;
//     double                 incubation_days;
//     char                   is_asymptomatic;
// };


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif
