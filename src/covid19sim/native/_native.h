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
#define HUMAN_SYMPTOM_SNEEZING 1<<0 /* Example */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declaration & Typedef */
typedef struct BaseEnvironmentObject BaseEnvironmentObject;
typedef struct BaseHumanObject       BaseHumanObject;

extern PyTypeObject BaseEnvironmentType;
extern PyTypeObject BaseHumanType;


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


/**
 * @brief The BaseHumanObject struct
 */

struct BaseHumanObject{
    PyObject_HEAD
    PyObject*              __dict__;
    PyObject*              __weakref__;
    BaseEnvironmentObject* env;
    double                 ts_birth;
    double                 ts_death;
    double                 ts_cold_symptomatic,  ts_flu_symptomatic,    ts_allergy_symptomatic;
    double                 ts_covid19_infection, ts_covid19_infectious, ts_covid19_symptomatic,
                           ts_covid19_recovery,  ts_covid19_immunity;
};


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif
