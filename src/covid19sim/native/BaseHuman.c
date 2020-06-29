/**
 * Includes
 */

#include "_native.h"        /* Because of "reasons", this header must be first. */



/* Defines */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Function Forward Declarations */
static int                    BaseHuman_set_env                 (BaseHumanObject* self, PyObject* val, void* closure);



/* BaseHuman Object Methods */
static BaseHumanObject*       BaseHuman_new                     (PyTypeObject* type,
                                                                 PyObject*     args,
                                                                 PyObject*     kwargs){
    BaseHumanObject* self;
    (void)args;
    (void)kwargs;
    
    self = (BaseHumanObject*)type->tp_alloc(type, 0);
    if(!self)
        return self;
    
    /* The __dict__ and __weaklist__ are empty for a new object, of course... */
    self->__dict__            = NULL;
    self->__weaklist__        = NULL;
    
    
    /* We set various timestamps to "never happened". */
    self->ts_birth               = INFINITY;
    self->ts_death               = INFINITY;
    self->ts_cold_symptomatic    = INFINITY;
    self->ts_flu_symptomatic     = INFINITY;
    self->ts_allergy_symptomatic = INFINITY;
    self->ts_covid19_infection   = INFINITY;
    self->ts_covid19_infectious  = INFINITY;
    self->ts_covid19_symptomatic = INFINITY;
    self->ts_covid19_recovery    = INFINITY;
    self->ts_covid19_immunity    = INFINITY;
    
    
    /* Return new object */
    return self;
}
static int                    BaseHuman_init                    (BaseHumanObject* self,
                                                                 PyObject*        args,
                                                                 PyObject*        kwargs){
    PyObject* env = NULL;
    
    /* Parse arguments */
    static char *kwargs_list[] = {"env", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwargs_list, &env))
        return -1;
    
    /**
     * Enforce that BaseHuman must receive a BaseEnvironment object as argument.
     * If it did, then increment its reference count and write it into the env
     * field.
     */
    
    if(BaseHuman_set_env(self, env, NULL) < 0)
        return -1;
    
    
    /* Init successful. */
    return 0;
}
static int                    BaseHuman_traverse          (BaseHumanObject* self,
                                                           visitproc        visit,
                                                           void*            arg){
    Py_VISIT(self->__dict__);
    Py_VISIT(self->__weaklist__);
    Py_VISIT(self->env);
    Py_VISIT(self->name);
    return 0;
}
static int                    BaseHuman_clear             (BaseHumanObject* self){
    Py_CLEAR(self->__dict__);
    Py_CLEAR(self->__weaklist__);
    Py_CLEAR(self->env);
    Py_CLEAR(self->name);
    return 0;
}
static void                   BaseHuman_dealloc           (BaseHumanObject* self){
    if(self->__weaklist__)
        PyObject_ClearWeakRefs((PyObject*)self);
    
    BaseHuman_clear(self);
    
    /**
     * Py_CLEAR() any other Python object members, which won't participate in
     * cycles.
     */
    
    /* ... */
    
    PyObject_GC_Del(self);
}

/* BaseHuman Object @properties (Getter/Setters) */
static PyObject*              BaseHuman_get_age                 (BaseHumanObject* self, void* closure){
    return PyLong_FromDouble((self->env->ts_now-self->ts_birth) / SECONDS_PER_TROPICAL_YEAR);
}
static int                    BaseHuman_set_age                 (BaseHumanObject* self, PyObject* val, void* closure){
    double years = PyFloat_AsDouble(val);
    if(PyErr_Occurred())
        return -1;
    self->ts_birth = self->env->ts_now - years*SECONDS_PER_TROPICAL_YEAR;
    return 0;
}
static PyObject*              BaseHuman_get_env                 (BaseHumanObject* self, void* closure){
    Py_XINCREF(self->env);
    return (PyObject*)self->env;
}
static int                    BaseHuman_set_env                 (BaseHumanObject* self, PyObject* val, void* closure){
    BaseEnvironmentObject* tmp;
    if(!PyObject_IsInstance((PyObject*)val, (PyObject*)&BaseEnvironmentType))
        return -1;
    Py_INCREF(val);
    tmp = self->env;
    self->env = (BaseEnvironmentObject*)val;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject*              BaseHuman_get_infection_timestamp (BaseHumanObject* self, void* closure){
    /**
     * The following implements the ugly logic of dynamically computing a datetime
     * infection timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_covid19_infection -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment.
     * 
     * NOTE: When datetime timestamps are ditched, this getter and its setter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* ts_plus_td    = NULL;
    
    if(self->ts_covid19_infection == INFINITY)
        Py_RETURN_NONE;
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)  goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)  goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj) goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)    goto fail;
    td            = PyObject_CallFunction(timedelta_obj, "ff", 0.0, self->ts_covid19_infection -
                                                                    self->env->ts_initial);
    if(!td)            goto fail;
    ts_plus_td    = PyNumber_Add(ts_initial, td);
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    return ts_plus_td;
}
static int                    BaseHuman_set_infection_timestamp (BaseHumanObject* self, PyObject* val, void* closure){
    /**
     * The following implements the ugly logic of dynamically setting a datetime
     * infection timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_covid19_infection -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment. To that end we compute the
     * floating-point timestamp that corresponds to that convoluted logic,
     * rather than through the straightforward conversion to a POSIX timestamp.
     * 
     * NOTE: When datetime timestamps are ditched, this setter and its getter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* td_seconds    = NULL;
    double    td_fp         = 0;
    int       ret           = -1;
    
    if(val == Py_None){
        self->ts_covid19_infection = INFINITY;
        return 0;
    }
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)    goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)    goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj)   goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)      goto fail;
    td            = PyNumber_Subtract(val, ts_initial);
    if(!td)              goto fail;
    td_seconds    = PyObject_CallMethod(td, "total_seconds", NULL);
    if(!td_seconds)      goto fail;
    td_fp         = PyFloat_AsDouble(td_seconds);
    if(PyErr_Occurred()) goto fail;
    
    self->ts_covid19_infection = self->env->ts_initial + td_fp;
    ret = 0;
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    Py_XDECREF(td_seconds);
    return ret;
}
static PyObject*              BaseHuman_get_recovered_timestamp (BaseHumanObject* self, void* closure){
    /**
     * The following implements the ugly logic of dynamically computing a datetime
     * recovered timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_covid19_recovery -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment.
     * 
     * To make things extra complicated, datetime.datetime.min and .max have
     * special meaning for recovered_timestamp.
     * 
     * NOTE: When datetime timestamps are ditched, this getter and its setter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* datetime_min  = NULL;
    PyObject* datetime_max  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* ts_plus_td    = NULL;
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)    goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)    goto fail;
    datetime_min  = PyObject_GetAttrString(datetime_obj, "min");
    if(!datetime_min)    goto fail;
    datetime_max  = PyObject_GetAttrString(datetime_obj, "max");
    if(!datetime_max)    goto fail;
    if(self->env->ts_now >= self->ts_death){
        /**
         * Death is unfortunately encoded as
         *     self.recovered_timestamp == datetime.datetime.max
         */
        
        Py_INCREF(datetime_max);
        ts_plus_td = datetime_max;
        goto success;
    }else if(self->env->ts_now < self->ts_covid19_recovery){
        /**
         * The not-yet-recovered condition is unfortunately encoded as
         *     self.recovered_timestamp == datetime.datetime.min
         */
        
        Py_INCREF(datetime_min);
        ts_plus_td = datetime_min;
        goto success;
    }
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj)   goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)      goto fail;
    td            = PyObject_CallFunction(timedelta_obj, "ff", 0.0, self->ts_covid19_recovery -
                                                                    self->env->ts_initial);
    if(!td)              goto fail;
    ts_plus_td    = PyNumber_Add(ts_initial, td);
    
    /* Success path. */
    success:
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(datetime_min);
    Py_XDECREF(datetime_max);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    return ts_plus_td;
}
static int                    BaseHuman_set_recovered_timestamp (BaseHumanObject* self, PyObject* val, void* closure){
    /**
     * The following implements the ugly logic of dynamically setting a datetime
     * recovery timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_covid19_recovery -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment. To that end we compute the
     * floating-point timestamp that corresponds to that convoluted logic,
     * rather than through the straightforward conversion to a POSIX timestamp.
     * 
     * To make things extra complicated, datetime.datetime.min and .max have
     * special meaning for recovered_timestamp.
     * 
     * NOTE: When datetime timestamps are ditched, this setter and its getter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* datetime_min  = NULL;
    PyObject* datetime_max  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* td_seconds    = NULL;
    double    td_fp         = 0;
    int       ret           = -1;
    
    if(val == Py_None){
        self->ts_covid19_infection = INFINITY;
        return 0;
    }
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)    goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)    goto fail;
    datetime_min  = PyObject_GetAttrString(datetime_obj, "min");
    if(!datetime_min)    goto fail;
    switch(PyObject_RichCompareBool(val, datetime_min, Py_EQ)){
        case -1: goto fail;
        case  1: self->ts_covid19_recovery = INFINITY; goto success;
        default: break;
    }
    datetime_max  = PyObject_GetAttrString(datetime_obj, "max");
    if(!datetime_max)    goto fail;
    switch(PyObject_RichCompareBool(val, datetime_max, Py_EQ)){
        case -1: goto fail;
        case  1:
            /**
             * In the Python code, self.recovered_timestamp == datetime.datetime.max
             * indicates death. We choose to represent it differently when using
             * POSIX timestamps.
             */
            
            self->ts_death            = self->env->ts_now;/* Kill */
            self->ts_covid19_recovery = INFINITY;
            goto success;
        default: break;
    }
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj)   goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)      goto fail;
    td            = PyNumber_Subtract(val, ts_initial);
    if(!td)              goto fail;
    td_seconds    = PyObject_CallMethod(td, "total_seconds", NULL);
    if(!td_seconds)      goto fail;
    td_fp         = PyFloat_AsDouble(td_seconds);
    if(PyErr_Occurred()) goto fail;
    
    self->ts_covid19_recovery = self->env->ts_initial + td_fp;
    
    /* Success path */
    success:
    ret = 0;
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(datetime_min);
    Py_XDECREF(datetime_max);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    Py_XDECREF(td_seconds);
    return ret;
}
static PyObject*              BaseHuman_get_cold_timestamp      (BaseHumanObject* self, void* closure){
    /**
     * The following implements the ugly logic of dynamically computing a datetime
     * cold timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_cold_symptomatic -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment.
     * 
     * NOTE: When datetime timestamps are ditched, this getter and its setter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* ts_plus_td    = NULL;
    
    if(self->ts_cold_symptomatic == INFINITY)
        Py_RETURN_NONE;
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)  goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)  goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj) goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)    goto fail;
    td            = PyObject_CallFunction(timedelta_obj, "ff", 0.0, self->ts_cold_symptomatic -
                                                                    self->env->ts_initial);
    if(!td)            goto fail;
    ts_plus_td    = PyNumber_Add(ts_initial, td);
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    return ts_plus_td;
}
static int                    BaseHuman_set_cold_timestamp      (BaseHumanObject* self, PyObject* val, void* closure){
    /**
     * The following implements the ugly logic of dynamically setting a datetime
     * cold timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_cold_symptomatic -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment. To that end we compute the
     * floating-point timestamp that corresponds to that convoluted logic,
     * rather than through the straightforward conversion to a POSIX timestamp.
     * 
     * NOTE: When datetime timestamps are ditched, this setter and its getter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* td_seconds    = NULL;
    double    td_fp         = 0;
    int       ret           = -1;
    
    if(val == Py_None){
        self->ts_cold_symptomatic = INFINITY;
        return 0;
    }
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)    goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)    goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj)   goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)      goto fail;
    td            = PyNumber_Subtract(val, ts_initial);
    if(!td)              goto fail;
    td_seconds    = PyObject_CallMethod(td, "total_seconds", NULL);
    if(!td_seconds)      goto fail;
    td_fp         = PyFloat_AsDouble(td_seconds);
    if(PyErr_Occurred()) goto fail;
    
    self->ts_cold_symptomatic = self->env->ts_initial + td_fp;
    ret = 0;
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    Py_XDECREF(td_seconds);
    return ret;
}
static PyObject*              BaseHuman_get_flu_timestamp       (BaseHumanObject* self, void* closure){
    /**
     * The following implements the ugly logic of dynamically computing a datetime
     * flu timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_flu_symptomatic -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment.
     * 
     * NOTE: When datetime timestamps are ditched, this getter and its setter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* ts_plus_td    = NULL;
    
    if(self->ts_flu_symptomatic == INFINITY)
        Py_RETURN_NONE;
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)  goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)  goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj) goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)    goto fail;
    td            = PyObject_CallFunction(timedelta_obj, "ff", 0.0, self->ts_flu_symptomatic -
                                                                    self->env->ts_initial);
    if(!td)            goto fail;
    ts_plus_td    = PyNumber_Add(ts_initial, td);
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    return ts_plus_td;
}
static int                    BaseHuman_set_flu_timestamp       (BaseHumanObject* self, PyObject* val, void* closure){
    /**
     * The following implements the ugly logic of dynamically setting a datetime
     * flu timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_flu_symptomatic -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment. To that end we compute the
     * floating-point timestamp that corresponds to that convoluted logic,
     * rather than through the straightforward conversion to a POSIX timestamp.
     * 
     * NOTE: When datetime timestamps are ditched, this setter and its getter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* td_seconds    = NULL;
    double    td_fp         = 0;
    int       ret           = -1;
    
    if(val == Py_None){
        self->ts_flu_symptomatic = INFINITY;
        return 0;
    }
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)    goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)    goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj)   goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)      goto fail;
    td            = PyNumber_Subtract(val, ts_initial);
    if(!td)              goto fail;
    td_seconds    = PyObject_CallMethod(td, "total_seconds", NULL);
    if(!td_seconds)      goto fail;
    td_fp         = PyFloat_AsDouble(td_seconds);
    if(PyErr_Occurred()) goto fail;
    
    self->ts_flu_symptomatic = self->env->ts_initial + td_fp;
    ret = 0;
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    Py_XDECREF(td_seconds);
    return ret;
}
static PyObject*              BaseHuman_get_allergy_timestamp   (BaseHumanObject* self, void* closure){
    /**
     * The following implements the ugly logic of dynamically computing a datetime
     * allergy timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_allergy_symptomatic -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment.
     * 
     * NOTE: When datetime timestamps are ditched, this getter and its setter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* ts_plus_td    = NULL;
    
    if(self->ts_allergy_symptomatic == INFINITY)
        Py_RETURN_NONE;
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)  goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)  goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj) goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)    goto fail;
    td            = PyObject_CallFunction(timedelta_obj, "ff", 0.0, self->ts_allergy_symptomatic -
                                                                    self->env->ts_initial);
    if(!td)            goto fail;
    ts_plus_td    = PyNumber_Add(ts_initial, td);
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    return ts_plus_td;
}
static int                    BaseHuman_set_allergy_timestamp   (BaseHumanObject* self, PyObject* val, void* closure){
    /**
     * The following implements the ugly logic of dynamically setting a datetime
     * allergy timestamp based on
     * 
     *     self.env.initial_timestamp + datetime.timedelta(self.ts_allergy_symptomatic -
     *                                                     self.env.ts_initial)
     * 
     * following the ugly logic of the Environment. To that end we compute the
     * floating-point timestamp that corresponds to that convoluted logic,
     * rather than through the straightforward conversion to a POSIX timestamp.
     * 
     * NOTE: When datetime timestamps are ditched, this setter and its getter
     *       should be deleted *IMMEDIATELY!*.
     */
    
    PyObject* datetime_mod  = NULL;
    PyObject* datetime_obj  = NULL;
    PyObject* timedelta_obj = NULL;
    PyObject* ts_initial    = NULL;
    PyObject* td            = NULL;
    PyObject* td_seconds    = NULL;
    double    td_fp         = 0;
    int       ret           = -1;
    
    if(val == Py_None){
        self->ts_allergy_symptomatic = INFINITY;
        return 0;
    }
    
    datetime_mod  = PyImport_ImportModule("datetime");
    if(!datetime_mod)    goto fail;
    datetime_obj  = PyObject_GetAttrString(datetime_mod, "datetime");
    if(!datetime_obj)    goto fail;
    timedelta_obj = PyObject_GetAttrString(datetime_mod, "timedelta");
    if(!timedelta_obj)   goto fail;
    ts_initial    = PyObject_CallMethod(datetime_obj, "fromtimestamp", "f", self->env->ts_initial);
    if(!ts_initial)      goto fail;
    td            = PyNumber_Subtract(val, ts_initial);
    if(!td)              goto fail;
    td_seconds    = PyObject_CallMethod(td, "total_seconds", NULL);
    if(!td_seconds)      goto fail;
    td_fp         = PyFloat_AsDouble(td_seconds);
    if(PyErr_Occurred()) goto fail;
    
    self->ts_allergy_symptomatic = self->env->ts_initial + td_fp;
    ret = 0;
    
    /* Abort path. */
    fail:
    Py_XDECREF(datetime_mod);
    Py_XDECREF(datetime_obj);
    Py_XDECREF(timedelta_obj);
    Py_XDECREF(ts_initial);
    Py_XDECREF(td);
    Py_XDECREF(td_seconds);
    return ret;
}
static PyObject*              BaseHuman_get_is_immune           (BaseHumanObject* self, void* closure){
    return PyBool_FromLong(self->env->ts_now >= self->ts_covid19_immunity);
}
static int                    BaseHuman_set_is_immune           (BaseHumanObject* self, PyObject* val, void* closure){
    switch(PyObject_IsTrue(val)){
        case -1: return -1;
        case  0: self->ts_covid19_immunity = INFINITY;          return 0;
        default: self->ts_covid19_immunity = self->env->ts_now; return 0;
    }
}
static PyObject*              BaseHuman_get_is_susceptible      (BaseHumanObject* self, void* closure){
    int is_removed = self->env->ts_now >= self->ts_covid19_immunity ||
                     self->env->ts_now >= self->ts_death;
    double td_infected = self->env->ts_now - self->ts_covid19_infection;
    
    return PyBool_FromLong(!is_removed && (td_infected  < 0));
}
static PyObject*              BaseHuman_get_is_exposed          (BaseHumanObject* self, void* closure){
    int is_removed = self->env->ts_now >= self->ts_covid19_immunity ||
                     self->env->ts_now >= self->ts_death;
    double td_infected = self->env->ts_now - self->ts_covid19_infection;
    double infectiousness_onset_secs = self->infectiousness_onset_days * SECONDS_PER_EPHEMERIS_DAY;
    
    return PyBool_FromLong(!is_removed && (td_infected >= 0) &&
                                          (td_infected  < infectiousness_onset_secs));
}
static PyObject*              BaseHuman_get_is_infectious       (BaseHumanObject* self, void* closure){
    int is_removed = self->env->ts_now >= self->ts_covid19_immunity ||
                     self->env->ts_now >= self->ts_death;
    double td_infected = self->env->ts_now - self->ts_covid19_infection;
    double infectiousness_onset_secs = self->infectiousness_onset_days * SECONDS_PER_EPHEMERIS_DAY;
    
    return PyBool_FromLong(!is_removed && (td_infected >= infectiousness_onset_secs));
}
static PyObject*              BaseHuman_get_is_removed          (BaseHumanObject* self, void* closure){
    int is_removed = self->env->ts_now >= self->ts_covid19_immunity ||
                     self->env->ts_now >= self->ts_death;
    
    return PyBool_FromLong(is_removed);
}
static PyObject*              BaseHuman_get_is_dead             (BaseHumanObject* self, void* closure){
    return PyBool_FromLong(self->env->ts_now >= self->ts_death);
}
static PyObject*              BaseHuman_get_is_incubated        (BaseHumanObject* self, void* closure){
    return PyBool_FromLong(!self->is_asymptomatic                          &&
                           (self->env->ts_now - self->ts_covid19_infection >=
                            self->incubation_days * SECONDS_PER_EPHEMERIS_DAY));
}
static PyObject*              BaseHuman_get_has_cold            (BaseHumanObject* self, void* closure){
    return PyBool_FromLong(self->ts_cold_symptomatic != INFINITY);
}
static PyObject*              BaseHuman_get_has_flu             (BaseHumanObject* self, void* closure){
    return PyBool_FromLong(self->ts_flu_symptomatic != INFINITY);
}
static PyObject*              BaseHuman_get_has_allergy_symptoms(BaseHumanObject* self, void* closure){
    return PyBool_FromLong(self->ts_allergy_symptomatic != INFINITY);
}
static PyObject*              BaseHuman_get_days_since_covid    (BaseHumanObject* self, void* closure){
    if(self->ts_covid19_infection == INFINITY)
        Py_RETURN_NONE;
    
    return PyLong_FromDouble((self->env->ts_now - self->ts_covid19_infection) / SECONDS_PER_EPHEMERIS_DAY);
}
static PyObject*              BaseHuman_get_days_since_cold     (BaseHumanObject* self, void* closure){
    if(self->ts_cold_symptomatic == INFINITY)
        Py_RETURN_NONE;
    
    return PyLong_FromDouble((self->env->ts_now - self->ts_cold_symptomatic) / SECONDS_PER_EPHEMERIS_DAY);
}
static PyObject*              BaseHuman_get_days_since_flu      (BaseHumanObject* self, void* closure){
    if(self->ts_flu_symptomatic == INFINITY)
        Py_RETURN_NONE;
    
    return PyLong_FromDouble((self->env->ts_now - self->ts_flu_symptomatic) / SECONDS_PER_EPHEMERIS_DAY);
}
static PyObject*              BaseHuman_get_days_since_allergies(BaseHumanObject* self, void* closure){
    if(self->ts_allergy_symptomatic == INFINITY)
        Py_RETURN_NONE;
    
    return PyLong_FromDouble((self->env->ts_now - self->ts_allergy_symptomatic) / SECONDS_PER_EPHEMERIS_DAY);
}


/**
 * PyMemberDef
 * 
 * Structure which describes an attribute of a type which corresponds to a C struct member. Its fields are:
 * Field   C Type      Meaning
 * name    char *      name of the member
 * type    int         the type of the member in the C struct
 * offset  Py_ssize_t  the offset in bytes that the member is located on the type’s object struct
 * flags   int         flag bits indicating if the field should be read-only or writable
 * doc     char *      points to the contents of the docstring
 * 
 * type can be one of many T_ macros corresponding to various C types. When the
 * member is accessed in Python, it will be converted to the equivalent Python type.
 * 
 * Macro name    C type         Macro name    C type
 * T_SHORT       short          T_CHAR        char
 * T_INT         int            T_BYTE        char
 * T_LONG        long           T_UBYTE       unsigned char
 * T_FLOAT       float          T_UINT        unsigned int
 * T_DOUBLE      double         T_USHORT      unsigned short
 * T_STRING      char*          T_ULONG       unsigned long
 * T_OBJECT      PyObject*      T_LONGLONG    long long
 * T_OBJECT_EX   PyObject*      T_ULONGLONG   unsigned long long
 * T_BOOL        char           T_PYSSIZET    Py_ssize_t
 * 
 * T_OBJECT and T_OBJECT_EX differ in that T_OBJECT returns None if the member is NULL
 * and T_OBJECT_EX raises an AttributeError. Try to use T_OBJECT_EX over T_OBJECT
 * because T_OBJECT_EX handles use of the del statement on that attribute more
 * correctly than T_OBJECT.
 * 
 * flags can be 0 for write and read access or READONLY for read-only access.
 * Using T_STRING for type implies READONLY. Only T_OBJECT and T_OBJECT_EX members
 * can be deleted. (They are set to NULL).
 */

static PyMemberDef BaseHuman_members[] = {
    {"name",                       T_OBJECT_EX, offsetof(BaseHumanObject, name),                      0,        "Name of Human."},
    {"ts_birth",                   T_DOUBLE,    offsetof(BaseHumanObject, ts_birth),                  0,        "POSIX timestamp of Human birth."},
    {"ts_death",                   T_DOUBLE,    offsetof(BaseHumanObject, ts_death),                  0,        "POSIX timestamp of Human death."},
    {"ts_cold_symptomatic",        T_DOUBLE,    offsetof(BaseHumanObject, ts_cold_symptomatic),       0,        "POSIX timestamp of Human cold symptoms."},
    {"ts_flu_symptomatic",         T_DOUBLE,    offsetof(BaseHumanObject, ts_flu_symptomatic),        0,        "POSIX timestamp of Human flu symptoms."},
    {"ts_allergy_symptomatic",     T_DOUBLE,    offsetof(BaseHumanObject, ts_allergy_symptomatic),    0,        "POSIX timestamp of Human allergy symptoms."},
    {"ts_covid19_infection",       T_DOUBLE,    offsetof(BaseHumanObject, ts_covid19_infection),      0,        "POSIX timestamp of Human COVID-19 infection."},
    {"ts_covid19_infectious",      T_DOUBLE,    offsetof(BaseHumanObject, ts_covid19_infectious),     0,        "POSIX timestamp of Human COVID-19 infectiousness."},
    {"ts_covid19_symptomatic",     T_DOUBLE,    offsetof(BaseHumanObject, ts_covid19_symptomatic),    0,        "POSIX timestamp of Human COVID-19 symptoms."},
    {"ts_covid19_recovery",        T_DOUBLE,    offsetof(BaseHumanObject, ts_covid19_recovery),       0,        "POSIX timestamp of Human COVID-19 recovery."},
    {"ts_covid19_immunity",        T_DOUBLE,    offsetof(BaseHumanObject, ts_covid19_immunity),       0,        "POSIX timestamp of Human COVID-19 immunity."},
    {"infectiousness_onset_days",  T_DOUBLE,    offsetof(BaseHumanObject, infectiousness_onset_days), 0,        "Time from infection to beginning of infectiousness, in days."},
    {"incubation_days",            T_DOUBLE,    offsetof(BaseHumanObject, incubation_days),           0,        "Time from infection to beginning of symptoms (incubation time), in days."},
    {"is_asymptomatic",            T_BOOL,      offsetof(BaseHumanObject, is_asymptomatic),           0,        "Whether Human is COVID-19 asymptomatic."},
    {NULL},
};

/**
 * struct PyMethodDef {
 *     const char* ml_name;   The name of the built-in function/method
 *     PyCFunction ml_meth;    The C function that implements it
 *     int         ml_flags;   Combination of METH_xxx flags, which mostly
 *                                describe the args expected by the C func
 *     const char* ml_doc;    The __doc__ attribute, or NULL
 * };
 * 
 * Possible flags are:
 *     METH_VARARGS                     PyObject* PyCFunction(PyObject* self, PyObject* args);
 *     METH_VARARGS | METH_KEYWORDS     PyObject* PyCFunctionWithKeywords(PyObject* self, PyObject* args, PyObject* kwargs);
 *     METH_NOARGS                      PyObject* PyCFunction(PyObject* self, NULL);
 *     METH_O                           PyObject* PyCFunction(PyObject* self, PyObject* arg);
 */

static PyMethodDef BaseHuman_methods[] = {
    {NULL},
};

/**
 * typedef struct PyGetSetDef {
 *     char *name;
 *     getter get;
 *     setter set;
 *     char *doc;
 *     void *closure;
 * } PyGetSetDef;
 */

static PyGetSetDef BaseHuman_getset[] = {
    {"__dict__",             (getter)PyObject_GenericGetDict,            (setter)PyObject_GenericSetDict},
    {"env",                  (getter)BaseHuman_get_env,                  (setter)BaseHuman_set_env,                 "Environment of which the Human is part."},
    {"age",                  (getter)BaseHuman_get_age,                  (setter)BaseHuman_set_age,                 "Age of Human."},
    {"infection_timestamp",  (getter)BaseHuman_get_infection_timestamp,  (setter)BaseHuman_set_infection_timestamp, "Infection timestamp, as datetime object."},
    {"recovered_timestamp",  (getter)BaseHuman_get_recovered_timestamp,  (setter)BaseHuman_set_recovered_timestamp, "Recovered timestamp, as datetime object."},
    {"cold_timestamp",       (getter)BaseHuman_get_cold_timestamp,       (setter)BaseHuman_set_cold_timestamp,      "Cold timestamp, as datetime object."},
    {"flu_timestamp",        (getter)BaseHuman_get_flu_timestamp,        (setter)BaseHuman_set_flu_timestamp,       "Flu timestamp, as datetime object."},
    {"allergy_timestamp",    (getter)BaseHuman_get_allergy_timestamp,    (setter)BaseHuman_set_allergy_timestamp,   "Allergy timestamp, as datetime object."},
    {"is_immune",            (getter)BaseHuman_get_is_immune,            (setter)BaseHuman_set_is_immune,           "Whether human is immune to COVID-19."},
    {"is_susceptible",       (getter)BaseHuman_get_is_susceptible,       NULL,                                      "Whether human is susceptible to being infected by COVID-19."},
    {"is_exposed",           (getter)BaseHuman_get_is_exposed,           NULL,                                      "Whether human has been exposed to COVID-19 but cannot yet infect anyone else."},
    {"is_infectious",        (getter)BaseHuman_get_is_infectious,        NULL,                                      "Whether human is infectious, i.e. can infect others with COVID-19."},
    {"is_removed",           (getter)BaseHuman_get_is_removed,           NULL,                                      "Whether human is removed, i.e. is dead or has acquired immunity from COVID-19."},
    {"is_dead",              (getter)BaseHuman_get_is_dead,              NULL,                                      "Whether human is dead."},
    {"is_incubated",         (getter)BaseHuman_get_is_incubated,         NULL,                                      "Whether human has spent enough time to become symptomatic."},
    {"has_cold",             (getter)BaseHuman_get_has_cold,             NULL,                                      "Whether human has a cold."},
    {"has_flu",              (getter)BaseHuman_get_has_flu,              NULL,                                      "Whether human has a flu."},
    {"has_allergy_symptoms", (getter)BaseHuman_get_has_allergy_symptoms, NULL,                                      "Whether human has allergy symptoms."},
    {"days_since_covid",     (getter)BaseHuman_get_days_since_covid,     NULL,                                      "Days since infection with COVID-19."},
    {"days_since_cold",      (getter)BaseHuman_get_days_since_cold,      NULL,                                      "Days since infection with cold."},
    {"days_since_flu",       (getter)BaseHuman_get_days_since_flu,       NULL,                                      "Days since infection with flu."},
    {"days_since_allergies", (getter)BaseHuman_get_days_since_allergies, NULL,                                      "Days since allergy symptoms manifested."},
    {NULL},
};

/**
 * BaseHuman Type Object
 */

PyTypeObject BaseHumanType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name             = "covid19sim.native._native.BaseHuman",
    .tp_doc              = "BaseHuman object documentation",
    .tp_flags            = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE,
    .tp_basicsize        = sizeof(BaseHumanObject),
    .tp_dictoffset       = offsetof(BaseHumanObject, __dict__),
    .tp_weaklistoffset   = offsetof(BaseHumanObject, __weaklist__),
    
    .tp_new              =      (newfunc)BaseHuman_new,
    .tp_init             =     (initproc)BaseHuman_init,
    .tp_dealloc          =   (destructor)BaseHuman_dealloc,
    .tp_methods          =               BaseHuman_methods,
    .tp_members          =               BaseHuman_members,
    .tp_getset           =               BaseHuman_getset,
    .tp_traverse         = (traverseproc)BaseHuman_traverse,
    .tp_clear            =      (inquiry)BaseHuman_clear,
};


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif
