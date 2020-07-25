/**
 * Includes
 */

#include "_native.h"        /* Because of "reasons", this header must be first. */



/* Defines */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* BaseEnvironment Object Methods */
static BaseEnvironmentObject* BaseEnvironment_new               (PyTypeObject* type,
                                                                 PyObject*     args,
                                                                 PyObject*     kwargs){
    BaseEnvironmentObject* self;
    (void)args;
    (void)kwargs;
    
    self = (BaseEnvironmentObject*)type->tp_alloc(type, 0);
    if(!self)
        return self;
    
    self->ts_now              = 0;
    self->ts_initial          = 0;
    self->ts_initial_midnight = 0;
    self->ts_initial_monday   = 0;
    self->__dict__            = NULL;
    self->__weaklist__        = NULL;
    if(!self->queue){
        self->queue = Py_None;
        Py_INCREF(Py_None);
    }
    if(!self->active_process){
        self->active_process = Py_None;
        Py_INCREF(Py_None);
    }
    
    return self;
}
static int                    BaseEnvironment_init              (BaseEnvironmentObject* self,
                                                                 PyObject*              args,
                                                                 PyObject*              kwargs){
    /* Default value of env.now == env.ts_initial is 0 */
    self->ts_initial = 0;
    
    /* Parse arguments */
    static char *kwargs_list[] = {"initial_time", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|d", kwargs_list, &self->ts_initial))
        return -1;
    
    /* Now is initial time */
    self->ts_now = self->ts_initial;
    
    /**
     * Compute extra timestamp bases, for speed.
     * 
     * NOTE: The code below is intellectually bankrupt, since it mostly
     * ignores such details as locale and DST. However, since the Python code
     * is just as bankrupt in its assumptions, the optimizations these
     * timebases enable are valid.
     */
    
    struct tm tm_initial;
    time_t ts_initial, ts_midnight, ts_sunday, ts_monday;
    ts_initial  = (time_t)floor(self->ts_initial);
    localtime_r(&ts_initial, &tm_initial);
    ts_midnight = ts_initial  - tm_initial.tm_sec *    1;/* Round down to midnight */
    ts_midnight = ts_midnight - tm_initial.tm_min *   60;
    ts_midnight = ts_midnight - tm_initial.tm_hour* 3600;
    ts_sunday   = ts_midnight - tm_initial.tm_wday*86400;/* Round down to Sunday */
    if(ts_initial-ts_sunday >= 1*86400)
        ts_monday = ts_sunday+1*86400;
    else
        ts_monday = ts_sunday-6*86400;
    self->ts_initial_midnight = ts_midnight;
    self->ts_initial_monday   = ts_monday;
    
    /* Init successful. */
    return 0;
}
static int                    BaseEnvironment_traverse          (BaseEnvironmentObject* self,
                                                                 visitproc              visit,
                                                                 void*                  arg){
    Py_VISIT(self->__dict__);
    Py_VISIT(self->__weaklist__);
    Py_VISIT(self->queue);
    Py_VISIT(self->active_process);
    return 0;
}
static int                    BaseEnvironment_clear             (BaseEnvironmentObject* self){
    Py_CLEAR(self->__dict__);
    Py_CLEAR(self->__weaklist__);
    Py_CLEAR(self->queue);
    Py_CLEAR(self->active_process);
    return 0;
}
static void                   BaseEnvironment_dealloc           (BaseEnvironmentObject* self){
    if(self->__weaklist__)
        PyObject_ClearWeakRefs((PyObject*)self);
    
    BaseEnvironment_clear(self);
    
    /**
     * Py_CLEAR() any other Python object members, which won't participate in
     * cycles.
     */
    
    /* ... */
    
    PyObject_GC_Del(self);
}

/* BaseEnvironment Object @properties (Getter/Setters) */
static PyObject*              BaseEnvironment_get_simulation_day(BaseEnvironmentObject* self, void* closure){
    return PyLong_FromLong((long)(self->ts_now-self->ts_initial_midnight)/86400);
}
static PyObject*              BaseEnvironment_get_minutes       (BaseEnvironmentObject* self, void* closure){
    long seconds_within_hour = (long)(self->ts_now-self->ts_initial_midnight) % 3600;
    return PyLong_FromLong(seconds_within_hour/60);
}
static PyObject*              BaseEnvironment_get_hour_of_day   (BaseEnvironmentObject* self, void* closure){
    long seconds_within_day = (long)(self->ts_now-self->ts_initial_midnight) % 86400;
    return PyLong_FromLong(seconds_within_day/3600);
}
static PyObject*              BaseEnvironment_get_day_of_week   (BaseEnvironmentObject* self, void* closure){
    long seconds_within_week = (long)(self->ts_now-self->ts_initial_monday) % (7*86400);
    return PyLong_FromLong(seconds_within_week/86400);
}
static PyObject*              BaseEnvironment_get_is_weekend    (BaseEnvironmentObject* self, void* closure){
    long seconds_within_week = (long)(self->ts_now-self->ts_initial_monday) % (7*86400);
    if(seconds_within_week >= 5*86400){
        Py_RETURN_TRUE;
    }else{
        Py_RETURN_FALSE;
    }
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

static PyMemberDef BaseEnvironment_members[] = {
    {"_now",           T_DOUBLE,    offsetof(BaseEnvironmentObject, ts_now),         0,        "Current simulation time, as POSIX timestamp."},
    {"now",            T_DOUBLE,    offsetof(BaseEnvironmentObject, ts_now),         READONLY, "Current simulation time, as POSIX timestamp."},
    {"ts_now",         T_DOUBLE,    offsetof(BaseEnvironmentObject, ts_now),         READONLY, "Current simulation time, as POSIX timestamp."},
    {"_initial",       T_DOUBLE,    offsetof(BaseEnvironmentObject, ts_initial),     0,        "Initial simulation time, as POSIX timestamp."},
    {"ts_initial",     T_DOUBLE,    offsetof(BaseEnvironmentObject, ts_initial),     READONLY, "Initial simulation time, as POSIX timestamp."},
    {"_active_proc",   T_OBJECT_EX, offsetof(BaseEnvironmentObject, active_process), 0,        "Active process."},
    {"active_process", T_OBJECT_EX, offsetof(BaseEnvironmentObject, active_process), READONLY, "Active process."},
    {"_queue",         T_OBJECT_EX, offsetof(BaseEnvironmentObject, queue),          0,        "Event queue."},
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

static PyMethodDef BaseEnvironment_methods[] = {
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

static PyGetSetDef BaseEnvironment_getset[] = {
    {"__dict__",       (getter)PyObject_GenericGetDict, (setter)PyObject_GenericSetDict},
    {"simulation_day", (getter)BaseEnvironment_get_simulation_day, NULL, "Current day of simultation (int, first day=0)."},
    {"minutes",        (getter)BaseEnvironment_get_minutes,        NULL, "Current simulation minutes of the day (int)."},
    {"hour_of_day",    (getter)BaseEnvironment_get_hour_of_day,    NULL, "Current simulation hour of the day (int)."},
    {"day_of_week",    (getter)BaseEnvironment_get_day_of_week,    NULL, "Current simulation day of the week (int, 0=Monday)."},
    {"is_weekend",     (getter)BaseEnvironment_get_is_weekend,     NULL, "Current simulation day is a weekend day (bool)."},
    {NULL},
};

/**
 * BaseEnvironment Type Object
 */

PyTypeObject BaseEnvironmentType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name             = "covid19sim.native._native.BaseEnvironment",
    .tp_doc              = "BaseEnvironment object documentation",
    .tp_flags            = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE,
    .tp_basicsize        = sizeof(BaseEnvironmentObject),
    .tp_dictoffset       = offsetof(BaseEnvironmentObject, __dict__),
    .tp_weaklistoffset   = offsetof(BaseEnvironmentObject, __weaklist__),
    
    .tp_new              =      (newfunc)BaseEnvironment_new,
    .tp_init             =     (initproc)BaseEnvironment_init,
    .tp_dealloc          =   (destructor)BaseEnvironment_dealloc,
    .tp_methods          =               BaseEnvironment_methods,
    .tp_members          =               BaseEnvironment_members,
    .tp_getset           =               BaseEnvironment_getset,
    .tp_traverse         = (traverseproc)BaseEnvironment_traverse,
    .tp_clear            =      (inquiry)BaseEnvironment_clear,
};


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif
