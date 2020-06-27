/**
 * Includes
 */

#include "_native.h" /* Because of "reasons", this header must be first. */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Module-level Methods Table */

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

static PyMethodDef _native_METHODS[] = {
    {NULL},  /* Sentinel */
};

/**
 * Python Module Definition
 */

static PyModuleDef _native_MODULE_DEF = {
    PyModuleDef_HEAD_INIT,
    "_native",           /* m_name */
    /* m_doc */
    "covid19sim.native._native bindings in C.",
    -1,                  /* m_size */
    _native_METHODS,     /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};


/**
 * Python _native Module Initialization Function
 */

PyMODINIT_FUNC PyInit__native(void){
    PyObject* m;
    
    
    /* Initialize Python module structures */
    m = PyModule_Create(&_native_MODULE_DEF);
    if(!m)
        return m;
    
    
    /* Add declared type */
    #define ADDTYPE(T)                                                \
        do{                                                           \
            if(PyType_Ready(&T##Type) < 0){                           \
                Py_DECREF(m);                                         \
                return NULL;                                          \
            }                                                         \
                                                                      \
            Py_INCREF(&T##Type);                                      \
                                                                      \
            if(PyModule_AddObject(m, #T, (PyObject*)&T##Type) < 0){   \
                Py_DECREF(&T##Type);                                  \
                Py_DECREF(m);                                         \
                return NULL;                                          \
            }                                                         \
        }while(0)
    ADDTYPE(BaseEnvironment);
    #undef ADDTYPE
    
    
    /* Add constants */
    #define ADDINTMACRO(T)                                            \
        do{                                                           \
            if(PyModule_AddIntConstant(m, #T, T) < 0){                \
                Py_DECREF(m);                                         \
                return NULL;                                          \
            }                                                         \
        }while(0)
    ADDINTMACRO(SECONDS_PER_MINUTE);
    ADDINTMACRO(SECONDS_PER_HOUR);
    ADDINTMACRO(SECONDS_PER_DAY);
    ADDINTMACRO(SECONDS_PER_EPHEMERIS_DAY);
    ADDINTMACRO(SECONDS_PER_WEEK);
    ADDINTMACRO(SECONDS_PER_YEAR);
    ADDINTMACRO(SECONDS_PER_TROPICAL_YEAR);
    ADDINTMACRO(SECONDS_PER_LEAP_YEAR);
    ADDINTMACRO(HUMAN_SYMPTOM_SNEEZING);
    #undef ADDINTMACRO
    
    
    /* Return finished module */
    return m;
}


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif
