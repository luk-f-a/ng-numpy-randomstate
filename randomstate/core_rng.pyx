#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import sys
import numpy as np
cimport numpy as np
cimport cython
import operator
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
try:
    from threading import Lock
except:
    from dummy_threading import Lock
np.import_array()

cdef extern from "Python.h":
    double PyFloat_AsDouble(object ob)
    long PyInt_AsLong(object ob)
    int PyErr_Occurred()
    void PyErr_Clear()

include "config.pxi"
include "src/common/binomial.pxi"

IF RNG_PCG32:
    include "shims/pcg-32/pcg-32.pxi"
IF RNG_PCG64:
    include "shims/pcg-64/pcg-64.pxi"
IF RNG_MT19937:
    include "shims/random-kit/random-kit.pxi"
IF RNG_XORSHIFT128:
    include "shims/xorshift128/xorshift128.pxi"
IF RNG_XORSHIFT1024:
    include "shims/xorshift1024/xorshift1024.pxi"
IF RNG_MRG32K3A:
    include "shims/mrg32k3a/mrg32k3a.pxi"
IF RNG_MLFG_1279_861:
    include "shims/mlfg-1279-861/mlfg-1279-861.pxi"
IF RNG_DUMMY:
    include "shims/dummy/dummy.pxi"

cdef extern from "core-rng.h":

    cdef uint64_t random_uint64(aug_state* state) nogil
    cdef uint32_t random_uint32(aug_state* state) nogil
    cdef uint64_t random_bounded_uint64(aug_state* state, uint64_t bound) nogil
    cdef uint32_t random_bounded_uint32(aug_state* state, uint32_t bound) nogil
    cdef int64_t random_bounded_int64(aug_state* state, int64_t low, int64_t high) nogil
    cdef int32_t random_bounded_int32(aug_state* state, int32_t low, int32_t high) nogil
    cdef int64_t random_positive_int64(aug_state* state) nogil
    cdef int32_t random_positive_int32(aug_state* state) nogil

    cdef long random_positive_int(aug_state* state) nogil
    cdef unsigned long random_uint(aug_state* state) nogil
    cdef unsigned long random_interval(aug_state* state, unsigned long max) nogil

    cdef void entropy_init(aug_state* state) nogil

    cdef double random_standard_uniform(aug_state* state) nogil
    cdef double random_gauss(aug_state* state) nogil
    cdef double random_gauss_zig(aug_state* state) nogil
    cdef double random_gauss_zig_julia(aug_state* state) nogil
    cdef double random_standard_exponential(aug_state* state) nogil
    cdef double random_standard_cauchy(aug_state* state) nogil

    cdef double random_exponential(aug_state *state, double scale) nogil
    cdef double random_standard_gamma(aug_state* state, double shape) nogil
    cdef double random_pareto(aug_state *state, double a) nogil
    cdef double random_weibull(aug_state *state, double a) nogil
    cdef double random_power(aug_state *state, double a) nogil
    cdef double random_rayleigh(aug_state *state, double mode) nogil
    cdef double random_standard_t(aug_state *state, double df) nogil
    cdef double random_chisquare(aug_state *state, double df) nogil

    cdef double random_normal(aug_state *state, double loc, double scale) nogil
    cdef double random_uniform(aug_state *state, double loc, double scale) nogil
    cdef double random_gamma(aug_state *state, double shape, double scale) nogil
    cdef double random_beta(aug_state *state, double a, double b) nogil
    cdef double random_f(aug_state *state, double dfnum, double dfden) nogil
    cdef double random_laplace(aug_state *state, double loc, double scale) nogil
    cdef double random_gumbel(aug_state *state, double loc, double scale) nogil
    cdef double random_logistic(aug_state *state, double loc, double scale) nogil
    cdef double random_lognormal(aug_state *state, double mean, double sigma) nogil
    cdef double random_noncentral_chisquare(aug_state *state, double df, double nonc) nogil
    cdef double random_wald(aug_state *state, double mean, double scale) nogil
    cdef double random_vonmises(aug_state *state, double mu, double kappa) nogil

    cdef double random_noncentral_f(aug_state *state, double dfnum, double dfden, double nonc) nogil
    cdef double random_triangular(aug_state *state, double left, double mode, double right) nogil

    cdef long random_poisson(aug_state *state, double lam) nogil
    cdef long random_negative_binomial(aug_state *state, double n, double p) nogil
    cdef long random_binomial(aug_state *state, double p, long n) nogil
    cdef long random_logseries(aug_state *state, double p) nogil
    cdef long random_geometric(aug_state *state, double p) nogil
    cdef long random_zipf(aug_state *state, double a) nogil
    cdef long random_hypergeometric(aug_state *state, long good, long bad, long sample) nogil

include "wrappers.pxi"

cdef enum ConstraintType:
    CONS_NONE
    CONS_NON_NEGATIVE
    CONS_POSITIVE
    CONS_BOUNDED_0_1
    CONS_BOUNDED_0_1_NOTNAN
    CONS_GT_1
    CONS_GTE_1
    CONS_POISSON

ctypedef ConstraintType constraint_type

cdef double POISSON_LAM_MAX = <uint64_t>(np.iinfo('l').max - np.sqrt(np.iinfo('l').max)*10)
cdef uint64_t MAXSIZE = <uint64_t>sys.maxsize

cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if np.any(np.less(val, 0)):
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE:
        if np.any(np.less_equal(val, 0)):
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1 or cons == CONS_BOUNDED_0_1_NOTNAN:
        if np.any(np.less_equal(val, 0)) or np.any(np.greater_equal(val, 1)):
            raise ValueError(name + " <= 0 or " + name + " >= 1")
        if cons == CONS_BOUNDED_0_1_NOTNAN:
            if np.any(np.isnan(val)):
                raise ValueError(name + ' contains NaNs')
    elif cons == CONS_GT_1:
        if np.any(np.less_equal(val, 1)):
            raise ValueError(name + " <= 1")
    elif cons == CONS_GTE_1:
        if np.any(np.less(val, 1)):
            raise ValueError(name + " < 1")
    elif cons == CONS_POISSON:
        if np.any(np.greater(val, POISSON_LAM_MAX)):
            raise ValueError(name + " value too large")
        if np.any(np.less(val, 0.0)):
            raise ValueError(name + " < 0")

    return 0



cdef int check_constraint(double val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if val < 0:
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE:
        if val <= 0:
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1 or cons == CONS_BOUNDED_0_1_NOTNAN:
        if val < 0 or val > 1:
            raise ValueError(name + " <= 0 or " + name + " >= 1")
        if cons == CONS_BOUNDED_0_1_NOTNAN:
            if np.isnan(val):
                raise ValueError(name + ' contains NaNs')
    elif cons == CONS_GT_1:
        if val <= 1:
            raise ValueError(name + " <= 1")
    elif cons == CONS_GTE_1:
        if val < 1:
            raise ValueError(name + " < 1")
    elif cons == CONS_POISSON:
        if val < 0:
            raise ValueError(name + " < 0")
        elif val > POISSON_LAM_MAX:
            raise ValueError(name + " value too large")

    return 0

cdef object cont_broadcast_1(aug_state* state, void* func, object size, object lock,
                             object a, object a_name, constraint_type a_constraint):

    cdef np.ndarray a_arr, randoms
    cdef np.broadcast it
    cdef random_double_1 f = (<random_double_1>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.double)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_DOUBLE)


    it = np.broadcast(randoms, a_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_2(aug_state* state, void* func, object size, object lock,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint):
    cdef np.ndarray a_arr, b_arr, randoms
    cdef np.broadcast it
    cdef random_double_2 f = (<random_double_2>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = np.empty(it.shape, np.double)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_3(aug_state* state, void* func, object size, object lock,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):
    cdef np.ndarray a_arr, b_arr, c_arr, randoms
    cdef np.broadcast it
    cdef random_double_3 f = (<random_double_3>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if c_constraint != CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)
        randoms = np.empty(it.shape, np.double)

    it = it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)
    # np.broadcast(randoms, a_arr, b_arr, c_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont(aug_state* state, void* func, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):

    cdef double _a = 0.0, _b = 0.0, _c = 0.0
    cdef bint is_scalar = True
    if narg > 0:
        _a = PyFloat_AsDouble(a)
        if _a == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        elif a_constraint != CONS_NONE:
            check_constraint(_a, a_name, a_constraint)
    if narg > 1 and is_scalar:
        _b = PyFloat_AsDouble(b)
        if _b == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        elif b_constraint != CONS_NONE:
            check_constraint(_b, b_name, b_constraint)
    if narg == 3 and is_scalar:
        _c = PyFloat_AsDouble(c)
        if _c == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        elif c_constraint != CONS_NONE:
            check_constraint(_c, c_name, c_constraint)
    if not is_scalar:
        PyErr_Clear()
        if narg == 1:
            return cont_broadcast_1(state, func, size, lock,
                                    a, a_name, a_constraint)
        elif narg == 2:
            return cont_broadcast_2(state, func, size, lock,
                                    a, a_name, a_constraint,
                                    b, b_name, b_constraint)
        else:
            return cont_broadcast_3(state, func, size, lock,
                                    a, a_name, a_constraint,
                                    b, b_name, b_constraint,
                                    c, c_name, c_constraint)

    if size is None:
        if narg == 0:
            return (<random_double_0>func)(state)
        elif narg == 1:
            return (<random_double_1>func)(state, _a)
        elif narg == 2:
            return (<random_double_2>func)(state, _a, _b)
        elif narg == 3:
            return (<random_double_3>func)(state, _a, _b, _c)

    cdef Py_ssize_t i, n = compute_numel(size)
    cdef double [::1] randoms = np.empty(n, np.double)
    cdef random_double_0 f0;
    cdef random_double_1 f1;
    cdef random_double_2 f2;
    cdef random_double_3 f3;

    with lock, nogil:
        if narg == 0:
            f0 = (<random_double_0>func)
            for i in range(n):
                randoms[i] = f0(state)
        elif narg == 1:
            f1 = (<random_double_1>func)
            for i in range(n):
                randoms[i] = f1(state, _a)
        elif narg == 2:
            f2 = (<random_double_2>func)
            for i in range(n):
                randoms[i] = f2(state, _a, _b)
        elif narg == 3:
            f3 = (<random_double_3>func)
            for i in range(n):
                randoms[i] = f3(state, _a, _b, _c)

    return np.asarray(randoms).reshape(size)

cdef object discrete_broadcast_d(aug_state* state, void* func, object size, object lock,
                                 object a, object a_name, constraint_type a_constraint):

    cdef np.ndarray a_arr, randoms
    cdef np.broadcast it
    cdef random_uint_d f = (<random_uint_d>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.long)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_LONG)

    it = np.broadcast(randoms, a_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_dd(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint):
    cdef np.ndarray a_arr, b_arr, randoms
    cdef np.broadcast it
    cdef random_uint_dd f = (<random_uint_dd>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)
    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = np.empty(size, np.long)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = np.empty(it.shape, np.long)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_LONG)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]

            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_di(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint):
    cdef np.ndarray a_arr, b_arr, randoms
    cdef np.broadcast it
    cdef random_uint_di f = (<random_uint_di>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_LONG, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = np.empty(size, np.long)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = np.empty(it.shape, np.long)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_LONG)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<long*>np.PyArray_MultiIter_DATA(it, 2))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_iii(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint,
                                  object c, object c_name, constraint_type c_constraint):
    cdef np.ndarray a_arr, b_arr, c_arr, randoms
    cdef np.broadcast it
    cdef random_uint_iii f = (<random_uint_iii>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_LONG, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_LONG, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_LONG, np.NPY_ALIGNED)
    if c_constraint != CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = np.empty(size, np.long)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        randoms = np.empty(it.shape, np.long)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_LONG)

    it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<long*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<long*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<long*>np.PyArray_MultiIter_DATA(it, 3))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_i(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint):
    cdef np.ndarray a_arr, randoms
    cdef np.broadcast it
    cdef random_uint_i f = (<random_uint_i>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_LONG, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.long)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_LONG)

    it = np.broadcast(randoms, a_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<long*>np.PyArray_MultiIter_DATA(it, 1))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

# Needs double <vec>, double-double <vec>, double-long<vec>, long <vec>, long-long-long
cdef object disc(aug_state* state, void* func, object size, object lock,
                 int narg_double, int narg_long,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):

    cdef double _da = 0, _db = 0
    cdef long _ia = 0, _ib = 0 , _ic = 0
    cdef bint is_scalar = True
    if narg_double > 0:
        _da = PyFloat_AsDouble(a)
        if _da == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        elif a_constraint != CONS_NONE:
            check_constraint(_da, a_name, a_constraint)

        if narg_double > 1 and is_scalar:
            _db = PyFloat_AsDouble(b)
            if _db == -1.0:
                if PyErr_Occurred():
                    is_scalar = False
            elif b_constraint != CONS_NONE:
                check_constraint(_db, b_name, b_constraint)
        if narg_long == 1 and is_scalar:
            _ib = PyInt_AsLong(b)
            if _ib == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif b_constraint != CONS_NONE:
                check_constraint(<double>_ib, b_name, b_constraint)
    else:
        if narg_long > 0:
            _ia = PyInt_AsLong(a)
            if _ia == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif a_constraint != CONS_NONE:
                check_constraint(<double>_ia, a_name, a_constraint)
        if narg_long > 1 and is_scalar:
            _ib = PyInt_AsLong(b)
            if _ib == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif b_constraint != CONS_NONE:
                check_constraint(<double>_ib, b_name, b_constraint)
        if narg_long > 2 and is_scalar:
            _ic = PyInt_AsLong(c)
            if _ic == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif c_constraint != CONS_NONE:
                check_constraint(<double>_ic, c_name, c_constraint)

    if not is_scalar:
        PyErr_Clear()
        if narg_long == 0:
            if narg_double == 1:
                return discrete_broadcast_d(state, func, size, lock,
                                            a, a_name, a_constraint)
            elif narg_double == 2:
                return discrete_broadcast_dd(state, func, size, lock,
                                             a, a_name, a_constraint,
                                             b, b_name, b_constraint)
        elif narg_long == 1:
            if narg_double == 0:
                return discrete_broadcast_i(state, func, size, lock,
                                            a, a_name, a_constraint)
            elif narg_double == 1:
                return discrete_broadcast_di(state, func, size, lock,
                                             a, a_name, a_constraint,
                                             b, b_name, b_constraint)
        else:
            raise NotImplementedError("No vector path available")

    if size is None:
        if narg_long == 0:
            if narg_double == 0:
                return (<random_uint_0>func)(state)
            elif narg_double == 1:
                return (<random_uint_d>func)(state, _da)
            elif narg_double == 2:
                return (<random_uint_dd>func)(state, _da, _db)
        elif narg_long == 1:
            if narg_double == 0:
                return (<random_uint_i>func)(state, _ia)
            if narg_double == 1:
                return (<random_uint_di>func)(state, _da, _ib)
        else:
            return (<random_uint_iii>func)(state, _ia, _ib, _ic)

    cdef Py_ssize_t i, n = compute_numel(size)
    cdef uint64_t [::1] randoms = np.empty(n, np.uint64)
    cdef random_uint_0 f0;
    cdef random_uint_d fd;
    cdef random_uint_dd fdd;
    cdef random_uint_di fdi;
    cdef random_uint_i fi;
    cdef random_uint_iii fiii;


    with lock, nogil:
        if narg_long == 0:
            if narg_double == 0:
                f0 = (<random_uint_0>func)
                for i in range(n):
                    randoms[i] = f0(state)
            elif narg_double == 1:
                fd = (<random_uint_d>func)
                for i in range(n):
                    randoms[i] = fd(state, _da)
            elif narg_double == 2:
                fdd = (<random_uint_dd>func)
                for i in range(n):
                    randoms[i] = fdd(state, _da, _db)
        elif narg_long == 1:
            if narg_double == 0:
                fi = (<random_uint_i>func)
                for i in range(n):
                    randoms[i] = fi(state, _ia)
            if narg_double == 1:
                fdi = (<random_uint_di>func)
                for i in range(n):
                    randoms[i] = fdi(state, _da, _ib)
        else:
            fiii = (<random_uint_iii>func)
            for i in range(n):
                randoms[i] = fiii(state, _ia, _ib, _ic)

    return np.asarray(randoms).reshape(size)


cdef double kahan_sum(double *darr, np.npy_intp n):
    cdef double c, y, t, sum
    cdef np.npy_intp i
    sum = darr[0]
    c = 0.0
    for i in range(1, n):
        y = darr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t
    return sum


cdef class RandomState:
    CLASS_DOCSTRING

    cdef binomial_t binomial_info
    cdef rng_t rng
    cdef aug_state rng_state
    cdef object lock
    poisson_lam_max = POISSON_LAM_MAX
    __MAXSIZE = <uint64_t>sys.maxsize

    IF RNG_SEED==1:
        def __init__(self, seed=None):
            self.rng_state.rng = &self.rng
            self.rng_state.binomial = &self.binomial_info
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            self.lock = Lock()
            if seed is not None:
                self.seed(seed)
            else:
                entropy_init(&self.rng_state)

    ELSE:
        def __init__(self, seed=None, inc=None):
            self.rng_state.rng = &self.rng
            self.rng_state.binomial = &self.binomial_info
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            self.lock = Lock()
            if seed is not None and inc is not None:
                self.seed(seed, inc)
            else:
                entropy_init(&self.rng_state)

    # Pickling support:
    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        # TODO: This probably is wrong
        # TODO: Removed np.random.__RandomState_ctor - This is needed on a RNG-by-RNG basis
        return ((), (), self.get_state())

    IF RNG_SEED==1:
        def seed(self, val=None):
            """
            seed(seed=None)

            Seed the generator.

            This method is called when `RandomState` is initialized. It can be
            called again to re-seed the generator. For details, see `RandomState`.

            Parameters
            ----------
            val : int, optional
                Seed for `RandomState`.

            Notes
            -----
            Acceptable range for seed depends on specifics of PRNG.  See
            class documentation for details.

            Seeds are hashed to produce the required number of bits.

            See Also
            --------
            RandomState

            """
            if val is not None:
                set_seed(&self.rng_state, val)
            else:
                entropy_init(&self.rng_state)
            self.rng_state.gauss = 0.0
            self.rng_state.has_gauss = 0
            self.rng_state.has_uint32= 0
            self.rng_state.uinteger = 0

    ELSE:
        def seed(self, val=None, inc=None):
            """
            seed(val=None, inc=None)

            Seed the generator.

            This method is called when `RandomState` is initialized. It can be
            called again to re-seed the generator. For details, see `RandomState`.

            Parameters
            ----------
            val : int, optional
                Seed for `RandomState`.
            inc : int, optional
                Increment to use for producing multiple streams

            See Also
            --------
            RandomState
            """
            if val is not None and inc is not None:
                set_seed(&self.rng_state, val, inc)
            else:
                entropy_init(&self.rng_state)
            self.rng_state.gauss = 0.0
            self.rng_state.has_gauss = 0
            self.rng_state.has_uint32= 0
            self.rng_state.uinteger = 0

    if RNG_ADVANCEABLE:
        def advance(self, rng_state_t delta):
            """
            advance(delta)

            Advance the PRNG as-if delta drawn have occurred.

            Parameters
            ----------
            delta : integer, positive
                Number of draws to advance the PRNG.

            Returns
            -------
            out : None
                Returns 'None' on success.

            Notes
            -----
            Advancing the prng state resets any pre-computed random numbers.
            This is required to ensure exact reproducibility.
            """
            advance(&self.rng_state, delta)
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            return None

    if RNG_JUMPABLE:
        def jump(self, uint32_t iter = 1):
            """
            jump(iter = 1)

            Jumps the random number generator by a pre-specified skip.  The size of the jump is
            rng-specific.

            Parameters
            ----------
            iter : integer, positive
                Number of times to jump the state of the rng.

            Returns
            -------
            out : None
                Returns 'None' on success.

            Notes
            -----
            Jumping the rng state resets any pre-computed random numbers. This is required to ensure
            exact reproducibility.
            """
            cdef Py_ssize_t i;
            for i in range(iter):
                jump(&self.rng_state)
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            return None

    def get_state(self):
        """
        get_state()

        Return a tuple representing the internal state of the generator.

        For more details, see `set_state`.

        Returns
        -------
        out : tuple(str, tuple, tuple, tuple)
            The returned tuple has the following items:

            1. the string containing the PRNG type.
            2. a tuple containing the PRNG-specific state
            3. a tuple containing two values :``has_gauss`` and ``cached_gaussian``
            4. a tuple containing two values :``has_uint32`` and ``cached_uint32``

        See Also
        --------
        set_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For information about the specific structure of the PRNG-specific
        component, see the class documentation.
        """
        return  {'name': RNG_NAME,
                 'state': _get_state(self.rng_state),
                 'gauss': {'has_gauss': self.rng_state.has_gauss, 'gauss': self.rng_state.gauss},
                 'uint32': {'has_uint32': self.rng_state.has_uint32, 'uint32': self.rng_state.uinteger}
                 }

    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator from a tuple.

        For use if one has reason to manually (re-)set the internal state of the
        pseudo-random number generating algorithm.

        Parameters
        ----------
        state : tuple(str, tuple, tuple, tuple)
            The returned tuple has the following items:

            1. the string containing the PRNG type.
            2. a tuple containing the PRNG-specific state
            3. a tuple containing two values :``has_gauss`` and ``cached_gaussian``
            4. a tuple containing two values :``has_uint32`` and ``cached_uint32``

        Returns
        -------
        out : None
            Returns 'None' on success.

        See Also
        --------
        get_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For information about the specific structure of the PRNG-specific
        component, see the class documentation.
        """
        rng_name = RNG_NAME
        if RNG_MT19937:
            if isinstance(state, tuple):
                if state[0] != 'MT19937':
                    raise ValueError('Not a ' + rng_name + ' RNG state')
                _set_state(self.rng_state, (state[1], state[2]))
                self.rng_state.has_gauss = state[3]
                self.rng_state.gauss = state[4]
                self.rng_state.has_uint32 = 0
                self.rng_state.uinteger = 0
                return None

        if state['name'] != rng_name:
            raise ValueError('Not a ' + rng_name + ' RNG state')
        _set_state(self.rng_state, state['state'])
        self.rng_state.has_gauss = state['gauss']['has_gauss']
        self.rng_state.gauss = state['gauss']['gauss']
        self.rng_state.has_uint32 = state['uint32']['has_uint32']
        self.rng_state.uinteger = state['uint32']['uint32']

    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098
        >>> type(np.random.random_sample())
        <type 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        return cont(&self.rng_state, &random_standard_uniform,
                    size, self.lock, 0,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def random_uintegers(self, size=None, int bits=64):
        """
        random_uintegers(size=None, bits=64)

        Return random unsigned integers

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        bits : int {32, 64}
            Size of the unsigned integer to return, either 32 bit or 64 bit.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method effectively exposes access to the raw underlying
        pseudo-random number generator since these all produce unsigned
        integers. In practice these are most useful for generating other
        random numbers.

        These should not be used to produce bounded random numbers by
        simple truncation.  Instead see ``random_bounded_integers``.
        """
        if bits == 64:
            return disc(&self.rng_state, &random_uint64, size, self.lock, 0, 0,
                            0, '', CONS_NONE, 0, '', CONS_NONE, 0, '', CONS_NONE)
        elif bits == 32:
            return uint0_32(&self.rng_state, &random_uint32, size, self.lock)
        else:
            raise ValueError('Unknown value of bits.  Must be either 32 or 64.')

    def random_bounded_uintegers(self, uint64_t high, size=None):
        cdef Py_ssize_t i, n
        cdef uint32_t [::1] randoms32
        cdef uint64_t [::1] randoms64
        if high < 4294967295:
            if size is None:
                return random_bounded_uint32(&self.rng_state, high)
            else:
                n = compute_numel(size)
                randoms32 = np.empty(n, np.uint32)
                with self.lock, nogil:
                    for i in range(n):
                        randoms32[i] = random_bounded_uint32(&self.rng_state, high)
                return np.asarray(randoms32).reshape(size)
        else:
            if size is None:
                return random_bounded_uint64(&self.rng_state, high)
            else:
                n = compute_numel(size)
                randoms64 = np.empty(n, np.uint64)
                with self.lock, nogil:
                    for i in range(n):
                        randoms64[i] = random_bounded_uint64(&self.rng_state, high)
                return np.asarray(randoms64).reshape(size)


    def random_bounded_integers(self, int64_t low, high=None, size=None):
        cdef int64_t _low, _high
        if high is None:
            _high = low
            _low = 0
        else:
            _high = high
            _low = low

        if _low >= -2147483648 and _high <= 2147483647:
            return int2_i_32(&self.rng_state, &random_bounded_int32,
                             <int32_t>_low, <int32_t>_high, size, self.lock)
        else:
            return int2_i(&self.rng_state, &random_bounded_int64, _low, _high, size, self.lock)


    def standard_normal(self, size=None, method='inv'):
        """
        standard_normal(size=None, method='inv')

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : str
            Either 'inv' or 'zig'. 'inv' uses the default FIXME method.  'zig' uses
            the much faster ziggurat method of FIXME.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
               -0.38672696, -0.4685006 ])                               #random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        """
        if method == 'inv':
            return cont(&self.rng_state, &random_gauss, size, self.lock, 0,
                        0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE)
        else:
            return cont(&self.rng_state, &random_gauss_zig_julia, size, self.lock, 0,
                        0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE)

    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        Output a 3x8000 array:

        >>> n = np.random.standard_exponential((3, 8000))

        """
        return cont(&self.rng_state, &random_standard_exponential, size, self.lock, 0,
                    0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        Notes
        -----
        The probability density function for the full Cauchy distribution is

        .. math:: P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[ 1+
                  (\\frac{x-x_0}{\\gamma})^2 \\bigr] }

        and the Standard Cauchy distribution just sets :math:`x_0=0` and
        :math:`\\gamma=1`

        The Cauchy distribution arises in the solution to the driven harmonic
        oscillator problem, and also describes spectral line broadening. It
        also describes the distribution of values at which a line tilted at
        a random angle will cut the x axis.

        When studying hypothesis tests that assume normality, seeing how the
        tests perform on data from a Cauchy distribution is a good indicator of
        their sensitivity to a heavy-tailed distribution, since the Cauchy looks
        very much like a Gaussian distribution, but with heavier tails.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
              Distribution",
              http://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
        .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
              Wolfram Web Resource.
              http://mathworld.wolfram.com/CauchyDistribution.html
        .. [3] Wikipedia, "Cauchy distribution"
              http://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------
        Draw samples and plot the distribution:

        >>> s = np.random.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_standard_cauchy, size, self.lock, 0,
                    0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE)

    def standard_gamma(self, object shape, size=None):
        """
        standard_gamma(shape, size=None)

        Draw samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        shape (sometimes designated "k") and scale=1.

        Parameters
        ----------
        shape : float
            Parameter, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        See Also
        --------
        scipy.stats.distributions.gamma : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma-distribution",
               http://en.wikipedia.org/wiki/Gamma-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 1. # mean and width
        >>> s = np.random.standard_gamma(shape, 1000000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/ \\
        ...                       (sps.gamma(shape) * scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_standard_gamma, size, self.lock, 1,
                    shape, 'shape', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def binomial(self, n, p, size=None):
        """
        binomial(n, p, size=None)
        Draw samples from a binomial distribution.
        Samples are drawn from a binomial distribution with specified
        parameters, n trials and p probability of success where
        n an integer >= 0 and p is in the interval [0,1]. (n may be
        input as a float, but it is truncated to an integer in use)
        Parameters
        ----------
        n : float (but truncated to an integer)
                parameter, >= 0.
        p : float
                parameter, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].
        See Also
        --------
        scipy.stats.distributions.binom : probability density function,
            distribution or cumulative density function, etc.
        Notes
        -----
        The probability density for the binomial distribution is
        .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N},
        where :math:`n` is the number of trials, :math:`p` is the probability
        of success, and :math:`N` is the number of successes.
        When estimating the standard error of a proportion in a population by
        using a random sample, the normal distribution works well unless the
        product p*n <=5, where p = population proportion estimate, and n =
        number of samples, in which case the binomial distribution is used
        instead. For example, a sample of 15 people shows 4 who are left
        handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
        so the binomial distribution should be used in this case.
        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics with R",
               Springer-Verlag, 2002.
        .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/BinomialDistribution.html
        .. [5] Wikipedia, "Binomial-distribution",
               http://en.wikipedia.org/wiki/Binomial_distribution
        Examples
        --------
        Draw samples from the distribution:
        >>> n, p = 10, .5  # number of trials, probability of each trial
        >>> s = np.random.binomial(n, p, 1000)
        # result of flipping a coin 10 times, tested 1000 times.
        A real world example. A company drills 9 wild-cat oil exploration
        wells, each with an estimated probability of success of 0.1. All nine
        wells fail. What is the probability of that happening?
        Let's do 20,000 trials of the model, and count the number that
        generate zero positive results.
        >>> sum(np.random.binomial(9, 0.1, 20000) == 0)/20000.
        # answer = 0.38885, or 38%.

        """
        return disc(&self.rng_state, &random_binomial, size, self.lock, 1, 1,
                    p, 'p', CONS_BOUNDED_0_1_NOTNAN,
                    n, 'n', CONS_POSITIVE,
                    0.0, '', CONS_NONE)


    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Draw samples from a standard Student's t distribution with `df` degrees
        of freedom.

        A special case of the hyperbolic distribution.  As `df` gets
        large, the result resembles that of the standard normal
        distribution (`standard_normal`).

        Parameters
        ----------
        df : int
            Degrees of freedom, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Drawn samples.

        Notes
        -----
        The probability density function for the t distribution is

        .. math:: P(x, df) = \\frac{\\Gamma(\\frac{df+1}{2})}{\\sqrt{\\pi df}
                  \\Gamma(\\frac{df}{2})}\\Bigl( 1+\\frac{x^2}{df} \\Bigr)^{-(df+1)/2}

        The t test is based on an assumption that the data come from a
        Normal distribution. The t test provides a way to test whether
        the sample mean (that is the mean calculated from the data) is
        a good estimate of the true mean.

        The derivation of the t-distribution was first published in
        1908 by William Gisset while working for the Guinness Brewery
        in Dublin. Due to proprietary issues, he had to publish under
        a pseudonym, and so he used the name Student.

        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics With R",
               Springer, 2002.
        .. [2] Wikipedia, "Student's t-distribution"
               http://en.wikipedia.org/wiki/Student's_t-distribution

        Examples
        --------
        From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
        women in Kj is:

        >>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \\
        ...                    7515, 8230, 8770])

        Does their energy intake deviate systematically from the recommended
        value of 7725 kJ?

        We have 10 degrees of freedom, so is the sample mean within 95% of the
        recommended value?

        >>> s = np.random.standard_t(10, size=100000)
        >>> np.mean(intake)
        6753.636363636364
        >>> intake.std(ddof=1)
        1142.1232221373727

        Calculate the t statistic, setting the ddof parameter to the unbiased
        value so the divisor in the standard deviation will be degrees of
        freedom, N-1.

        >>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(s, bins=100, normed=True)

        For a one-sided t-test, how far out in the distribution does the t
        statistic appear?

        >>> np.sum(s<t) / float(len(s))
        0.0090699999999999999  #random

        So the p-value is about 0.009, which says the null hypothesis has a
        probability of about 99% of being true.

        """
        return cont(&self.rng_state, &random_standard_t, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0, '', CONS_NONE,
                    0, '', CONS_NONE)

    def bytes(self, Py_ssize_t length):
        """
        bytes(length)

        Return random bytes.
        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : str
            String of length `length`.

        Examples
        --------
        >>> np.random.bytes(10)
        ' eh\\x85\\x022SZ\\xbf\\xa4' #random
        """
        cdef Py_ssize_t n_uint32 = ((length - 1) // 4 + 1)
        return self.random_uintegers(n_uint32, bits=32).tobytes()[:length]

    def randn(self, *args, method='inv'):
        """
        randn(d0, d1, ..., dn)

        Return a sample (or samples) from the "standard normal" distribution.

        If positive, int_like or int-convertible arguments are provided,
        `randn` generates an array of shape ``(d0, d1, ..., dn)``, filled
        with random floats sampled from a univariate "normal" (Gaussian)
        distribution of mean 0 and variance 1 (if any of the :math:`d_i` are
        floats, they are first converted to integers by truncation). A single
        float randomly sampled from the distribution is returned if no
        argument is provided.

        This is a convenience function.  If you want an interface that takes a
        tuple as the first argument, use `numpy.random.standard_normal` instead.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should be all positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        Z : ndarray or float
            A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
            the standard normal distribution, or a single such float if
            no parameters were supplied.

        See Also
        --------
        random.standard_normal : Similar, but takes a tuple as its argument.

        Notes
        -----
        For random samples from :math:`N(\mu, \sigma^2)`, use:

        ``sigma * np.random.randn(...) + mu``

        Examples
        --------
        >>> np.random.randn()
        2.1923875335537315 #random

        Two-by-four array of samples from N(3, 6.25):

        >>> 2.5 * np.random.randn(2, 4) + 3
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random
        """
        return self.standard_normal(size=args, method=method)

    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        Create an array of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should all be positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Notes
        -----
        This is a convenience function. If you want an interface that
        takes a shape-tuple as the first argument, refer to
        np.random.random_sample .

        Examples
        --------
        >>> np.random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random
        """
        return self.random_sample(size=args)

    def uniform(self, object low=0.0, object high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=None)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float
            Upper boundary of the output interval.  All values generated will be
            less than high.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed
                          interval ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats,
               uniformly distributed over ``[0, 1)``.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        Examples
        --------
        Draw samples from the distribution:

        >>> s = np.random.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True
        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 15, normed=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()
        """

        return cont(&self.rng_state, &random_uniform, size, self.lock, 2,
                    low, '', CONS_NONE,
                    high - low, 'high - low', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    # Shuffling and permutations:
    def shuffle(self, object x):
        """
        shuffle(x)

        Modify a sequence in-place by shuffling its contents.

        Parameters
        ----------
        x : array_like
            The array or list to be shuffled.

        Returns
        -------
        None

        Examples
        --------
        >>> arr = np.arange(10)
        >>> np.random.shuffle(arr)
        >>> arr
        [1 7 5 2 9 4 3 6 0 8]

        This function only shuffles the array along the first index of a
        multi-dimensional array:

        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.shuffle(arr)
        >>> arr
        array([[3, 4, 5],
               [6, 7, 8],
               [0, 1, 2]])

        """
        cdef Py_ssize_t i
        cdef long j

        i = len(x) - 1

        # Logic adapted from random.shuffle()
        if isinstance(x, np.ndarray) and \
           (x.ndim > 1 or x.dtype.fields is not None):
            # For a multi-dimensional ndarray, indexing returns a view onto
            # each row. So we can't just use ordinary assignment to swap the
            # rows; we need a bounce buffer.
            buf = np.empty_like(x[0])
            with self.lock:
                while i > 0:
                    j = random_interval(&self.rng_state, i)
                    buf[...] = x[j]
                    x[j] = x[i]
                    x[i] = buf
                    i = i - 1
        else:
            # For single-dimensional arrays, lists, and any other Python
            # sequence types, indexing returns a real object that's
            # independent of the array contents, so we can just swap directly.
            with self.lock:
                while i > 0:
                    j = random_interval(&self.rng_state, i)
                    x[i], x[j] = x[j], x[i]
                    i = i - 1

    def permutation(self, object x):
        """
        permutation(x)

        Randomly permute a sequence, or return a permuted range.

        If `x` is a multi-dimensional array, it is only shuffled along its
        first index.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        Examples
        --------
        >>> np.random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

        >>> np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12])

        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.permutation(arr)
        array([[6, 7, 8],
               [0, 1, 2],
               [3, 4, 5]])

        """
        if isinstance(x, (int, long, np.integer)):
            arr = np.arange(x)
        else:
            arr = np.array(x)
        self.shuffle(arr)
        return arr

    def tomaxint(self, size=None):
        """
        tomaxint(size=None)

        Random integers between 0 and ``sys.maxint``, inclusive.

        Return a sample of uniformly distributed random integers in the interval
        [0, ``sys.maxint``].

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Uniform sampling over a given half-open interval of integers.
        random_integers : Uniform sampling over a given closed interval of
            integers.

        Examples
        --------
        >>> RS = np.random.mtrand.RandomState() # need a RandomState object
        >>> RS.tomaxint((2,2,2))
        array([[[1170048599, 1600360186],
                [ 739731006, 1947757578]],
               [[1871712945,  752307660],
                [1601631370, 1479324245]]])
        >>> import sys
        >>> sys.maxint
        2147483647
        >>> RS.tomaxint((2,2,2)) < sys.maxint
        array([[[ True,  True],
                [ True,  True]],
               [[ True,  True],
                [ True,  True]]], dtype=bool)

        """
        cdef Py_ssize_t n
        cdef np.long_t [::1] randoms

        if size is None:
            return random_positive_int(&self.rng_state)

        n = compute_numel(size)
        randoms = np.empty(n, np.long)
        for i in range(n):
            randoms[i] = random_positive_int(&self.rng_state)
        return np.asarray(randoms).reshape(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        normal(loc=0.0, scale=1.0, size=None)

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float
            Mean ("centre") of the distribution.
        scale : float
            Standard deviation (spread or "width") of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        See Also
        --------
        scipy.stats.distributions.norm : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard
        deviation. The square of the standard deviation, :math:`\\sigma^2`,
        is called the variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        `numpy.random.normal` is more likely to return samples lying close to
        the mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               http://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
               Random Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 0, 0.1 # mean and standard deviation
        >>> s = np.random.normal(mu, sigma, 1000)

        Verify the mean and the variance:

        >>> abs(mu - np.mean(s)) < 0.01
        True

        >>> abs(sigma - np.std(s, ddof=1)) < 0.01
        True

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_normal, size, self.lock, 2,
                    loc, '', CONS_NONE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        Draw samples from a Beta distribution.

        The Beta distribution is a special case of the Dirichlet distribution,
        and is related to the Gamma distribution.  It has the probability
        distribution function

        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                         (1 - x)^{\\beta - 1},

        where the normalisation, B, is the beta function,

        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                     (1 - t)^{\\beta - 1} dt.

        It is often seen in Bayesian inference and order statistics.

        Parameters
        ----------
        a : float
            Alpha, non-negative.
        b : float
            Beta, non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Array of the given shape, containing values drawn from a
            Beta distribution.

        """
        return cont(&self.rng_state, &random_beta, size, self.lock, 2,
                    a, 'a', CONS_POSITIVE,
                    b, 'b', CONS_POSITIVE,
                    0.0, '', CONS_NONE)


    def exponential(self, scale=1.0, size=None):
        """
        exponential(scale=1.0, size=None)

        Draw samples from an exponential distribution.

        Its probability density function is

        .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

        for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
        which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
        The rate parameter is an alternative, widely used parameterization
        of the exponential distribution [3]_.

        The exponential distribution is a continuous analogue of the
        geometric distribution.  It describes many common situations, such as
        the size of raindrops measured over many rainstorms [1]_, or the time
        between page requests to Wikipedia [2]_.

        Parameters
        ----------
        scale : float
            The scale parameter, :math:`\\beta = 1/\\lambda`.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        References
        ----------
        .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
               Random Signal Principles", 4th ed, 2001, p. 57.
        .. [2] "Poisson Process", Wikipedia,
               http://en.wikipedia.org/wiki/Poisson_process
        .. [3] "Exponential Distribution, Wikipedia,
               http://en.wikipedia.org/wiki/Exponential_distribution

        """
        return cont(&self.rng_state, &random_exponential, size, self.lock, 1,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def gamma(self, shape, scale=1.0, size=None):
        """
        gamma(shape, scale=1.0, size=None)

        Draw samples from a Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        `shape` (sometimes designated "k") and `scale` (sometimes designated
        "theta"), where both parameters are > 0.

        Parameters
        ----------
        shape : scalar > 0
            The shape of the gamma distribution.
        scale : scalar > 0, optional
            The scale of the gamma distribution.  Default is equal to 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray, float
            Returns one sample unless `size` parameter is specified.

        See Also
        --------
        scipy.stats.distributions.gamma : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma-distribution",
               http://en.wikipedia.org/wiki/Gamma-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 2. # mean and dispersion
        >>> s = np.random.gamma(shape, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> y = bins**(shape-1)*(np.exp(-bins/scale) /
        ...                      (sps.gamma(shape)*scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_gamma, size, self.lock, 2,
                    shape, 'shape', CONS_POSITIVE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    def f(self, dfnum, dfden, size=None):
        """
        f(dfnum, dfden, size=None)

        Draw samples from an F distribution.

        Samples are drawn from an F distribution with specified parameters,
        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
        freedom in denominator), where both parameters should be greater than
        zero.

        The random variate of the F distribution (also known as the
        Fisher distribution) is a continuous probability distribution
        that arises in ANOVA tests, and is the ratio of two chi-square
        variates.

        Parameters
        ----------
        dfnum : float
            Degrees of freedom in numerator. Should be greater than zero.
        dfden : float
            Degrees of freedom in denominator. Should be greater than zero.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Samples from the Fisher distribution.

        See Also
        --------
        scipy.stats.distributions.f : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The F statistic is used to compare in-group variances to between-group
        variances. Calculating the distribution depends on the sampling, and
        so it is a function of the respective degrees of freedom in the
        problem.  The variable `dfnum` is the number of samples minus one, the
        between-groups degrees of freedom, while `dfden` is the within-groups
        degrees of freedom, the sum of the number of samples in each group
        minus the number of groups.

        References
        ----------
        .. [1] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [2] Wikipedia, "F-distribution",
               http://en.wikipedia.org/wiki/F-distribution

        Examples
        --------
        An example from Glantz[1], pp 47-40:

        Two groups, children of diabetics (25 people) and children from people
        without diabetes (25 controls). Fasting blood glucose was measured,
        case group had a mean value of 86.1, controls had a mean value of
        82.2. Standard deviations were 2.09 and 2.49 respectively. Are these
        data consistent with the null hypothesis that the parents diabetic
        status does not affect their children's blood glucose levels?
        Calculating the F statistic from the data gives a value of 36.01.

        Draw samples from the distribution:

        >>> dfnum = 1. # between group degrees of freedom
        >>> dfden = 48. # within groups degrees of freedom
        >>> s = np.random.f(dfnum, dfden, 1000)

        The lower bound for the top 1% of the samples is :

        >>> sort(s)[-10]
        7.61988120985

        So there is about a 1% chance that the F statistic will exceed 7.62,
        the measured value is 36, so the null hypothesis is rejected at the 1%
        level.

        """
        return cont(&self.rng_state, &random_f, size, self.lock, 2,
                    dfnum, 'dfnum', CONS_POSITIVE,
                    dfden, 'dfden', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """
        noncentral_f(dfnum, dfden, nonc, size=None)

        Draw samples from the noncentral F distribution.

        Samples are drawn from an F distribution with specified parameters,
        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
        freedom in denominator), where both parameters > 1.
        `nonc` is the non-centrality parameter.

        Parameters
        ----------
        dfnum : int
            Parameter, should be > 1.
        dfden : int
            Parameter, should be > 1.
        nonc : float
            Parameter, should be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : scalar or ndarray
            Drawn samples.

        Notes
        -----
        When calculating the power of an experiment (power = probability of
        rejecting the null hypothesis when a specific alternative is true) the
        non-central F statistic becomes important.  When the null hypothesis is
        true, the F statistic follows a central F distribution. When the null
        hypothesis is not true, then it follows a non-central F statistic.

        References
        ----------
        .. [1] Weisstein, Eric W. "Noncentral F-Distribution."
               From MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/NoncentralF-Distribution.html
        .. [2] Wikipedia, "Noncentral F distribution",
               http://en.wikipedia.org/wiki/Noncentral_F-distribution

        Examples
        --------
        In a study, testing for a specific alternative to the null hypothesis
        requires use of the Noncentral F distribution. We need to calculate the
        area in the tail of the distribution that exceeds the value of the F
        distribution for the null hypothesis.  We'll plot the two probability
        distributions for comparison.

        >>> dfnum = 3 # between group deg of freedom
        >>> dfden = 20 # within groups degrees of freedom
        >>> nonc = 3.0
        >>> nc_vals = np.random.noncentral_f(dfnum, dfden, nonc, 1000000)
        >>> NF = np.histogram(nc_vals, bins=50, normed=True)
        >>> c_vals = np.random.f(dfnum, dfden, 1000000)
        >>> F = np.histogram(c_vals, bins=50, normed=True)
        >>> plt.plot(F[1][1:], F[0])
        >>> plt.plot(NF[1][1:], NF[0])
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_noncentral_f, size, self.lock, 3,
                    dfnum, 'dfnum', CONS_POSITIVE,
                    dfden, 'dfden', CONS_POSITIVE,
                    nonc, 'nonc', CONS_NON_NEGATIVE)

    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard normal
        distributions (mean 0, variance 1), are squared and summed, the
        resulting distribution is chi-square (see Notes).  This distribution
        is often used in hypothesis testing.

        Parameters
        ----------
        df : int
             Number of degrees of freedom.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        output : ndarray
            Samples drawn from the distribution, packed in a `size`-shaped
            array.

        Raises
        ------
        ValueError
            When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
            is given.

        Notes
        -----
        The variable obtained by summing the squares of `df` independent,
        standard normally distributed random variables:

        .. math:: Q = \\sum_{i=0}^{\\mathtt{df}} X^2_i

        is chi-square distributed, denoted

        .. math:: Q \\sim \\chi^2_k.

        The probability density function of the chi-squared distribution is

        .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}
                         x^{k/2 - 1} e^{-x/2},

        where :math:`\\Gamma` is the gamma function,

        .. math:: \\Gamma(x) = \\int_0^{-\\infty} t^{x - 1} e^{-t} dt.

        References
        ----------
        .. [1] NIST "Engineering Statistics Handbook"
               http://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

        Examples
        --------
        >>> np.random.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272])

        """
        return cont(&self.rng_state, &random_chisquare, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def noncentral_chisquare(self, df, nonc, size=None):
        """
        noncentral_chisquare(df, nonc, size=None)

        Draw samples from a noncentral chi-square distribution.

        The noncentral :math:`\\chi^2` distribution is a generalisation of
        the :math:`\\chi^2` distribution.

        Parameters
        ----------
        df : int
            Degrees of freedom, should be > 0 as of Numpy 1.10,
            should be > 1 for earlier versions.
        nonc : float
            Non-centrality, should be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Notes
        -----
        The probability density function for the noncentral Chi-square
        distribution is

        .. math:: P(x;df,nonc) = \\sum^{\\infty}_{i=0}
                               \\frac{e^{-nonc/2}(nonc/2)^{i}}{i!}
                               \\P_{Y_{df+2i}}(x),

        where :math:`Y_{q}` is the Chi-square with q degrees of freedom.

        In Delhi (2007), it is noted that the noncentral chi-square is
        useful in bombing and coverage problems, the probability of
        killing the point target given by the noncentral chi-squared
        distribution.

        References
        ----------
        .. [1] Delhi, M.S. Holla, "On a noncentral chi-square distribution in
               the analysis of weapon systems effectiveness", Metrika,
               Volume 15, Number 1 / December, 1970.
        .. [2] Wikipedia, "Noncentral chi-square distribution"
               http://en.wikipedia.org/wiki/Noncentral_chi-square_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> import matplotlib.pyplot as plt
        >>> values = plt.hist(np.random.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, normed=True)
        >>> plt.show()

        Draw values from a noncentral chisquare with very small noncentrality,
        and compare to a chisquare.

        >>> plt.figure()
        >>> values = plt.hist(np.random.noncentral_chisquare(3, .0000001, 100000),
        ...                   bins=np.arange(0., 25, .1), normed=True)
        >>> values2 = plt.hist(np.random.chisquare(3, 100000),
        ...                    bins=np.arange(0., 25, .1), normed=True)
        >>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
        >>> plt.show()

        Demonstrate how large values of non-centrality lead to a more symmetric
        distribution.

        >>> plt.figure()
        >>> values = plt.hist(np.random.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, normed=True)
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_noncentral_chisquare, size, self.lock, 2,
                    df, 'df', CONS_POSITIVE,
                    nonc, 'nonc', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE)

    def vonmises(self, mu, kappa, size=None):
        """
        vonmises(mu, kappa, size=None)

        Draw samples from a von Mises distribution.

        Samples are drawn from a von Mises distribution with specified mode
        (mu) and dispersion (kappa), on the interval [-pi, pi].

        The von Mises distribution (also known as the circular normal
        distribution) is a continuous probability distribution on the unit
        circle.  It may be thought of as the circular analogue of the normal
        distribution.

        Parameters
        ----------
        mu : float
            Mode ("center") of the distribution.
        kappa : float
            Dispersion of the distribution, has to be >=0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : scalar or ndarray
            The returned samples, which are in the interval [-pi, pi].

        See Also
        --------
        scipy.stats.distributions.vonmises : probability density function,
            distribution, or cumulative density function, etc.

        Notes
        -----
        The probability density for the von Mises distribution is

        .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)},

        where :math:`\\mu` is the mode and :math:`\\kappa` the dispersion,
        and :math:`I_0(\\kappa)` is the modified Bessel function of order 0.

        The von Mises is named for Richard Edler von Mises, who was born in
        Austria-Hungary, in what is now the Ukraine.  He fled to the United
        States in 1939 and became a professor at Harvard.  He worked in
        probability theory, aerodynamics, fluid mechanics, and philosophy of
        science.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
               Mathematical Functions with Formulas, Graphs, and Mathematical
               Tables, 9th printing," New York: Dover, 1972.
        .. [2] von Mises, R., "Mathematical Theory of Probability
               and Statistics", New York: Academic Press, 1964.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, kappa = 0.0, 4.0 # mean and dispersion
        >>> s = np.random.vonmises(mu, kappa, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy.special import i0
        >>> plt.hist(s, 50, normed=True)
        >>> x = np.linspace(-np.pi, np.pi, num=51)
        >>> y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
        >>> plt.plot(x, y, linewidth=2, color='r')
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_vonmises, size, self.lock, 2,
                    mu, 'mu', CONS_NONE,
                    kappa, 'kappa', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE)



    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto II or Lomax distribution with
        specified shape.

        The Lomax or Pareto II distribution is a shifted Pareto
        distribution. The classical Pareto distribution can be
        obtained from the Lomax distribution by adding 1 and
        multiplying by the scale parameter ``m`` (see Notes).  The
        smallest value of the Lomax distribution is zero while for the
        classical Pareto distribution it is ``mu``, where the standard
        Pareto distribution has location ``mu = 1``.  Lomax can also
        be considered as a simplified version of the Generalized
        Pareto distribution (available in SciPy), with the scale set
        to one and the location set to zero.

        The Pareto distribution must be greater than zero, and is
        unbounded above.  It is also known as the "80-20 rule".  In
        this distribution, 80 percent of the weights are in the lowest
        20 percent of the range, while the other 20 percent fill the
        remaining 80 percent of the range.

        Parameters
        ----------
        shape : float, > 0.
            Shape of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        See Also
        --------
        scipy.stats.distributions.lomax.pdf : probability density function,
            distribution or cumulative density function, etc.
        scipy.stats.distributions.genpareto.pdf : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Pareto distribution is

        .. math:: p(x) = \\frac{am^a}{x^{a+1}}

        where :math:`a` is the shape and :math:`m` the scale.

        The Pareto distribution, named after the Italian economist
        Vilfredo Pareto, is a power law probability distribution
        useful in many real world problems.  Outside the field of
        economics it is generally referred to as the Bradford
        distribution. Pareto developed the distribution to describe
        the distribution of wealth in an economy.  It has also found
        use in insurance, web page access statistics, oil field sizes,
        and many other problems, including the download frequency for
        projects in Sourceforge [1]_.  It is one of the so-called
        "fat-tailed" distributions.


        References
        ----------
        .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
               Sourceforge projects.
        .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
        .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
               Values, Birkhauser Verlag, Basel, pp 23-30.
        .. [4] Wikipedia, "Pareto distribution",
               http://en.wikipedia.org/wiki/Pareto_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a, m = 3., 2.  # shape and mode
        >>> s = (np.random.pareto(a, 1000) + 1) * m

        Display the histogram of the samples, along with the probability
        density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 100, normed=True)
        >>> fit = a*m**a / bins**(a+1)
        >>> plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_pareto, size, self.lock, 1,
                    a, 'a', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def weibull(self, a, size=None):
        """
        weibull(a, size=None)

        Draw samples from a Weibull distribution.

        Draw samples from a 1-parameter Weibull distribution with the given
        shape parameter `a`.

        .. math:: X = (-ln(U))^{1/a}

        Here, U is drawn from the uniform distribution over (0,1].

        The more common 2-parameter Weibull, including a scale parameter
        :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.

        Parameters
        ----------
        a : float
            Shape of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray

        See Also
        --------
        scipy.stats.distributions.weibull_max
        scipy.stats.distributions.weibull_min
        scipy.stats.distributions.genextreme
        gumbel

        Notes
        -----
        The Weibull (or Type III asymptotic extreme value distribution
        for smallest values, SEV Type III, or Rosin-Rammler
        distribution) is one of a class of Generalized Extreme Value
        (GEV) distributions used in modeling extreme value problems.
        This class includes the Gumbel and Frechet distributions.

        The probability density for the Weibull distribution is

        .. math:: p(x) = \\frac{a}
                         {\\lambda}(\\frac{x}{\\lambda})^{a-1}e^{-(x/\\lambda)^a},

        where :math:`a` is the shape and :math:`\\lambda` the scale.

        The function has its peak (the mode) at
        :math:`\\lambda(\\frac{a-1}{a})^{1/a}`.

        When ``a = 1``, the Weibull distribution reduces to the exponential
        distribution.

        References
        ----------
        .. [1] Waloddi Weibull, Royal Technical University, Stockholm,
               1939 "A Statistical Theory Of The Strength Of Materials",
               Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
               Generalstabens Litografiska Anstalts Forlag, Stockholm.
        .. [2] Waloddi Weibull, "A Statistical Distribution Function of
               Wide Applicability", Journal Of Applied Mechanics ASME Paper
               1951.
        .. [3] Wikipedia, "Weibull distribution",
               http://en.wikipedia.org/wiki/Weibull_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 5. # shape
        >>> s = np.random.weibull(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(1,100.)/50.
        >>> def weib(x,n,a):
        ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

        >>> count, bins, ignored = plt.hist(np.random.weibull(5.,1000))
        >>> x = np.arange(1,100.)/50.
        >>> scale = count.max()/weib(x, 1., 5.).max()
        >>> plt.plot(x, weib(x, 1., 5.)*scale)
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_weibull, size, self.lock, 1,
                    a, 'a', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def power(self, a, size=None):
        """
        power(a, size=None)

        Draws samples in [0, 1] from a power distribution with positive
        exponent a - 1.

        Also known as the power function distribution.

        Parameters
        ----------
        a : float
            parameter, > 0
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The returned samples lie in [0, 1].

        Raises
        ------
        ValueError
            If a < 1.

        Notes
        -----
        The probability density function is

        .. math:: P(x; a) = ax^{a-1}, 0 \\le x \\le 1, a>0.

        The power function distribution is just the inverse of the Pareto
        distribution. It may also be seen as a special case of the Beta
        distribution.

        It is used, for example, in modeling the over-reporting of insurance
        claims.

        References
        ----------
        .. [1] Christian Kleiber, Samuel Kotz, "Statistical size distributions
               in economics and actuarial sciences", Wiley, 2003.
        .. [2] Heckert, N. A. and Filliben, James J. "NIST Handbook 148:
               Dataplot Reference Manual, Volume 2: Let Subcommands and Library
               Functions", National Institute of Standards and Technology
               Handbook Series, June 2003.
               http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/powpdf.pdf

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 5. # shape
        >>> samples = 1000
        >>> s = np.random.power(a, samples)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, bins=30)
        >>> x = np.linspace(0, 1, 100)
        >>> y = a*x**(a-1.)
        >>> normed_y = samples*np.diff(bins)[0]*y
        >>> plt.plot(x, normed_y)
        >>> plt.show()

        Compare the power function distribution to the inverse of the Pareto.

        >>> from scipy import stats
        >>> rvs = np.random.power(5, 1000000)
        >>> rvsp = np.random.pareto(5, 1000000)
        >>> xx = np.linspace(0,1,100)
        >>> powpdf = stats.powerlaw.pdf(xx,5)

        >>> plt.figure()
        >>> plt.hist(rvs, bins=50, normed=True)
        >>> plt.plot(xx,powpdf,'r-')
        >>> plt.title('np.random.power(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, normed=True)
        >>> plt.plot(xx,powpdf,'r-')
        >>> plt.title('inverse of 1 + np.random.pareto(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, normed=True)
        >>> plt.plot(xx,powpdf,'r-')
        >>> plt.title('inverse of stats.pareto(5)')

        """
        return cont(&self.rng_state, &random_power, size, self.lock, 1,
                    a, 'a', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """
        laplace(loc=0.0, scale=1.0, size=None)

        Draw samples from the Laplace or double exponential distribution with
        specified location (or mean) and scale (decay).

        The Laplace distribution is similar to the Gaussian/normal distribution,
        but is sharper at the peak and has fatter tails. It represents the
        difference between two independent, identically distributed exponential
        random variables.

        Parameters
        ----------
        loc : float, optional
            The position, :math:`\\mu`, of the distribution peak.
        scale : float, optional
            :math:`\\lambda`, the exponential decay.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or float

        Notes
        -----
        It has the probability density function

        .. math:: f(x; \\mu, \\lambda) = \\frac{1}{2\\lambda}
                                       \\exp\\left(-\\frac{|x - \\mu|}{\\lambda}\\right).

        The first law of Laplace, from 1774, states that the frequency
        of an error can be expressed as an exponential function of the
        absolute magnitude of the error, which leads to the Laplace
        distribution. For many problems in economics and health
        sciences, this distribution seems to model the data better
        than the standard Gaussian distribution.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
               Mathematical Functions with Formulas, Graphs, and Mathematical
               Tables, 9th printing," New York: Dover, 1972.
        .. [2] Kotz, Samuel, et. al. "The Laplace Distribution and
               Generalizations, " Birkhauser, 2001.
        .. [3] Weisstein, Eric W. "Laplace Distribution."
               From MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/LaplaceDistribution.html
        .. [4] Wikipedia, "Laplace Distribution",
               http://en.wikipedia.org/wiki/Laplace_distribution

        Examples
        --------
        Draw samples from the distribution

        >>> loc, scale = 0., 1.
        >>> s = np.random.laplace(loc, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> x = np.arange(-8., 8., .01)
        >>> pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
        >>> plt.plot(x, pdf)

        Plot Gaussian for comparison:

        >>> g = (1/(scale * np.sqrt(2 * np.pi)) *
        ...      np.exp(-(x - loc)**2 / (2 * scale**2)))
        >>> plt.plot(x,g)

        """
        return cont(&self.rng_state, &random_laplace, size, self.lock, 2,
                    loc, 'loc', CONS_NONE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """
        gumbel(loc=0.0, scale=1.0, size=None)

        Draw samples from a Gumbel distribution.

        Draw samples from a Gumbel distribution with specified location and
        scale.  For more information on the Gumbel distribution, see
        Notes and References below.

        Parameters
        ----------
        loc : float
            The location of the mode of the distribution.
        scale : float
            The scale parameter of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar

        See Also
        --------
        scipy.stats.gumbel_l
        scipy.stats.gumbel_r
        scipy.stats.genextreme
        weibull

        Notes
        -----
        The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
        Value Type I) distribution is one of a class of Generalized Extreme
        Value (GEV) distributions used in modeling extreme value problems.
        The Gumbel is a special case of the Extreme Value Type I distribution
        for maximums from distributions with "exponential-like" tails.

        The probability density for the Gumbel distribution is

        .. math:: p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/
                  \\beta}},

        where :math:`\\mu` is the mode, a location parameter, and
        :math:`\\beta` is the scale parameter.

        The Gumbel (named for German mathematician Emil Julius Gumbel) was used
        very early in the hydrology literature, for modeling the occurrence of
        flood events. It is also used for modeling maximum wind speed and
        rainfall rates.  It is a "fat-tailed" distribution - the probability of
        an event in the tail of the distribution is larger than if one used a
        Gaussian, hence the surprisingly frequent occurrence of 100-year
        floods. Floods were initially modeled as a Gaussian process, which
        underestimated the frequency of extreme events.

        It is one of a class of extreme value distributions, the Generalized
        Extreme Value (GEV) distributions, which also includes the Weibull and
        Frechet.

        The function has a mean of :math:`\\mu + 0.57721\\beta` and a variance
        of :math:`\\frac{\\pi^2}{6}\\beta^2`.

        References
        ----------
        .. [1] Gumbel, E. J., "Statistics of Extremes,"
               New York: Columbia University Press, 1958.
        .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
               Values from Insurance, Finance, Hydrology and Other Fields,"
               Basel: Birkhauser Verlag, 2001.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, beta = 0, 0.1 # location and scale
        >>> s = np.random.gumbel(mu, beta, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp( -np.exp( -(bins - mu) /beta) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Show how an extreme value distribution can arise from a Gaussian process
        and compare to a Gaussian:

        >>> means = []
        >>> maxima = []
        >>> for i in range(0,1000) :
        ...    a = np.random.normal(mu, beta, 1000)
        ...    means.append(a.mean())
        ...    maxima.append(a.max())
        >>> count, bins, ignored = plt.hist(maxima, 30, normed=True)
        >>> beta = np.std(maxima) * np.sqrt(6) / np.pi
        >>> mu = np.mean(maxima) - 0.57721*beta
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp(-np.exp(-(bins - mu)/beta)),
        ...          linewidth=2, color='r')
        >>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi))
        ...          * np.exp(-(bins - mu)**2 / (2 * beta**2)),
        ...          linewidth=2, color='g')
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_gumbel, size, self.lock, 2,
                    loc, 'loc', CONS_NONE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """
        logistic(loc=0.0, scale=1.0, size=None)

        Draw samples from a logistic distribution.

        Samples are drawn from a logistic distribution with specified
        parameters, loc (location or mean, also median), and scale (>0).

        Parameters
        ----------
        loc : float

        scale : float > 0.

        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.logistic : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Logistic distribution is

        .. math:: P(x) = P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2},

        where :math:`\\mu` = location and :math:`s` = scale.

        The Logistic distribution is used in Extreme Value problems where it
        can act as a mixture of Gumbel distributions, in Epidemiology, and by
        the World Chess Federation (FIDE) where it is used in the Elo ranking
        system, assuming the performance of each player is a logistically
        distributed random variable.

        References
        ----------
        .. [1] Reiss, R.-D. and Thomas M. (2001), "Statistical Analysis of
               Extreme Values, from Insurance, Finance, Hydrology and Other
               Fields," Birkhauser Verlag, Basel, pp 132-133.
        .. [2] Weisstein, Eric W. "Logistic Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/LogisticDistribution.html
        .. [3] Wikipedia, "Logistic-distribution",
               http://en.wikipedia.org/wiki/Logistic_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> loc, scale = 10, 1
        >>> s = np.random.logistic(loc, scale, 10000)
        >>> count, bins, ignored = plt.hist(s, bins=50)

        #   plot against distribution

        >>> def logist(x, loc, scale):
        ...     return exp((loc-x)/scale)/(scale*(1+exp((loc-x)/scale))**2)
        >>> plt.plot(bins, logist(bins, loc, scale)*count.max()/\\
        ... logist(bins, loc, scale).max())
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_logistic, size, self.lock, 2,
                    loc, 'loc', CONS_NONE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE)


    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """
        lognormal(mean=0.0, sigma=1.0, size=None)

        Draw samples from a log-normal distribution.

        Draw samples from a log-normal distribution with specified mean,
        standard deviation, and array shape.  Note that the mean and standard
        deviation are not the values for the distribution itself, but of the
        underlying normal distribution it is derived from.

        Parameters
        ----------
        mean : float
            Mean value of the underlying normal distribution
        sigma : float, > 0.
            Standard deviation of the underlying normal distribution
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or float
            The desired samples. An array of the same shape as `size` if given,
            if `size` is None a float is returned.

        See Also
        --------
        scipy.stats.lognorm : probability density function, distribution,
            cumulative density function, etc.

        Notes
        -----
        A variable `x` has a log-normal distribution if `log(x)` is normally
        distributed.  The probability density function for the log-normal
        distribution is:

        .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}}
                         e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard
        deviation of the normally distributed logarithm of the variable.
        A log-normal distribution results if a random variable is the *product*
        of a large number of independent, identically-distributed variables in
        the same way that a normal distribution results if the variable is the
        *sum* of a large number of independent, identically-distributed
        variables.

        References
        ----------
        .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
               Distributions across the Sciences: Keys and Clues,"
               BioScience, Vol. 51, No. 5, May, 2001.
               http://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
        .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
               Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 3., 1. # mean and standard deviation
        >>> s = np.random.lognormal(mu, sigma, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 100, normed=True, align='mid')

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, linewidth=2, color='r')
        >>> plt.axis('tight')
        >>> plt.show()

        Demonstrate that taking the products of random samples from a uniform
        distribution can be fit well by a log-normal probability density
        function.

        >>> # Generate a thousand samples: each is the product of 100 random
        >>> # values, drawn from a normal distribution.
        >>> b = []
        >>> for i in range(1000):
        ...    a = 10. + np.random.random(100)
        ...    b.append(np.product(a))

        >>> b = np.array(b) / np.min(b) # scale values to be positive
        >>> count, bins, ignored = plt.hist(b, 100, normed=True, align='mid')
        >>> sigma = np.std(np.log(b))
        >>> mu = np.mean(np.log(b))

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, color='r', linewidth=2)
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_lognormal, size, self.lock, 2,
                    mean, 'mean', CONS_NONE,
                    sigma, 'sigma', CONS_POSITIVE,
                    0.0, '', CONS_NONE)


    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Draw samples from a Rayleigh distribution.

        The :math:`\\chi` and Weibull distributions are generalizations of the
        Rayleigh.

        Parameters
        ----------
        scale : scalar
            Scale, also equals the mode. Should be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Notes
        -----
        The probability density function for the Rayleigh distribution is

        .. math:: P(x;scale) = \\frac{x}{scale^2}e^{\\frac{-x^2}{2 \\cdotp scale^2}}

        The Rayleigh distribution would arise, for example, if the East
        and North components of the wind velocity had identical zero-mean
        Gaussian distributions.  Then the wind speed would have a Rayleigh
        distribution.

        References
        ----------
        .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
               http://www.brighton-webs.co.uk/distributions/rayleigh.asp
        .. [2] Wikipedia, "Rayleigh distribution"
               http://en.wikipedia.org/wiki/Rayleigh_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> values = hist(np.random.rayleigh(3, 100000), bins=200, normed=True)

        Wave heights tend to follow a Rayleigh distribution. If the mean wave
        height is 1 meter, what fraction of waves are likely to be larger than 3
        meters?

        >>> meanvalue = 1
        >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
        >>> s = np.random.rayleigh(modevalue, 1000000)

        The percentage of waves larger than 3 meters is:

        >>> 100.*sum(s>3)/1000000.
        0.087300000000000003

        """
        return cont(&self.rng_state, &random_rayleigh, size, self.lock, 1,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def wald(self, mean, scale, size=None):
        """
        wald(mean, scale, size=None)

        Draw samples from a Wald, or inverse Gaussian, distribution.

        As the scale approaches infinity, the distribution becomes more like a
        Gaussian. Some references claim that the Wald is an inverse Gaussian
        with mean equal to 1, but this is by no means universal.

        The inverse Gaussian distribution was first studied in relationship to
        Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
        because there is an inverse relationship between the time to cover a
        unit distance and distance covered in unit time.

        Parameters
        ----------
        mean : scalar
            Distribution mean, should be > 0.
        scale : scalar
            Scale parameter, should be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Drawn sample, all greater than zero.

        Notes
        -----
        The probability density function for the Wald distribution is

        .. math:: P(x;mean,scale) = \\sqrt{\\frac{scale}{2\\pi x^3}}e^
                                    \\frac{-scale(x-mean)^2}{2\\cdotp mean^2x}

        As noted above the inverse Gaussian distribution first arise
        from attempts to model Brownian motion. It is also a
        competitor to the Weibull for use in reliability modeling and
        modeling stock returns and interest rate processes.

        References
        ----------
        .. [1] Brighton Webs Ltd., Wald Distribution,
               http://www.brighton-webs.co.uk/distributions/wald.asp
        .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
               Distribution: Theory : Methodology, and Applications", CRC Press,
               1988.
        .. [3] Wikipedia, "Wald distribution"
               http://en.wikipedia.org/wiki/Wald_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(np.random.wald(3, 2, 100000), bins=200, normed=True)
        >>> plt.show()

        """
        return cont(&self.rng_state, &random_wald, size, self.lock, 2,
                    mean, 'mean', CONS_POSITIVE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE)

    def randint(self, low, high=None, size=None):
        """
        randint(low, high=None, size=None)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution in the
        "half-open" interval [`low`, `high`). If `high` is None (the default),
        then results are from [0, `low`).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.random_integers : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is
            omitted. In particular, this other one is the one to use to generate
            uniformly distributed discrete non-integers.

        Examples
        --------
        >>> np.random.randint(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        >>> np.random.randint(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> np.random.randint(5, size=(2, 4))
        array([[4, 0, 2, 1],
               [3, 2, 2, 0]])

        """
        if high is not None and low >= high:
            raise ValueError("low >= high")

        if high is None:
            high = low
            low = 0

        return self.random_integers(low, high - 1, size)


    def negative_binomial(self, n, p, size=None):
        """
        negative_binomial(n, p, size=None)

        Draw samples from a negative binomial distribution.

        Samples are drawn from a negative binomial distribution with specified
        parameters, `n` trials and `p` probability of success where `n` is an
        integer > 0 and `p` is in the interval [0, 1].

        Parameters
        ----------
        n : int
            Parameter, > 0.
        p : float
            Parameter, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : int or ndarray of ints
            Drawn samples.

        Notes
        -----
        The probability density for the negative binomial distribution is

        .. math:: P(N;n,p) = \\binom{N+n-1}{n-1}p^{n}(1-p)^{N},

        where :math:`n-1` is the number of successes, :math:`p` is the
        probability of success, and :math:`N+n-1` is the number of trials.
        The negative binomial distribution gives the probability of n-1
        successes and N failures in N+n-1 trials, and success on the (N+n)th
        trial.

        If one throws a die repeatedly until the third time a "1" appears,
        then the probability distribution of the number of non-"1"s that
        appear before the third "1" is a negative binomial distribution.

        References
        ----------
        .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/NegativeBinomialDistribution.html
        .. [2] Wikipedia, "Negative binomial distribution",
               http://en.wikipedia.org/wiki/Negative_binomial_distribution

        Examples
        --------
        Draw samples from the distribution:

        A real world example. A company drills wild-cat oil
        exploration wells, each with an estimated probability of
        success of 0.1.  What is the probability of having one success
        for each successive well, that is what is the probability of a
        single success after drilling 5 wells, after 6 wells, etc.?

        >>> s = np.random.negative_binomial(1, 0.1, 100000)
        >>> for i in range(1, 11):
        ...    probability = sum(s<i) / 100000.
        ...    print i, "wells drilled, probability of one success =", probability

        """
        return disc(&self.rng_state, &random_negative_binomial, size, self.lock, 2, 0,
                        n, 'n', CONS_POSITIVE,
                        p, 'p', CONS_BOUNDED_0_1,
                        0.0, '', CONS_NONE)



    def random_integers(self, low, high=None, size=None):
        """
        random_integers(low, high=None, size=None)

        Return random integers between `low` and `high`, inclusive.

        Return random integers from the "discrete uniform" distribution in the
        closed interval [`low`, `high`].  If `high` is None (the default),
        then results are from [1, `low`].

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, the largest (signed) integer to be drawn from the
            distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.randint : Similar to `random_integers`, only for the half-open
            interval [`low`, `high`), and 0 is the lowest value if `high` is
            omitted.

        Notes
        -----
        To sample from N evenly spaced floating-point numbers between a and b,
        use::

          a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)

        Examples
        --------
        >>> np.random.random_integers(5)
        4
        >>> type(np.random.random_integers(5))
        <type 'int'>
        >>> np.random.random_integers(5, size=(3.,2.))
        array([[5, 4],
               [3, 3],
               [4, 5]])

        Choose five random numbers from the set of five evenly-spaced
        numbers between 0 and 2.5, inclusive (*i.e.*, from the set
        :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

        >>> 2.5 * (np.random.random_integers(5, size=(5,)) - 1) / 4.
        array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ])

        Roll two six sided dice 1000 times and sum the results:

        >>> d1 = np.random.random_integers(1, 6, 1000)
        >>> d2 = np.random.random_integers(1, 6, 1000)
        >>> dsums = d1 + d2

        Display results as a histogram:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(dsums, 11, normed=True)
        >>> plt.show()

        """
        if high is not None and low > high:
            raise ValueError("low > high")

        cdef long lo, hi, rv
        cdef unsigned long diff
        cdef np.long_t [::1] randoms
        cdef Py_ssize_t n, i

        if high is None:
            lo = 1
            hi = low
        else:
            lo = low
            hi = high

        diff = <unsigned long>hi - <unsigned long>lo
        if size is None:
            with self.lock:
                rv = lo + <long>random_interval(&self.rng_state, diff)
            return rv

        n = compute_numel(size)
        randoms = np.empty(n, np.long)
        with self.lock, nogil:
            for i in range(n):
                randoms[i] = lo + <long>random_interval(&self.rng_state, diff)
        return np.asarray(randoms).reshape(size)


    def logseries(self, p, size=None):
        """
        logseries(p, size=None)

        Draw samples from a logarithmic series distribution.

        Samples are drawn from a log series distribution with specified
        shape parameter, 0 < ``p`` < 1.

        Parameters
        ----------
        loc : float

        scale : float > 0.

        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.logser : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Log Series distribution is

        .. math:: P(k) = \\frac{-p^k}{k \\ln(1-p)},

        where p = probability.

        The log series distribution is frequently used to represent species
        richness and occurrence, first proposed by Fisher, Corbet, and
        Williams in 1943 [2].  It may also be used to model the numbers of
        occupants seen in cars [3].

        References
        ----------
        .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional
               species diversity through the log series distribution of
               occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,
               Volume 5, Number 5, September 1999 , pp. 187-195(9).
        .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The
               relation between the number of species and the number of
               individuals in a random sample of an animal population.
               Journal of Animal Ecology, 12:42-58.
        .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small
               Data Sets, CRC Press, 1994.
        .. [4] Wikipedia, "Logarithmic-distribution",
               http://en.wikipedia.org/wiki/Logarithmic-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = .6
        >>> s = np.random.logseries(a, 10000)
        >>> count, bins, ignored = plt.hist(s)

        #   plot against distribution

        >>> def logseries(k, p):
        ...     return -p**k/(k*log(1-p))
        >>> plt.plot(bins, logseries(bins, a)*count.max()/
                     logseries(bins, a).max(), 'r')
        >>> plt.show()

        """
        return disc(&self.rng_state, &random_logseries, size, self.lock, 1, 0,
                 p, 'p', CONS_BOUNDED_0_1,
                 0.0, '', CONS_NONE,
                 0.0, '', CONS_NONE)


    def geometric(self, p, size=None):
        """
        geometric(p, size=None)

        Draw samples from the geometric distribution.

        Bernoulli trials are experiments with one of two outcomes:
        success or failure (an example of such an experiment is flipping
        a coin).  The geometric distribution models the number of trials
        that must be run in order to achieve success.  It is therefore
        supported on the positive integers, ``k = 1, 2, ...``.

        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1 - p)^{k - 1} p

        where `p` is the probability of success of an individual trial.

        Parameters
        ----------
        p : float
            The probability of success of an individual trial.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Samples from the geometric distribution, shaped according to
            `size`.

        Examples
        --------
        Draw ten thousand values from the geometric distribution,
        with the probability of an individual success equal to 0.35:

        >>> z = np.random.geometric(p=0.35, size=10000)

        How many trials succeeded after a single run?

        >>> (z == 1).sum() / 10000.
        0.34889999999999999 #random

        """

        return disc(&self.rng_state, &random_geometric, size, self.lock, 1, 0,
                        p, 'p', CONS_BOUNDED_0_1,
                        0.0, '', CONS_NONE,
                        0.0, '', CONS_NONE)


    def zipf(self, a, size=None):
        """
        zipf(a, size=None)

        Draw samples from a Zipf distribution.

        Samples are drawn from a Zipf distribution with specified parameter
        `a` > 1.

        The Zipf distribution (also known as the zeta distribution) is a
        continuous probability distribution that satisfies Zipf's law: the
        frequency of an item is inversely proportional to its rank in a
        frequency table.

        Parameters
        ----------
        a : float > 1
            Distribution parameter.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : scalar or ndarray
            The returned samples are greater than or equal to one.

        See Also
        --------
        scipy.stats.distributions.zipf : probability density function,
            distribution, or cumulative density function, etc.

        Notes
        -----
        The probability density for the Zipf distribution is

        .. math:: p(x) = \\frac{x^{-a}}{\\zeta(a)},

        where :math:`\\zeta` is the Riemann Zeta function.

        It is named for the American linguist George Kingsley Zipf, who noted
        that the frequency of any word in a sample of a language is inversely
        proportional to its rank in the frequency table.

        References
        ----------
        .. [1] Zipf, G. K., "Selected Studies of the Principle of Relative
               Frequency in Language," Cambridge, MA: Harvard Univ. Press,
               1932.

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 2. # parameter
        >>> s = np.random.zipf(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        Truncate s values at 50 so plot is interesting
        >>> count, bins, ignored = plt.hist(s[s<50], 50, normed=True)
        >>> x = np.arange(1., 50.)
        >>> y = x**(-a)/sps.zetac(a)
        >>> plt.plot(x, y/max(y), linewidth=2, color='r')
        >>> plt.show()

        """
        return disc(&self.rng_state, &random_zipf, size, self.lock, 1, 0,
                        a, 'a', CONS_GT_1,
                        0.0, '', CONS_NONE,
                        0.0, '', CONS_NONE)


    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution
        for large N.

        Parameters
        ----------
        lam : float or sequence of float
            Expectation of interval, should be >= 0. A sequence of expectation
            intervals must be broadcastable over the requested size.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples, of shape *size*, if it was provided.

        Notes
        -----
        The Poisson distribution

        .. math:: f(k; \\lambda)=\\frac{\\lambda^k e^{-\\lambda}}{k!}

        For events with an expected separation :math:`\\lambda` the Poisson
        distribution :math:`f(k; \\lambda)` describes the probability of
        :math:`k` events occurring within the observed
        interval :math:`\\lambda`.

        Because the output is limited to the range of the C long type, a
        ValueError is raised when `lam` is within 10 sigma of the maximum
        representable value.

        References
        ----------
        .. [1] Weisstein, Eric W. "Poisson Distribution."
               From MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/PoissonDistribution.html
        .. [2] Wikipedia, "Poisson distribution",
               http://en.wikipedia.org/wiki/Poisson_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> import numpy as np
        >>> s = np.random.poisson(5, 10000)

        Display histogram of the sample:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 14, normed=True)
        >>> plt.show()

        Draw each 100 values for lambda 100 and 500:

        >>> s = np.random.poisson(lam=(100., 500.), size=(100, 2))

        """
        return disc(&self.rng_state, &random_poisson, size, self.lock, 1, 0,
                        lam, 'lam', CONS_POISSON,
                        0.0, '', CONS_NONE,
                        0.0, '', CONS_NONE)

    def dirichlet(self, object alpha, size=None):
        """
        dirichlet(alpha, size=None)

        Draw samples from the Dirichlet distribution.

        Draw `size` samples of dimension k from a Dirichlet distribution. A
        Dirichlet-distributed random variable can be seen as a multivariate
        generalization of a Beta distribution. Dirichlet pdf is the conjugate
        prior of a multinomial in Bayesian inference.

        Parameters
        ----------
        alpha : array
            Parameter of the distribution (k dimension for sample of
            dimension k).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray,
            The drawn samples, of shape (size, alpha.ndim).

        Notes
        -----
        .. math:: X \\approx \\prod_{i=1}^{k}{x^{\\alpha_i-1}_i}

        Uses the following property for computation: for each dimension,
        draw a random sample y_i from a standard gamma generator of shape
        `alpha_i`, then
        :math:`X = \\frac{1}{\\sum_{i=1}^k{y_i}} (y_1, \\ldots, y_n)` is
        Dirichlet distributed.

        References
        ----------
        .. [1] David McKay, "Information Theory, Inference and Learning
               Algorithms," chapter 23,
               http://www.inference.phy.cam.ac.uk/mackay/
        .. [2] Wikipedia, "Dirichlet distribution",
               http://en.wikipedia.org/wiki/Dirichlet_distribution

        Examples
        --------
        Taking an example cited in Wikipedia, this distribution can be used if
        one wanted to cut strings (each of initial length 1.0) into K pieces
        with different lengths, where each piece had, on average, a designated
        average length, but allowing some variation in the relative sizes of
        the pieces.

        >>> s = np.random.dirichlet((10, 5, 3), 20).transpose()

        >>> plt.barh(range(20), s[0])
        >>> plt.barh(range(20), s[1], left=s[0], color='g')
        >>> plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
        >>> plt.title("Lengths of Strings")

        """

        #=================
        # Pure python algo
        #=================
        #alpha   = N.atleast_1d(alpha)
        #k       = alpha.size

        #if n == 1:
        #    val = N.zeros(k)
        #    for i in range(k):
        #        val[i]   = sgamma(alpha[i], n)
        #    val /= N.sum(val)
        #else:
        #    val = N.zeros((k, n))
        #    for i in range(k):
        #        val[i]   = sgamma(alpha[i], n)
        #    val /= N.sum(val, axis = 0)
        #    val = val.T

        #return val

        cdef np.npy_intp   k
        cdef np.npy_intp   totsize
        cdef np.ndarray    alpha_arr, val_arr
        cdef double     *alpha_data
        cdef double     *val_data
        cdef np.npy_intp   i, j
        cdef double     acc, invacc

        k           = len(alpha)
        alpha_arr   = <np.ndarray>np.PyArray_FROM_OTF(alpha, np.NPY_DOUBLE, np.NPY_ALIGNED)
        alpha_data  = <double*>np.PyArray_DATA(alpha_arr)

        if size is None:
            shape = (k,)
        else:
            try:
                shape = (operator.index(size), k)
            except:
                shape = tuple(size) + (k,)

        diric   = np.zeros(shape, np.float64)
        val_arr = <np.ndarray>diric
        val_data= <double*>np.PyArray_DATA(val_arr)

        i = 0
        totsize = np.PyArray_SIZE(val_arr)
        with self.lock, nogil:
            while i < totsize:
                acc = 0.0
                for j in range(k):
                    val_data[i+j] = random_standard_gamma(&self.rng_state, alpha_data[j])
                    acc             = acc + val_data[i + j]
                invacc  = 1/acc
                for j in range(k):
                    val_data[i + j]   = val_data[i + j] * invacc
                i = i + k

        return diric

    def multinomial(self, np.npy_intp n, object pvals, size=None):
        """
        multinomial(n, pvals, size=None)

        Draw samples from a multinomial distribution.

        The multinomial distribution is a multivariate generalisation of the
        binomial distribution.  Take an experiment with one of ``p``
        possible outcomes.  An example of such an experiment is throwing a dice,
        where the outcome can be 1 through 6.  Each sample drawn from the
        distribution represents `n` such experiments.  Its values,
        ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
        outcome was ``i``.

        Parameters
        ----------
        n : int
            Number of experiments.
        pvals : sequence of floats, length p
            Probabilities of each of the ``p`` different outcomes.  These
            should sum to 1 (however, the last element is always assumed to
            account for the remaining probability, as long as
            ``sum(pvals[:-1]) <= 1)``.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Examples
        --------
        Throw a dice 20 times:

        >>> np.random.multinomial(20, [1/6.]*6, size=1)
        array([[4, 1, 7, 5, 2, 1]])

        It landed 4 times on 1, once on 2, etc.

        Now, throw the dice 20 times, and 20 times again:

        >>> np.random.multinomial(20, [1/6.]*6, size=2)
        array([[3, 4, 3, 3, 4, 3],
               [2, 4, 3, 4, 0, 7]])

        For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
        we threw 2 times 1, 4 times 2, etc.

        A loaded die is more likely to land on number 6:

        >>> np.random.multinomial(100, [1/7.]*5 + [2/7.])
        array([11, 16, 14, 17, 16, 26])

        The probability inputs should be normalized. As an implementation
        detail, the value of the last entry is ignored and assumed to take
        up any leftover probability mass, but this should not be relied on.
        A biased coin which has twice as much weight on one side as on the
        other should be sampled like so:

        >>> np.random.multinomial(100, [1.0 / 3, 2.0 / 3])  # RIGHT
        array([38, 62])

        not like:

        >>> np.random.multinomial(100, [1.0, 2.0])  # WRONG
        array([100,   0])

        """
        cdef np.npy_intp d
        cdef np.ndarray parr "arrayObject_parr", mnarr "arrayObject_mnarr"
        cdef double *pix
        cdef long *mnix
        cdef np.npy_intp i, j, dn, sz
        cdef double Sum

        d = len(pvals)
        parr = <np.ndarray>np.PyArray_ContiguousFromObject(pvals, np.NPY_DOUBLE, 1, 1)
        pix = <double*>np.PyArray_DATA(parr)

        if kahan_sum(pix, d-1) > (1.0 + 1e-12):
            raise ValueError("sum(pvals[:-1]) > 1.0")

        if size is None:
            shape = (d,)
        else:
            try:
                shape = (operator.index(size), d)
            except:
                shape = tuple(size) + (d,)

        multin = np.zeros(shape, dtype=np.long)
        mnarr = <np.ndarray>multin
        mnix = <long*>np.PyArray_DATA(mnarr)
        sz = np.PyArray_SIZE(mnarr)

        with self.lock, nogil:
            i = 0
            while i < sz:
                Sum = 1.0
                dn = n
                for j in range(d-1):
                    mnix[i+j] = random_binomial(&self.rng_state, pix[j]/Sum, dn)
                    dn = dn - mnix[i+j]
                    if dn <= 0:
                        break
                    Sum = Sum - pix[j]
                if dn > 0:
                    mnix[i+d-1] = dn
                i = i + d

        return multin


    def multivariate_normal(self, mean, cov, size=None):
        """
        multivariate_normal(mean, cov[, size])

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalization of the one-dimensional normal distribution to higher
        dimensions.  Such a distribution is specified by its mean and
        covariance matrix.  These parameters are analogous to the mean
        (average or "center") and variance (standard deviation, or "width,"
        squared) of the one-dimensional normal distribution.

        Parameters
        ----------
        mean : 1-D array_like, of length N
            Mean of the N-dimensional distribution.
        cov : 2-D array_like, of shape (N, N)
            Covariance matrix of the distribution. It must be symmetric and
            positive-semidefinite for proper sampling.
        size : int or tuple of ints, optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
            If no shape is specified, a single (`N`-D) sample is returned.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Notes
        -----
        The mean is a coordinate in N-dimensional space, which represents the
        location where samples are most likely to be generated.  This is
        analogous to the peak of the bell curve for the one-dimensional or
        univariate normal distribution.

        Covariance indicates the level to which two variables vary together.
        From the multivariate normal distribution, we draw N-dimensional
        samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
        "spread").

        Instead of specifying the full covariance matrix, popular
        approximations include:

          - Spherical covariance (*cov* is a multiple of the identity matrix)
          - Diagonal covariance (*cov* has non-negative elements, and only on
            the diagonal)

        This geometrical property can be seen in two dimensions by plotting
        generated data-points:

        >>> mean = [0, 0]
        >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

        Diagonal covariance means that points are oriented along x or y-axis:

        >>> import matplotlib.pyplot as plt
        >>> x, y = np.random.multivariate_normal(mean, cov, 5000).T
        >>> plt.plot(x, y, 'x')
        >>> plt.axis('equal')
        >>> plt.show()

        Note that the covariance matrix must be positive semidefinite (a.k.a.
        nonnegative-definite). Otherwise, the behavior of this method is
        undefined and backwards compatibility is not guaranteed.

        References
        ----------
        .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
               Processes," 3rd ed., New York: McGraw-Hill, 1991.
        .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
               Classification," 2nd ed., New York: Wiley, 2001.

        Examples
        --------
        >>> mean = (1, 2)
        >>> cov = [[1, 0], [0, 1]]
        >>> x = np.random.multivariate_normal(mean, cov, (3, 3))
        >>> x.shape
        (3, 3, 2)

        The following is probably true, given that 0.6 is roughly twice the
        standard deviation:

        >>> list((x[0,0,:] - mean) < 0.6)
        [True, True]

        """
        from numpy.dual import svd

        # Check preconditions on arguments
        mean = np.array(mean)
        cov = np.array(cov)
        if size is None:
            shape = []
        elif isinstance(size, (int, long, np.integer)):
            shape = [size]
        else:
            shape = size

        if len(mean.shape) != 1:
               raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
               raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
               raise ValueError("mean and cov must have same length")

        # Compute shape of output and create a matrix of independent
        # standard normally distributed random numbers. The matrix has rows
        # with the same length as mean and as many rows are necessary to
        # form a matrix of shape final_shape.
        final_shape = tuple(shape[:])
        final_shape += (mean.shape[0],)
        x = self.standard_normal(final_shape).reshape(-1, mean.shape[0])

        # Transform matrix of standard normals into matrix where each row
        # contains multivariate normals with the desired covariance.
        # Compute A such that dot(transpose(A),A) == cov.
        # Then the matrix products of the rows of x and A has the desired
        # covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
        # decomposition of cov is such an A.
        #
        # Also check that cov is positive-semidefinite. If so, the u.T and v
        # matrices should be equal up to roundoff error if cov is
        # symmetrical and the singular value of the corresponding row is
        # not zero. We continue to use the SVD rather than Cholesky in
        # order to preserve current outputs. Note that symmetry has not
        # been checked.
        (u, s, v) = svd(cov)
        neg = (np.sum(u.T * v, axis=1) < 0) & (s > 0)
        if np.any(neg):
            s[neg] = 0.
            import warnings
            warnings.warn("covariance is not positive-semidefinite.",
                          RuntimeWarning)

        x = np.dot(x, np.sqrt(s)[:, None] * v)
        x += mean
        x.shape = tuple(final_shape)
        return x

    def triangular(self, left, mode, right, size=None):
        """
        triangular(left, mode, right, size=None)

        Draw samples from the triangular distribution.

        The triangular distribution is a continuous probability
        distribution with lower limit left, peak at mode, and upper
        limit right. Unlike the other distributions, these parameters
        directly define the shape of the pdf.

        Parameters
        ----------
        left : scalar
            Lower limit.
        mode : scalar
            The value where the peak of the distribution occurs.
            The value should fulfill the condition ``left <= mode <= right``.
        right : scalar
            Upper limit, should be larger than `left`.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The returned samples all lie in the interval [left, right].

        Notes
        -----
        The probability density function for the triangular distribution is

        .. math:: P(x;l, m, r) = \\begin{cases}
                  \\frac{2(x-l)}{(r-l)(m-l)}& \\text{for $l \\leq x \\leq m$},\\\\
                  \\frac{2(r-x)}{(r-l)(r-m)}& \\text{for $m \\leq x \\leq r$},\\\\
                  0& \\text{otherwise}.
                  \\end{cases}

        The triangular distribution is often used in ill-defined
        problems where the underlying distribution is not known, but
        some knowledge of the limits and mode exists. Often it is used
        in simulations.

        References
        ----------
        .. [1] Wikipedia, "Triangular distribution"
               http://en.wikipedia.org/wiki/Triangular_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(np.random.triangular(-3, 0, 8, 100000), bins=200,
        ...              normed=True)
        >>> plt.show()

        """
        cdef bint is_scalar = True
        cdef double fleft, fmode, fright
        cdef np.ndarray oleft, omode, oright

        fleft = PyFloat_AsDouble(left)
        fright = PyFloat_AsDouble(right)
        fmode = PyFloat_AsDouble(mode)
        if fleft == -1.0 or fright == -1.0 or fmode == -1.0:
            if PyErr_Occurred():
                PyErr_Clear()
                is_scalar = False

        if is_scalar:
            if fleft > fmode:
                raise ValueError("left > mode")
            if fmode > fright:
                raise ValueError("mode > right")
            if fleft == fright:
                raise ValueError("left == right")
            return cont(&self.rng_state, &random_triangular, size, self.lock, 3,
                        fleft, '', CONS_NONE,
                        fmode, '', CONS_NONE,
                        fright, '', CONS_NONE)

        oleft = <np.ndarray>np.PyArray_FROM_OTF(left, np.NPY_DOUBLE, np.NPY_ALIGNED)
        omode = <np.ndarray>np.PyArray_FROM_OTF(mode, np.NPY_DOUBLE, np.NPY_ALIGNED)
        oright = <np.ndarray>np.PyArray_FROM_OTF(right, np.NPY_DOUBLE, np.NPY_ALIGNED)

        if np.any(np.greater(oleft, omode)):
            raise ValueError("left > mode")
        if np.any(np.greater(omode, oright)):
            raise ValueError("mode > right")
        if np.any(np.equal(oleft, oright)):
            raise ValueError("left == right")

        return cont_broadcast_3(&self.rng_state, &random_triangular, size, self.lock,
                            oleft, '', CONS_NONE,
                            omode, '', CONS_NONE,
                            oright, '', CONS_NONE)


    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """
        hypergeometric(ngood, nbad, nsample, size=None)

        Draw samples from a Hypergeometric distribution.

        Samples are drawn from a hypergeometric distribution with specified
        parameters, ngood (ways to make a good selection), nbad (ways to make
        a bad selection), and nsample = number of items sampled, which is less
        than or equal to the sum ngood + nbad.

        Parameters
        ----------
        ngood : int or array_like
            Number of ways to make a good selection.  Must be nonnegative.
        nbad : int or array_like
            Number of ways to make a bad selection.  Must be nonnegative.
        nsample : int or array_like
            Number of items sampled.  Must be at least 1 and at most
            ``ngood + nbad``.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.hypergeom : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Hypergeometric distribution is

        .. math:: P(x) = \\frac{\\binom{m}{n}\\binom{N-m}{n-x}}{\\binom{N}{n}},

        where :math:`0 \\le x \\le m` and :math:`n+m-N \\le x \\le n`

        for P(x) the probability of x successes, n = ngood, m = nbad, and
        N = number of samples.

        Consider an urn with black and white marbles in it, ngood of them
        black and nbad are white. If you draw nsample balls without
        replacement, then the hypergeometric distribution describes the
        distribution of black balls in the drawn sample.

        Note that this distribution is very similar to the binomial
        distribution, except that in this case, samples are drawn without
        replacement, whereas in the Binomial case samples are drawn with
        replacement (or the sample space is infinite). As the sample space
        becomes large, this distribution approaches the binomial.

        References
        ----------
        .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/HypergeometricDistribution.html
        .. [3] Wikipedia, "Hypergeometric-distribution",
               http://en.wikipedia.org/wiki/Hypergeometric_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> ngood, nbad, nsamp = 100, 2, 10
        # number of good, number of bad, and number of samples
        >>> s = np.random.hypergeometric(ngood, nbad, nsamp, 1000)
        >>> hist(s)
        #   note that it is very unlikely to grab both bad items

        Suppose you have an urn with 15 white and 15 black marbles.
        If you pull 15 marbles at random, how likely is it that
        12 or more of them are one color?

        >>> s = np.random.hypergeometric(15, 15, 15, 100000)
        >>> sum(s>=12)/100000. + sum(s<=3)/100000.
        #   answer = 0.003 ... pretty unlikely!

        """
        cdef bint is_scalar = True
        cdef np.ndarray ongood, onbad, onsample
        cdef long lngood, lnbad, lnsample

        lngood = PyInt_AsLong(ngood)
        lnbad = PyInt_AsLong(nbad)
        lnsample = PyInt_AsLong(nsample)
        if lngood == -1 or lnbad == -1 or lnsample == -1:
            if PyErr_Occurred():
                PyErr_Clear()
                is_scalar = False

        if is_scalar:
            if lngood < 0:
                raise ValueError("ngood < 0")
            if lnbad < 0:
                raise ValueError("nbad < 0")
            if lnsample < 1:
                raise ValueError("nsample < 1")
            if lngood + lnbad < lnsample:
                raise ValueError("ngood + nbad < nsample")
            return disc(&self.rng_state, &random_hypergeometric, size, self.lock, 0, 3,
                        lngood, 'ngood', CONS_NON_NEGATIVE,
                        lnbad, 'nbad', CONS_NON_NEGATIVE,
                        lnsample, 'nsample', CONS_GTE_1)

        PyErr_Clear()

        ongood = <np.ndarray>np.PyArray_FROM_OTF(ngood, np.NPY_LONG, np.NPY_ALIGNED)
        onbad = <np.ndarray>np.PyArray_FROM_OTF(nbad, np.NPY_LONG, np.NPY_ALIGNED)
        onsample = <np.ndarray>np.PyArray_FROM_OTF(nsample, np.NPY_LONG, np.NPY_ALIGNED)

        if np.any(np.less(np.add(ongood, onbad),onsample)):
            raise ValueError("ngood + nbad < nsample")
        return discrete_broadcast_iii(&self.rng_state, &random_hypergeometric, size, self.lock,
                                      ongood, 'ngood', CONS_NON_NEGATIVE,
                                      onbad, nbad, CONS_NON_NEGATIVE,
                                      onsample, 'nsample', CONS_GTE_1)


    @cython.wraparound(True)
    def choice(self, a, size=None, replace=True, p=None):
        """
        choice(a, size=None, replace=True, p=None)

        Generates a random sample from a given 1-D array

                .. versionadded:: 1.7.0

        Parameters
        -----------
        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a was np.arange(n)
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        replace : boolean, optional
            Whether the sample is with or without replacement
        p : 1-D array-like, optional
            The probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all
            entries in a.

        Returns
        --------
        samples : 1-D ndarray, shape (size,)
            The generated random samples

        Raises
        -------
        ValueError
            If a is an int and less than zero, if a or p are not 1-dimensional,
            if a is an array-like of size 0, if p is not a vector of
            probabilities, if a and p have different lengths, or if
            replace=False and the sample size is greater than the population
            size

        See Also
        ---------
        randint, shuffle, permutation

        Examples
        ---------
        Generate a uniform random sample from np.arange(5) of size 3:

        >>> np.random.choice(5, 3)
        array([0, 3, 4])
        >>> #This is equivalent to np.random.randint(0,5,3)

        Generate a non-uniform random sample from np.arange(5) of size 3:

        >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        array([3, 3, 0])

        Generate a uniform random sample from np.arange(5) of size 3 without
        replacement:

        >>> np.random.choice(5, 3, replace=False)
        array([3,1,0])
        >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

        Generate a non-uniform random sample from np.arange(5) of size
        3 without replacement:

        >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        array([2, 3, 0])

        Any of the above can be repeated with an arbitrary array-like
        instead of just integers. For instance:

        >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
        >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
        array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
              dtype='|S11')

        """

        # Format and Verify input
        a = np.array(a, copy=False)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError:
                raise ValueError("a must be 1-dimensional or an integer")
            if pop_size <= 0:
                raise ValueError("a must be greater than 0")
        elif a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        else:
            pop_size = a.shape[0]
            if pop_size is 0:
                raise ValueError("a must be non-empty")

        if p is not None:
            d = len(p)

            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.floating):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))

            p = <np.ndarray>np.PyArray_FROM_OTF(p, np.NPY_DOUBLE, np.NPY_ALIGNED)
            pix = <double*>np.PyArray_DATA(p)

            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("a and p must have same size")
            if np.logical_or.reduce(p < 0):
                raise ValueError("probabilities are not non-negative")
            if abs(kahan_sum(pix, d) - 1.) > atol:
                raise ValueError("probabilities do not sum to 1")

        shape = size
        if shape is not None:
            size = np.prod(shape, dtype=np.intp)
        else:
            size = 1

        # Actual sampling
        if replace:
            if p is not None:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = self.random_sample(shape)
                idx = cdf.searchsorted(uniform_samples, side='right')
                idx = np.array(idx, copy=False) # searchsorted returns a scalar
            else:
                idx = self.randint(0, pop_size, size=shape)
        else:
            if size > pop_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when 'replace=False'")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=np.int)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.rand(size - n_uniq)
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side='right')
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq:n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                idx = self.permutation(pop_size)[:size]
                if shape is not None:
                    idx.shape = shape

        if shape is None and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        #Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if shape is not None and idx.ndim == 0:
            # If size == () then the user requested a 0-d array as opposed to
            # a scalar object when size is None. However a[idx] is always a
            # scalar and not an array. So this makes sure the result is an
            # array, taking into account that np.array(item) may not work
            # for object arrays.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        return a[idx]


_rand = RandomState()
seed = _rand.seed
get_state = _rand.get_state
set_state = _rand.set_state
random_sample = _rand.random_sample
choice = _rand.choice
randint = _rand.randint
bytes = _rand.bytes
uniform = _rand.uniform
rand = _rand.rand
randn = _rand.randn
random_integers = _rand.random_integers
standard_normal = _rand.standard_normal
normal = _rand.normal
beta = _rand.beta
exponential = _rand.exponential
standard_exponential = _rand.standard_exponential
standard_gamma = _rand.standard_gamma
gamma = _rand.gamma
f = _rand.f
noncentral_f = _rand.noncentral_f
chisquare = _rand.chisquare
noncentral_chisquare = _rand.noncentral_chisquare
standard_cauchy = _rand.standard_cauchy
standard_t = _rand.standard_t
vonmises = _rand.vonmises
pareto = _rand.pareto
weibull = _rand.weibull
power = _rand.power
laplace = _rand.laplace
gumbel = _rand.gumbel
logistic = _rand.logistic
lognormal = _rand.lognormal
rayleigh = _rand.rayleigh
wald = _rand.wald
triangular = _rand.triangular

binomial = _rand.binomial
negative_binomial = _rand.negative_binomial
poisson = _rand.poisson
zipf = _rand.zipf
geometric = _rand.geometric
hypergeometric = _rand.hypergeometric
logseries = _rand.logseries

multivariate_normal = _rand.multivariate_normal
multinomial = _rand.multinomial
dirichlet = _rand.dirichlet

shuffle = _rand.shuffle
permutation = _rand.permutation