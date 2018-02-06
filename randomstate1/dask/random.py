from __future__ import absolute_import, division, print_function

from itertools import product
from numbers import Integral
from operator import getitem

import numpy as np

from dask.array.core import (normalize_chunks, Array, slices_from_chunks, asarray,
                   broadcast_shapes, broadcast_to)
from dask import sharedict
from dask.base import tokenize
from dask.utils import ignoring, random_state_data

from randomstate.prng import xoroshiro128plus

Random_State_class = xoroshiro128plus.RandomState

def doc_wraps(func):
    """ Copy docstring from one function to another """
    def _(func2):
        if func.__doc__ is not None:
            func2.__doc__ = func.__doc__.replace('>>>', '>>').replace('...', '..')
        return func2
    return _


class RandomState(object):
    """
    Mersenne Twister pseudo-random number generator

    This object contains state to deterministically generate pseudo-random
    numbers from a variety of probability distributions.  It is identical to
    ``Random_State_class`` except that all functions also take a ``chunks=``
    keyword argument.

    Examples
    --------

    >>> import dask.array as da
    >>> state = da.random.RandomState(1234)  # a seed
    >>> x = state.normal(10, 0.1, size=3, chunks=(2,))
    >>> x.compute()
    array([ 10.01867852,  10.04812289,   9.89649746])

    See Also:
        Random_State_class
    """
    def __init__(self, seed=None):
        self._numpy_state = Random_State_class(seed)

    def seed(self, seed=None):
        self._numpy_state.seed(seed)

    def _wrap(self, func, *args, **kwargs):
        """ Wrap numpy random function to produce dask.array random function

        extra_chunks should be a chunks tuple to append to the end of chunks
        """
        size = kwargs.pop('size', None)
        chunks = kwargs.pop('chunks')
        extra_chunks = kwargs.pop('extra_chunks', ())

        if size is not None and not isinstance(size, (tuple, list)):
            size = (size,)

        args_shapes = {ar.shape for ar in args
                       if isinstance(ar, (Array, np.ndarray))}
        args_shapes.union({ar.shape for ar in kwargs.values()
                           if isinstance(ar, (Array, np.ndarray))})

        shapes = list(args_shapes)
        if size is not None:
            shapes += [size]
        # broadcast to the final size(shape)
        size = broadcast_shapes(*shapes)
        chunks = normalize_chunks(chunks, size)
        slices = slices_from_chunks(chunks)

        def _broadcast_any(ar, shape, chunks):
            if isinstance(ar, Array):
                return broadcast_to(ar, shape).rechunk(chunks)
            if isinstance(ar, np.ndarray):
                return np.ascontiguousarray(np.broadcast_to(ar, shape))

        # Broadcast all arguments, get tiny versions as well
        # Start adding the relevant bits to the graph
        dsk = {}
        dsks = []
        lookup = {}
        small_args = []
        for i, ar in enumerate(args):
            if isinstance(ar, (np.ndarray, Array)):
                res = _broadcast_any(ar, size, chunks)
                if isinstance(res, Array):
                    dsks.append(res.dask)
                    lookup[i] = res.name
                elif isinstance(res, np.ndarray):
                    name = 'array-{}'.format(tokenize(res))
                    lookup[i] = name
                    dsk[name] = res
                small_args.append(ar[tuple(0 for _ in ar.shape)])
            else:
                small_args.append(ar)

        small_kwargs = {}
        for key, ar in kwargs.items():
            if isinstance(ar, (np.ndarray, Array)):
                res = _broadcast_any(ar, size, chunks)
                if isinstance(res, Array):
                    dsks.append(res.dask)
                    lookup[key] = res.name
                elif isinstance(res, np.ndarray):
                    name = 'array-{}'.format(tokenize(res))
                    lookup[key] = name
                    dsk[name] = res
                small_kwargs[key] = ar[tuple(0 for _ in ar.shape)]
            else:
                small_kwargs[key] = ar

        # Get dtype
        small_kwargs['size'] = (0,)
        dtype = func(xoroshiro128plus.RandomState(), *small_args,
                     **small_kwargs).dtype

        sizes = list(product(*chunks))
        state_data = random_state_data(len(sizes), self._numpy_state)
        token = tokenize(state_data, size, chunks, args, kwargs)
        name = 'da.random.{0}-{1}'.format(func.__name__, token)

        keys = product([name], *([range(len(bd)) for bd in chunks] +
                                 [[0]] * len(extra_chunks)))
        blocks = product(*[range(len(bd)) for bd in chunks])
        vals = []
        for state, size, slc, block in zip(state_data, sizes, slices, blocks):
            arg = []
            for i, ar in enumerate(args):
                if i not in lookup:
                    arg.append(ar)
                else:
                    if isinstance(ar, Array):
                        arg.append((lookup[i], ) + block)
                    else:   # np.ndarray
                        arg.append((getitem, lookup[i], slc))
            kwrg = {}
            for k, ar in kwargs.items():
                if k not in lookup:
                    kwrg[k] = ar
                else:
                    if isinstance(ar, Array):
                        kwrg[k] = (lookup[k], ) + block
                    else:   # np.ndarray
                        kwrg[k] = (getitem, lookup[k], slc)
            vals.append((_apply_random, func.__name__, state, size, arg, kwrg))
        dsk.update(dict(zip(keys, vals)))
        dsk = sharedict.merge((name, dsk), *dsks)
        return Array(dsk, name, chunks + extra_chunks, dtype=dtype)

    @doc_wraps(Random_State_class.beta)
    def beta(self, a, b, size=None, chunks=None):
        return self._wrap(Random_State_class.beta, a, b,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.binomial)
    def binomial(self, n, p, size=None, chunks=None):
        return self._wrap(Random_State_class.binomial, n, p,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.chisquare)
    def chisquare(self, df, size=None, chunks=None):
        return self._wrap(Random_State_class.chisquare, df,
                          size=size, chunks=chunks)

    with ignoring(AttributeError):
        @doc_wraps(Random_State_class.choice)
        def choice(self, a, size=None, replace=True, p=None, chunks=None):
            dsks = []
            # Normalize and validate `a`
            if isinstance(a, Integral):
                # On windows the output dtype differs if p is provided or
                # absent, see https://github.com/numpy/numpy/issues/9867
                dummy_p = np.array([1]) if p is not None else p
                dtype = np.random.choice(1, size=(), p=dummy_p).dtype
                len_a = a
                if a < 0:
                    raise ValueError("a must be greater than 0")
            else:
                a = asarray(a).rechunk(a.shape)
                dtype = a.dtype
                if a.ndim != 1:
                    raise ValueError("a must be one dimensional")
                len_a = len(a)
                dsks.append(a.dask)
                a = a.__dask_keys__()[0]

            # Normalize and validate `p`
            if p is not None:
                if not isinstance(p, Array):
                    # If p is not a dask array, first check the sum is close
                    # to 1 before converting.
                    p = np.asarray(p)
                    if not np.isclose(p.sum(), 1, rtol=1e-7, atol=0):
                        raise ValueError("probabilities do not sum to 1")
                    p = asarray(p)
                else:
                    p = p.rechunk(p.shape)

                if p.ndim != 1:
                    raise ValueError("p must be one dimensional")
                if len(p) != len_a:
                    raise ValueError("a and p must have the same size")

                dsks.append(p.dask)
                p = p.__dask_keys__()[0]

            if size is None:
                size = ()
            elif not isinstance(size, (tuple, list)):
                size = (size,)

            chunks = normalize_chunks(chunks, size)
            sizes = list(product(*chunks))
            state_data = random_state_data(len(sizes), self._numpy_state)

            name = 'da.random.choice-%s' % tokenize(state_data, size, chunks,
                                                    a, replace, p)
            keys = product([name], *(range(len(bd)) for bd in chunks))
            dsk = {k: (_choice, state, a, size, replace, p) for
                   k, state, size in zip(keys, state_data, sizes)}

            return Array(sharedict.merge((name, dsk), *dsks),
                         name, chunks, dtype=dtype)

    # @doc_wraps(Random_State_class.dirichlet)
    # def dirichlet(self, alpha, size=None, chunks=None):

    @doc_wraps(Random_State_class.exponential)
    def exponential(self, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.exponential, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.f)
    def f(self, dfnum, dfden, size=None, chunks=None):
        return self._wrap(Random_State_class.f, dfnum, dfden,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.gamma)
    def gamma(self, shape, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.gamma, shape, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.geometric)
    def geometric(self, p, size=None, chunks=None):
        return self._wrap(Random_State_class.geometric, p,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.gumbel)
    def gumbel(self, loc=0.0, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.gumbel, loc, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.hypergeometric)
    def hypergeometric(self, ngood, nbad, nsample, size=None, chunks=None):
        return self._wrap(Random_State_class.hypergeometric,
                          ngood, nbad, nsample,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.laplace)
    def laplace(self, loc=0.0, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.laplace, loc, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.logistic)
    def logistic(self, loc=0.0, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.logistic, loc, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.lognormal)
    def lognormal(self, mean=0.0, sigma=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.lognormal, mean, sigma,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.logseries)
    def logseries(self, p, size=None, chunks=None):
        return self._wrap(Random_State_class.logseries, p,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.multinomial)
    def multinomial(self, n, pvals, size=None, chunks=None):
        return self._wrap(Random_State_class.multinomial, n, pvals,
                          size=size, chunks=chunks,
                          extra_chunks=((len(pvals),),))

    @doc_wraps(Random_State_class.negative_binomial)
    def negative_binomial(self, n, p, size=None, chunks=None):
        return self._wrap(Random_State_class.negative_binomial, n, p,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.noncentral_chisquare)
    def noncentral_chisquare(self, df, nonc, size=None, chunks=None):
        return self._wrap(Random_State_class.noncentral_chisquare, df, nonc,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.noncentral_f)
    def noncentral_f(self, dfnum, dfden, nonc,  size=None, chunks=None):
        return self._wrap(Random_State_class.noncentral_f,
                          dfnum, dfden, nonc,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.normal)
    def normal(self, loc=0.0, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.normal, loc, scale,
                          size=size, chunks=chunks, method='zig')

    @doc_wraps(Random_State_class.pareto)
    def pareto(self, a, size=None, chunks=None):
        return self._wrap(Random_State_class.pareto, a,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.poisson)
    def poisson(self, lam=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.poisson, lam,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.power)
    def power(self, a, size=None, chunks=None):
        return self._wrap(Random_State_class.power, a,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.randint)
    def randint(self, low, high=None, size=None, chunks=None):
        return self._wrap(Random_State_class.randint, low, high,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.random_integers)
    def random_integers(self, low, high=None, size=None, chunks=None):
        return self._wrap(Random_State_class.random_integers, low, high,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.random_sample)
    def random_sample(self, size=None, chunks=None):
        return self._wrap(Random_State_class.random_sample,
                          size=size, chunks=chunks)

    random = random_sample

    @doc_wraps(Random_State_class.rayleigh)
    def rayleigh(self, scale=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.rayleigh, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.standard_cauchy)
    def standard_cauchy(self, size=None, chunks=None):
        return self._wrap(Random_State_class.standard_cauchy,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.standard_exponential)
    def standard_exponential(self, size=None, chunks=None):
        return self._wrap(Random_State_class.standard_exponential,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.standard_gamma)
    def standard_gamma(self, shape, size=None, chunks=None):
        return self._wrap(Random_State_class.standard_gamma, shape,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.standard_normal)
    def standard_normal(self, size=None, chunks=None):
        return self._wrap(Random_State_class.standard_normal,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.standard_t)
    def standard_t(self, df, size=None, chunks=None):
        return self._wrap(Random_State_class.standard_t, df,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.tomaxint)
    def tomaxint(self, size=None, chunks=None):
        return self._wrap(Random_State_class.tomaxint,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.triangular)
    def triangular(self, left, mode, right, size=None, chunks=None):
        return self._wrap(Random_State_class.triangular, left, mode, right,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.uniform)
    def uniform(self, low=0.0, high=1.0, size=None, chunks=None):
        return self._wrap(Random_State_class.uniform, low, high,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.vonmises)
    def vonmises(self, mu, kappa, size=None, chunks=None):
        return self._wrap(Random_State_class.vonmises, mu, kappa,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.wald)
    def wald(self, mean, scale, size=None, chunks=None):
        return self._wrap(Random_State_class.wald, mean, scale,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.weibull)
    def weibull(self, a, size=None, chunks=None):
        return self._wrap(Random_State_class.weibull, a,
                          size=size, chunks=chunks)

    @doc_wraps(Random_State_class.zipf)
    def zipf(self, a, size=None, chunks=None):
        return self._wrap(Random_State_class.zipf, a,
                          size=size, chunks=chunks)


def _choice(state_data, a, size, replace, p):
    state = Random_State_class(state_data)
    return state.choice(a, size=size, replace=replace, p=p)


def _apply_random(func, state_data, size, args, kwargs):
    """Apply RandomState method with seed"""
    state = Random_State_class(state_data)
    func = getattr(state, func)
    return func(*args, size=size, **kwargs)


_state = RandomState()


seed = _state.seed


beta = _state.beta
binomial = _state.binomial
chisquare = _state.chisquare
if hasattr(_state, 'choice'):
    choice = _state.choice
exponential = _state.exponential
f = _state.f
gamma = _state.gamma
geometric = _state.geometric
gumbel = _state.gumbel
hypergeometric = _state.hypergeometric
laplace = _state.laplace
logistic = _state.logistic
lognormal = _state.lognormal
logseries = _state.logseries
multinomial = _state.multinomial
negative_binomial = _state.negative_binomial
noncentral_chisquare = _state.noncentral_chisquare
noncentral_f = _state.noncentral_f
normal = _state.normal
pareto = _state.pareto
poisson = _state.poisson
power = _state.power
rayleigh = _state.rayleigh
random_sample = _state.random_sample
random = random_sample
randint = _state.randint
random_integers = _state.random_integers
triangular = _state.triangular
uniform = _state.uniform
vonmises = _state.vonmises
wald = _state.wald
weibull = _state.weibull
zipf = _state.zipf

"""
Standard distributions
"""

standard_cauchy = _state.standard_cauchy
standard_exponential = _state.standard_exponential
standard_gamma = _state.standard_gamma
standard_normal = _state.standard_normal
standard_t = _state.standard_t
