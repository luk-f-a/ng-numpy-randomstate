import os
import sys
from os.path import join
from unittest import TestCase

import numpy as np
import randomstate.mlfg_1279_861 as mlfg_1279_861
import randomstate.mrg32k3a as mrg32k3a
import randomstate.mt19937 as mt19937
import randomstate.pcg32 as pcg32
import randomstate.pcg64 as pcg64
import randomstate.xorshift1024 as xorshift1024
import randomstate.xorshift128 as xorshift128
from numpy.testing import assert_equal, assert_allclose

if (sys.version_info > (3, 0)):
    long = int

pwd = os.path.dirname(os.path.abspath(__file__))


def uniform_from_uint(x, bits):
    if bits == 64:
        return uniform_from_uint64(x)
    elif bits == 32:
        return uniform_from_uint32(x)
    elif bits == 31:
        return uniform_from_uint32(x)


def uniform_from_uint64(x):
    a = x >> 37
    b = (x & 0xFFFFFFFF) >> 6
    return (a * 67108864.0 + b) / 9007199254740992.0


def uniform_from_uint32(x):
    out = np.empty(len(x) // 2)
    for i in range(0, len(x), 2):
        a = x[i] >> 5
        b = x[i + 1] >> 6
        out[i // 2] = (a * 67108864.0 + b) / 9007199254740992.0
    return out


def uint32_from_uint31(x):
    out = np.empty(len(x) // 2, dtype=np.uint32)
    for i in range(0, len(x), 2):
        a = x[i] & 0xffff0000
        b = x[i + 1] >> 16
        out[i // 2] = a | b
    return out


def gauss_from_uint(x, n, bits):
    if bits == 64:
        doubles = uniform_from_uint64(x)
    else:
        doubles = uniform_from_uint32(x)
    gauss = []
    loc = 0
    while len(gauss) < n:
        r2 = 2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * doubles[loc] - 1.0;
            x2 = 2.0 * doubles[loc + 1] - 1.0;
            r2 = x1 * x1 + x2 * x2;
            loc += 2

        f = np.sqrt(-2.0 * np.log(r2) / r2);
        gauss.append(f * x2)
        gauss.append(f * x1)

    return gauss[:n]


class Base(object):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = xorshift128.RandomState
        cls.bits = 64
        cls.dtype = np.uint64

    @classmethod
    def _read_csv(cls, filename):
        with open(filename) as csv:
            seed = csv.readline()
            seed = seed.split(',')
            seed = [long(s) for s in seed[1:]]
            data = []
            for line in csv:
                data.append(long(line.split(',')[-1]))
            return {'seed': seed, 'data': np.array(data, dtype=cls.dtype)}

    def test_raw(self):
        rs = self.RandomState(*self.data1['seed'])
        uints = rs.random_uintegers(1000, bits=self.bits)
        assert_equal(uints, self.data1['data'])

        rs = self.RandomState(*self.data2['seed'])
        uints = rs.random_uintegers(1000, bits=self.bits)
        assert_equal(uints, self.data2['data'])

    def test_double(self):
        rs = self.RandomState(*self.data1['seed'])
        vals = uniform_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random_sample(len(vals))
        assert_allclose(uniforms, vals)

        rs = self.RandomState(*self.data2['seed'])
        vals = uniform_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random_sample(len(vals))
        assert_allclose(uniforms, vals)

    def test_gauss_inv(self):
        n = 25
        rs = self.RandomState(*self.data1['seed'])
        gauss = rs.standard_normal(n, method='inv')
        assert_allclose(gauss,
                        gauss_from_uint(self.data1['data'], n, self.bits))

        rs = self.RandomState(*self.data2['seed'])
        gauss = rs.standard_normal(25, method='inv')
        assert_allclose(gauss,
                        gauss_from_uint(self.data2['data'], n, self.bits))


class TestXorshift128(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = xorshift128.RandomState
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/xorshift128-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/xorshift128-testset-2.csv'))


class TestXorshift1024(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = xorshift1024.RandomState
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/xorshift1024-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/xorshift1024-testset-2.csv'))


class TestMT19937(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = mt19937.RandomState
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/randomkit-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/randomkit-testset-2.csv'))

class TestPCG32(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = pcg32.RandomState
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/pcg32-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg32-testset-2.csv'))

class TestPCG64(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = pcg64.RandomState
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64-testset-2.csv'))



class TestMRG32K3A(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = mrg32k3a.RandomState
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/mrg32k3a-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/mrg32k3a-testset-2.csv'))


class TestMLFG(Base, TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = mlfg_1279_861.RandomState
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/mlfg-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/mlfg-testset-2.csv'))

    def test_raw(self):
        rs = self.RandomState(*self.data1['seed'])
        vals = uint32_from_uint31(self.data1['data'])
        uints = rs.random_uintegers(len(vals), bits=self.bits)
        assert_equal(uints, vals)

        rs = self.RandomState(*self.data2['seed'])
        vals = uint32_from_uint31(self.data2['data'])
        uints = rs.random_uintegers(len(vals), bits=self.bits)
        assert_equal(uints, vals)