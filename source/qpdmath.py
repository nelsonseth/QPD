'''
Base calculations for quantile paramaterized distributions (qpd).

The math for the Johnson system qpd's are based on the phd dissertation work 
by Christopher Hadlock (2017). 
 
References:
- [1] Hadlock, C.C., J.E. Bickel. 2017. Johnson Quantile-Parameterized 
    Distributions. Decision Analysis 14(1) 35-64.
- [2] Hadlock, C.C. 2017. Quantile-Parameterized Methods for Quantifying 
    Uncertainty in Decision Analysis. Ph.D. dissertation, University of Texas.
'''
# MIT License

# Copyright (c) 2023 Seth M. Nelson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


__author__ = ('Seth M. Nelson <github.com/nelsonseth>')

__all__ = [
    'JQPDBounded',
    'JQPDSemiBounded',
]

# imports

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
from builtins import Exception

import numpy as np
from numpy import sqrt, exp, log, arccosh, sinh, arcsinh, sign, cosh
import scipy.stats as stat
from numpy.typing import ArrayLike

_default_size = 200


# Normal unit functions used to build below calculations.
# Unit normal cdf.
def _ncdf(x: ArrayLike) -> ArrayLike: 
    '''
    Unit normal cdf. Equivalent to and uses `scipy.stats.norm.cdf(x)`.

    Math: ``1/2 + 1/2 * erf(x / sqrt(2))``

    NOTE: The scipy default loc and scale parameters (0,1) are already the unit norm. 
    '''
    return stat.norm.cdf(x)

# Unit normal pdf
def _npdf(x: ArrayLike) -> ArrayLike: 
    '''
    Unit normal pdf. Equivalent to and uses `scipy.stats.norm.pdf(x)`.
    
    Math: ``1 / sqrt(2*pi) * exp(-1/2 * x**2)``

    NOTE: The scipy default loc and scale parameters (0,1) are already the unit norm. 
    '''
    return stat.norm.pdf(x)

# Unit normal inverse cdf (quantile function).
def _nppf(q: ArrayLike) -> ArrayLike:
    '''
    Unit normal inverse cdf. Equivalent to and uses `scipy.stats.norm.ppf(q)`.
    
    Math: ``sqrt(2) * erfinv(2*q - 1)``

    NOTE: The scipy default loc and scale parameters (0,1) are already the unit norm. 
    '''
    return stat.norm.ppf(q)


class InputError(Exception):
    '''Error on user inputs.'''

@dataclass
class _BaseInputs:

    '''Dataclass for _QPD core inputs.
    
    Provides routine logic checks of core inputs while _QPD subclasses are active.
    '''
    x: list[float]
    q: list[float]
    bounds: list[float, None]

    # auto check inputs after first initialization.
    # active instances will need to call check_args() if needed.
    def __post_init__(self):
        self.check_args()

    def check_args(
            self,
            new_x: Union[list, None] = None,
            new_q: Union[list, None] = None,
            new_bounds: Union[list, None] = None,
            ):
        
        # allows for check of new inputs without updating the current self._* just yet
        if new_x:
            _x = new_x
        else:
            _x = self.x

        if new_q:
            _q = new_q
        else:
            _q = self.q

        if new_bounds:
            _bounds = new_bounds
        else:
            _bounds = self.bounds

        # x must be strictly increasing
        if not all(a < b for a,b in zip(_x, _x[1:])):
            raise InputError('x values must be strictly increasing.')
        
        # q must be strictly increasing
        if not all(a < b for a,b in zip(_q, _q[1:])):
            raise InputError('q values must be strictly increasing.')
        
        # q must be between 0 and 1
        # TODO: 0,1 inclusive vs exclusive
        if _q[0] <= 0 or _q[-1] >= 1:
            raise InputError('q values must be between 0 and 1.')

        if len(_x) != len(_q):
            raise InputError('Different length of inputs for x and q.')

        if _bounds[0] >= _x[0]:
            raise InputError('Lower bound must be less than all x.')
        
        # upper bound could be None
        if _bounds[1]:
            if _bounds[1] <= _x[-1]:
                raise InputError('Upper bound must be greater than all x.')

        return True
    

# Quantile Paramaterized Distribution abstract base class. 
# Forces inclusion of cdf, pdf, ppf for subsequent qpd versions.
class _QPD(ABC):
    '''Parent class for Quantile Parameterized Distribution functions.'''
    
    def __init__(
            self,
            x: list,
            q: list,
            bounds: list)-> None:
        
        self._base = _BaseInputs(x, q, bounds)


    @property
    def x_base(self):
        return self._base.x
    
    @x_base.setter
    def x_base(self, value:list):
        # check input first
        self._base.check_args(new_x=value)
        # if good, then change
        self._base.x = value
        
    @property
    def q_base(self):
        return self._base.q
    
    @q_base.setter
    def q_base(self, value:list):
        # check input first
        self._base.check_args(new_q=value)
        # if good, then change
        self._base.q = value
        
    @property
    def bounds(self):
        return self._base.bounds
    
    @bounds.setter
    def bounds(self, value:list):
        # check input first
        self._base.check_args(new_bounds=value)
        # if good, then change
        self._base.bounds = value


    @abstractmethod
    def cdf(*args, **kwargs):
        '''
        Cumulative density function at ``x`` of the given distribution.
    
        Parameters
        ----------
        ``x``: array_like, float, Optional
            Quantiles for the given range. If not specified, cdf is computed for number
            of quantiles specified in ``size`` across the bounds range. Note: ``x`` inputs
            outside of bounds will result in a ``nan`` result.

        ``size``: int, Optional
            Size of array to be calculated. If ``x`` is specified, size is ignored.
            Default = 200.
        
        Returns
        -------
        cdf : array_like, float
            Cumulative density function evaluated at ``x``.
        '''
        pass

    @abstractmethod
    def pdf(*args, **kwargs):
        '''
        Probability density function at ``x`` of the given distribution.
    
        Parameters
        ----------
        ``x``: array_like, float, Optional
            Quantiles for the given range. If not specified, pdf is computed for number
            of quantiles specified in ``size`` across the bounds range.

        ``size``: int, Optional
            Size of array to be calculated. If ``x`` is specified, size is ignored.
            Default = 200.

        ``normalized``: bool, optional
            Normalization choice. If false (default), returns un-normalized pdf.
            If true, returns normalized pdf (ie. integral = 1).

        Returns
        -------
        pdf : array_like, float
            Probability density function evaluated at ``x``.
        '''
        pass

    @abstractmethod
    def ppf(*args, **kwargs):
        '''
        Percent point function (inverse of cdf) at ``q`` of the given distribution. 
        Also called the quantile function.

        Parameters
        ----------
        ``q``: array_like, float
            Lower tail probability. Must be in the range 0 to 1.
        
        Returns
        -------
        x: array_like, float
            Quantile corresponding to the lower tail probability ``q``.
        '''
        pass

    def __repr__(self):
        
        msg = f'''
        QPD Instance with basis:

        Quantile  Probability
            {self._base.x[0]},      {self._base.q[0]}
            {self._base.x[1]},      {self._base.q[1]}
            {self._base.x[2]},      {self._base.q[2]}

        Bounds: {self._base.bounds[0]} to {self._base.bounds[1]}
        '''

        return msg

class JQPDBounded(_QPD):
    '''Johnson QPD - Bounded Class

    Quantile parameterized distribution using the Johnson SU distribution as a
    basis for construction. The JQPDBounded distribution has finite lower and 
    upper support bounds, [``lower``, ``upper``], and is parameterized by any 
    compatible symmetric quantile triplet.

    Parameters
    ----------
    ``xlow``: float
        Value for the lower symmetric quantile, q(``alpha``). 

    ``x50``: float
        Value for the base quantile, q(0.5).

    ``xhigh``: float
        Value for the higher symmetric quantile, q(1 - ``alpha``).

    ``alpha``: float, optional
        Quantile parameter element of (0, 0.5), exclusive. Defines the symmetry about the 
            base quantile. For example, ``alpha`` = 0.1 --> (``xlow``, ``x50``, ``xhigh``) 
            correspond to quantiles of (0.1, 0.5, 0.9), respectively. (Default = 0.1).

    ``lower``: float, optional
        Value for lower support bound. (Default = 0).

    ``upper``: float, optional
        Value for upper support bound. (Default = 1).

    Returns
    -------
    Instantiated JQPDBounded class object.

    Notes
    -----
    - A quantile triplet is compatible if and only if:
        ``lower`` < ``xlow`` < ``x50`` < ``xhigh`` < ``upper``
    - ``alpha`` must be within the bounds 0 < ``alpha`` < 0.5

    References
    ----------
    - [1] Hadlock, C.C., J.E. Bickel. 2017. Johnson Quantile-Parameterized Distributions. 
        Decision Analysis 14(1) 35-64.
    - [2] Hadlock, C.C. 2017. Quantile-Parameterized Methods for Quantifying Uncertainty in 
        Decision Analysis. Ph.D. dissertation, University of Texas.
    '''

    def __init__(
        self,
        xlow: float, # low quantile
        x50: float, # base quantile
        xhigh: float, # high quantile
        alpha: float = 0.1, # triplet definition a=0.1 -> Q(0.1, 0.5, 0.9)
        lower: float = 0.0, # lower bound
        upper: float = 1.0 # upper bound
    ) -> None:
        
        # check alpha input before having to check everything else.
        if alpha <= 0 or alpha >= 0.5:
            raise InputError(f'alpha must be in the range (0, 0.5)')

        # defines self._base and checks inputs
        super().__init__(
            [xlow, x50, xhigh],
            [alpha, 0.5, 1-alpha],
            [lower, upper])

        self._alpha = alpha

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):

        if value <= 0 or value >= 0.5:
            raise InputError(f'alpha must be in the range (0, 0.5)')

        self.q_base = [value, 0.5, 1-value]
        self._alpha = value


    def _get_calc_args(self):
        '''Internal argument calcs.'''
        # Grab current base values
        xlow, x50, xhigh = self._base.x
        lower, upper = self._base.bounds
        alpha = self._alpha

        # Starting from the JSU quantile function, we have to solve for the
        # parameters xi, lambda, delta, and gamma.

        # JSU shape parameter gamma is held constant to a value of either
        # -c, 0, or +c, where c is:
        c = _nppf(1 - alpha)
        # NOTE: sign of 'c' determined by 'n' below

        # xlow factor
        lf = _nppf((xlow - lower) / (upper - lower))
        # x50 factor
        mf = _nppf((x50 - lower) / (upper - lower))
        # xhigh factor
        hf = _nppf((xhigh - lower) / (upper - lower))
        # NOTE: factors represented as 'L', 'B', and 'H' by Hadlock

        # JSU shape parameter gamma is defined as n*c.
        # The sign of gamma is determined by n, where
        # n is -1, 0, or 1, based on lf, mf, and hf
        n = sign(lf + hf - 2*mf)
        
        # JSU location parameter, xi.
        # xi = low_fact if n=1, mid_fact if n=0, or high_fact if n=-1
        xi = (1/2) * (lf*(1 + n) + hf*(1 - n)) 

        # JSU shape parameter, delta
        # delta must be > 0. If n = 0 -> delta = 0. This violates the parameter
        # rules, but by using the limit as delta -> 0, an exception case is 
        # defined later.
        delta = (1/c) * arccosh((hf - lf) / (2*min(mf - lf, hf - mf)))
        
        # JSU scale parameter, lambda
        lam = (hf - lf) / sinh(2*delta*c)

        # Package the necessary parameters into a tuple to use later. This 
        # just allows me to still access them without having to write self
        # a bunch of times in each method.
        return (c, lf, mf, hf, n, xi, delta, lam)


    def ppf(self, q: Union[ArrayLike, float]) -> Union[ArrayLike, float]:
        # Quantile function - JQPD-Bounded.
        # This is the result of an inverse-probit Q-transformation applied to
        # the Johnson SU quantile function. Additional shifting and scaling
        # applied with (lower, upper).

        # Method named 'ppf' to be consistent with Scipy terminology.
        _q = np.asarray(q)

        if np.any(_q) < 0 or np.any(_q) > 1:
            raise InputError('Probability input must be in the range [0, 1]')

        # Unpackage internal args from tuple.
        c, lf, mf, hf, n, xi, delta, lam = self._get_calc_args()

        lower, upper = self._base.bounds

        if n == 0: # special exception when n = 0 -> delta = 0
            out = lower + (upper - lower) * _ncdf(mf + ((hf - lf)/2*c) * _nppf(_q))
        else: # delta is > 0
            out = (lower + (upper - lower)
                * _ncdf(xi + lam*sinh(delta * (_nppf(_q) + n*c))))
        return out


    def cdf(self,
        x: Union[ArrayLike, float] = [],
        size: int = _default_size,
        ) -> Union[ArrayLike, float]:
        # Cumulative Density Function - JQPD - Bounded.
        # Direct inverse of quantile function (ppf).

        lower, upper = self._base.bounds

        _x = np.asarray(x)
        if _x.size == 0: # no x input given
            # create array with length 'size' exclusive to lower and upper by wiggle.
            _x = np.linspace(lower, upper, size)

        # Unpackage internal args from tuple.
        c, lf, mf, hf, n, xi, delta, lam = self._get_calc_args()

        # Suppress numpy divide by zero errors when x == lower or x == upper.
        # Those instances will result in a nan value within the array. I wanted
        # to maintain the size of the original array so that it plays nice with 
        # plotting and other size dependent checks.
        with np.errstate(divide='ignore', invalid='ignore'):
            if n == 0: # special exception when n = 0 -> delta = 0
                out = _ncdf((2*c/(hf - lf)) * (_nppf((_x - lower)/(upper - lower)) - mf))
            else:
                out = (_ncdf((1/delta) * arcsinh((1/lam) 
                    * (_nppf((_x - lower) / (upper - lower)) - xi)) - n*c))
            return out


    def pdf(self,
        x: Union[ArrayLike, float] = [],
        size: int = _default_size,
        normalized: bool=False,
        ) -> Union[ArrayLike, float]:
        # Probability density function - JQPD-Bounded.
        # derivative of cdf; done by chain rule.

        lower, upper = self._base.bounds

        _x = np.asarray(x)

        if _x.size == 0: # no x input given
            # create array with length 'size'
            _x = np.linspace(lower, upper, size)
   
        # Unpackage internal args from tuple.
        c, lf, mf, hf, n, xi, delta, lam = self._get_calc_args()

        # Suppress numpy divide by zero errors when x == lower or x == upper.
        # Those instances will result in a nan value within the array. I wanted
        # to maintain the size of the original array so that it plays nice with 
        # plotting and other size dependent checks.
        with np.errstate(divide='ignore', invalid='ignore'):
            # first two parts of chain rule
            part1 = 1 / (upper - lower)
            
            part2 = 1 / (_npdf(_nppf((_x - lower)/(upper - lower))))

            # 3rd and 4th parts of chain rule
            if n == 0: # special exception when n = 0, delta = 0
                part3 = 2*c / (hf - lf) * _nppf((_x - lower)/(upper - lower))
                part4 = _npdf((2*c/(hf - lf))*(_nppf((_x - lower)/(upper - lower)) - mf))

            else:
                part3 = (1 / (delta * sqrt((_nppf((_x - lower)/(upper - lower))
                    - xi)**2 + lam**2)))
                part4 = (_npdf((1/delta) * arcsinh((1/lam)
                    * (_nppf((_x - lower) / (upper - lower)) - xi)) - n*c))

            # return combined derivative
            pdf_out = np.asarray(part1*part2*part3*part4)
            
            if normalized: #normalize pdf
                pdf_out = pdf_out / pdf_out.sum()

            return pdf_out


class JQPDSemiBounded(_QPD):
    '''Johnson QPD - Semi-Bounded Class

    Quantile parameterized distribution using the Johnson SU distribution as a
    basis for construction. The JQPDSemiBounded distribution has a finite lower bound and 
    infinite upper bound, [``lower``, infinity), and is parameterized by any compatible symmetric 
    quantile triplet.

    Parameters
    ----------
    ``xlow``: float
        Value for the lower symmetric quantile, q(``alpha``). 

    ``x50``: float
        Value for the base quantile, q(0.5).

    ``xhigh``: float
        Value for the higher symmetric quantile, q(1 - ``alpha``).

    ``alpha``: float, optional
        Quantile parameter element of (0, 0.5), exclusive. Defines the symmetry about the 
        base quantile. For example, ``alpha`` = 0.1 --> (``xlow``, ``x50``, ``xhigh``) 
        correspond to quantiles of (0.1, 0.5, 0.9), respectively. (Default = 0.1).

    ``lower``: float, optional
        Value for lower support bound. (Default = 0).

    Returns
    -------
    Instantiated JQPDSemiBounded class object.

    Notes
    -----
    - A quantile triplet is compatible if and only if:
        ``lower`` < ``xlow`` < ``x50`` < ``xhigh``
    - ``alpha`` must be within the bounds 0 < ``alpha`` < 0.5

    References
    ----------
    - [1] Hadlock, C.C., J.E. Bickel. 2017. Johnson Quantile-Parameterized Distributions. 
        Decision Analysis 14(1) 35-64.
    - [2] Hadlock, C.C. 2017. Quantile-Parameterized Methods for Quantifying Uncertainty in 
        Decision Analysis. Ph.D. dissertation, University of Texas.
    '''

    def __init__(
        self,
        xlow: float, # low quantile
        x50: float, # base quantile
        xhigh: float, # high quantile
        alpha: float = 0.1, # triplet definition a=0.1 -> Q(0.1, 0.5, 0.9)
        lower: float = 0.0, # lower bound
    ) -> None:

        # check alpha input before having to check everything else.
        if alpha <= 0 or alpha >= 0.5:
            raise InputError(f'alpha must be in the range (0, 0.5)')

        # defines self._base and checks inputs
        super().__init__(
            [xlow, x50, xhigh],
            [alpha, 0.5, 1-alpha],
            [lower, None])

        self._alpha = alpha


    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):

        if value <= 0 or value >= 0.5:
            raise InputError(f'alpha must be in the range (0, 0.5)')

        self.q_base = [value, 0.5, 1-value]
        self._alpha = value


    def _get_calc_args(self):
        '''Internal argument calcs.'''
        # Grab current base values
        xlow, x50, xhigh = self._base.x
        lower = self._base.bounds[0]
        alpha = self._alpha

        # starting from the JSU quantile function, we have to solve for the
        # parameters xi, lambda, delta, and gamma.

        # JSU shape parameter gamma is held constant to a value of either
        # -c*delta, 0, or +c*delta, where c is:
        c = _nppf(1 - alpha)

        # xlow factor
        lf = log(xlow - lower)
        # x50 factor
        mf = log(x50 - lower)
        #  xhigh factor
        hf = log(xhigh - lower)
        # NOTE: factors represented as 'L', 'B', and 'H' by Hadlock
        
        # JSU shape parameter gamma is defined as n*c*delta
        # n is -1, 0, or 1, based on lf, mf, and hf.
        n = sign(lf + hf - 2*mf)

        # JSU location parameter, xi.
        # Transformed by theta = exp(xi)
        # theta = (xlow-lower) if n=1 or (xhigh-lower) if n=-1
        theta = 1/2 * ((xlow - lower)*(1 + n) + (xhigh - lower)*(1 - n)) 

        # JSU shape parameter, delta
        # delta must be > 0. If n = 0 -> delta = 0. This violates the parameter
        # rules, but by using the limit as delta -> 0, an exception case is 
        # defined.
        delta = (1 / c) * sinh(arccosh((hf - lf) / (2 * min([mf - lf, hf - mf]))))
        
        # JSU scale parameter, lambda
        lam = (1 / (delta * c)) * min([mf - lf, hf - mf])
        
        # additional shape parameter via Hadlock ?
        s1 = x50 / xhigh

        # package the necessary parameters into a tuple to use later. This 
        # just allows me to still access them without having to write self
        # a bunch of times in each method.
        return (c, n, theta, delta, lam, s1)


    def ppf(self, q: Union[ArrayLike, float]) -> Union[ArrayLike, float]:
        # Quantile function - JQPD-Semi-Bounded.
        # This is the result of an exponential Q-transformation applied to
        # the Johnson SU quantile function. Additional shifting 
        # applied with lower.

        # Method named 'ppf' to be consistent with Scipy terminology.

        # quick check on q inputs
        _q = np.asarray(q)

        if np.any(_q) < 0 or np.any(_q) > 1:
            raise InputError('Probability input must be in the range [0, 1].')

        # Unpackage internal args from tuple.
        c, n, theta, delta, lam, s1 = self._get_calc_args()

        lower = self._base.bounds[0]
        x50 = self._base.x[1]

        if n == 0: # special exception when n = 0, delta = 0
            out = lower + x50 * (s1**(-1 * _nppf(_q) / c))
        else:
            out = (lower + theta*exp(lam*sinh(arcsinh(delta*_nppf(_q))
                + arcsinh(n*c*delta))))

        return out


    def cdf(self,
        x: Union[ArrayLike, float] = [],
        size: int = _default_size,
        ) -> Union[ArrayLike, float]:
        # Cumulative Density Function - JQPD - Semi-Bounded.
        # Direct inverse of quantile function (ppf).

        lower = self._base.bounds[0]

        _x = np.asarray(x)

        if _x.size == 0: # no x input given
            # create upper limit of twice ppf(0.999)
            upper = self.ppf(0.999) * 2
            # create array with length 'size'
            _x = np.linspace(lower, upper, size)

        # Unpackage internal args from tuple.
        c, n, theta, delta, lam, s1 = self._get_calc_args()
            
        x50 = self._base.x[1]
    
        # Suppress numpy divide by zero errors when x == lower.
        # Those instances will result in a nan value within the array. I wanted
        # to maintain the size of the original array so that it plays nice with 
        # plotting and other size dependent checks.
        with np.errstate(divide='ignore', invalid='ignore'):
            s3 = log((_x - lower)/x50)
            s4 = log((_x - lower)/theta)

            if n == 0: # special exception when n = 0, delta = 0
                out = _ncdf((-c / log(s1)) * s3)
            else:
                out = (_ncdf((1/(lam*delta)) * (s4*sqrt((n*c*delta)**2 + 1)
                    - (n*c*delta)*sqrt(s4**2 + lam**2))))

            return out


    def pdf(self,
        x: Union[ArrayLike, float] = [],
        size: int = _default_size,
        normalized: bool=False,
        ) -> Union[ArrayLike, float]:
        # Cumulative Density Function - JQPD - Semi-Bounded.
        # Direct inverse of quantile function (ppf).

        lower = self._base.bounds[0]

        _x = np.asarray(x)

        if _x.size == 0:
            # create upper limit of twice ppf(0.999)
            upper = self.ppf(0.999) * 2
            # create array with length 'size' 
            _x = np.linspace(lower, upper, size)

        # Unpackage internal args from tuple.
        c, n, theta, delta, lam, s1 = self._get_calc_args()
            
        x50 = self._base.x[1]
        
        # Suppress numpy divide by zero errors when x == lower.
        # Those instances will result in a nan value within the array. I wanted
        # to maintain the size of the original array so that it plays nice with 
        # plotting and other size dependent checks.
        with np.errstate(divide='ignore', invalid='ignore'):
            s3 = log((_x - lower)/x50)
            s4 = log((_x - lower)/theta)

            if n == 0: # special exception when n = 0, delta = 0
                out = _npdf((-c / log(s1)) * s3)
            else:
                part1 = ((1/((lam*delta)*(_x - lower))) * (sqrt((n*c*delta)**2 + 1)
                    - n*c*delta*s4 / sqrt((s4)**2 + lam**2)))

                part2 = (_npdf((1/(lam*delta)) * (s4*sqrt((n*c*delta)**2 + 1)
                    - (n*c*delta)*sqrt(s4**2 + lam**2))))

                out = part1*part2

            pdf_out = np.asarray(out)

            if normalized == True:
                pdf_out = pdf_out / pdf_out.sum()

            return pdf_out


    def _pdf2(self,
        x: Union[ArrayLike, float] = [],
        size: int = _default_size,
        normalized: bool=False,
        ) -> Union[ArrayLike, float]:
        # Cumulative Density Function - JQPD - Semi-Bounded.
        # Direct inverse of quantile function (ppf).

        lower = self._base.bounds[0]

        _x = np.asarray(x)

        if _x.size == 0:
            # create upper limit of twice ppf(0.999)
            upper = self.ppf(0.999) * 2
            # create array with length 'size'.
            _x = np.linspace(lower, upper, size)

        # Unpackage internal args from tuple.
        c, n, theta, delta, lam, s1 = self._get_calc_args()
            
        x50 = self._base.x[1]

        # Suppress numpy divide by zero errors when x == lower or x == upper.
        # Those instances will result in a nan value within the array. I wanted
        # to maintain the size of the original array so that it plays nice with 
        # plotting and other size dependent checks.
        with np.errstate(divide='ignore', invalid='ignore'):
            s3 = log((_x - lower)/x50)
            s4 = log((_x - lower)/theta)

            if n == 0: # special exception when n = 0, delta = 0
                part1 = 1/x50
                part2 = -c/log(s1) * x50/(_x-lower)
                part3 = _npdf(((-c / log(s1)) * s3))
                out = part1*part2*part3
            else:
                part1 = 1/theta
                part2 = theta / (lam * (_x - lower))
                part3 = 1 / sqrt(1/lam**2 * s4**2 + 1)
                part4 = 1/delta * cosh(arcsinh(1/lam*s4) - arcsinh(n*c*delta))
                part5 = _npdf(1/delta*sinh(arcsinh(1/lam*s4)-arcsinh(n*c*delta)))
                out = part1*part2*part3*part4*part5  

            pdf_out = np.asarray(out)

            if normalized == True:
                pdf_out = pdf_out / pdf_out.sum()

            return pdf_out

#-------------------------------------------------------------------------------

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    xlowB = stat.beta.ppf(0.1, 2, 4)
    x50B = stat.beta.ppf(0.5, 2, 4)
    xhighB = stat.beta.ppf(0.9, 2, 4)

    distB = JQPDBounded(xlowB, x50B, xhighB)

    xB = np.linspace(0, 1, 200)
    yB_pdf = distB.pdf(normalized=True)
    yB_cdf = distB.cdf()
    yBeta_pdf = stat.beta.pdf(xB, 2, 4)
    yBeta_pdf = yBeta_pdf / np.sum(yBeta_pdf)
    yBeta_cdf = stat.beta.cdf(xB, 2, 4)

    xlowSB = stat.weibull_max.ppf(0.1, 2, 5)
    x50SB = stat.weibull_max.ppf(0.5, 2, 5)
    xhighSB = stat.weibull_max.ppf(0.9, 2, 5)
   
    distSB = JQPDSemiBounded(xlowSB, x50SB, xhighSB)

    xSB = np.logspace(0, 1, 200)
    ySB_pdf = distSB.pdf(xSB, normalized=True)
    ySB_cdf = distSB.cdf(xSB)
    yW_pdf = stat.weibull_max.pdf(xSB, 2, 5)
    yW_pdf = yW_pdf / np.sum(yW_pdf)
    yW_cdf = stat.weibull_max.cdf(xSB, 2, 5)

    fig, (axB, axSB) = plt.subplots(2,2)
    axB[0].plot(xB, yB_pdf, xB, yBeta_pdf, '--')
    axB[1].plot(xB, yB_cdf, xB, yBeta_cdf, '--')
    axSB[0].plot(xSB, ySB_pdf, xSB, yW_pdf, '--')
    axSB[1].plot(xSB, ySB_cdf, xSB, yW_cdf, '--')
    axB[0].sharex(axB[1])
    axSB[0].sharex(axSB[1])

    plt.show()
