import pandas as pd
import numpy as np
from math import factorial


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.
    d: dictionary
    options: keyword args to add to d
    :return: modified d
    """
    for key, val in options.items():
        d.setdefault(key, val)

class PDF(pd.Series):
    def __init__(self, *args, **kwargs):
        """Initialize a Pmf.
        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        underride(kwargs, name="")
        if args or ("index" in kwargs):
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)
        


    def pmf(self, normalize=False):
        n = self.value_counts()
        if normalize:
            return (n/self.shape[0]).sort_index()
        return n.sort_index()

    def combination(self,n, k):
        result = factorial(n)/(factorial(k)*factorial(n-k))
        return result

    def binom(self, n, k, p):
        if p > 1:
            raise Exception("Probabily should be between 0 and 1")
        elif k > n:
            raise Exception("number of options should be smaller than all options")
        result = self.combination(n, k)*(p**k)*(1-p)**(n-k)
        return result

