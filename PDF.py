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
    
    def binomic_plot(self,n,p):
        """this function calculates all probabilities for all 
        values given the success rate

        Args:
            n (int): number of total values to find the binomial probability
            p (float): probability of success, or success rate

        Returns:
            tuple: returns a tuple of values one for values from range 0 to number
            of values, and second array for probablities regarding each value.
        """
    
        x = np.arange(0, n+1)
        y = np.array([self.binom(n, i, p) for i in x])
        return (x,y)
    
        
    def binsum(self,length,p, size):
        """this method first creates an array of all arrays returned by binary array
        genrator 'lbiner' then it returns sum of each array in a numpy array,
        which lists all sums. then you can see the binomial distribution for a 
        given success rate.

        Args:
            length (int): it is length of each array to be created by array-
            generator function 'lbiner'
            p (float): probability of success between 0-1
            size (int): the number of samples to generate

        Returns:
            numpy.ndarray: returns an array of sum of successes in each sample-
            space 
        """
        def lbiner(size, p):
            "creates an array of returnees of biner function given the size"
            def biner(number, p):
                """returns a binary number for given random number"""
                if (p > 1) or (p < 0):
                    raise Exception('Probablity should be between 0 and 1')
                if number < p:
                    return 1
                elif number >= p:
                    return 0

            vbiner = np.vectorize(biner) #vectorizes the function to work with numpy arrays
            r = np.random.random
            l = np.array([r() for i in range(size)])
            brnli = vbiner(l, p)
            return brnli

        pulls = np.array([lbiner(length, p) for i in range(size)])
        pull_result = np.array([x.sum() for x in pulls])
        return pull_result


