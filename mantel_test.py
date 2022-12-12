# This is a python version of the original 'mantel.test.R' file from
# ape package (https://www.rdocumentation.org/packages/ape/versions/5.6-2/topics/mantel.test).
# mantel.test.R (2019-02-25)
# Mantel Test for Similarity of Two Matrices
# Copyright 2002-2011 Ben Bolker and Julien Claude, 2019 Emmanuel Paradis
# This file (mantel.test.R) is part of the R-package `ape'.
# See the file ../COPYING for licensing issues.
import numpy as np
import pandas as pd
from random import sample


# permutation the index of a matrix; m1 is a data frame
def perm_rowscols(m1, n):
    m1_row_num = m1.shape[0]
    if (m1_row_num == m1.shape[1]) and (n <= m1_row_num):
        s = sample(range(m1_row_num), n)
        return m1.iloc[s, s]
    else:
        return pd.DataFrame([])


# calculate the Mantel z-statistic for two square matrices m1 and m2; m1 and m2 are two data frames
def mant_zstat(m1, m2):
    m1 = pd.DataFrame.to_numpy(m1)
    np.fill_diagonal(m1, 0)
    m2 = pd.DataFrame.to_numpy(m2)
    np.fill_diagonal(m2, 0)
    m1m2 = np.multiply(m1, m2)
    return np.sum(m1m2)/2


# the mantel test of two matrices, m1 and m2 are two data frames
def mant_test(m1, m2, nperm=999, alternative="two.sided"):
    mantOut = {"z.stat": -1, "p": -1, "alternative": alternative}
    # check whether the number of rows and columns are same between m1 and m2
    if (m1.shape[0] == m2.shape[0]) and (m1.shape[1] == m2.shape[1]):
        n = m2.shape[0]
        realz = mant_zstat(m1, m2)
        nullstats = []
        for ii in range(nperm):
            m2perm = perm_rowscols(m2, n)
            permz = mant_zstat(m1, m2perm)
            nullstats.append(permz)
        lessNum = [num for num in nullstats if num <= realz]
        greaterNum = [num for num in nullstats if num >= realz]
        if alternative == "less":
            pval = len(lessNum)
        elif alternative == "greater":
            pval = len(greaterNum)
        elif alternative == "two.sided":
            pval = 2*min([len(lessNum), len(greaterNum)])
        else:
            pval = -1*nperm
            print("Please set the alternative as 'less', 'greater', or 'two.sided'\n.")
        # 'realz' is included in 'nullstats'
        pval = (pval + 1)/(nperm + 1)
        if (alternative == "two.sided") and (pval > 1):
            pval = 1
        mantOut["z.stat"] = realz
        mantOut["p"] = pval
    else:
        print("The dimensions of two matrices are not equal.\n")
    return mantOut

# example
# x = pd.DataFrame({"a":[0,1,2], "b":[1,0,3], "c":[2,3,0]})
# y = pd.DataFrame({"a":[0,2,7], "b":[2,0,6], "c":[7,6,0]})
# mant_test(x,y)
# {'z.stat': 34.0, 'p': 0.687, 'alternative': 'two.sided'}
