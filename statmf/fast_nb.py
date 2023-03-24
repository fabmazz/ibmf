
import numpy as np
from numba import njit
@njit()
def _inner_calc_loop(pi, pnoi, rows, cols, data):
    for j,i,l in zip(rows,cols, data):
        pnoi[i]*=(1- pi[j]*l)

def calc_loop_p_nb(pi, lambs):
    pnoi = np.ones(len(pi))
    mcc=lambs.tocoo()
    _inner_calc_loop(pi,pnoi,mcc.row,mcc.col, mcc.data)
    return pnoi