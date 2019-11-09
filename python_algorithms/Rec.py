""" Estimate missing entries in ratings matrix using
alternating Lagrangian method
"""

import math as ma
import numpy as np


def IALM(R, rho, flag=0, itermax=1000):
    """ Incomplete alternating Lagrangian method (IALM)
    for solving the matrix completion problem
    Input:
    R: Incomplete ratings matrix
    rho: scaling factor for updating mu
    flag: Missing entries in R satisfy R=flag
    itermax: maximum number of iterations

    Output:
    A,E: Estimated complete ratings matrix and auxillary matrix
    Original code developed by Prof. P. Schmid
    """

    # thresholds
    ep1 = 1.e-7
    ep2 = 1.e-6
    Rnorm = np.linalg.norm(R, 'fro')

    mu = 1. / np.linalg.norm(R, 2)  # initial guess

    # projector matrix
    PP = (R == flag)
    P = PP.astype(np.float)

    # initialization
    m, n = np.shape(R)
    Y = np.zeros((m, n))
    Eold = np.zeros((m, n))

    # iteration
    for i in range(1, itermax):

        # Stage 1: compute SVD---
        tmp = R - Eold + Y / mu
        U, S, V = np.linalg.svd(tmp, full_matrices=False)

        # threshold and patch matrix back together
        ss = S - (1 / mu)
        s2 = np.clip(ss, 0, max(ss))
        A = np.dot(U, np.dot(np.diag(s2), V))

        # Stages 2 and 3: project------
        Enew = P * (R - A + 0.5 * Y / mu)
        RAE = R - A - Enew
        Y += 2 * mu * RAE

        # check residual and (maybe) exit
        r1 = np.linalg.norm(RAE, 'fro')
        resi = r1 / Rnorm
        print(i, ' residual ', resi)
        if resi < ep1:
            break

        # Stage 4: adjust mu-factor
        muf = np.linalg.norm((Enew - Eold), 'fro')
        fac = min(mu, ma.sqrt(mu)) * (muf / Rnorm)
        if fac < ep2:
            mu *= rho

        # update E and go back
        Eold = np.copy(Enew)

    E = np.copy(Enew)
    return A, E
# -----------------------


def lecture_matrix():
    """Generate example ratings matrix from lecture
    """
    Don = [5, -10, 1, -10, -10]
    Liz = [0, 4, 5, -10, -10]
    Kamala = [2, -10, 3, -10, 5]
    Beto = [1, 5, 4, 5, -10]

    return np.vstack([Don, Liz, Kamala, Beto])
# -------------------------


def generic_matrix(m=500, n=150):
    """Generate random incomplete low-rank m x n matrix, R
    from full low-rank matrix, A
    """
    r = int(round(min(m, n) / 3))  # rank

    # construct low-rank matrix
    U = np.random.random((m, r))
    V = np.random.random((r, n))
    A = np.dot(U, V)

    # sampling matrix
    PP = (np.random.random((m, n)) > 0.433)
    P = PP.astype(np.float)
    # number of non-zero elements

    # data matrix
    R = P * A

    return R, A
# -------------------------------------------


if __name__ == '__main__':

    example1, example2 = False, False

    if example1:
        # Generic example
        R_out, A_out = generic_matrix()
        rho_out = 1.25
        AA, EE = IALM(R_out, rho_out)

    if example2:
        # Small example from lecture
        R_out = lecture_matrix()
        rho_out = 1.25
        AA, EE = IALM(R_out, rho_out, flag=-10)

    if example1 or example2:
        print('\n')
        print('Data matrix')
        print(R_out[0:5, 0:5])
        print('\n')
        print('Recovered matrix')
        print(AA[0:5, 0:5])
        print('\n')
        if example1:
            print('Original matrix')
            print(A_out[0:5, 0:5])
