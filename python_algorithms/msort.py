""" Module contains 2 functions:
merge: used by mergesort
mergesort: recursive implementation of merge sort algorithm
Running this module will run mergesort with a random sequence
of 8 integers
"""


def merge(L, R):
    """Merge 2 sorted lists provided as input
    into a single sorted list
    """
    M = []  # Merged list, initially empty
    indL, indR = 0, 0  # start indices
    nL, nR = len(L), len(R)

    # Add one element to M per iteration until an entire sublist
    # has been addedz
    for i in range(nL + nR):
        if L[indL] < R[indR]:
            M.append(L[indL])
            indL = indL + 1
            if indL >= nL:
                M.extend(R[indR:])
                break
        else:
            M.append(R[indR])
            indR = indR + 1
            if indR >= nR:
                M.extend(L[indL:])
                break
    return M


def mergesort(X):
    """Given a unsorted list, sort list using
    merge sort algorithm and return sorted list
    """

    n = len(X)

    if n == 1:
        return X
    L = mergesort(X[:n // 2])
    R = mergesort(X[n // 2:])
    return merge(L, R)


if __name__ == '__main__':
    """
    Executing "run msort" within a python terminal will run the example
    code below
    """
    import numpy as np
    A = list(np.random.randint(0, 20, 8))
    S = mergesort(A)
    print("Initial list:", A)
    print("Sorted list:", S)
