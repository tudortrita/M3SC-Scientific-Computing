def isort(X):
    """Sort X using insertion sort algorithm and return sorted array
    """

    S = X.copy()

    for i, x in enumerate(X[1:], 1):
        # place x appropriately in partially sorted array, S
        for j in range(i - 1, -1, -1):
            if S[j + 1] < S[j]:
                S[j], S[j + 1] = S[j + 1], S[j]
            else:
                break
    return S


if __name__ == '__main__':
    """
    Executing "run isort" within a python terminal will run the example
    code below
    """
    import numpy as np
    A = list(np.random.randint(0, 20, 8))
    S_out = isort(A)
    print("Initial list:", A)
    print("Sorted list:", S_out)
