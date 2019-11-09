"""M345SC Homework 1, Part 1
Tudor Trita Trita
MSci Mathematics Year 3
CID: 01199397
"""


def ksearch(S, k, f, x):
    """
    Search for frequently-occurring k-mers within a DNA sequence
    and find the number of point-x mutations for each frequently
    occurring k-mer.
    Input:
    S: A string consisting of A,C,G, and Ts
    k: The size of k-mer to search for (an integer)
    f: frequency parameter -- the search should identify k-mers which
    occur at least f times in S
    x: the location within each frequently-occurring k-mer where
    point-x mutations will differ.

    Output:
    L1: list containing the strings corresponding to the frequently-occurring k-mers
    L2: list containing the locations of the frequent k-mers
    L3: list containing the number of point-x mutations for each k-mer in L1.

    Discussion:

    a) My algorithm works like this:

    1. An empty dictionary is created, and various parameters are set up to be used
    for the rest of the algorithm.

    2. We run through S checking if the current k-mer is in the dictionary. If it is not,
    then we create a new key, and add it's location in the dictionary corresponding to it.
    If it is already in the dictionary, we append it's current location to the dictionary.

    3. For this step, we loop across the length of the dictionary, and for each iteration:
        3.1. We can get the number of times it appears in S by checking the length of it's
        locations in the dictionary. If this length is >= f, then we proceed with the loop, else
        we skip the iteration (and following steps in 3).
        3.2. We append the current k-mer to L1.
        3.3. We the locations of current k-mer to L2.
        3.4. We append the negative of the no. of appereances of the current k-mer to L3. This
        is to balance it out for step 3.5
        3.5. We sum the appearances of other mutations of L3 by
        Repeat for other k-mers in the dictionary.

    Return L1, L2, L3. END

    b) Analysis: (N = len(S)))

    The two computationally expensive portions of code are in steps 2. and 3.

    Step 2 iterates N - k + 1 times. For each iteration, we have a string slicing and
    comparison which is O(k) time complexity. Then we check if the kmer is in the dictionary
    which is of O(1) complexity and then we append in the dictionary, which is again
    O(1) complexity. Putting it together, for this step we have approx. O(k*(N - k)) complexity.

    Step 3:
        The number of possible k-mers of length k is 4^k in general. So the length of
        our dictionary is at most the minimum of 4^k and N - k + 1. This is followed
        by some operations each of O(1) complexity.
        Inside step 3, we have another loop for checking mutations.
        Inside this loop, we have some operations, some of which of O(1), and some
        of which of O(k).

    Combining these, we have an overall complexity for Step 3 of O(min(4^k, N-k)).

    For the whole algorithm, we have a time complexity of around O(k*(N - k) + min(4^k, N-k))

    The leading order term in this is k(N-k+1), and this is dependent on both k and N.

    Summary:
    When k << N, we have that the algorithm is approx O(N) in complexity, because the leading-order
    term is N.
    When k is large and close in size to N, we have that N-k is small, and therefore the leading
    term is k, so the algorithm is approx. O(k). But since k is close in size to N,
    this is equivalent to saying that the algorithm is of O(N) complexity again.
    When is k is approx N/2, the algorithm will run in approx. N/2*N/2 operations which is
    O(N^2) complexity, which is the worst case scenario in this instance.

    For the other parameters x and f:
    Varying x does not change the number of operations involved and therefore does not affect
    the running time.
    Varying f does change the amount of operations in Step 3: The worst case scenario is if f = 1,
    as then every k-mer will be present in L1. For larger f, L1 decreases dramatically and therefore
    Step 3 does not take as many steps. Still, there algorithms still requires O(kN) operations
    from step 2, and thus the summary still holds true.

    Conclusion:
    Searching for k-mers in S involves keeping track of every possible k-mer in S. To
    be able to succesfully fill L1, L2 and L3, we have to keep track of the locations
    of every k-mer. I have decided to use a Python dictionary to keep track of the
    locations for each k-mer, as it is a hash table and searching scales well with
    the size of S, because a search in the dictionary is always of O(1) no matter its size.
    The reason why I chose this method, is that it works well for different orders of magnitude of
    k and takes a reasonably small number of operations to compute L1, L2 and L3.
    """

    # 1. Setting up parameters:

    dict_kmer = {}  # Creating empty dictionary to store locations of kmers
    N = len(S)
    iterates = N - k + 1
    L1, L2, L3 = [], [], []
    bases = ['A', 'C', 'G', 'T']

    # 2. Iterating over N - k + 1 to get information about kmers:

    for i in range(iterates):

        k_mer = S[i:i + k]  # Get current kmer

        if k_mer in dict_kmer:
            dict_kmer[k_mer].append(i)  # If kmer is inside dictionary already

        else:
            dict_kmer[k_mer] = [i]  # If kmer is not inside dictionary

    # 3. Begin process of filling L1, L2 and L3
    counter = 0  # Needed to keep track of kmer in L1

    # Iterating over dictionary:
    for kr, locs in dict_kmer.items():
        if len(locs) >= f:
            L1.append(kr)
            L2.append(dict_kmer[kr])
            # Accounting for current base in following loop
            L3.append(-len(dict_kmer[kr]))
            beg = kr[:x]
            end = kr[x + 1:]
            for ltr in bases:
                if beg + ltr + end in dict_kmer:
                    L3[-1] += len(dict_kmer[beg + ltr + end])

            counter += 1

    return L1, L2, L3


def testing():
    """ Function for testing part 1.
    """
    import time

    k = 5
    x = 1
    f = 3
    infile = open('test_sequence.txt', 'r')
    S = infile.read()
    infile.close()
    t1 = time.time()
    # timeit ksearch(S, k, f, x)
    t2 = time.time()
    print(t2 - t1)

    return None
