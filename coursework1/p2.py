"""M345SC Homework 1, Part 2
Tudor Trita Trita
MSci Mathematics Year 3
CID: 01199397
"""

import time
import numpy as np
import matplotlib.pyplot as plt

def nsearch(L, P, target):
    """Main function to search for target"""

    N = len(L)
    M = len(L[0])
    Lout=[]

    # loop over each sublist inside L
    for i in range(N):

        # Binary search for sorted part of sublists

        Lprime = L[i][:P]
        midp = bsearch(Lprime, target)

        if midp == 'nothere':
            pass
        else:
            Lout.append([i, midp])

            # Checking whether there are any other targets in the sorted part of the list
            # If there are any - they should be next to each other as the first
            # elements in the list should be sorted

            midp2 = midp + 1

            while L[i][midp2] == target and midp2 < P:
                Lout.append([i, midp2])
                midp2 += 1

            midp2 = midp - 1

            while L[i][midp2] == target and midp2 >= 0:
                Lout.append([i, midp2])
                midp2 -= 1

        # Linear search for rest of list
        for j in range(M-P):
            if L[i][P+j] == target:
                Lout.append([i, P+j])

        Lout.sort()
    return Lout



def bsearch(L, x, istart=0, iend=-1000):
    """Return target from a list using binary search.
    Developed in lab2
    """

    #Set initial start and end indices for full list if iend=-1000
    if iend == -1000:
        iend = len(L)-1
    imid = int(0.5*(istart+iend))

    #Check if search has "converged", otherwise search in appropriate "half" of data
    if istart > iend:
        return 'nothere'

    #Add code below for:
    # 1) comparison between x and L[imid] and 2) 2 recursive calls
    if x == L[imid]:
        return imid
    elif x < L[imid]:
        iend = imid-1
        return bsearch(L, x, istart, iend)
    else:
        istart = imid+1
        return bsearch(L, x, istart, iend)


def Lgen(start, finish, N, M, P):
    """Generates L. This function returns L where numbers are integers
    between start and finish.
    """
    L = []
    for i in range(N):
        aux1 = list(np.random.randint(start, finish, P))
        aux2 = list(np.random.randint(start, finish, M-P))
        aux1.sort()
        L.append(aux1 + aux2)
    return L

def nsearch_time():
    """ Discussion: (Points answered as in per CW description)

    My implementation assumes that there may be multiple
    instances of target inside each sub-list in L.

    a) My algorithm works like this. For each sub-list, it performs a binary
    search on the sorted portion of the list (L[i][:P]). After the binary search
    is complete, it returns the index (midp) of where Target was found, and then it
    checks if there are any other instances of Target within L[i][:P] by checking
    'right' of midp until it doesn't find an instance of Target, then moves on
    to the 'left' portion, doing the same procedure.

    It then moves on to the unsorted part of the list L[i][P:]. As the list is
    unsorted, and there may be multiple instances of Target inside it, the only
    choice is to do a full linear search checking if every element is Target.
    So the algorithm does exactly this.

    On both searches, once an instance of Target is found, its location gets stored
    in Lout as per the coursework description.

    b) The running times depends on N, M and P, but also on the amount of times Target
    is present in the binary search portion.

    For a list of size N, the binary search would run in O(log(N)) complexity, whilst
    the linear search would run in O(N) complexity.

    Taking each sub-list separately, we can consider the worst case scenario, which is
    that the whole of L[i][:P] is filled with Target. In that case, the binary operation
    runs in O(log P) operations but when checking for adjacent terms, we end up having to
    do O(P) extra operations. Therefore, the complexity is O(P + log P), and asymptotycally
    this is O(P) operations. For the linear part, this always runs in O(M-P) operations. Combining
    these two results, for the worst case scenario we have that the algorithm runs in O(P + M - P)
    = O(M) complexity for each sublist. Combining this with the rest of L, and if every sublist
    is in this 'worst case' scenario, we will have O(N*M) complexity.

    For the best case scenario for each sublist, which will be if there is only 1
    instance of Target in L[i][:P], the binary search will run in O(log P) complexity.
    The linear search will still run in O(M - P) time so combining, for each sublist,
    we will have O(log P + M - P). Asymptotycally, this is O(M), and therefore, for the whole
    algorithm, the complexity will be O(N*M) again. (This is because we are assuming M - P is large)

    Therefore, we can conclude, that asymptotycally the algorithm is O(N*M) complexity.

    c) I believe my algorithm is as efficient as possible in this problem setting.
    When performing the search for the sorted part of the list, binary search
    is the most efficient algorithm possible running in O(log(P)). I chose to go with
    the binary search approach and not the alternative, which would be doing a
    upper-lower bound approach of two binary searches, and then filling the middle
    in, thus skipping the comparing adjacent targets, as I assume that the probability
    of target appearing in the list is relatively low.

    For the unsorted part, although linear search is inefficient by nature, it is the
    best choice here, as the list is unsorted, and if I were to sort L[i][P:]
    first and then use binary search, this would have complexity O(M * log M),
    which is bigger than O(M).

    d) and e)

    Figure 1 illustrates the relationship between running times and N.
    The relationship is linear, and this is expected, as times will increase
    proportionally with N.

    Figure 2 illustrates the relationship between actual running times of the
    function and M and it is linear too. This is expected again, as times will increase
    proportionally with M.

    Figure 3 illustrates the relationship between actual running times of the function
    and P as a proportion of M. We can see, that the larger P is in comparison to M,
    the total time taken goes down. This is makes sense, as the binary search
    part gets larger, and this one is quicker than the linear search, so this
    relationship is not surprising.

    Note: time.perf_counter is used, as time.time() does not
    have enough accuracy in Windows systems. This is not the case for Unix-based systems
    """

    # Figure 1:

    times_ave = []
    times_single = []
    Narray = np.linspace(10, 5000, 11, dtype=int)
    start = 0
    finish = 1000
    M = 1000
    P = 400
    target = 400

    for i in range(len(Narray)): # Number of N's
        L = Lgen(start,finish,Narray[i],M,P)
        for j in range(50): # Number of times for averaging
            t1 = time.perf_counter()
            nsearch(L,P,target)
            t2 = time.perf_counter()
            times_single.append(t2-t1)
        times_ave.append(np.average(times_single))
        times_single = []

    fig1 = plt.figure(figsize=(15, 12))
    plt.plot(Narray, times_ave)
    plt.xlabel('N')
    plt.ylabel('Time, seconds')
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 1, Function: nsearch_time \n Plot of average execution times against N, M = 1000, P = 400')
    plt.grid(True)
    plt.show()

    # Figure 2

    times_ave = []
    times_single = []
    Marray = np.linspace(10,5000,11,dtype=int)
    start = 0
    finish = 1000
    N = 50
    Parray = Marray - 1 # np.linspace(5,100,20,dtype=int)
    target = 500

    for i in range(len(Marray)): # Number of N's
        L = Lgen(start,finish,N,Marray[i],Parray[i])
        for j in range(1000): # Number of times for averaging
            t1 = time.perf_counter()
            nsearch(L,Parray[i],target)
            t2 = time.perf_counter()
            times_single.append(t2-t1)
        times_ave.append(np.average(times_single))
        times_single = []

    fig2 = plt.figure(figsize=(15, 12))
    plt.plot(Marray, times_ave)
    plt.xlabel('M')
    plt.ylabel('Time, seconds')
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2, Function: nsearch_time \n Plot of average execution times against M \n when P is half of M')
    plt.grid(True)
    plt.show()

    # Figure 3

    times_ave = []
    times_single = []
    Parray = np.linspace(100,1000,10,dtype=int)
    start = 0
    finish = 1000
    N = 20
    M = 1000
    target = 400

    for i in range(len(Parray)): # Number of N's
        L = Lgen(start,finish,N,M,Parray[i])
        for j in range(50): # Number of times for averaging
            t1 = time.perf_counter()
            nsearch(L,Parray[i],target)
            t2 = time.perf_counter()
            times_single.append(t2-t1)
        times_ave.append(np.average(times_single))
        times_single = []

    fig3 = plt.figure(figsize=(15, 12))
    plt.plot(Parray, times_ave)
    plt.xlabel('P')
    plt.ylabel('Time, seconds')
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 3, Function: nsearch_time \n Plot of average execution times against P \n with other variables fixed (M = 1000, N = 20).')
    plt.grid(True)
    plt.show()
    return None

if __name__ == '__main__':

    nsearch_time()  # Warning may take a little while to do plot 1
