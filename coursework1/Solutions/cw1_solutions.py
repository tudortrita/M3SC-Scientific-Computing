""" M345SC Coursework 1 Solutions

Example solutions for Coursework 1
"""

#Part 1

def ksearch(S,k,f,x):
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

    Discussion: Add analysis here
    """

    #Build dictionary of k-mers
    D = {}
    n = len(S)
    for i in range(n-k+1):
        if i%10**6==0: print('i=',i)
        key = S[i:i+k]
        if key in D:
            D[key].append(i)
        else:
            D[key]=[i]

    #Extract frequently-occurring k-mers
    L1,L2=[],[]
    for l,v in D.items():
        if len(v)>=f:
            L1.append(l)
            L2.append(v)

    #Find point-x mutations
    L3=[]
    for l in L1:
        L3.append(0)
        for c in "ACGT":
            if c != l[x]:
                l2 = l[:x] + c + l[x+1:]
                if l2 in D: L3[-1]+=len(D[l2])

    return L1,L2,L3
	
# Rolling Hash Table for Part 1:

def ksearch2(S,k,f,x):
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

    Discussion: Add analysis here
    """
    n=len(S)
    #Below k=10 or so, no need to hash
    prime = 9999889

    #Convert string to base4

    L = char2base4(S)

    #Initialize hash table
    T = [[] for i in range(prime)]

    #Precompute factor for rolling hash
    bm = 4**k%prime

    #Compute 1st hash
    ind=0
    hi = heval(L[ind:ind+k],4,prime)

    #---Build Hash Table--------
    #Iterate through base4 number
    for ind in range(0,n-k+1):
        #Check for hash collision
        if len(T[hi])==0:
            #if new, create new sublist containing: [index]
            T[hi].append([ind])
        else:
            #print("collision, ind=",ind,"hi=",hi)
            found=0
            for j,l in enumerate(T[hi]):
                Lind = l[0]
                if L[Lind:Lind+k]==L[ind:ind+k]:
                    T[hi][j].append(ind)
                    found=1
                    break
            if found==0: T[hi].append([ind])
        #if not, sequentially compare and append new sublist if new
        #or append to old sublist if not

        #update hash
        if ind<n-k: hi = (4*hi - (L[ind])*bm + (L[ind+k])) % prime

    L1=[]
    L2=[]

    #---Extract frequently-occurring k-mers----
    for t in T:
        if len(t)>0:
            for l in t:
                if len(l)>=f:
                    ind=l[0]
                    L1.append(S[ind:ind+k])
                    L2.append(l)

    #Find point-x mutations
    L3=[]
    for l in L1:
        L3.append(0)
        for c in "ACGT": #iterate through point mutations
            if c != l[x]:
                l2 = l[:x] + c + l[x+1:]
                l3 = char2base4(l2)
                #compute hash of mutation
                hi = heval(l3,4,prime)

                #check if hash is present in table
                if len(T[hi])>0:
                    #scan through table entry and find exact match
                    for l4 in T[hi]:
                        Lind = l4[0]
                        if S[Lind:Lind+k]==l2:
                            L3[-1]+=len(l4)
                            break

    return L1,L2,L3

def char2base4(S):
    """Convert gene test_sequence
    string to list of ints between 0 and 4
    """
    c2b = {}
    c2b['A']=0
    c2b['C']=1
    c2b['G']=2
    c2b['T']=3
    L=[]
    for s in S:
        L.append(c2b[s])
    return L



def heval(L,Base,Prime):
    """Convert list L to base-10 number mod Prime
    where Base specifies the base of L
    """
    f = 0
    for l in L[:-1]:
        f = Base*(l+f)

    h = (f + (L[-1])) % Prime
    return h

# Part 2

def nsearch(L,P,target):
    """Input:
    L: list containing *N* sub-lists of length M. Each sub-list
    contains M numbers (floats or ints), and the first P elements
    of each sub-list can be assumed to have been sorted in
    ascending order (assume that P<M). L[i][:p] contains the sorted elements in
    the i+1th sub-list of L
    P: The first P elements in each sub-list of L are assumed
    to be sorted in ascending order
    target: the number to be searched for in L

    Output:
    Lout: A list consisting of Q 2-element sub-lists where Q is the number of
    times target occurs in L. Each sub-list should contain 1) the index of
    the sublist of L where target was found and 2) the index within the sublist
    where the target was found. So, Lout = [[0,5],[0,6],[1,3]] indicates
    that the target can be found at L[0][5],L[0][6],L[1][3]. If target
    is not found in L, simply return an empty list (as in the code below)
    """

    Lout=[]

    #iterate through sublists in L
    for i,Lsub in enumerate(L):


        #run binary search on Lsub[:p]------------------
        #Set initial start and end indices for full sublist
        istart = 0
        iend = P-1

        #Iterate and contract to "active" portion of list
        while istart<=iend:

            imid = int(0.5*(istart+iend))

            if target==Lsub[imid]:
                Lout.append([i,imid])
                j=imid+1
                #Linear search within active portion of list--
                while target==Lsub[j]:
                    Lout.append([i,j])
                    j=j+1
                j = imid-1
                while target==Lsub[j]:
                    Lout.append([i,j])
                    j=j-1
                break
                #--------------------------
            elif target < Lsub[imid]:
                iend = imid-1
            else:
                istart = imid+1
        #Finished binary search
        #----------------------

        #Run linear search on Lsub[P:]-----------
        j = P
        for x in Lsub[P:]:
            if x==target: Lout.append([i,j])
            j=j+1
        #Finished linear search-------------

    return Lout
	
	
def nsearch_time(Nsize=11,Msize=11,Psize=11):
    """Analyze the running time of nsearch.
    Add input/output as needed, add a call to this function below to generate
    the figures you are submitting with your codes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time

    #Vary N, M,P fixed--------------
    Narray = np.logspace(1,4,Nsize,dtype=int)
    M=1000
    P=500
    dtarray_N = np.zeros(Nsize)
    for i,N in enumerate(Narray):
        print("i,N=",i,N)
        Linput = generate_input(N,M,P)
        t1=time()
        Lout = nsearch(Linput,P,-1)
        t2 = time()
        dtarray_N[i] = t2-t1

    #Display results and compare to expected linear trend
    plt.figure()
    plt.loglog(Narray,dtarray_N,'x--')
    x = np.logspace(1,4,1001)
    plt.plot(x,x/x[0]*dtarray_N[0],'k:')
    plt.xlabel('N')
    plt.ylabel('walltime (sec)')
    plt.title('Walltime vs N for M=%d,P=%d' %(M,P))
    plt.grid()
    plt.legend(('computation','linear'))
    #----------------------------------------------

    #Vary M-P, N,P fixed---------------------
    Marray = np.logspace(1,5,Msize,dtype=int)
    N=100
    P=500
    Marray += P
    dtarray_M = np.zeros(Msize)
    for i,M in enumerate(Marray):
        print("i,M=",i,M)
        Linput = generate_input(N,M,P)
        t1=time()
        Lout = nsearch(Linput,P,-1)
        t2 = time()
        dtarray_M[i]= t2-t1

    #Display results and compare to expected linear trend
    plt.figure()
    plt.loglog(Marray-P,dtarray_M,'x--')
    x = np.logspace(1,5,1001)
    plt.plot(x,x/x[-1]*dtarray_M[-1],'k:')
    plt.xlabel('M-P')
    plt.ylabel('walltime (sec)')
    plt.title('Walltime vs M-P for N=%d,P=%d' %(N,P))
    plt.grid()
    plt.legend(('computation','linear'))
    #-----------------------------

    #Vary P, M-P fixed
    Parray = np.logspace(3,7,Psize,dtype=int)
    N=10
    dtarray_P = np.zeros(Psize)
    for i,P in enumerate(Parray):
        M = P
        print("i,P,M=",i,P,M)
        Linput = generate_input(N,M,P)
        t1=time()
        Lout = nsearch(Linput,P,-1)
        t2 = time()
        dtarray_P[i]= t2-t1

    #Display results and compare to expected logarithmic trend
    plt.figure()
    plt.semilogx(Parray,dtarray_P,'x--')
    x = np.logspace(3,7,1001)
    plt.semilogx(x,np.log2(x)/np.log2(x[0])*dtarray_P[0],'k:')
    plt.xlabel('P')
    plt.ylabel('walltime (sec)')
    plt.title('Walltime vs P for N=%d,M-P=%d' %(N,M-P))
    plt.grid()
    plt.legend(('computation','log2(P)'))


    return Narray,dtarray_N,Marray,dtarray_M,Parray,dtarray_P #Modify as needed

	