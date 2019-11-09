"""M345SC Homework 2, Part 1
Tudor Trita Trita
CID: 01199397
MSci Mathematics Year 3
"""

############################################################################################################################################################################
# Part 1 Question 1: Scheduler, using a recursive DFS-like approach
############################################################################################################################################################################

def scheduler(L):
    """
    Question 1.1
    Schedule tasks using dependency list provided as input

    Input:
    L: Dependency list for tasks. L contains N sub-lists, and L[i] is a sub-list
    containing integers (the sub-list may also be empty). An integer, j, in this
    sub-list indicates that task j must be completed before task i can be started.

    Output:
    S: A list of integers corresponding to the schedule of tasks. L[i] indicates
    the day on which task i should be carried out. Days are numbered starting
    from 0.

    Discussion:

    My implementation uses a recursive-DFS implementation to aggresively search
    into the schedule L by thinking of L as a graph, where the nodes of the graph
    correspond to the tasks L[i] and the edges correspond to links between tasks
    which can be found in L[i][j].

    To perform the calculation of the schedule, first we note that if task I is scheduled
    for day 'a' and if task J depends on task I, then task I will have to be scheduled
    at day >= a + 1. In particular, if we find the highest day scheduled for task J's
    dependencies, and we let this = b, then task J will be scheduled for day b+1.

    Using this line of thought, we can efficiently move through the graph, visiting each
    node and edge only once. Thus, taking N = nodes in L and M = edges in L,
    then the complexity of the algorithm is O(N + M).
    """

    # Initialising parameters:
    N = len(L)  # Number of total tasks
    S = [0]*N   # List of schedule of tasks initialized (Initialy all set to 0 in case no dependencies everywhere)
    Tasks_to_explore = [False]*N  # Tasks explored: All false for now

    def tasker(List, S, Tasks_to_explore, task):
        """ Recursive function for part1 q1. """
        maximum = 0
        if Tasks_to_explore[task]:  # Check if the node has been explored
            return S, Tasks_to_explore
        Tasks_to_explore[task] = True  # Set as explored
        if List[task] == []:  # Check for no tasks pre-planned
            return S, Tasks_to_explore

        for dependency in List[task]:
            if not Tasks_to_explore[dependency]:
                S, Tasks_to_explore = tasker(List, S, Tasks_to_explore, dependency)  # Recursive call
            if maximum < S[dependency]:
                maximum = S[dependency]  # Calculating the maximum at each iteration
        S[task] = maximum + 1  # Setting current day
        return S, Tasks_to_explore

    for task in range(N):  # Iterating through each node
        S, Tasks_to_explore = tasker(L, S, Tasks_to_explore, task)  # Main call
    return S

############################################################################################################################################################################
# Part 1 Question 2 (i): findPath function, using a modified BFS implementation
############################################################################################################################################################################

def findPath(A, a0, amin, J1, J2):
    """
    Question 1.2 i)
    Search for feasible path for successful propagation of signal
    from node J1 to J2

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    a0: Initial amplitude of signal at node J1

    amin: If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine if the signal can successfully reach node J2 from node J1

    Output:
    L: A list of integers corresponding to a feasible path from J1 to J2.

    Discussion:

    The algorithm used here is very similar to the BFS code provided in the lectures.
    The key modification is that we introduce a parameter min_threshold inside the main
    while loop of the bfs algorithm to be able to discriminate between feasible and
    unfeasable paths. L2 and L4 are the same as in lectures, corresponding to arrays
    to keep track of already-explored nodes and paths that have been visited. Note
    that while the algorithm will output a path, there may be other feasible paths
    which are available too.

    The implementation uses a deque object to be able to add/remove elements in
    a list efficiently. The loop terminates if J2 has been found, as to
    keep iterating one J2 has been found makes no sense.

    Complexity: The algorithm iterates through each node until it finds J2, and
    for each node it will check every edge, so in the worst case scenario, where
    BFS will cover the whole graph, this is of order O(N + M). In reality, if J2
    is encountered early, the execution time will be shorter.
    """
    from collections import deque

    # Initialising Lists used:
    L2 = [0 for i in A]    # Array of keeping track of explored nodes
    L4 = [[] for i in A]   # Array of keeping track of paths
    L2[J1] = 1  # Mark initial value as explored
    L4[J1] = [J1] # Initial starting path (from J1)

    # Initialise queue as a deque object:
    Q = deque()
    Q.append(J1)

    min_threshold = amin/a0  # Lowest threshold value in adjacency list

    while Q: # While not empty
        x = Q.popleft()   # Extracting node from the front of the queue
        for node, Lij in A[x]:  # Iterating through A[x]
            # Following only executes when weight is large enough:
            if Lij >= min_threshold and L2[node] == 0:  # If Lij large enough + node is not explored
                L2[node] = 1    # Mark as explored
                L4[node].extend(L4[x])  # Add path to node x and 'node' to the path
                L4[node].append(node)   # Append for 'node'
                Q.append(node)  # Adding current node to queue
                if node == J2:  # Checking if we have reached J2 already
                    return L4[J2]  # If so, exit the function
    return L4[J2]

############################################################################################################################################################################
# Part 1 Question 2 (ii): a0min function, implemented using an adaptation Dijkstra's algorithm found in the lectures
############################################################################################################################################################################


def a0min(A, amin, J1, J2):
    """
    Question 1.2 ii)
    Find minimum initial amplitude needed for signal to be able to
    successfully propagate from node J1 to J2 in network (defined by adjacency list, A)

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    amin: Threshold for signal boost
    If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine min(a0) needed so the signal can successfully
    reach node J2 from node J1

    Output:
    (a0min,L) a two element tuple containing:
    a0min: minimum initial amplitude needed for signal to successfully reach J2 from J1
    L: A list of integers corresponding to a feasible path from J1 to J2 with
    a0=a0min
    If no feasible path exists for any a0, return output as shown below.

    Discussion:

    Implementation: The algorithm used here is a modified version of Dijkstra's algorithm
    seen in the lectures. The key difference is that while in lectures we tried to find
    the minimum weight of each edge, here we try to find the maximum L_ij which minimizes
    the amplitude a0.

    The implementation uses 2 dictionaries to keep track of explored nodes and unexplored
    nodes and moves from node to node by going through the path with largest L_ij. This
    L_ij is stored as a temporary distance, and we can carry on in the same way as Dijkstra's
    until we reach J2.

    Once J2 has been reached, the while loop breaks, and we can move on to extracting the
    full path using the previous findPath function. The inputs to findPath are A: the graph,
    a0min: calculated by dividing amin by dmax, and amin, J1 and J2.

    findPath returns the required path on which a0 has been minimized.

    The function then returns a0min and the path as wanted.

    Complexity: The Dijkstra's algorithm runs in O(N) time, and this modified version of it
    runs in O(N^2 + (N + M)), where the O(N^2) comes from the while loop and the O(N + M)
    comes from calling findPath. Asymptotically however, this is O(N^2).
    """

    N = len(A)  # Number of nodes in A
    dinit = -1  # Initial distance
    Edict = {}  # Explored nodes dictionary
    Udict = {}  # Unexplored nodes dictionary

    for i in range(N):
        Udict[i] = dinit  # Initialising every distance negatively
    Udict[J1] = 0  # Setting source distance equal to 0

    # Main Search:
    while Udict:  # While nodes are still unexplored
        #Find node with min d in Udict and move to Edict
        dmax = dinit    # Set minimum distance
        for node, weight in Udict.items():
            if weight > dmax:  # Finiding largest distance (opposite of Lecture's Dijkstra)
                dmax = weight  # New distance
                nmax = node  # Current node selected

        if nmax == J2:
            break

        Edict[nmax] = Udict.pop(nmax)  # Unexplored --> explored
        # Update other distances for unexplored nodes next to nmin
        for node, weight in A[nmax]:
            if node in Udict:  # Check if we have already explored this:
                dcomp = min(dmax, weight)
                if dcomp > Udict[node] or Udict[node] == -1:
                    Udict[node] = dcomp
                    if dmax == 0:
                        Udict[node] = weight

    if dmax == 0:
        return 'No path found'
    # Once we have reached J2:
    a0min = amin/dmax
    return a0min, findPath(A, a0min, amin, J1, J2)
