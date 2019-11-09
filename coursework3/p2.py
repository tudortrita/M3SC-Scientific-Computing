"""M345SC Homework 3, part 2
Tudor Trita Trita
CID: 01199397
"""
import numpy as np
import networkx as nx
import scipy.linalg as linalg


def growth1(G, params=(0.02, 6, 1, 0.1, 0.1), T=6):
    """
    Question 2.1
    Find maximum possible growth, G = e(t=T)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    growth (G): Maximum growth (Changed from G as it was annoying to
    repeat name of growth and graph).
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V

    Discussion:
    The system of equations can be turned to a linear system of ODEs of the
    form dy = A*y, where A is the matrix that includes the coefficients alpha,
    theta, gamma, kappa, tau and F. The solution to this system is given by
    y = expm(A*t)*y0, where expm is the matrix exponential and y0 is the
    initial condition. We note that the second sum involving F and SIV becomes
    tau*SIV.

    We are tasked to find the maximum growth of perturbations. It is useful to
    note that the 'perturbation energy' e(t) can be written as
    np.dot(y(t).T, y(t)), and so we are interested in finding an initial
    condition y0 such that (expm(A*t) * y0)**2 / |y0|**2 is maximised.

    This task can be achieved by using SVD as in the lecture notes (Lec.12),
    because we can find y0 by doing SVD on expm(A*t), and if we let the outputs
    of svd(expm(A*t)) be U, SIG, VT, then y0 can be found in VT[0], which is
    the evector corresponding to the largest evalue SIG[0]. The desired
    growth is simply SIG[0] squared.
    """
    alpha, theta, gamma, kappa, tau = params
    N = G.number_of_nodes()

    # Construct flux matrix
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1 / Pden
    Q = tau * np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F * np.outer(Q, Pden)

    # Initalising sums for speed
    gkt = gamma + kappa + tau
    akt = alpha + kappa + tau
    k_min_t = kappa - tau

    A = np.block([[F - np.eye(N)*gkt, alpha*np.eye(N), np.zeros((N, N))],
                  [theta*np.eye(N), F - np.eye(N)*akt, np.zeros((N, N))],
                  [-theta*np.eye(N), np.zeros((N, N)), F + k_min_t*np.eye(N)]])

    B = linalg.expm(A * T)  # Exponential matrix
    U, SIG, VH = linalg.svd(B)  # Singlar value decomposition of B
    growth = SIG[0]**2  # Largest eigenvalue squared
    y = VH[0]  # Eigenvector corresponding to largest evalues
    return growth, y


def growth2(G, params=(0.02, 6, 1, 0.1, 0.1), T=6):
    """
    Question 2.2
    Find maximum possible growth, G=sum(Ii^2)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    growth (G): Maximum growth (Changed from G as it was annoying to
    repeat name of growth and graph).
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V

    Discussion:
    To find growth2, we use a very similar approach to the one in the growth1
    function. This time, we are only intereted in the biggest growth of I.
    To achieve this, we can perform SVD on expm(A*t)[N:2*N], which is the
    part that corresponds to the vector I. We find the results in the same
    manner as in growth1.
    """
    alpha, theta, gamma, kappa, tau = params
    N = G.number_of_nodes()

    # Construct flux matrix
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1 / Pden
    Q = tau * np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F * np.outer(Q, Pden)

    akt = alpha + kappa + tau
    gkt = gamma + kappa + tau
    k_min_t = kappa - tau

    A = np.block([[F - np.eye(N)*gkt, alpha*np.eye(N), np.zeros((N, N))],
                  [theta*np.eye(N), F - np.eye(N)*akt, np.zeros((N, N))],
                  [-theta*np.eye(N), np.zeros((N, N)), F + k_min_t*np.eye(N)]])

    B = linalg.expm(A * T)

    Imat = B[N:2*N, :]  # Extracting rows which correspond to I
    U, SIG, VH = linalg.svd(Imat)  # svd performed on rows corresponding to I
    growth = SIG[0]**2
    y = VH[0]
    return growth, y


def growth3(G, params=(2, 2.8, 1, 1.0, 0.5), T=6):
    """
    Question 2.3
    Find maximum possible growth, G=sum(Si Vi)/e(t=0)
    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    growth (G): Maximum growth (Changed from G as it was annoying to
    repeat name of growth and graph).

    Discussion:
    To find growth3, we have to be a more careful about our approach
    than in finding growth1 and growth2. This time, we have to perform
    some linear algebra to get the correct value.

    Let B = expm(A*t), then if we let B1 = B[:N] and B3 = B[2*N:], these two
    matrices correspond to values of S and V respectively. We are interested
    in manipulating B1 and B3 to become symmetric, to be able to find highest
    eigenvalue.

    First we note that S.T*V = (B1*y0).T*(B3*y0), V.T*S = (B3*y0).T*(B1*y0)
    Then combining these, we have:

    S.T*V + V.T*S = (B1*y0).T*(B3*y0) + (B3*y0).T*(B1*y0)
                  = y0.T*B1.T*B3*y0 + y0.T*B3.T*B1*y0
                  = y0.T * (B1.T*B3 + B3.T*B1) * y0

    There are two key things to note here, first is that S.T*V = V.T*S, and
    the second is that B1.T*B3 + B3.T*B1 is symmetric.

    Thus S.T*V + V.T*S = 2*S.T*V, and therefore

    S.T*V = y0.T * (0.5*(B1.T*B3 + B3.T*B1)) * y0

    The LHS of this equation is precisely sum(S_i, T_i), which is what we want
    to find. Thus to find the maximum growth, we simply have to find the
    largest eigenvalue of 0.5*(B1.T*B3 + B3.T*B1) and we are done.
    """
    alpha, theta, gamma, kappa, tau = params
    N = G.number_of_nodes()

    # Construct flux matrix
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1 / Pden
    Q = tau * np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F * np.outer(Q, Pden)

    gkt = gamma + kappa + tau
    akt = alpha + kappa + tau
    k_min_t = kappa - tau

    A = np.block([[F - np.eye(N)*gkt, alpha*np.eye(N), np.zeros((N, N))],
                  [theta*np.eye(N), F - np.eye(N)*akt, np.zeros((N, N))],
                  [-theta*np.eye(N), np.zeros((N, N)), F + k_min_t*np.eye(N)]])

    B = linalg.expm(A * T)

    # We use a method of eigenvalues
    Smat = B[:N, :]
    Vmat = B[2 * N:, :]

    P = (np.matmul(Smat.T, Vmat) + np.matmul(Vmat.T, Smat))/2
    evals = linalg.eigvalsh(P)
    growth = max(evals)  # Find maximum eigenvalue which is growth.
    return growth


def Inew(D):
    """
    Question 2.4

    Input:
    D: N x M array, each column contains I for an N-node network

    Output:
    I: N-element array, approximation to D containing "large-variance"
    behavior

    Discussion:
    We are given some data 'D' as input, consisting of measurements at one
    time of I_i for M different organisms in the early stages of infection.

    To find I, we perform principal value decomposition on D. We begin
    by constructing a matrix A, which is equal to D minus the column
    means of D. We can then find the total variance of D by taking the
    trace of the dot product of A and A transposed, and then scaling
    it by dividing by N.

    We find this variance to be approx. 0.402 for the test dataset.

    To find Ihat, we begin by performing the SVD of A. The hint in the
    clarification of the coursework gives away how to find I. We want
    to find a unit vector Z in R^m such that I = B*Z, where B is a matrix.
    Furthermore, we know that C = B*B.T, where C is the covariance matrix.
    The matrix A found before fits this description, and thus B = A, and
    after performing the SVD of A, we know that VH contains unit vectors
    in R^m. Thus, we take the vector in VH which corresponds to the
    largest eigenvalue, namely VH[0], which will reflect the total
    variance in the system best.

    We can then find I = A*VH[0], and this vector is the one we need.
    We find that the variance of I is approx. 0.322, which is an estimate
    of approx. 80% of the total variance in the dataset.

    We also note that if we do I**squared, this approximates the row-variance
    of D, namely np.var(D, axis=1).
    """
    N, M = D.shape

    A = D - D.mean(axis=0)  # D scaled, subtracting column means

    VAR_total = np.trace(np.dot(A, A.T))/N  # Total variance in D
    print('Total Variance of D = ', VAR_total)  # Printing variance

    U, SIG, VH = linalg.svd(A)  # Singular Value Decomposition
    I = np.dot(A, VH[0])  # Calculating Ihat

    Ihat_var = np.var(I)
    print('Variance of Ihat = ', Ihat_var)
    return I


if __name__ == '__main__':
    N_in, M_in = 100, 5
    G_in = nx.barabasi_albert_graph(N_in, M_in, seed=1)
    g1, y1 = growth1(G_in)
    g2, y2 = growth2(G_in)
    g3 = growth3(G_in)
    D_in = np.loadtxt('q22test.txt')
    Iout = Inew(D_in)
