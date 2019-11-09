"""M345SC Homework 2, part 2
Tudor Trita Trita
CID: 01199397
MSci Mathematics Year 3
"""

import numpy as np
import networkx as nx
from scipy.integrate import odeint
import scipy
import matplotlib.pyplot as plt

############################################################################################################################################################################
# Part 2 Question 1 - Model1
############################################################################################################################################################################

def model1(G, x=0, params=(50, 80, 105, 71, 1, 0), tf=6, Nt=400, display=False):
    """
    Question 2.1
    Simulate model with tau = 0

    Input:
    G: Networkx graph

    params: contains model parameters, see code below.

    tf, Nt: Solutions Nt time steps from t=0 to t=tf (see code below)

    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    S: Array containing S(t) for infected node
    """

    # Initialising parameters:
    tarray = np.linspace(0, tf, Nt+1)
    y0 = [0.1, 0.05, 0.05]  # (V,I,S) initial condition
    alpha, theta0, theta1, gamma, kappa = params[:-1]

    # As tau = 0, flux matrix is equal to 0, therefore we ignore it in the equations
    # and concentrate on only 1 node

    def RHS_model1(vis, t):
        """
        Defines the differential equations for the situation in Part 2.1
        Arguments:
            vis: vector of the variables:
                    vis = [V_i, I_i, S_i]
            t: time
            params: Vector of parameters
        """
        theta = theta0 + theta1*(1 - np.sin(2*np.pi*t))

        dS_dt = alpha*vis[1] - (gamma + kappa)*vis[2]
        dI_dt = theta*vis[2]*vis[0] - (kappa + alpha)*vis[1]
        dV_dt = kappa*(1 - vis[0]) - theta*vis[2]*vis[0]

        dy = (dV_dt, dI_dt, dS_dt)
        return dy

    y = odeint(RHS_model1, y0, tarray)
    S = y[:, 2]

    if display:
        plt.figure(figsize=(13, 8))
        plt.plot(tarray, S, 'k')
        plt.xlabel('t')
        plt.ylabel('S(t)')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Function: model1 \n Plot of S(t) against t for model1')
        plt.show()
    return S

############################################################################################################################################################################
# Part 2 Question 2 - ModelN
############################################################################################################################################################################


def modelN(G, x=0, params=(50, 80, 105, 71, 1, 0.01), tf=6, Nt=400, display=False):
    """
    Question 2.2
    Simulate model with tau=0.01

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true
    Note: display has been changed to be an list of boolean variables corresponding
    to the plot of Smean and Svar separately

    x: node which is initially infected

    Output:
    Smean,Svar: Array containing mean and variance of S across network nodes at
                each time step.
    """
    alpha, theta0, theta1, gamma, kappa, tau = params
    tarray = np.linspace(0, tf, Nt+1)

    # Setting initial conditions corresponding to each node:
    N = nx.number_of_nodes(G)
    S0, I0, V0 = np.zeros(N), np.zeros(N), np.ones(N)
    S0[x], I0[x], V0[x] = 0.05, 0.05, 0.1
    SIV0 = np.concatenate([S0, I0, V0])  # Finalised initial condition

    # Computing F:
    Q = np.asarray([j for i, j in G.degree()])  # Matrix of degrees
    A = nx.adjacency_matrix(G)

    F = tau*scipy.sparse.diags(Q)*A*scipy.sparse.diags(1/(A*Q)) # Calculating F kept in scipy sparse form

    def RHS(y, t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        Discussion:

        For the discussion, we take N = no. of nodes in G, M = no. of edges in G.

        Below is calculated as the following:
        Scalar*Vector takes N operations
        Matrix*Vector takes 4M operations (no. of edges due to sparse format):
        One addition and one mult per entry, and there are 2M entries.
        Scalar+Vector takes N operations
        Scalar+Scalar takes 1 operation

        Preambles:
        1. Unpacking S, I and V takes 3N operations (helped by precomputing two_n)
        2. Calculating theta takes 6 operations
        3. Precomputing theta*S*V takes 2N operations
        Total: 5N + 6 operations.

        Computing dSi_dt, dIi_dt, dVi_dt:

        dSi_dt: N + N + N + N + 2M = 4N + 4M operations.
        dIi_dt: N + N + N + 2M = 3N + 4M operations.
        dVi_dt: (N + N) + N + N + 2M + N = 5N + 4M operations.
        Total to compute dy: 12(N + M) operations.

        Total arithmetic operations in RHS: 14N + 12M + 6.
        Therefore the complexity is O(N + M).
        (Note, assignment operations have not been taken into account.
        If they were, it would add 6N + 2 op.)

        For added speed, we use the fact that the sum of the columns of F is equal
        to tau, precompute F, gamma + kappa + tau, kappa + alpha + tau and 2pi.
        """

        #Unpacking yvalues to S, I and V
        S = y[:N]
        I = y[N:two_n]
        V = y[two_n:]

        # Theta for timestep t
        theta = theta0 + theta1*(1 - np.sin(pi_2*t))
        tSV = theta*S*V

        # Notation more compact than formulas in the notes due to precomputing
        # and Sum(F_ij) = tau:
        dy[:N] = alpha*I - gkt*S + F*S
        dy[N:two_n] = tSV - kat*I + F*I
        dy[two_n:] = kappa*(1 - V) - tSV + F*V - tau*V
        return dy

    # Initialising dy for the RHS function and precomputing constants for RHS
    dy = np.zeros(3*N)
    two_n = 2*N # Prevents from calculating 2*N every timestep inside RHS
    gkt = gamma + kappa + tau
    kat = kappa + alpha + tau
    pi_2 = 2*np.pi

    y = odeint(RHS, SIV0, tarray)
    Smean = np.mean(y[:, :N], axis=1)
    Svar = np.var(y[:, :N], axis=1)

    if display:
        plt.figure(figsize=(10, 5))
        plt.plot(tarray, Smean, 'k')
        plt.xlabel('t')
        plt.ylabel('Mean of S(t)')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Function: modelN \n Plot of <S(t)> against t')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(tarray, Svar, 'k')
        plt.xlabel('t')
        plt.ylabel('Variance of S(t)')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Function: modelN \n Plot of Variance of S(t) against t')
        plt.show()

    return Smean, Svar

############################################################################################################################################################################
# Part 2 Question 3 - Diffusion Question
############################################################################################################################################################################


def diffusion():
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.

    Discussion:

    My discussion will mostly involve talking about the mean and variance of S, I and V
    for different inputs of tau, theta0 and a different initial condition comparing
    both of the models. All my plots have D=tau and are execute on a single
    BA(100, 5) network.

    From Figure 1:

    Figure 1 illustrates changes to the mean and variance for both modelN (with new parameters)
    and diffusion models for different values of tau.

    1.1 and 1.2: we can see that the mean of S is constant at 0.005
    for varying values of tau. This is to be expected, as when alpha, kappa, theta1
    and gamma are zero, there is no change in the amount of Survivors in the system,
    thus the mean across the whole graph will remain constant, and this value is:
    Initial condition at node i=x/no.of nodes in the graph.

    2.1 and 2.2: These plots show how the variance changes with time for different tau.
    One immediate trend we see is that the bigger tau is, the variance will drop to zero
    faster. Regardless, in all cases of tau, the variance always tends to zero for both
    models. Another thing we can see is that the variance in the diffusion model
    drops much faster than in the modelN model, thus suggesting that if both model's
    variance were to decrease at similar time, D would have to equal approx. tau/10.
    The reason why the variance tends to zero for S(t) as t goes to infinity is that
    as the model is run, the Spreaders 'diffuse' throughout the network, and eventually
    reach stability, where Spreaders do not move around nodes anymore, bringing down the
    variance.

    From Figure 2:
    Figure 2 illustrates how changing the initial infected node in the model affects
    the mean and variance of S. Here, I picked 3 cases: x_min - node with least degree,
    x_random - a random node, x_max - node with the biggest degree.

    Again, the mean of S is constant here no matter the starting point for the same
    reasons as above.

    The variance of S is a more interesting story. For the modelN model, the variance
    follows a similar curve for all different x's. For the diffusion model, it is
    a different story. The larger the degree of the starting point, the faster
    the variance will drop to zero, showing that the spreaders 'spread' more
    succesfully if they start from a node with lots of connections. This is not
    the case for modelN, thus indicating that the model doesn't reflect
    diffusivity 'very well'.

    Moving on to Figure 3:

    This figure contains 12 subplots. Each row corresponds to S, I and V respectively and
    each column corresponds to Smean for modelN, Smean for diffusion, Svar for modelN and
    Svar for diffusion

    The top row is similar to Figure 1 so I will not look into it.

    Looking at plots 2.1 and 3.1, we can see that asymptotically, I(t) tends to
    1-0.005 as t goes to infinity, and V(t) tends to zero as t goes to infinity.
    Interestingly, the curve of V(t) = 0.995 - I(t), thus this is the relationship
    between them.

    Plots 1.4, 2.4 and 3.4 are the same, as expected because the equations for diffusion
    are the same.

    The most interesting plots in this figure are 2.3 and 3.3.

    Looking at 3.3, we can see that the variance of V(t) follows what appears to be
    a Gamma distribution-like shape. This is because, for small t, the change from
    Vulnerable to Infected is fastest, and therefore the variance reflects on this trend.

    More interestingly still is plot 2.3. This shows that the variance of I(t)
    actually increases, approaching a value of approx. 2.1, and stabilises there
    as t goes to infinity. This can be explained in the following way: the solution
    to the differential equation dI_dt given in the coursework description has a
    period behaviour, sinesoidal-like (but sharper) and this is what causes the
    variance to be constant for large t.

    This claim is backed up by Figure 4. Here I plotted the behaviour of I(t) for a
    random node (x=11 in this case). We can see the behaviour explained above, which
    explains the variance of Figure 4, plot 2.3.

    Other linear trends which I have not included a plot for, but that I found, was that
    varying theta0 doesn't affect the plots of Smean or Svar.
    """

    # Parameters and graph:
    alpha, theta0, theta1, gamma, kappa = (0, 80, 0, 0, 0)
    G_diff = nx.barabasi_albert_graph(100, 5)
    tf1 = 10
    tf2 = 70
    Nt = 400

    ############################################################################################################################################################################
    # Figure 1: Plots of Smean for modelN and Diffusion varying Tau
    ############################################################################################################################################################################

    tarray1 = np.linspace(0, tf1, Nt+1)
    tarray2 = np.linspace(0, tf2, Nt+1)

    colours = ['b', 'g--', 'r:', 'c', 'm--', 'y:', 'k:']

    tau_array = [0.03, 0.05, 0.1, 0.2, 0.5, 1, 2]

    fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    for i, tau in enumerate(tau_array):
        Smean1, Svar1 = modelN(G_diff, params=(alpha, theta0, theta1, gamma, kappa, tau), tf=tf2)
        Smean2, Svar2 = model_diff1(G_diff, tau, tf=tf1)
        axes[0, 0].plot(tarray2, np.round(Smean1, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird errors (terms at e-15)
        axes[0, 1].plot(tarray1, np.round(Smean2, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird errors (terms at e-15)
        axes[1, 0].plot(tarray2, Svar1, colours[i], label=('$τ = $' + str(tau)))
        axes[1, 1].plot(tarray1, Svar2, colours[i], label=('$τ = $' + str(tau)))

    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('<S(t)>')
    axes[0, 0].grid(True)
    axes[0, 0].set_title('1.1: Mean of S(t) for ModelN')

    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('<S(t)>')
    axes[0, 1].grid(True)
    axes[0, 1].set_title('1.2: Mean of S(t) for Diffusion Model')

    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('Var(S(t))')
    axes[1, 0].grid(True)
    axes[1, 0].set_title('2.1: Variance of S(t) for ModelN')

    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('Var(S(t))')
    axes[1, 1].grid(True)
    axes[1, 1].set_title('2.2: Variance of S(t) for Diffusion Model')

    fig1.suptitle('Name: Tudor Trita Trita, CID:01199397 \n Figure 1, Function: diffusion \n Plots of Mean and Variance of S for ModelN and Diffusion Model ($θ _0 = 80$)')
    plt.show()

    ############################################################################################################################################################################
    # Figure 2: Illustrating effect of choosing a different node with various connections, illustrating behaviour of well-connected nodes
    ############################################################################################################################################################################

    tf1 = 300
    tf2 = 75
    tarray1 = np.linspace(0, tf1, Nt+1)
    tarray2 = np.linspace(0, tf2, Nt+1)

    x_max = max(dict(G_diff.degree()).keys(), key=(lambda k: dict(G_diff.degree())[k]))  # Node with highest degree
    x_min = min(dict(G_diff.degree()).keys(), key=(lambda k: dict(G_diff.degree())[k]))  # Node with lowest degree
    x_random = np.random.choice(range(100)) # Random node for comparison
    x_array = [x_min, x_random, x_max]
    deg_array = [G_diff.degree(x_min), G_diff.degree(x_random), G_diff.degree(x_max)]
    tau = 0.01
    fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    for i, y in enumerate(x_array):
        Smean1, Svar1 = modelN(G_diff, x=y, params=(alpha, theta0, theta1, gamma, kappa, tau), tf=tf1)
        Smean2, Svar2 = model_diff1(G_diff, tau, x=y, tf=tf2)
        axes[0, 0].plot(tarray1, np.round(Smean1, 6), colours[i], label=('$x = $' + str(y) + ', deg = ' + str(deg_array[i])))
        axes[0, 1].plot(tarray2, np.round(Smean2, 6), colours[i], label=('$x = $' + str(y) + ', deg = ' + str(deg_array[i])))
        axes[1, 0].plot(tarray1, Svar1, colours[i], label=('$x = $' + str(y) + ', deg = ' + str(deg_array[i])))
        axes[1, 1].plot(tarray2, Svar2, colours[i], label=('$x = $' + str(y) + ', deg = ' + str(deg_array[i])))

    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('<S(t)>')
    axes[0, 0].grid(True)
    axes[0, 0].set_title('1.1: Mean of S(t) for ModelN')

    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('<S(t)>')
    axes[0, 1].grid(True)
    axes[0, 1].set_title('1.2: Mean of S(t) for Diffusion Model')

    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('Var(S(t))')
    axes[1, 0].grid(True)
    axes[1, 0].set_title('1.3: Variance of S(t) for ModelN')

    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('Var(S(t))')
    axes[1, 1].grid(True)
    axes[1, 1].set_title('1.4: Variance of S(t) for Diffusion Model')

    fig2.suptitle('Name: Tudor Trita Trita, CID:01199397 \n Figure 2, Function: diffusion \n Plots of Mean and Variance of S for ModelN and Diffusion Model ($ τ = D= 0.01$)')
    plt.show()

    ############################################################################################################################################################################
    # Figure 3: Illustrating effect of other parameters (I, V) for different tau
    ############################################################################################################################################################################

    fig3, axes = plt.subplots(nrows=3, ncols=4, figsize=(35, 17))
    tf1 = 65
    tf2 = 15
    tau_array = [0.05, 0.1, 0.2, 0.5, 1, 2]
    tarray1 = np.linspace(0, tf1, Nt+1)
    tarray2 = np.linspace(0, tf2, Nt+1)


    for i, tau in enumerate(tau_array):
        Smean1, Svar1, Imean1, Ivar1, Vmean1, Vvar1 = modelN_SIV(G_diff, params=(alpha, theta0, theta1, gamma, kappa, tau), tf=tf1)
        Smean2, Svar2, Imean2, Ivar2, Vmean2, Vvar2 = model_diff2(G_diff, tau, tf=tf2)
        axes[0, 0].plot(tarray1, np.round(Smean1, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird scale (terms at e-15)
        axes[0, 1].plot(tarray2, np.round(Smean2, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird scale (terms at e-15)
        axes[0, 2].plot(tarray1, Svar1, colours[i], label=('$τ = $' + str(tau)))
        axes[0, 3].plot(tarray2, Svar2, colours[i], label=('$τ = $' + str(tau)))
        axes[1, 0].plot(tarray1, np.round(Imean1, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird scale (terms at e-15)
        axes[1, 1].plot(tarray2, np.round(Imean2, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird scale (terms at e-15)
        axes[1, 2].plot(tarray1, Ivar1, colours[i], label=('$τ = $' + str(tau)))
        axes[1, 3].plot(tarray2, Ivar2, colours[i], label=('$τ = $' + str(tau)))
        axes[2, 0].plot(tarray1, np.round(Vmean1, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird scale (terms at e-15)
        axes[2, 1].plot(tarray2, np.round(Vmean2, 6), colours[i], label=('$τ = $' + str(tau)))    # Rounding due to weird scale (terms at e-15)
        axes[2, 2].plot(tarray1, Vvar1, colours[i], label=('$τ = $' + str(tau)))
        axes[2, 3].plot(tarray2, Vvar2, colours[i], label=('$τ = $' + str(tau)))

    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('<S(t)>')
    axes[0, 0].grid(True)
    axes[0, 0].set_title('1.1: Mean of S(t) for ModelN')

    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('<S(t)>')
    axes[0, 1].grid(True)
    axes[0, 1].set_title('1.2: Mean of S(t) for Diffusion Model')

    axes[0, 2].legend(loc='upper right')
    axes[0, 2].set_xlabel('t')
    axes[0, 2].set_ylabel('Var(S(t))')
    axes[0, 2].grid(True)
    axes[0, 2].set_title('1.3: Var(S(t)) for ModelN')

    axes[0, 3].legend(loc='upper right')
    axes[0, 3].set_xlabel('t')
    axes[0, 3].set_ylabel('Var(S(t))')
    axes[0, 3].grid(True)
    axes[0, 3].set_title('1.4: Var(S(t)) for Diffusion Model')

    ######

    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('<I(t)>')
    axes[1, 0].grid(True)
    axes[1, 0].set_title('2.1: Mean of I(t) for ModelN')

    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('<I(t)>')
    axes[1, 1].grid(True)
    axes[1, 1].set_title('2.2: Mean of I(t) for Diffusion Model')

    axes[1, 2].legend(loc='upper right')
    axes[1, 2].set_xlabel('t')
    axes[1, 2].set_ylabel('Var(I(t))')
    axes[1, 2].grid(True)
    axes[1, 2].set_title('2.3: Var(I(t)) for ModelN')

    axes[1, 3].legend(loc='upper right')
    axes[1, 3].set_xlabel('t')
    axes[1, 3].set_ylabel('Var(I(t))')
    axes[1, 3].grid(True)
    axes[1, 3].set_title('2.4: Var(I(t)) for Diffusion Model')

    ###########

    axes[2, 0].legend(loc='upper right')
    axes[2, 0].set_xlabel('t')
    axes[2, 0].set_ylabel('<V(t)>')
    axes[2, 0].grid(True)
    axes[2, 0].set_title('3.1: Mean of V(t) for ModelN')

    axes[2, 1].legend(loc='upper right')
    axes[2, 1].set_xlabel('t')
    axes[2, 1].set_ylabel('<V(t)>')
    axes[2, 1].grid(True)
    axes[2, 1].set_title('3.2: Mean of V(t) for Diffusion Model')

    axes[2, 2].legend(loc='upper right')
    axes[2, 2].set_xlabel('t')
    axes[2, 2].set_ylabel('Var(V(t))')
    axes[2, 2].grid(True)
    axes[2, 2].set_title('3.3: Var(V(t)) for ModelN')

    axes[2, 3].legend(loc='upper right')
    axes[2, 3].set_xlabel('t')
    axes[2, 3].set_ylabel('Var(V(t))')
    axes[2, 3].grid(True)
    axes[2, 3].set_title('3.4: Var(V(t)) for Diffusion Model')

    fig3.suptitle('Name: Tudor Trita Trita, CID:01199397 \n Figure 3, Function: diffusion \n Plots of Mean and Variance of S, I and V for ModelN and Diffusion Model ($θ _0 = 80$)')
    plt.show()

    ############################################################################################################################################################################
    # Figure 4: Illustrating oscillations of I_i in modelN
    ############################################################################################################################################################################

    Nt = 400
    tf = 20
    y = modelN_SIV2(G_diff, tf=tf, Nt=Nt)
    tau = 0.5
    tarray = np.linspace(0, tf, Nt+1)

    fig4 = plt.figure(figsize=(13, 8))
    plt.plot(tarray, y[:, 110], 'b:')
    plt.ylabel('I(t)')
    plt.xlabel('t')
    plt.grid(True)
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 4, Function: diffusion \n Plot of I_11 against t to illustrate fluctuatory behaviour.')
    plt.show()
    return None

############################################################################################################################################################################
# Helper Functions and Diffusion model to be called at diffusion()
############################################################################################################################################################################


def model_diff1(G, D, x=0, SIV_input=(0.05, 0.05, 0.1), tf=6, Nt=400):
    """ Function for the diffusion model.
    """

    L = nx.laplacian_matrix(G)
    N = nx.number_of_nodes(G)
    S0 = np.zeros(N)
    S0[x] = SIV_input[0]
    tarray = np.linspace(0, tf, Nt+1)

    def RHS_diff(y, t):
        """ RHS for the Laplacian model.
        """
        return -D*L*y

    y = odeint(RHS_diff, S0, tarray)
    Smean = np.mean(y[:, :N], axis=1)
    Svar = np.var(y[:, :N], axis=1)
    return Smean, Svar


def model_diff2(G, D, x=0, SIV_input=(0.05, 0.05, 0.1), tf=6, Nt=400):
    """ Function for the diffusion model.
    """

    L = nx.laplacian_matrix(G)
    N = nx.number_of_nodes(G)
    S0, I0, V0 = np.zeros(N), np.zeros(N), np.ones(N)
    S0[x], I0[x], V0[x] = SIV_input
    SIV0 = np.concatenate([S0, I0, V0])
    tarray = np.linspace(0, tf, Nt+1)

    def RHS_diff2(y, t):
        """ RHS for the Laplacian model.
        """
        dy[:N] = -D*L*y[:N]
        dy[N:two_n] = -D*L*y[N:two_n]
        dy[two_n:] = -D*L*y[two_n:]
        return dy

    dy = np.zeros(3*N)
    two_n = 2*N
    y = odeint(RHS_diff2, SIV0, tarray)
    Smean = np.mean(y[:, :N], axis=1)
    Imean = np.mean(y[:, N:2*N], axis=1)
    Vmean = np.mean(y[:, 2*N:], axis=1)
    Svar = np.var(y[:, :N], axis=1)
    Ivar = np.var(y[:, N:2*N], axis=1)
    Vvar = np.var(y[:, 2*N:], axis=1)

    return Smean, Svar, Imean, Ivar, Vmean, Vvar


def modelN_SIV(G, x=0, params=(50, 80, 105, 71, 1, 0.01), tf=6, Nt=400):
    """ Function that returns mean and variance of all parameters."""
    alpha, theta0, theta1, gamma, kappa, tau = params
    tarray = np.linspace(0, tf, Nt+1)

    # Setting initial conditions corresponding to each node:
    N = nx.number_of_nodes(G)
    S0, I0, V0 = np.zeros(N), np.zeros(N), np.ones(N)
    S0[x], I0[x], V0[x] = 0.05, 0.05, 0.1
    SIV0 = np.concatenate([S0, I0, V0])  # Finalised initial condition

    # Computing F:
    Q = np.asarray([j for i, j in G.degree()])  # Matrix of degrees
    A = nx.adjacency_matrix(G)
    F = tau*scipy.sparse.diags(Q)*A*scipy.sparse.diags(1/(A*Q)) # Calculating F

    def RHS3(y, t):
        #Unpacking yvalues to S, I and V
        S = y[:N]
        I = y[N:two_n]
        V = y[two_n:]

        # Theta for timestep t
        theta = theta0 + theta1*(1 - np.sin(pi_2*t))
        tSV = theta*S*V

        dy[:N] = alpha*I - gkt*S + F*S
        dy[N:two_n] = tSV - kat*I + F*I
        dy[two_n:] = kappa*(1 - V) - tSV + F*V - tau*V

        return dy

    # Initialising dy for the RHS function
    dy = np.zeros(3*N)
    two_n = 2*N # Prevents from calculating 2*N every timestep inside RHS
    gkt = gamma + kappa + tau
    kat = kappa + alpha + tau
    pi_2 = 2*np.pi

    y = odeint(RHS3, SIV0, tarray)
    Smean = np.mean(y[:, :N], axis=1)
    Imean = np.mean(y[:, N:2*N], axis=1)
    Vmean = np.mean(y[:, 2*N:], axis=1)
    Svar = np.var(y[:, :N], axis=1)
    Ivar = np.var(y[:, N:2*N], axis=1)
    Vvar = np.var(y[:, 2*N:], axis=1)
    return Smean, Svar, Imean, Ivar, Vmean, Vvar

def modelN_SIV2(G, x=0, params=(50, 80, 105, 71, 1, 0.01), tf=6, Nt=400):
    """ Function that returns y"""
    alpha, theta0, theta1, gamma, kappa, tau = params
    tarray = np.linspace(0, tf, Nt+1)

    # Setting initial conditions corresponding to each node:
    N = nx.number_of_nodes(G)
    S0, I0, V0 = np.zeros(N), np.zeros(N), np.ones(N)
    S0[x], I0[x], V0[x] = 0.05, 0.05, 0.1
    SIV0 = np.concatenate([S0, I0, V0])  # Finalised initial condition

    # Computing F:
    Q = np.asarray([j for i, j in G.degree()])  # Matrix of degrees
    A = nx.adjacency_matrix(G)
    F = tau*scipy.sparse.diags(Q)*A*scipy.sparse.diags(1/(A*Q)) # Calculating F

    def RHS4(y, t):
        #Unpacking yvalues to S, I and V
        S = y[:N]
        I = y[N:two_n]
        V = y[two_n:]

        # Theta for timestep t
        theta = theta0 + theta1*(1 - np.sin(pi_2*t))
        tSV = theta*S*V

        dy[:N] = alpha*I - gkt*S + F*S
        dy[N:two_n] = tSV - kat*I + F*I
        dy[two_n:] = kappa*(1 - V) - tSV + F*V - tau*V

        return dy

    # Initialising dy for the RHS function
    dy = np.zeros(3*N)
    two_n = 2*N # Prevents from calculating 2*N every timestep inside RHS
    gkt = gamma + kappa + tau
    kat = kappa + alpha + tau
    pi_2 = 2*np.pi

    y = odeint(RHS4, SIV0, tarray)
    return y


if __name__ == '__main__':
    diffusion()
