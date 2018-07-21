'''
Written by: Jun Hao Hu
Date: 07/20/2018

Python program that uses a simple feed-forward Neural Network (NN) to solve Laplace's equation with
mixed Dirichlet boundary conditions, on the domain [0,1] x [0,1]
'''


'''
Import statements.
We will use the autograd module to calculate the Jacobians efficiently.
'''

import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr

from matplotlib import pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

'''
Define the parameters for the numerical method.
We will use a step size of 1/10 in both the x and y directions.
'''

nx, ny = 10, 10

dx, dy = 1./nx, 1./ny

x_space, y_space = np.linspace(0,1,nx), np.linspace(0,1,ny)

'''
Plot the analytic solution to the PDE. Refer to the paper for the analytic solution to the PDE.
'''

# Procedure that outputs the value of the analytic solution at some given x in our domain of solution.

def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
    	np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))

# Create the surface for which we are going to plot our solution on.

surface = np.zeros((ny, nx))

# Procedure that populates surface with the values the analytic solution takes on.

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])

# Create a new figure and plot the surface on the figure

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

# Set the limits of the figure axes

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# Plot the surface

plt.show()

'''
Define all the procedures that are going to be used to solve the PDE.
The following procedures will be defined:
1. sigmoid
2. neural_network
3. neural_network_x
4. A
5. psi_trial
6. loss_function

As a note, there should also be a function that returns the values of the function f, which is the right-hand side
of the PDE. However, since we are dealing with a homogeneous problem, we will skip defining this function.
'''

# Procedure that returns the value of the sigmoid function.

def sigmoid(x):
    return 1./(1.+np.exp(-x))

# Procedure that returns the value output by the sigmoid function. For this case, we are using the simple feed-forward architecture.

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

# Procedure that returns the value output by the sigmoid function, but takes in only a point x.

def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

# Procedure that returns the value of the function A in the trial function. The form of the function
# will differ from problem to problem. One area that this methods leaves for improvement is finding the
# best trial function according to the set up of the problem.

def A(x):
    return x[1] * np.sin(np.pi * x[0])

# Procedure that returns the value of the trial function at some point x, with parameters of the network.
# The network parameters will be intialized using some other method, and then will be learned by the network.

def psi_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1- x[1]) * net_out

# Procedure that returns the value of the loss function, defined in the paper. Note that we are using
# the L^2 loss function. One area of exploration is to explore the different types of loss function
# that can be used.
# We will call this function in order to perform gradient descent.

def loss_function(W, x, y):
    loss_sum = 0

    # Procedure that iterates through the entire domain and performs gradient descent.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])

            # Calculate the necessary information (first and second derivatives) needed for gradient
            # descent, for the neural network.

            net_out = neural_network(W, input_point)[0]
            net_out_jacobian = jacobian(neural_network_x)(input_point)
            net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)

            # Calculate the necessary information (first and second derivatives) needed for gradient
            # descent, for the trial function

            psi_t = psi_trial(input_point, net_out)
            psi_t_jacobian = jacobian(psi_trial)(input_point, net_out)
            psi_t_hessian = jacobian(jacobian(psi_trial))(input_point, net_out)

            # The Hessian is the matrix of second derivatives; hence, we need the requisite
            # pure second derivatives.

            grad_trial_d2x = psi_t_hessian[0][0]
            grad_trial_d2y = psi_t_hessian[1][1]

            # As a note, here is where we would call the function f and evaluate it at the input point.
            # As for this specific case, we have f = 0 identically, we do not call f.

            # Explicitly compute the error at this step, with the given loss function.

            err = ((grad_trial_d2x + grad_trial_d2y))**2
            loss_sum += err

    return loss_sum

'''
We now perform the actual training. We will use the method of gradient descent to perform the training.
'''

# Initialize the weights randomly. One area for improvement is to decide in what way the weights
# should be initialized.

W = [npr.randn(2, 10), npr.randn(10, 1)]
alph = 0.001

for i in range(100):
    print('We are on iteration', i)
    loss_grad = grad(loss_function)(W, x_space, y_space)

    W[0] = W[0] - alph * loss_grad[0]
    W[1] = W[1] - alph * loss_grad[1]

'''
We now plot the results and examine the error rate.
'''

# Print how much the loss function outputs

print(loss_function(W, x_space, y_space))

# Populate the surfaces. The first surface contains the analytic solution. The second surface contains
# the approximate solution.

surface_analytic = np.zeros((ny, nx))
surface_approx = np.zeros((ny, nx))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface_analytic[i][j] = analytic_solution([x, y])

for i, x in enumerate(x_space):
    for j,y in enumerate(y_space):
        net_out_surface = neural_network(W, [x, y])[0]
        surface_approx[i][j] = psi_trial([x, y], net_out_surface)

# Plot the surface and set the axes properties as required. This plots the analytic solution.

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface_analytic, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# Plot the surface and set the axes properties as required. This plots the approximate solution.

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface_approx, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()
