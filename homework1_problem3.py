import numpy as np
import time
import math
from typing import Tuple

import matplotlib.pyplot as plt

def problem3a():
    """
    returns derivative of the function: f(x) at that a given point x
    """
    def del_f(x):
        if x < -0.1:
            return -3*x**2
        if x < 3 and -0.1 <= x:
            return -0.03
        if x < 5 and 3 <= x:
            return -3*((x-3.1)**2)
        if x >= 5:
            return 10.83*(x-6)
        
    """
    f(x) definition
    """
    def function_x(x):
        if x < -0.1:
            return -x**3
        if x < 3 and -0.1 <= x:
            return -0.03*x - (1/500)
        if x >= 3 and x < 5:
            return -(x-31/10)**3 - 23/250
        if x >= 5:
            return 5.415*((x-6)**2) - (6183/500)

    def gradient_descent(x, y, learning_rate):
        if del_f(x) == 0:
            return x, y, True
        new_x = x - (learning_rate*del_f(x))
        new_y = function_x(new_x)
        return new_x, new_y, False

    # starting point
    x = -3
    y = function_x(x)

    learning_rates = [0.24, 1e-3, 1e-2, 1e-1, 1e0]
    x_vals = np.linspace(-4, 10, 400)
    f_vals = np.array([function_x(x) for x in x_vals])

    for learning_rate in learning_rates:
        plt.plot(x_vals, f_vals, label='f(x)')

        plt.plot(-3, function_x(-3), 'ro', label='start')

        for i in range(0, 100):
            learning_rate = learning_rate*0.97
            x, y, converged = gradient_descent(x, y, learning_rate)
            if converged:
                print(i, x)
                print('Converged')
                break
            plt.plot(x, y, 'rx', label='learning_rate = learning_rate*0.97')
            if learning_rate == 1e-3:
                plt.plot(x, y, 'rx', label='learning_rate = 1e-3')
            elif learning_rate == 1e-2:
                plt.plot(x, y, 'bx', label='learning_rate = 1e-2')
            elif learning_rate == 1e-1:
                plt.plot(x, y, 'gx', label='learning_rate = 1e-1')
            elif learning_rate == 1e0:
                plt.plot(x, y, 'yx', label='learning_rate = 1e0')
            
        plt.plot(x, y, 'go', label='end')
        plt.legend(['f(x)', 'start', f'decaying learning_rate of 0.24'])

        plt.show()
    
def problem3b():
    """
    observe f(x1, x2) = a1(x1 - c1)^2 + a2(x2 - c2)^2
    """
    a1 = 10.5
    a2 = 1
    c1 = 0.5
    c2 = 4.0

    def f(x1, x2):
        return a1 * (x1 - c1)**2 + a2 * (x2 - c2)**2

    # Generate a grid of points
    x1 = np.linspace(-4, 5, 400)
    x2 = np.linspace(-4, 5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    # Load the gradient descent sequence
    sequence = np.loadtxt('gradient_descent_sequence.txt')
    x_seq, y_seq = sequence[:, 0], sequence[:, 1]

    # Plotting the contour and the sequence
    plt.contour(X1, X2, Z, levels=50)
    plt.scatter(x_seq, y_seq, color='blue', s=10)
    plt.plot(x_seq, y_seq, color='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour plot of f(x1, x2) with Gradient Descent Sequence')
    plt.axis('equal')
    plt.xlim(-5, 5)  
    plt.ylim(-2, 6)  
    plt.axis('equal')
    plt.show()

def problem3c():
    """
    Observing gradient descent for f(x) = (2/3)*|x|^(3/2)
    """
    def func(x) -> float:
        return (2/3)*(np.abs(x)**(3/2))
    
    def gradient_descent(x, learning_rate) -> Tuple[float, float, bool]:
        if x >= 0:
            new_x = x - learning_rate * np.sqrt(x)
        else:
            new_x = x + learning_rate * np.sqrt(-x)
        if new_x == 0:
            return new_x, func(new_x), True
        else:
            return new_x, func(new_x), False

    learning_rates = [1, 1e-3, 1e-2, 1e-1 ]

    # Generate the function values
    x_vals = np.linspace(-3, 3, 10000)
    f_vals = np.array([func(x) for x in x_vals])

    # starting points
    x = 1
    y = func(x)
    plt.plot(x, y, 'ro', label='start')

    gradient_descent_pts = []
    for learning_rate in learning_rates:
        plt.plot(x_vals, f_vals, label='f(x)')
        for i in range(0, 1000):
            plt.plot(x, y, 'rx', label='learning_rate = learning_rate')
            x, y, converged = gradient_descent(x, learning_rate)
            gradient_descent_pts.append((x, y))
            if converged:
                print(i, x)
                print('Converged')
                plt.plot(x, y, 'go', label='end')
                break
        plt.title(f'learning_rate = {learning_rate}')
        plt.show()
    return gradient_descent_pts

def problem3d():
    def func(x) -> float:
        return x**3 + 3*x**2

    def gradient_descent(x, learning_rate) -> Tuple[float, float, bool]:
        new_x = x - learning_rate*(3*x**2 + 6*x)
        if new_x == 0:
            return new_x, func(new_x), True
        else:
            return new_x, func(new_x), False
        
    learning_rate = 1/3

    #starting point
    x = 1
    y = func(x)

    # Generate the function values
    x_vals = np.linspace(-3, 3, 10000)  
    f_vals = np.array([func(x) for x in x_vals])

    plt.plot(x_vals, f_vals, label='f(x)')

    plt.plot(x, y, 'ro', label='start')
    
    for i in range(0, 1000):
        print(x, y)
        x, y, converged = gradient_descent(x, learning_rate)
            
        if converged:
            print(i, x)
            print('Converged')
            break
        plt.plot(x, y, 'rx')
        
    plt.plot(x, y, 'go', label='end')
    plt.show()