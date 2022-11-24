import numpy as np
import matplotlib.pyplot as plt



# this is the cost function
def f(x):
    return x*x


# this is the derivative of cost function
def df(x):
    return 2*x

# gradient descent algorithm -n is the number of iterations
# and alpha is the learning rate
def gradient_descent(start, end, n, alpha=0.1, momentum=0.0):
    # we track the results (x and f(X)) values as well
    x_values = []
    y_values = []
    # generate random starting point 
    x = np.random.uniform(start, end)

    for i in range(n):
        x = x - alpha * df(x) - momentum*x

        # we store the x and f(X) values
        x_values.append(x)
        y_values.append(f(x))
        print('#%d  f(%s) = %s' %(i, x, f(x)))

    return [x_values, y_values]

if __name__ =='__main__':
    solutions, scores = gradient_descent(-1,1,n=50,alpha=0.1, momentum=0.3)

    # sample input range uniformly at 0.1 increments to plot the function
    inputs = np.arange(-1, 1.1, 0.1)

    # create a plot of input vs result
    plt.plot(inputs, f(inputs))
    plt.plot(solutions, scores, '.-', color='green')
    plt.show()
