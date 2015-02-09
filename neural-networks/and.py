#!/usr/bin/env python
import sys
import numpy as np

# Neural network for calculating x1 AND x2
#
# Architecture
# ------------
# Layer 1 (input)  : 2 nodes (x1, x2) + bias node x0
# Layer 2 (output) : 1 node
# 
# example of using backpropagation to learn
# weights for theta (1x3)
#
# x1  x2  | h(x)
# ______________
#  0   0  |  0
#  0   1  |  0
#  1   0  |  0
#  1   1  |  1
#
# Usage
# -----
#   and.py train [m] [epochs]
#   - initializes theta to random values
#   - creates training set of m randomly-selected values in table above
#   - propagates forward
#   - calculates error and back-propagates
#   - outputs test using updated theta

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z*0.3))

def threshold(n):
    if n > 0.5:
        return 1
    else:
        return 0
vthreshold = np.vectorize(threshold, otypes=[np.dtype('f8')]) 

def train(m, epochs):
    def _init_theta(m, n):
        EPS = 0.0001
        return np.random.random((m,n)) * (2.0*EPS) - EPS

    def _create_training_set(m):
        # row format:    [ x1, x2, y ]
        data = np.matrix([ [0, 0, 0], 
                           [0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 1] ]) 
        train = np.vstack( [ data[np.random.randint(0,4)] for i in range(m) ] )
        bias  = np.ones(m).reshape(m,1)
        a1 = np.hstack([ bias, train[:, 0:2] ])
        y  = train[:, -1]
        return a1, y

    def _forward(x, y, theta):
        z2 = x * theta.transpose()
        a2 = sigmoid(z2)
        hx = vthreshold(a2)
        m  = x.shape[0]
        p1 = np.multiply(y, np.log1p(hx))
        p2 = np.multiply(1-y, np.log1p(1-hx))
        cost = -sum(p1+p2)/m 
        return cost, a2

    def _backward(inputs, y, alpha=0.5):
        a1, a2 = inputs
        m = a1.shape[0]
        dW = np.zeros((3,1))
        for i in range(m):
            hx = vthreshold(a2[i])
            d2 = hx - y[i]
            dW += a1[i].transpose() * d2
        dW = dW.transpose()
        print '=================='
        print "dW\n", dW/m
        print '=================='
        print
        return dW

    def _gradient_approx(x, y, theta, eps=0.0001):
        m = x.shape[0]
        print '******************'
        print "m\n", m
        for i in range(m):
            grad_approx = np.zeros(theta.shape)
            for j in range(3):
                theta_plus = theta.copy()
                theta_plus[0,j] += eps
                theta_minus = theta.copy()
                theta_minus[0,j] -= eps
                c_plus, foo  = _forward(x[i], y[i], theta_plus)
                c_minus, bar = _forward(x[i], y[i], theta_minus)
                grad_approx[0,j] = (c_plus - c_minus)/ (2*eps)
                #print c_plus
                #print c_minus
                #print 2*eps
            print grad_approx/m
        print '******************'

    print "training NN with dataset of %d rows, over %d epochs" % (m, epochs)
    theta = _init_theta(1,3)
    print "theta, initial\n", theta
    print

    for e in range(epochs):
        a1, y = _create_training_set(m)

        ### PROPAGATE FORWARD
        cost, a2 = _forward(a1, y, theta)
        print "epoch %d, cost: %f" % (e+1, cost)

        ### BACK-PROPAGATION
        d = _backward((a1, a2), y, 0.5)

        ### GRADIENT CHECK!
        #print 
        #print ".... gradient check!"
        #_gradient_approx(a1, y, theta)
        #print "...."
        #
        theta -= (0.5 * d/m)
        print '******************'
        print "theta, updated\n", theta
        print '******************'
        print


    print
    print "TEST"
    print "theta, final\n", theta
    check = [(0,0), (0,1), (1,0), (1,1)]
    for p in check:
        result = calculate(p, theta)
        print '%d AND %d = %d' % (p[0], p[1], result)

def calculate(tup, theta):
    a1 = np.hstack([ [[1]], np.mat(tup) ])
    z2 = a1 * theta.transpose()
    a2 = vthreshold(sigmoid(z2))
    return a2.item(0)

def usage():
    sys.stderr.write("Usage: %s <train [m] [epochs]|calculate [x1] [x2]>" % (argv[0],))
    return 1

def main(argv):
    if len(argv) < 2:
        return usage()

    if argv[1]=='train':
        m      = int(argv[2])
        epochs = int(argv[3])
        train(m, epochs)
        return 0
    elif argv[1]=='calculate':
        theta = np.matrix([ [-30.0,  20.0,  20.0] ])
        x1, x2 = int(argv[2]), int(argv[3])
        result = calculate((x1, x2), theta)
        print '%d AND %d = %d' % (x1, x2, result)
        return 0
    else:
        return usage()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
