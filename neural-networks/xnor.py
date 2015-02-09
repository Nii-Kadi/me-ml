#!/usr/bin/env python
import sys
import numpy as np

# Neural network for calculating x1 XNOR x2
#
# Architecture
# ------------
# Input Layer : 2 nodes (x1, x2) + bias node x0
# Layer 2     : 2 nodes (a1, a2) + bias node a0
# Output Layer: 1 node  
# 
# example of using backpropagation to learn
# weights for Theta1 (2x3) and Theta2 (1x3)
#
# x1  x2  |  a1  a2  | h(x)
# _________________________
#  0   0  |   0   1  |  1
#  0   1  |   0   0  |  0
#  1   0  |   0   0  |  0
#  1   1  |   1   0  |  1

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

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
        # row format: [ x1, x2, y ]
        data = np.matrix([ [0, 0, 1], 
                           [0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 1] ]) 
        train = np.vstack( [ data[np.random.randint(0,4)] for i in range(m) ] )
        a1 = train[:, 0:2]
        y  = train[:, -1]
        return a1, y

    def _forward(x, y, theta):
        m      = x.shape[0]
        theta1 = theta[0]
        theta2 = theta[1]

        bias  = np.ones(m).reshape(m,1)
        a1 = np.hstack([ bias, x ])

        z2 = a1 * theta1.transpose()
        a2 = sigmoid(z2)

        a2 = np.hstack([ bias, a2 ])
        z3 = a2 * theta2.transpose()
        a3 = sigmoid(z3)
        hx = vthreshold(a3)

        p1 = np.multiply(y, np.log1p(hx))
        p2 = np.multiply(1-y, np.log1p(1-hx))
        cost = -sum(p1+p2)/m 
        return cost, (a1, a2, a3)
    
    def _backward(input, outputs, thetas, y, nrows, alpha):
        pass

    print "training NN with dataset of %d rows, over %d epochs" % (m, epochs)
    theta1 = _init_theta(2,3)
    theta2 = _init_theta(1,3)
    print "theta1, initial\n", theta1
    print "theta2, initial\n", theta2
    print

    for e in range(epochs):
        x, y = _create_training_set(m)

        ### PROPAGATE FORWARD
        cost, (a1, a2, a3) = _forward(x, y, (theta1, theta2))
        print "epoch %d, cost: %f" % (e, cost)

        ### BACK-PROPAGATION
        gp3 = np.multiply(a3, 1-a3)
        d3  = np.multiply(-(y - vthreshold(a3)), gp3)
        dW2 = np.multiply(d3, a2)
        #print "dW theta 2"
        #print 0.5 * sum(dW2)/m

        gp2_1 = np.multiply(a2[:, 1], 1-a2[:, 1])
        d2_1  = np.multiply(np.multiply(theta2, d3), gp2_1)
        dW1_1 = np.multiply(d2_1, a1)
        #print "dW theta 1"
        #print 0.5 * sum(dW1_1)/m

        gp2_2 = np.multiply(a2[:, 2], 1-a2[:, 2])
        d2_2  = np.multiply(np.multiply(theta2, d3), gp2_2)
        dW1_2 = np.multiply(d2_2, a1)

        DT2 = 0.8 * sum(dW2)/m
        DT1 = np.vstack([ 0.8 * sum(dW1_1)/m, 0.8 * sum(dW1_2)/m ])

        theta2 -= DT2
        theta1 -= DT1

    print
    print "TEST"
    print "theta1, final\n", theta1
    print "theta2, final\n", theta2
    check = [(0,0), (0,1), (1,0), (1,1)]
    for p in check:
        result = calculate(p, theta1, theta2)
        print '%d AND %d = %d' % (p[0], p[1], result)

def calculate(tup, theta1, theta2):
    bias  = np.matrix([[1]])
    # Layer 1 -> 2 
    a1  = np.hstack([ bias, np.mat(tup) ])
    z2  = a1 * theta1.transpose()
    ## Layer 2 -> 3
    a2  = sigmoid(z2)
    a2  = np.hstack([ bias, a2 ])
    z3  = a2 * theta2.transpose()
    a3  = vthreshold(sigmoid(z3))
    return a3.item(0)

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
        theta1 = np.matrix([ [-30.0,  20.0,  20.0],
                             [ 10.0, -20.0, -20.0] ])
        theta2 = np.matrix([ [-10.0,  20.0,  20.0] ])
        x1, x2 = int(argv[2]), int(argv[3])
        result = calculate((x1, x2), theta1, theta2)
        print '%d XNOR %d = %d' % (x1, x2, result)
        return 0
    else:
        return usage()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
