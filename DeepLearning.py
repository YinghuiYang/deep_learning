import numpy as np


# weight initialization
def weight_initialization(layer_dimension):
    w = []
    for i in range(1, len(layer_dimension)):
        w.append(np.random.rand(layer_dimension[i], layer_dimension[i-1])/5)

    return w

# bias initialization
def b_initialization(layer_dimension):
    b = []
    for i in range(1,len(layer_dimension)):
        b.append(np.zeros([layer_dimension[i], 1]))
    return b

def cost(a, y):
    cost =  - np.sum(y * np.log(a) + (1-y)*np.log(1-a))/len(y)
    return cost

def activation(string,x):
    if string=="relu":
        return abs(x)*(x>0)
    elif string=="tanh":
        return np.tanh(x)
    elif string=="sigmoid":
        return 1 / (1 + np.exp(-x))

def forward_propagation(w, b, x, hidden_activation, output_activation):
    a = []
    z = []

    layers = len(w)

    z.append(np.dot(w[0],x) + b[0])
    a.append(activation(hidden_activation, z[0]))

    for i in range(1,layers-1):
        z.append(np.dot(w[i],a[i-1])+b[i])
        a.append(activation(hidden_activation, z[i]))

    z.append(np.dot(w[layers-1],a[layers-2])+b[layers-1])
    a.append(activation(output_activation,z[layers-1]))
    return [a, z]

def derivative(activation,x):
    if activation=="relu":
        return 1*(x>0)
    elif activation=="tanh":
        return 1-np.square(np.tanh(x))
    elif activation == "sigmoid":
        return (1 / (1 + np.exp(-x)))*(1-(1 / (1 + np.exp(-x))))




def backward_propagation(w, b, a, z, x, y, alpha, hidden_activation):

    dw = []
    db = []
    dz = []

    layers = len(w)
    m = len(y)

    for i in range(layers):
        dw.append(np.zeros(np.shape(w[i])))
        db.append(np.zeros(np.shape(b[i])))
        dz.append(np.zeros(np.shape(z[i])))

    dz[layers-1] = a[layers-1] - y
    dw[layers-1] = np.dot(dz[layers-1],a[layers - 2].T)/m
    db[layers-1] = np.sum(dz[layers-1], axis=1)/m

    for i in range(layers-2, 0, -1):
        dz[i] = np.dot(w[i+1].T,dz[i+1])*derivative(hidden_activation,z[i])
        dw[i] = np.dot(dz[i],a[i-1].T)/m
        db[i] = np.sum(dz[i], axis=1)/m

    dz[0] = w[1].T.dot(dz[1])*derivative(hidden_activation, z[0])
    dw[0] = np.dot(dz[0],x.T)/m
    db[0] =  np.sum(dz[0], axis=1)/m

    for j in range(layers):
        w[j] = w[j] - alpha * dw[j]
        # db[j] is 1d array, transpose b[j] so the broadcast is happy
        b[j] = b[j].T - alpha * db[j]
        b[j] = b[j].T

    return [w, b]