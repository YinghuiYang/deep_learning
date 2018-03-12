import numpy as np

def weight_initialization(layer_dimension):
    w = []
    for i in range(1, len(layer_dimension)):
        w.append(np.random.normal(0, 0.05, [layer_dimension[i], layer_dimension[i-1]]))

    w.append(np.random.normal(0, 0.05, [1, layer_dimension[i]]))
    return w

def b_initialization(layer_dimension):
    b = []
    for i in range(len(layer_dimension)):
        b.append(np.zeros([layer_dimension[i], 1]))
    return b

def cost(a, y):
    cost =  - np.sum(y * np.log(a) + (1-y)*np.log(1-a))/len(y)
    return cost

def activation(string,x)
    if string=="ReLU":
        return 1*(x>0)
    elif string=="tanh":
        return np.tanh(x)
    elif string=="sigmoid":
        return 1 / (1 + np.exp(-x))

def forward_propagation(w, b, x, hidden_activation, output_activation):
    a = w
    z = w

    layers = len(w)

    z[0] = w[0].dot(x) + b[0]
    a[0] = activation(hidden_activation, z[0])

    for i in range(1,layers-1):
        z[i] = w[i].dot(a[i-1])+b[i]
        a[i] = activation(hidden_activation, z[i])

    z[layers-1] = w[layers-1].dot(a[layers-2])+b[layers-1]
    a[layers-1] = activation(output_activation,z[layers-1])
    return [a, z]

def derivative(activation,x):
    if activation=="ReLU":
        return 1*(x>0)
    elif activation=="tanh":
        return 1-np.square(np.tanh(x))
    elif activation == "sigmoid":




def backward_propagation(w, b, a, z, x, y, alpha, hidden_activation, output_activation):

    dw = w
    db = b
    dz = z

    layers = len(w)

    dz[layers-1] = a[layers-1] - y
    dw[layers-1] = dz[layers-1].dot(a[layers - 2].T) / len(dz[layers-1])
    db[layers-1] = np.sum(dz[layers-1], axis=1) / len(dz[layers-1])
    for i in reversed(layers-2, 0, -1):
        dz[i] = w[i+1].T.dot(dz[i])*derivative(hidden_activation,i)
        dw[i] = dz[i].dot(a[i-1].T) / len(dz[i])
        db[i] = np.sum(dz[i], axis=1) / len(dz[i])

    dz[0] = w[0].T.dot(dz[1])*derivative(hidden_activation, 0)
    dw[0] = dz[0].dot(x.T) / len(dz[0])
    db[0] =  np.sum(dz[0], axis=1) / len(dz[0])

    for j in range(layers):
        w[j] = w[j] - alpha * dw[j]
        b[j] = b[j] - alpha * db[j]

    return [w, b]