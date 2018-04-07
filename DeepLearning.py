import numpy as np


# weight initialization
def weight_initialization(layer_dimension, sample):
    w = []
    for i in range(1, len(layer_dimension)):
        w.append(np.random.normal(0, 1/np.sqrt(sample),(layer_dimension[i], layer_dimension[i-1])))

    return w

# bias initialization
def b_initialization(layer_dimension):
    b = []
    for i in range(1,len(layer_dimension)):
        b.append(np.zeros([layer_dimension[i], 1]))
    return b

def cost(a, y):
    cost =  - np.sum(y * np.log(a) + (1-y)*np.log(1-a))/np.size(y)
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


def gradient(w, b, a, z, x, y, hidden_activation):
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
    db[0] = np.sum(dz[0], axis=1)/m

    return dw,db



def backward_propagation(w, b, dw, db, alpha):

    layers = len(w)

    for j in range(layers):
        w[j] = w[j] - alpha * dw[j]
        # db[j] is 1d array, transpose b[j] so the broadcast is happy
        if np.shape(b[j]) != np.shape(db[j]):
            b[j] = b[j].T - alpha * db[j]
            b[j] = b[j].T
        else:
            b[j] = b[j] - alpha*db[j]

    return [w, b]

def l2_regularized_cost(cof, w, a, y):
    cost = (cof/2)*np.dot(w,w.T) - np.sum(y * np.log(a) + (1-y)*np.log(1-a))/np.size(y)
    cost = cost[0,0]
    return cost


def l2_regularized_backward_propagation(w, b, a, z, x, y, alpha, cof, hidden_activation):

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
    dw[layers-1] = cof*w[layers-1] + np.dot(dz[layers-1],a[layers - 2].T)/m
    db[layers-1] = np.sum(dz[layers-1], axis=1)/m

    for i in range(layers-2, 0, -1):
        dz[i] = np.dot(w[i+1].T,dz[i+1])*derivative(hidden_activation,z[i])
        dw[i] = cof*w[i] + np.dot(dz[i],a[i-1].T)/m
        db[i] = np.sum(dz[i], axis=1)/m

    dz[0] = w[1].T.dot(dz[1])*derivative(hidden_activation, z[0])
    dw[0] = cof*w[0] + np.dot(dz[0],x.T)/m
    db[0] = np.sum(dz[0], axis=1)/m

    for j in range(layers):
        w[j] = w[j] - alpha * dw[j]
        # db[j] is 1d array, transpose b[j] so the broadcast is happy
        b[j] = b[j].T - alpha * db[j]
        b[j] = b[j].T

    return [w, b]


# p: drop out probability
def drop_out_init(p, m, dim):
    d = []
    for i in range(1, len(dim)):
        temp = np.random.rand(dim[i],m)
        temp = 1*(temp>p)
        d.append(temp)

    return d

def drop_forward_propagation(w, b, x, d, p, hidden_activation, output_activation):
    a = []
    z = []

    layers = len(w)

    z.append(np.dot(w[0],x) + b[0])
    a.append(activation(hidden_activation, z[0])*d[0]/p)

    for i in range(1,layers-1):
        z.append(np.dot(w[i],a[i-1])+b[i])
        a.append(activation(hidden_activation, z[i])*d[i]/p)

    z.append(np.dot(w[layers-1],a[layers-2])+b[layers-1])
    a.append(activation(output_activation,z[layers-1]))
    return [a, z]

def drop_backward_propagation(w, b, a, z, x, y, d, alpha, p, hidden_activation):
    da = []
    dw = []
    db = []
    dz = []

    layers = len(w)
    m = len(y)

    for i in range(layers):
        da.append(np.zeros(np.shape(a[i])))
        dw.append(np.zeros(np.shape(w[i])))
        db.append(np.zeros(np.shape(b[i])))
        dz.append(np.zeros(np.shape(z[i])))

    dz[layers-1] = a[layers-1] - y
    dw[layers-1] = np.dot(dz[layers-1],a[layers - 2].T)/m
    db[layers-1] = np.sum(dz[layers-1], axis=1)/m

    for i in range(layers-2, 0, -1):
        da[i] = np.dot(w[i+1].T,dz[i+1])*d[i]/p
        dz[i] = da[i]*derivative(hidden_activation,z[i])
        dw[i] = np.dot(dz[i],a[i-1].T)/m
        db[i] = np.sum(dz[i], axis=1)/m

    da[0] = np.dot(w[1].T, dz[1]) * d[0] / p
    dz[0] = da[0]*derivative(hidden_activation, z[0])
    dw[0] = np.dot(dz[0],x.T)/m
    db[0] = np.sum(dz[0], axis=1)/m

    for j in range(layers):
        w[j] = w[j] - alpha * dw[j]
        # db[j] is 1d array, transpose b[j] so the broadcast is happy
        b[j] = b[j].T - alpha * db[j]
        b[j] = b[j].T

    return [w, b]


def momentum(w,b,dw,db, velocity_w, velocity_b, alpha,gamma):
    for j in range(len(w)):
        velocity_w[j] = alpha*velocity_w[j]-gamma * dw[j]
        velocity_b[j] = alpha * velocity_b[j].T - gamma * db[j]
    for j in range(len(w)):
        w[j] = w[j] + velocity_w[j]
        b[j] = b[j].T + velocity_b[j]
        b[j] = b[j].T

    return w,b,velocity_w,velocity_b


def accuracy(a,y):
    a = 1*(a>0.5)
    inacc = np.absolute(y-a)
    miss = np.sum(inacc)
    return 1 - miss/np.size(y)


