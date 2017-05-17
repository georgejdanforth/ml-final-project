import os
import h5py
import numpy as np
import scipy.signal as sg

from scipy.misc import logsumexp


class Module(object):
    def __init__(self):
        self.train = True
        return

    def forward(self, _input):
        pass

    def backward(self, _input, _gradOutput):
         pass

    def parameters(self):
         pass

    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False


class Sequential(Module):
    def __init__(self):
        Module.__init__(self)
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def size(self):
        return len(self.layers)

    def forward(self, _input):
        self._inputs = [_input]

        for i in range(self.size()):
            self._inputs.append(self.layers[i].forward(self._inputs[i]))

        self._output = self._inputs[-1]
        return self._output

    def backward(self, _input, _gradOutput):
        self._gradInputs = [None] * (self.size() + 1)
        self._gradInputs[self.size()] = _gradOutput

        for i in range(self.size(), 0, -1):
            self._gradInputs[i-1] = self.layers[i-1].backward(self._inputs[i-1],
                                                              self._gradInputs[i])
        self._gradInput = self._gradInputs[0]
        return self._gradInput

    def parameters(self):
        params = []
        gradParams = []
        for m in self.layers:
            _p, _g = m.parameters()
            if _p is not None:
                params.append(_p)
                gradParams.append(_g)
        return params, gradParams

    def training(self):
        Module.training(self)
        for m in self.layers:
            m.training()

    def evaluate(self):
        Module.evaluate(self)
        for m in self.layers:
            m.evaluate()

class Convolutional(Module):
    def __init__(self, inputLength, inputDepth, filterLength, filterDepth):
        Module.__init__(self)
        stdv = 1./np.sqrt(inputLength)
        
        self.weight = np.random.uniform(-stdv, stdv, (filterLength, inputDepth, filterDepth))
        self.gradWeight = np.ndarray((filterLength, inputDepth, filterDepth))
        self.bias = np.random.uniform(-stdv, stdv, filterDepth)
        self.gradBias = np.ndarray(filterDepth)
        
    def forward(self, _input):
        inputDepth = self.weight.shape[1]
        filterDepth = self.weight.shape[2]
        
        _input = _input.reshape(_input.shape[0],-1,inputDepth)
        
        self._output = []
        for i in range(filterDepth):
            self._output.append(sg.correlate(_input, np.expand_dims(np.take(self.weight,i,2),0), 'valid') + self.bias[i])
        self._output = np.array(self._output)
        self._output = self._output.reshape(self._output.shape[1],self._output.shape[2],self._output.shape[0])

        return self._output
    
    def backward(self, _input, _gradOutput):
        filterLength = self.weight.shape[0]
        inputDepth = self.weight.shape[1]
        filterDepth = self.weight.shape[2]
        
        _input = _input.reshape(_input.shape[0],-1,inputDepth)
        _gradOutput = _gradOutput.reshape(_gradOutput.shape[0],-1,filterDepth)
        
        for i in range(filterDepth):
            self.gradWeight[:,:,i] = np.squeeze(sg.correlate(_input, np.expand_dims(np.take(_gradOutput,i,2),2), 'valid'))
        
        self.gradBias = np.sum(_gradOutput, axis=(0,1))
        
        self._gradInput = []
        for i in range(inputDepth):
            self._gradInput.append(sg.correlate(np.pad(_gradOutput, ((0,0),(filterLength-1,filterLength-1),(0,0)), 'constant'), np.expand_dims(np.take(self.weight,i,1),0), 'valid'))
        self._gradInput = np.array(self._gradInput)
        self._gradInput = self._gradInput.reshape(self._gradInput.shape[1],self._gradInput.shape[2],self._gradInput.shape[0])

        return self._gradInput
        
    def parameters(self):
        return [self.weight, self.bias], [self.gradWeight, self.gradBias]

class FullyConnected(Module):
    def __init__(self, inputSize, outputSize):
        Module.__init__(self)
        stdv = 1. / np.sqrt(inputSize)

        self.weight = np.random.uniform(-stdv, stdv, (inputSize, outputSize))
        self.gradWeight = np.ndarray((inputSize, outputSize))
        self.bias = np.random.uniform(-stdv, stdv, outputSize)
        self.gradBias = np.ndarray(outputSize)

    def forward(self, _input):
    	_input = _input.reshape(_input.shape[0],-1)

        self._output = np.dot(_input, self.weight) + self.bias
        return self._output

    def backward(self, _input, _gradOutput):
        _input = _input.reshape(_input.shape[0],-1)
        _gradOutput = _gradOutput.reshape(_gradOutput.shape[0],-1)

        self.gradWeight = np.dot(_input.T, _gradOutput)
        self.gradBias = np.sum(_gradOutput, axis=0)

        self._gradInput = np.dot(_gradOutput, self.weight.T)

        return self._gradInput

    def parameters(self):
        return [self.weight, self.bias], [self.gradWeight, self.gradBias]


class ReLU(Module):
    def __init__(self):
        Module.__init__(self)
        return

    def forward(self, _input):
    	_input = _input.reshape(_input.shape[0],-1)

        self._output = np.maximum(0, _input)
        return self._output

    def backward(self, _input, _gradOutput):
    	_input = _input.reshape(_input.shape[0],-1)
        _gradOutput = _gradOutput.reshape(_gradOutput.shape[0],-1)

        self._gradInput = _gradOutput * (_input > 0)
        return self._gradInput

    def parameters(self):
        return None, None


class SoftMaxLoss(object):
    def __init__(self):
        return

    def forward(self, _input, _label):
        _input -= np.amax(_input, axis=1)[:, np.newaxis]
        z = _label * (_input - logsumexp(_input, axis=1).reshape(1, -1).T)
        self._output = -np.sum(z) #/ np.size(z)
        return self._output

    def backward(self, _input, _label):
        _input -= np.amax(_input, axis=1)[:, np.newaxis]
        self._gradInput = np.exp(_input)/np.sum(np.exp(_input), axis=1).reshape(1, -1).T - _label
        return self._gradInput


def predict(X, model):
    return model.forward(X)


def error_rate(X, Y, model):
    model.evaluate()
    res = 1 - (model.forward(X).argmax(-1) == Y.argmax(-1)).mean()
    model.training()
    return res


def sgd(x, dx, lr, weight_decay=0):
    if type(x) is list:
        assert len(x) == len(dx), "Should be the same"
        for _x, _dx in zip(x, dx):
            sgd(_x, _dx, lr)
    else:
        x -= lr * (dx + 2 * weight_decay * x)


def runTrainVal(X, Y, model, Xval, Yval, trainopt, crit):
    eta = trainopt["eta"]
    N = X.shape[0]
    minValError = np.inf

    shuffled_idx = np.random.permutation(N)
    start_idx = 0

    for iteration in range(trainopt["maxiter"]):
        if iteration % int(trainopt["eta_frac"] * trainopt["maxiter"]) == 0:
            eta *= trainopt["etadrop"]
        stop_idx = min(start_idx + trainopt["batch_size"], N)
        batch_idx = range(N)[int(start_idx):int(stop_idx)]
        bX = X[shuffled_idx[batch_idx], :]
        bY = Y[shuffled_idx[batch_idx]]

        score = model.forward(bX)
        loss = crit.forward(score, bY)
        dscore = crit.backward(score, bY)
        model.backward(bX, dscore)

        params, gradParams = model.parameters()
        sgd(params, gradParams, eta, weight_decay=trainopt["lambda"])
        start_idx = stop_idx % N

        if (iteration % trainopt["display_iter"]) == 0:
            trainError = 100 * error_rate(X, Y, model)
            valError = 100 * error_rate(Xval, Yval, model)
            print(
                '{:8} batch loss: {:.3f} train error: {:.3f} val error: {:.3f}'\
                .format(iteration, loss, trainError, valError)
            )


def build_model(inputLength, inputDepth, outputSize):
    model = Sequential()
    model.add(Convolutional(inputLength,inputDepth, 21,30))
    model.add(ReLU())
    model.add(Convolutional(100,30, 51,5))
    model.add(ReLU())
    model.add(FullyConnected(250,outputSize))
    model.add(ReLU())
    return model

def load_data():
    MSD_DIR = os.path.join(
        os.path.expanduser("~"),
        "Documents",
        "Homework",
        "Machine Learning",
        "Project",
        "msd_data_10_proc.hdf5"
        #"msd_data_10.hdf5"
    )

    with h5py.File(MSD_DIR, "r") as f:
        """
        stop_val = np.zeros(f["/data"][0].shape)
        i = 0
        while not (f["/data"][i] == stop_val).all():
            i += 1
        j = int(0.9 * i)
        Xtrain = f["/data"][:j, :]
        Xval = f["/data"][j:i, :]
        Ytrain = np.zeros((j, 10))
        Yval = np.zeros((i-j, 10))
        Ytrain[np.arange(j), f["/labels"][:j]] += 1
        Yval[np.arange(i-j), f["/labels"][j:i]] += 1
        """
        X = f["/data"]
        Y = f["/labels"]
        N = X.shape[0]
        i = int(0.9 * N)
        Xtrain = X[:i, :]
        Ytrain = Y[:i, :]
        Xval = X[i:N, :]
        Yval = Y[i:N, :]

    return Xtrain, Ytrain, Xval, Yval

def load_mnist():
    MNIST_DIR = os.path.join(
        os.path.expanduser("~"),
        "Dropbox",
        "machine_learning",
        "ml-final-project",
        "data",
        "CLEAN_MNIST_SUBSETS.h5"
    )

    with h5py.File(MNIST_DIR, "r") as f:
        Xtrain = f["large_train/data"][:].astype(np.float64)
        Ytrain = f["large_train/labels"][:].astype(np.int32)
        Xval = f["val/data"][:].astype(np.float64)
        Yval = f["val/labels"][:].astype(np.int32)

    return Xtrain, Ytrain, Xval, Yval


def main():

    Xtrain, Ytrain, Xval, Yval = load_data()
    #Xtrain, Ytrain, Xval, Yval = load_mnist()

    trainopt = {
        "eta": 1e-4,
        "maxiter": 20000,
        "display_iter": 500,
        "batch_size": 100,
        "etadrop": 0.5,
        "eta_frac": 0.25
    }

    feature_length = 120
    feature_depth = 25
    categories = 10

    Xtrain = Xtrain.reshape(-1,feature_length,feature_depth)

    lambdas = np.array([0, 0.001, 0.01, 0.1])
    hidden_sizes = np.array([10])

    for lambda_ in lambdas:
        for hidden_size_ in hidden_sizes:
            trainopt["lambda"] = lambda_
            model = build_model(feature_length, feature_depth, categories)
            crit = SoftMaxLoss()
            runTrainVal(Xtrain, Ytrain, model, Xval, Yval, trainopt, crit)
            print("=" * 80)

if __name__ == "__main__":
    main()
