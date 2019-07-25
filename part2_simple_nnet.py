
import numpy as np

# Part 2: A simple neural network
# a. A fully connected neuron and its parameters
# equation: $$ output = w * x + b $$
imgfile='number5.jpg'
from PIL import Image
img_pil = Image.open(imgfile)
data = np.asarray(img_pil).astype(np.float32).reshape((784,1))
# scale from 0-1
data=data/255.0
# random data/label matrices can be very useful
#data = np.random.random((784,1))
weights = np.random.randn(10, 784)
biases = np.random.randn(10, 1)
output = np.dot(weights,data) + biases
print(output.shape)
assert output.shape == (10,1), 'Shape is not what we expected'

# b. Activation functions
sigmoid = lambda x: 1/(1+np.exp(-x))
relu = lambda x: np.maximum(x, 0.0)
activations_relu = relu(output)
activations_sigmoid = sigmoid(output)

# c. A simple neural network for digit classification
w_scale=0.01
b_scale = 0.0
# two layer fc, first layer is 784 inputs and 512 outputs, second layer is 512 inputs, 10 outputs
sizes=[784, 512, 10]
biases = [0.01*(2*np.random.randn(y, 1)-1) for y in sizes[1:]]
weights = [0.00001*(2*np.random.randn(y, x)-1) for x, y in zip(sizes[:-1], sizes[1:])]
print('Weights shapes: ',[w.shape for w in weights])
print('Biases shapes:  ',[b.shape for b in biases])
def forward(data, weights, biases, activation=sigmoid):
    activations=[data]
    for i in range(len(weights)):
        feat = np.dot(weights[i], activations[-1]) + biases[i]
        # only apply relu activation on penultimate layers
        act = activation(feat) if i < (len(weights)-1) or activation==sigmoid else feat
        activations.append(act)
    return activations[-1]

act_fn=relu
print('Output (relu): ', forward(data, weights, biases, act_fn))
act_fn=sigmoid
print('Output (sigmoid): ', forward(data, weights, biases, act_fn))

# d. Brief aside on regression vs classification
# two of the most popular loss/feedback methods are regression and classification
# typically, regression involves metric distance between label and output and results in a scaled floating point value(s)
# classification involves finding the maximum (or minimum) output of a network and how much larger it is relative to other outputs
softmax = lambda x: np.exp(x)/np.nansum(np.exp(x), axis=0)
print('Softmaxed output: ', softmax(forward(data, weights, biases, act_fn)))
maxind = np.argmax(softmax(forward(data, weights, biases, act_fn)), axis=0)[0]
print('Maximum index: ', maxind)
print('"Probability": ', softmax(forward(data, weights, biases, act_fn))[maxind][0])

# e. Load tuned weights and run a forward pass
import dill
with open('nnet_params.dill','rb') as fp:
    params = dill.load(fp)
weights = [params[0][0], params[1][0]]
biases = [params[0][1], params[1][1]]
act_fn=relu
print('Output (trained): ', forward(data, weights, biases, act_fn))
print('Softmaxed ()trained) output: ', softmax(forward(data, weights, biases, act_fn)))
maxind = np.argmax(softmax(forward(data, weights, biases, act_fn)), axis=0)[0]
print('Maximum (trained) index: ', maxind)
print('"Probability": ', softmax(forward(data, weights, biases, act_fn))[maxind][0])


