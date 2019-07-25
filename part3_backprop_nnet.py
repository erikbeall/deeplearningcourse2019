
import numpy as np

# Part 3: Finding the parameters
# a few helper functions
sigmoid = lambda x: 1.0/(1.0000000000001 + np.exp(-x))
relu = lambda x: np.maximum(x, 0.0)
softmax = lambda x: np.exp(x)/np.nansum(np.exp(x), axis=0)
data = np.random.random((784,1))
# derivative of the activation functions
sigmoid_delta = lambda x, sigmoid: sigmoid(x)*(1-sigmoid(x))
relu_delta = lambda x: np.maximum(x+0.0001, 0.0)/np.maximum(x, 0.0001)-0.0001

# a. Abstractions for parameters and layers
print('')
print('************************************************')
print('Building abstractions for working with layers')
import collections
def get_params(input_pixels=784, n_hidden=[512], output_classes=10, w_scale=0.01, b_scale=0.0, activations=None):
    fc_sizes=[input_pixels]
    fc_sizes.extend(n_hidden if type(n_hidden) == type(list()) else [n_hidden])
    fc_sizes.append(output_classes)
    '''
    # How would you abstract your layers?
    # One option, build a named collection with lists attached to each
    params=collections.namedtuple('params',['weights','biases','activations'])
    params.weights=[w_scale*np.random.randn(y, x) for x, y in zip(fc_sizes[:-1], fc_sizes[1:])]
    params.biases=[b_scale*np.random.randn(y, 1) for y in fc_sizes[1:]]
    params.activations = activations # if activations is not None else [[]]*(len(fc_sizes)-1)
    '''
    # Another option, layer-indexed params list
    # benefit is each layer is an index into the params
    params=[]
    for i in range(len(fc_sizes)-1):
        layer=collections.namedtuple('layer',['weight','bias','activation'])
        layer.weights = w_scale * np.random.randn(fc_sizes[i+1], fc_sizes[i])
        layer.bias  = b_scale * np.random.randn(fc_sizes[i+1], 1)
        layer.activation = activations[i]
        params.append(layer)
    # Other methods could include a dict or collection of layers, where the input and outputs have specific names
    # and each layer object includes the layers that feed into and take output from it (e.g. Caffe)
    return params

# get a network (set of initialized parameters)
activation_fn=relu
delta_activation_fn=relu_delta
params = get_params(n_hidden=512, activations=[activation_fn, None])

# re-create forward pass on nnet with this layer-based param structure
def forward(data, params, forward_only=True):
    activations=[data]
    features=[]
    for layer in params:
        features.append(np.dot(layer.weights, activations[-1]) + layer.bias)
        activations.append(layer.activation(features[-1]) if layer.activation is not None else features[-1])
    if forward_only:
        return activations[-1]
    else:
        return activations,features

print('Testing forward with layer-based param structure')
maxind = np.argmax(softmax(forward(data, params)), axis=0)[0]
print('Maximum index: ', maxind)
print('"Probability": ', softmax(forward(data, params))[maxind][0])


# b. Loss and backpropagation
print('')
print('************************************************')
print('Loss functions and calculations')
# Cross-entropy loss function, log of the softmaxed output at the correct class.
# Goal of training is to increase it (to 1.0) and decrease all others
labels=np.asarray([[8]])
# this isn't strictly cross-entropy, its actually the log-likelihood, but when used with softmax, this is what it comes down to
# cross-entropy loss is difference between label and output distributions as used with sigmoid
# while log-likelihood is what it looks like with softmax, and softmax is useful for reporting "probabilities" (which they aren't really)
ce_loss = lambda labels,output: -np.nansum(np.log(softmax(output))* one_hot(labels, output.shape[0]), axis=0)
# turn a label into a one-hot vector (0's for non-label, 1 for the label)
one_hot = lambda labels, n_classes: np.eye(n_classes)[np.array(labels).reshape(-1)].transpose()


# what might softmax cross-entropy loss look like in a real training run?
# recall softmax produces outputs that sum to 1.0, they look like probabilities from 0.0 to 1.0
# realistically, a random network with 10 classes will produce ten outputs about 1/10 = 0.10
print('Typical starting softmax cross-entropy loss (p = 0.1): ', -np.log(0.1))
print('Decent ending softmax cross-entropy loss (p ~ 0.95 or so): ', -np.log(0.95))
import matplotlib.pylab as plt
plt.ion()
plt.plot(-np.log(np.linspace(0.1,0.99,100)))
plt.title('Softmax cross-entropy loss starting at chance, ending at confident 99% prediction')
plt.show()
plt.pause(2)
plt.close()
# recall loss is sensitive to two things with the __relative values__ outputted by the network:
#     maximizing the correctly classed example and minimizing the others
# Therefore, it is possible for accuracy and loss to not track each other perfectly. Accuracy
# can stop improving while loss continues to decrease, typically meaning the network is becoming more confident than we want.
# But, these aren't really confidences, and it would be trivial to re-map these relative values to a more meaningful
# (e.g. more useful) set of confidences in that they more match experience with real world data. Its a sort of illusion
# to consider these as confidences, they only relate to relative scaling. The amount of confidence scaling applied with
# softmax can be manipulated with the "temperature" parameter. See online for examples.

# Also note with softmax cross-entropy, due to the softmax, the loss can continue to decrease while accuracy remains constant
# because the top-returned class can continue being correct at the same rate while the network learns to become less equivocal
# overly-certain in its output. A reduction in this kind of loss without an increase in accuracy may mean the network is becoming
# more brittle

# Regression loss, L2 and L1
l2_loss = lambda labels,output: -0.5*np.nanmean(np.power(output - labels, 2.0))
l1_loss = lambda labels,output: -np.nanmean(np.abs(output - labels))
# because we nearly always consider loss as a scalar, typically we take the mean above
# however, gradient functions need to remain un-summed
l2_loss_delta = lambda labels,output: -(output - labels)
l1_loss_delta = lambda labels,output: -np.where(output>labels, np.ones((len(labels),)), -np.ones((len(labels),)))
# random data with and without outliers
y = np.arange(100)
loss_data = 5*np.random.random((100,))+y
loss_data_outliers = np.copy(loss_data)
inds=[4,8,15]
loss_data_outliers[inds] = 100
print('L1 Losses on clean, corrrupted data: ',[l1_loss(loss_data, y), l1_loss(loss_data_outliers, y)])
print('L2 Losses on clean, corrrupted data: ',[l2_loss(loss_data, y), l2_loss(loss_data_outliers, y)])
y = y.astype(np.float)
# cross-entropy loss formulated for use in regression
ce_regression_loss = lambda labels, output: -np.nansum(labels * np.log(sigmoid(output)) + (1 - labels) * np.log(1 - sigmoid(output)))
print('CE Losses on clean, corrrupted data: ',[ce_regression_loss(loss_data/100.0, y/100.0), ce_regression_loss(loss_data_outliers/100.0, y/100.0)])

# c. The gradient
print('')
print('************************************************')
print('Backpropagation calculations')
# example fully connected layer with 1024 hidden neurons, with 784 inputs and 10 outputs
# Two layers: w1 = [512, 784], b1 = [512,1], w2 = [10,512], b2 = [10,1]
# Hidden layer output (feature map): z1_j = (w1_ij . x_i) + b1_j
# activation of hidden layer output: a1_j = g(z1_j)
# Output feature map: z2_k = (w2_jk . a1_j) + b2_k
# Softmax on map: S_k = exp(z2_k) / sum_k(exp(z2_k))
#                     = exp([w2_jk * g((w1_ij * x_i) + b_j)] +b2_k)/sum_k(exp([w2_jk * g((w_ij * x_i) + b_j)] +b2_k))
# Loss is -log(S_i) and is a scalar, where i==label
# we'll be computing dL/dz_in, where L=loss, z_in is the feature map from the previous layer into this layer
#    also note, additional term in loss for weight decay: L_reg = wd*0.5*sum(weights*weights)
#    this second term will be added for each layer's weights gradient term: dL_reg/dW = wd*weights
#
# gradient of the softmax loss w.r.t. activation coming from penultimate layer
#  dL/dz_in = dL/dS * dS/dz_in = dL/dS * dS/da_in * da_in/dz_in
# gradient w.r.t. weight
#  dL/dW_i = dL/dS * dS/da_in * da_in/dz_in * dz_in/dW_i, where S=softmaxed output, a=feature output at layer i, W=weight at layer i
# gradient w.r.t. bias
#  dL/db_i = dL/dS * dS/da_in * da_in/db_i, where b_i is bias at layer i
#  df/dW = d(W . input_activation + bias)/dW = input_activation
#  df/db = d(W . input_activation + bias)/dbias = 1
#  dL/dS = d(-log(S_i))/dS = -1/S_i (scalar, where i==label index)
#  dS/da_in = softmax derivative, which has two cases: y==i and y!=i
#           = S_k - S_k*S_i (k==i) and -S_k*S_i (k!=i)
#           = S_i - S_i*S_i (k==i) and -S_k*S_i (k!=i)
#           = S_i*one_hot(label) - S_k*S_i, where S_k is the full softmaxed output vector and S_i is (a scalar) S_k indexed at the label index
# we'll define a gradient lambda for the combined gradient of the ce_loss and softmax
#
# note, if we were doing L2 loss, the gradient of the output w.r.t. the weight would involve the derivative of L2:
#  dL2/dW_i = dL2/da_in * da_in/dW_i (no softmax, but may or may not be an activation layer, e.g. sigmoid can be common for squashed outputs)
#                            e.g. for 1/depth outputs, for which it can be helpful to scale and cut-off below some small number
#                            depth_activation = lambda output, depth_label, scale=0.3, cutoff=(1.0/1000): np.minimum(scale * sigmoid(output), cutoff) - cutoff
#  dL_L2/da_in = d(L2)/da_in = (output_activation - label_vector)
ce_softmax_delta = lambda labels,output: output - one_hot(labels, output.shape[0])
# so ultimately:
# dL/dW = (-1/S_i) * (S_i*one_hot(label) - S_k*S_i) * (input_activation) + wd*weights
# dL/dW = (S_k - one_hot(label)) * input_activation + wd*weights = gradient at the weights in this layer
# dL/db = S_k - one_hot(label) = gradient at the bias

# LAST LAYER: run a forward pass and get back all the activations
activations,features = forward(data, params, forward_only=False)
softmax_outputs = softmax(activations[-1])
loss = ce_loss(labels, activations[-1])
print('Single forward pass loss on initialized network: ',loss)
# ignoring L2 norm of weights for now
grad_last_layer = (softmax_outputs - one_hot(labels, softmax_outputs.shape[0])) # defined above as ce_softmax_delta(labels, outputs)
# if we had used relu on last layer (or sigmoid), we'd multiply last_layer grad by its grad on the feature map that had been passed thru it originally:
# grad_last_layer *= delta_activation_fn(features[-1])
grad_last_biases = grad_last_layer
# note, the input activation must be transposed here, we're actually taking the current output error (number of output channels)
# and expanding it via the data that originally flowed into these weights (number of input channels) giving grads for all (output x input) channels
grad_last_weights = np.dot(grad_last_layer, activations[-2].transpose())
print(grad_last_weights.shape)
assert grad_last_weights.shape == params[-1].weights.shape, 'gradient and weights shapes do not match'
# get the weight-decay term and sum into the weights gradient
wd=0.0005
l2_last_weights = wd * params[-1].weights
grad_last_weights += l2_last_weights
print('Backprop, grad_last_layer shape: ',grad_last_layer.shape)

# FIRST LAYER
# for next layer back, we need the gradient of the activation and chain rule them again...
# dz_out/dz_in = dz_out/da_in * da_in/dz_in = np.dot(next layer's weights, grad_last_layer above) * da_in/dz_in
# df_last/df_p = delta_activation(on the input_features from the previous layer into this layer)
# df_p/dW_p    = input_features from previous layer (input data in the case this is the first layer)
# this is just error propagation and there are good texts on error propagation (copy of Bevington in the lab, Jack currently has it)
# essentially, the gradient at a layer back is equal to the gradient at the next layer, backwardly multiplied by the next layer's weights
# times any partial derivative for anything happening between this layer and the next (e.g. any derivative due to the activation layer and
# as in the above case, the current layer's input activation). Recall the dot product used forward direction takes the
# input channels and transforms by inner product sum to get the output channels,
# so taking the dot of the transpose takes the gradient in the form of the layer's output channels and gets us back to the input channel dimension.
grad_first_layer = np.dot(params[-1].weights.transpose(), grad_last_layer) * relu_delta(features[-2])
# thus to keep pushing back further, to get the gradient at the previous layer, we first multiply the transpose of the current layer's weights
# by the current layer's gradient to get the gradient at the activation layer between the previous and current layers, and then multiply
# by the activation layer gradient to get the gradient at said layer.
grad_first_biases = grad_first_layer
grad_first_weights = np.dot(grad_first_layer, activations[-3].transpose())
assert grad_first_weights.shape == params[0].weights.shape, 'gradient and weights shapes do not match'
# again get the weight-decay term (assuming we're using weight decay)
l2_first_weights = wd * params[0].weights
grad_first_weights += l2_first_weights
print('Backprop, grad_first_layer shape: ',grad_first_layer.shape)
# look over the above until you understand it well enough to write it out (at least in words, accurately describing the gradient flow)

# a few takeaways, small activations result in small gradients, and small weights in the next layer will result in small backprop gradients.
# large activations can blow up if nothing is done to prevent them (e.g. weights regularization, gradient clamping).
# with relu if something isn't activated the gradients will also be clamped.
# with most of the other nonlinear activations (e.g. sigmoid, tanh) that squish output nonlinearly, the derivative is very small for
# either large or small feature values, a condition commonly described as a "saturated" neuron.
# relu solves one end of this, leaky relu or prelu or selu were motivated to solve this issue.
#
#      grad_at_some_layer = np.recursive_dot( (next_layer_weights.T, next_layer_grads) * delta_activation(previous_layer_forward_output) )
#      grad_at_some_layer_weights = np.dot(grad_at_some_layer, activations_into_this_layer.T)
#
# this means that the backpropagated gradients grow geometrically in dependence on the subsequent weights and changes in activation as you move back
# next look at the gradient at the output we started with, cross-entropy vs L2
#
#     ce_grad_last_layer = (softmax(outputs) - one_hot(labels, outputs.shape[0]))
#     l2_grad_last_layer = (outputs - labels) * sigmoid_delta()
#
# unfortunately, the sigmoid_delta can become 0 or 1 easily, and at that point, sigmoid_delta(close to 0 or close to 1) = 0
# and the output gradient we want to start with becomes zero easily. Softmax cross-entropy does not have this issue.
# we can avoid this by avoiding a sigmoid on the output, but this results in an unconstrained output (can be handled in other ways)


# d. Backpropagation in full
# activation_delta is a lambda hand-crafted to encompass the softmax cross-entropy loss (in this case)
# if using L2 or L1, must be hand-crafted to encompass that loss and any activation applied at the final layer before the loss
# modern platforms provide layers with auto-differentiation so you only need to worry about tying the layers together
# and the platform handles gradient computation for you
def forward_hybrid_onesample(data, label, params, loss_fn, loss_fn_delta, activation, activation_delta, wd=0.0005):
    # activations (input to the next layer, equal to feature when no activation used)
    a_l = [data]
    # feature maps
    f_l = []
    delta_a_l = []
    # compute the forward calculations and store the features and activations and the derivative of the parameters at these values
    for layer in params:
        # append feature and activation outputs
        f_l.append(np.dot(layer.weights, a_l[-1]) + layer.bias)
        # calc activation if specified, else just the raw feature map
        a_l.append(layer.activation(f_l[-1]) if layer.activation is not None else f_l[-1])
        # same for gradient, else just ones
        delta_a_l.append(activation_delta(f_l[-1]) if layer.activation is not None else f_l[-1]/f_l[-1])

    loss = loss_fn(label, a_l[-1])
    # gradient at the loss
    layerback = 1
    # gradient at the last layer is the cost gradient times the derivative of the last activation layer
    # in our toy network, we did no activation on last layer...
    gradient_layer = loss_fn_delta(label, a_l[-1]) * delta_a_l[-1]
    # gradients (in reverse order)
    grad_w_reverse=[]
    grad_b_reverse=[]
    # store the gradients, bias is simply the gradient we compute above at this layer
    grad_b_reverse.append(gradient_layer)
    # weights gradient is this layer's gradient dotted with the previous layer's activation output (e.g. what this layer saw as input)
    grad_w_reverse.append(np.dot(gradient_layer, a_l[-2].transpose()) + wd * params[-1].weights)

    # backward further, subsequent layers depend on the following layer's activation_delta
    N_layers = len(params)
    for layerback in range(1, N_layers):
        current_layer=N_layers - layerback
        previous_layer=current_layer - 1
        gradient_layer = np.dot(params[current_layer].weights.transpose(), gradient_layer) * delta_a_l[previous_layer]
        # backpropagation: gradient at a given layer is equal to the next layer's weights dotted with the next layer's gradient
        # then the bias' gradient is said gradient, and weights' gradient is said gradient dotted with the activations that were inputted to this layer
        grad_b_reverse.append(gradient_layer)
        grad_w_reverse.append(np.dot(gradient_layer, a_l[previous_layer].transpose()) + wd * params[previous_layer].weights)
    return (grad_w_reverse, grad_b_reverse, loss, a_l[-1])

# get params for a one-hidden layer network with 64 units
activation_fn=relu
delta_activation_fn=relu_delta
params = get_params(n_hidden=[64], activations=[activation_fn, None])
# grab gradients for a single made-up example
gradw_r,gradb_r,loss,output = forward_hybrid_onesample(data, labels, params, ce_loss, ce_softmax_delta, activation_fn, delta_activation_fn)
print('Single pass thru initialized network, loss = ',loss)

# check gradients numerically
# iterate over points in data and run forward_hybrid_onesample() twice, with data+eps and data-eps and compute delta = diff/(2*eps)
import matplotlib.pylab as plt
plt.ion()
import copy
'''
print('')
print('************************************************')
print('Checking gradients')
eps=0.0001
dgwp=[]
dgbp=[]
for i in range(784):
    datap=copy.deepcopy(data)
    datan=copy.deepcopy(data)
    datap[i] += eps
    datan[i] -= eps
    gwp,gbp,lp,op =  forward_hybrid_onesample(datap, labels, params, ce_loss, ce_softmax_delta, relu, relu_delta)
    gwn,gbn,ln,on =  forward_hybrid_onesample(datan, labels, params, ce_loss, ce_softmax_delta, relu, relu_delta)
    dgwp.append(sum([np.nansum(p-n) for p,n in zip(gwp,gwn)])/(2*eps*sum([np.prod(p.shape) for p in gwp])))
    dgbp.append(sum([np.nansum(p-n) for p,n in zip(gbp,gbn)])/(2*eps*sum([np.prod(p.shape) for p in gbp])))
dgwp=np.asarray(dgwp)
dgbp=np.asarray(dgbp)
print([np.prod(p.shape) for p in gbp])
print([np.prod(p.shape) for p in gwp])
plt.figure()
plt.subplot(2,1,1)
plt.plot(dgbp)
plt.title('bias gradient numeric check')
plt.subplot(2,1,2)
plt.plot(dgwp)
plt.title('weights gradient numeric check')
plt.show()
plt.pause(2)
plt.close()
print([np.min(dgwp), np.max(dgwp), np.nanstd(dgwp)])
print([np.min(dgbp), np.max(dgbp), np.nanstd(dgbp)])
'''
# modern platforms are pretty robust, but from time to time have bugs appear from some commit, becoming evident weeks or more later
# they now include numeric checks in their CI/automated tests, so this may only be used if you make new layers or check a new loss function for stability

# e. Simple data loader for digits
print('')
print('************************************************')
print('Constructing data loader for MNIST')
import gzip
import struct
datasets=['data/train-labels-idx1-ubyte.gz', 'data/train-images-idx3-ubyte.gz']
with gzip.open(datasets[0], 'rb') as fp:
    # labels
    magic=struct.unpack(">L", fp.read(size=4))[0]
    assert magic == 2049, 'magic number in header does not match expectations'
    num_items=struct.unpack(">L", fp.read(size=4))[0]
    mnist_labels=np.asarray(list(fp.read()))

with gzip.open(datasets[1], 'rb') as fp:
    magic=struct.unpack(">L", fp.read(size=4))[0]
    assert magic == 2051, 'magic number in header does not match expectations'
    num_imgs=struct.unpack(">L", fp.read(size=4))[0]
    num_rows=struct.unpack(">L", fp.read(size=4))[0]
    num_cols=struct.unpack(">L", fp.read(size=4))[0]
    data=np.asarray(list(fp.read()))
    mnist_data=data.reshape((num_imgs, 28, 28, 1))

# randomize index and get data loader
index=np.arange(num_items)
np.random.shuffle(index)
data_loader = zip([d for d in mnist_data[index,:,:,:]], [l for l in mnist_labels[index]])
# not really a normalization just a rescaling to 0-1
normalize_image = lambda img: (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)) if np.nanmax(img) > 0.0 else img
print('created one-time use simple data_loader with n_images = ', num_imgs)

print('')
print('************************************************')
print('Running a single epoch of MNIST data in our simple network')
plt.figure()
batch_size=64
batch_index=0
wgrads_mov_avg=None
# SGD with momentum
update_mov_avg = lambda grads, batch_size, grads_mov_avg, momentum: [momentum * m + (1-momentum)*g/batch_size for g,m in zip(grads, grads_mov_avg)]
loss_mov_avg = 0.0
lr=0.1
acc=[]
grad_w_r_acc=None
params_copy = copy.deepcopy(params)
momentum=0.9
for batchnum,batch in enumerate(data_loader):
    img,lbl= batch
    img=img.reshape((784,1)).astype(np.float32)
    nimg=normalize_image(img)
    gradw_r,gradb_r,loss,output = forward_hybrid_onesample(nimg, lbl, params_copy, ce_loss, ce_softmax_delta, activation_fn, delta_activation_fn)
    loss_mov_avg += loss
    maxind = np.argmax(softmax(output), axis=0)[0]
    acc.append(maxind==lbl)
    # initialize the batch gradient accumulator, will be re-set after each batch_size
    if grad_w_r_acc is None:
        gradw_r_acc = gradw_r
        gradb_r_acc = gradb_r
    else:
        gradw_r_acc = [g1+g2 for g1,g2 in zip(gradw_r_acc, gradw_r)]
        gradb_r_acc = [g1+g2 for g1,g2 in zip(gradb_r_acc, gradb_r)]
    batch_index +=1
    if batch_index==batch_size:
        # accumulate a batch into the gradient tracker (SGD)
        # recall the accumulator must be divided by batch_size as it is simply a sum over the batch
        if wgrads_mov_avg is None:
            wgrads_mov_avg = [w/batch_size for w in gradw_r_acc]
            bgrads_mov_avg = [b/batch_size for b in gradb_r_acc]
        else:
            wgrads_mov_avg = update_mov_avg(gradw_r_acc, batch_size, wgrads_mov_avg, momentum)
            bgrads_mov_avg = update_mov_avg(gradb_r_acc, batch_size, bgrads_mov_avg, momentum)
        # Update parameters with SGD (learning_rate times the momentum-moving-averaged gradient)
        for j in range(len(params_copy)):
            params_copy[j].weights -= lr*wgrads_mov_avg[-j-1]
            params_copy[j].bias    -= lr*bgrads_mov_avg[-j-1]
        # reset the batch gradient accumulator
        gradw_r_acc=None
        batch_index = 0
    if batchnum % (batch_size*100) == 1:
        print('Batch %d, Loss avg: %.02f, acc: %.2f'%(int(batchnum/batch_size), loss_mov_avg/batchnum, sum(acc)/len(acc)))
        if batchnum>batch_size*400:
            break
    # inspect the first few
    if batchnum>=2:
        continue
    print('output: ', output)
    plt.imshow(img.reshape((28,28)))
    plt.title(lbl)
    plt.show()
    plt.pause(0.5)
    plt.close()

print('Ran one epoch')

# save parameters to a dill file (supposedly better than pickle)
import dill
with open('nnet_params_oneepoch.dill','wb') as fp:
    listparams=[[p.weights,p.bias] for p in params_copy]
    dill.dump(listparams, fp)


print('')
print('************************************************')
# f. Training a simple network in numpy
# play with batch size and learning rate
#batch_size=64
#lr=0.1
epochs=20
# check out the leaderboard for MNIST networks at http://yann.lecun.com/exdb/mnist/

reduction_epochs=[15, 20, 25]
#activation_fn=sigmoid
#delta_activation_fn=sigmoid_delta
#params = get_params(n_hidden=[512, 256], activations=[activation_fn, activation_fn, None])
params = get_params(n_hidden=[64], activations=[activation_fn, None])
# accuracy accumulator
acc_acc = []
grad_w_r_acc=None
wgrads_mov_avg=None
for epoch in range(epochs):
  if epoch in reduction_epochs:
      lr=lr/1.5
  batch_index=0
  acc=[]
  loss_mov_avg = 0.0
  # randomize index and get data loader
  index=np.arange(num_items)
  np.random.shuffle(index)
  data_loader = zip([d for d in mnist_data[index,:,:,:]], [l for l in mnist_labels[index]])
  grad_w_r_acc=None
  wgrads_mov_avg=None
  # main training loop
  for batchnum,batch in enumerate(data_loader):
    img,lbl= batch
    img=img.reshape((784,1)).astype(np.float32)
    nimg=normalize_image(img)
    randnoise=np.random.randn(784).reshape((784,1))*0.025
    nimg=nimg+randnoise
    # what would happen if we modified the image here before passing to forward, how would that affect the network's ability to learn?
    # try adding some random noise...
    gradw_r,gradb_r,loss,output = forward_hybrid_onesample(nimg, lbl, params, ce_loss, ce_softmax_delta, activation_fn, delta_activation_fn)
    loss_mov_avg += loss
    maxind = np.argmax(softmax(output), axis=0)[0]
    acc.append(maxind==lbl)
    # initialize the gradient accumulators
    if grad_w_r_acc is None:
        gradw_r_acc = gradw_r
        gradb_r_acc = gradb_r
    else:
        gradw_r_acc = [g1+g2 for g1,g2 in zip(gradw_r_acc, gradw_r)]
        gradb_r_acc = [g1+g2 for g1,g2 in zip(gradb_r_acc, gradb_r)]
    batch_index +=1
    if batch_index==batch_size:
        # initialize the moving-average gradient tracker for SGD
        # recall, accumulator needs to be divided by batch_size so be careful using it blindly
        if wgrads_mov_avg is None:
            wgrads_mov_avg = [w/batch_size for w in gradw_r_acc]
            bgrads_mov_avg = [b/batch_size for b in gradb_r_acc]
        else:
            wgrads_mov_avg = update_mov_avg(gradw_r_acc, batch_size, wgrads_mov_avg, momentum)
            bgrads_mov_avg = update_mov_avg(gradb_r_acc, batch_size, bgrads_mov_avg, momentum)
        for j in range(len(params)):
            params[j].weights -= lr*wgrads_mov_avg[-j-1]
            params[j].bias    -= lr*bgrads_mov_avg[-j-1]
        gradw_r_acc=None
        batch_index = 0
    if batchnum % (batch_size*100) == 1:
        print(' Batch %d, Loss avg: %.02f, acc: %.2f'%(int(batchnum/batch_size), loss_mov_avg/batchnum, sum(acc)/len(acc)))
    # Note, we'll miss the last partial batch - can you think of a simple way to make use of these last images?
    # many ways to do it, see if you can do it in as little as two lines moved above

  acc_acc.append(sum(acc)/len(acc))
  print('Epoch %d, Loss avg: %.02f, acc: %.2f'%(epoch, loss_mov_avg/batchnum, acc_acc[-1]))

# save parameters to a dill file (supposedly better than pickle)
import dill
with open('nnet_params.dill','wb') as fp:
    listparams=[[p.weights,p.bias] for p in params]
    dill.dump(listparams, fp)

print('Ran training loop over specified epochs and saved params')
print('')

