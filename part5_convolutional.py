
import numpy as np
from scipy import signal
# define Activation layers and derivatives
Sigmoid = lambda x: 1.0/(1.0000000000001 + np.exp(-x))
sigmoid_delta = lambda x, Sigmoid: Sigmoid(x)*(1-Sigmoid(x))
Relu = lambda x: np.maximum(x, 0.0)
relu_delta = lambda x: np.maximum(x+0.0001, 0.0)/np.maximum(x, 0.0001)-0.0001

# define a MaxPooling layer with downsampling factor
MaxPool = lambda data, ds=2: data[:, :(data.shape[1] // ds)*ds, :(data.shape[2] // ds)*ds].reshape(data.shape[0], data.shape[1] // ds, ds, data.shape[2] // ds, ds).max(axis=(2, 4))

softmax = lambda x: np.exp(x)/np.nansum(np.exp(x), axis=0)
one_hot = lambda labels, n_classes: np.eye(n_classes)[np.array(labels).reshape(-1)].transpose()
ce_loss = lambda labels,output: -np.nansum(np.log(softmax(output))* one_hot(labels, output.shape[0]), axis=0)


#Part 5: Convolutional neural networks
#a. Convolutions
imgfile='number5.jpg'
from PIL import Image
img_pil = Image.open(imgfile)
img_pil_32x32= img_pil.resize((32,32), Image.LINEAR)
# reshape as having a single input channel, we're going to be working with input and output channels as we go thru the network
data = np.asarray(img_pil_32x32).astype(np.float32).reshape((32,32,1))
labels=np.asarray([[5]])

# conv weights are in form channels_in (I), channels_out (J), kernel_width, kernel_height and convolution will
# proceed with one kernel (e.g. weight[i,j,:,:]) over each channel i of the I channels of data/activation flowing into this layer
# and these will be summed over all i to generate the jth output feature map/image, resulting in J output channels.
in_channels=1
out_channels=8
# typically (but not necessarily) the filter is symmetric
filter_size=5
weights = np.random.randn(in_channels, out_channels, filter_size, filter_size)
# Bias on the other hand is just like the fully-connecteds, there will be one bias per output channel
biases = np.random.randn(out_channels, 1)
# so we will be doing something like this:
'''
for j in range(out_channels):
    temp=[]
    for i in range(in_channels):
        temp += conv(weights[i,j,:,:],data[:,:,i])
    output = sum(temp) + biases[j]
'''

#b. Convolutional layer and its variants
# one-liner with list comprehension
Conv2D = lambda data, weights, biases: \
    np.asarray([sum( \
        [signal.convolve2d(data[i,:,:], weights[i,j,:,:].squeeze(), mode='same') \
         for i in range(weights.shape[0])]) \
        + biases[j] for j in range(weights.shape[1])])
# 'same' mode means a constant padding (zeros, but can be adjusted) is applied to expand the image so the output will end up the same size as the input
#         this creates a padding = (filter_size//2)
# 'valid' mode means the filter is only applied in regions of the image where the kernel can be contained, meaning the edges will get cut off
#         size_output = size_input - 2*(filter_size//2)
# 'full' is the default, meaning the mathematical convolution will be applied, which will expand the image to
#         size_output = size_input + 2*(filter_size//2)
# Note its not uncommon to skip biases for convolutional layers (usually with something like use_bias=False in one of the platforms)

# pass data thru conv layer defined by the weights and biases
data=data.transpose((2,0,1))
output = Conv2D(data, weights, biases)
print('Filter-matched padding, input data shape '+str(data.shape)+', output data shape: '+str(output.shape))

# padding
Conv2DNoPad = lambda data, weights, biases, pad=0: np.asarray([sum( [signal.convolve2d(data[i,:,:], weights[i,j,:,:].squeeze(), mode='valid') for i in range(weights.shape[0])] )+biases[j] for j in range(weights.shape[1])])
print('No padding, output data shape: '+str(Conv2DNoPad(data, weights, biases).shape))

# stride (please note, this is not a robust implementation, ideally we'd want to define starts/stops of strides, and limit needless computation)
Conv2DStride = lambda data, weights, biases, stride=2: np.asarray([sum( [signal.convolve2d(data[i,:,:], weights[i,j,:,:].squeeze(), mode='same') for i in range(weights.shape[0])] )+biases[j] for j in range(weights.shape[1])])[:,::stride,::stride]
print('Padding and stride==2, output data shape: '+str(Conv2DNoPad(data, weights, biases).shape))

# group (with groups=C) convolution (or depthwise convolution), the number of filters must match the number
# of input channels and the output is same size (and channels) as input. Each filter is applied
# to ONLY one channel, meaning convolution operations are done separately per channel in this layer
# note, there are only C-channels, no input/output channels in the weights
# ALSO note, from here on, we've redefined data to start with the channel dimension first
depthwise_weights = np.random.randn(out_channels, filter_size, filter_size)
Conv2DDepthwise = lambda data, weights, biases: np.asarray([signal.convolve2d(data[i,:,:], depthwise_weights[i,:,:].squeeze(), mode='same')+biases[i] for i in range(weights.shape[0])] )
print('Input to depthwise: '+str(output.shape)+', output: '+str(Conv2DDepthwise(output, depthwise_weights, biases).shape))


# depth-separable convolution, do a MxN with C-groupwise (depthwise) convolution followed by a 1x1 regular
# convolution with as many channels as makes sense, here lets double the number
weights_1x1 = np.random.randn(out_channels, 2*out_channels, 1, 1)
biases_1x1 = np.random.randn(2*out_channels, 1)
# however, scipy's convolve2d doesn't really support 1x1 (non-2D) convolutions, so we have to implement it differently
output_depthwise = Conv2DDepthwise(output, depthwise_weights, biases)
try:
    output_depth_sep = Conv2D(output_depthwise, weights_1x1, biases_1x1)
except:
    print('Nope, scipy convolve2d only supports actual 2D convolution')

Conv2D_1x1 = lambda data, weights, biases: np.asarray([sum([data[i,:,:]*weights[i, c, 0, 0] for i in range(weights.shape[0])]) for c in range(weights.shape[1])])
output_depth_sep = Conv2D_1x1(output_depthwise, weights_1x1, biases_1x1)
print('input to depth-separable pair of convolutions: '+str(output.shape)+', output: '+str(output_depth_sep.shape))


# dilated/atrous convolution, insert "holes" in the filter before applying, equivalent to expanding the filter and filling with zeros
# so a 3x3 filter with dilation=2 becomes a 5x5 without any added computation (assuming its implemented well, we're going to be lazy and
# keep using scipy's convolve2d), and a 3x3 with dilation=3 becomes a 7x7 filter. You could say it expands the "receptive area" of a filter.
dilation=2
dilated_filter_size = dilation*(filter_size-1)+1
dilated_weights = np.zeros( (weights.shape[0], weights.shape[1], (weights.shape[2]-1)*dilation + 1, (weights.shape[3]-1)*dilation + 1) )
dilated_weights[:, :, ::dilation, ::dilation]=weights
output_atrous = Conv2D(data, dilated_weights, biases)


# up-volution (fractional-strided convolution), apply the filter to the input data but here the input data is dilated (atrous) like above
# it is also called transposed convolution, and also sometimes called deconvolution, which it isn't, but this is what they're referring to
fmap = output
dilated_fmap = np.zeros( (fmap.shape[0], fmap.shape[1]*dilation, fmap.shape[2]*dilation) )
dilated_fmap[:, ::dilation, ::dilation]=fmap
# note, like some of these layers, this implementation involves a choice of where to index from, so
# platforms can differ in the fine details without you knowing anything about it



# network parameter builder function
import collections
def get_CNN_params(w_scale=0.01, b_scale=0.0, activations=[], conv_sizes=[], maxpool_sizes=[], fc_sizes=[]):
    # default is lenet-5
    if len(conv_sizes)==0:
        conv_sizes=[(1,6,5,5), 0, (6,16,5,5), 0, 0, 0, 0]
        maxpool_sizes=[1, 2, 1, 2, 1, 1, 1]
        fc_sizes=[0, 0, 0, 0, 400, 120, 84, 10]
    if len(activations)==0:
        activations=[Sigmoid, None, Sigmoid, None, Sigmoid, Sigmoid, None]
    params=[]
    first_fc=True
    for i in range(len(fc_sizes)-1):
        layer=collections.namedtuple('layer',['type', 'pool','weight','bias','activation'])
        if fc_sizes[i]>0:
            if first_fc:
                first_fc=False
                layer=collections.namedtuple('layer',['type', 'pool','weight','bias','activation'])
                layer.type='Flatten'
                layer.activation = None
                params.append(layer)
                layer=collections.namedtuple('layer',['type', 'pool','weight','bias','activation'])
            layer.type='FullyConnected'
            layer.weights = w_scale * np.random.randn(fc_sizes[i+1], fc_sizes[i])
            print('layer weights ',layer.weights.shape)
            layer.bias  = b_scale * np.random.randn(fc_sizes[i+1], 1)
            print('layer bias ',layer.bias.shape)
        elif maxpool_sizes[i]>1:
            layer.type='MaxPool'
            layer.weight = None
            layer.bias  = None
        else:
            layer.type='Conv2D'
            layer.weights = w_scale * np.random.random(conv_sizes[i])
            layer.bias  = b_scale * np.random.randn(conv_sizes[i][1], 1)
        layer.activation = activations[i]
        params.append(layer)
    if params[-1].type=='Conv2D':
        # if last layer was Conv, append a flatten layer
        layer=collections.namedtuple('layer',['type', 'pool','weight','bias','activation'])
        layer.type='Flatten'
        layer.activation = None
        params.append(layer)
    return params

# setup network config, use valid pooling from here on
Conv2D = lambda data, weights, biases: np.asarray([sum( [signal.convolve2d(data[i,:,:], weights[i,j,:,:].squeeze(), mode='valid') for i in range(weights.shape[0])] )+biases[j] for j in range(weights.shape[1])])
Conv2D_1x1 = lambda data, weights, biases: np.asarray([sum([data[i,:,:]*weights[i, c, 0, 0] for i in range(weights.shape[0])]) for c in range(weights.shape[1])])
activation_fn=Sigmoid
delta_activation_fn=sigmoid_delta

# fully convolutional (i.e., no fc layers)
# 32x32x1 -> 8x28x28 (ds to 14x14) -> 32x10x10 (ds to 5x5) -> 64x1x1 -> 10x1x1 -> flatten
params_fcn = get_CNN_params(activations=[activation_fn, None, activation_fn, None, activation_fn, None], conv_sizes=[(1,8,5,5),0,(8,32,5,5),0,(32,64,5,5),(64,10,1,1)], maxpool_sizes=[1,2,1,2,1,1,1], fc_sizes=[0,0,0,0,0,0,0])

# get the LeNet-5 network arch
params_lenet5 = get_CNN_params(activations=[activation_fn, None, activation_fn, None, activation_fn, activation_fn, None])

# re-create forward pass on nnet with this layer-based param structure
def forward(data, params, forward_only=True):
    activations=[data]
    features=[]
    for layer in params:
        print(layer.type)
        print('activations flowing into this layer: ', activations[-1].shape)
        if layer.type == 'Flatten':
            features.append(activations[-1].reshape((-1, 1)))
            # next layer for dot product MUST have shape (N,1) not (N,)
            activations.append(features[-1])
            continue
        elif layer.type == 'Conv2D':
            if layer.weights.shape[-1]>1:
                features.append(Conv2D(activations[-1], layer.weights, layer.bias))
            else:
                features.append(Conv2D_1x1(activations[-1], layer.weights, layer.bias))
        elif layer.type == 'FullyConnected':
            features.append(np.dot(layer.weights, activations[-1]) + layer.bias)
        elif layer.type == 'MaxPool':
            features.append(MaxPool(activations[-1]))
        activations.append(layer.activation(features[-1]) if layer.activation is not None else features[-1])
    if forward_only:
        return activations[-1]
    else:
        return activations,features

#c. Backpropagation with convolutions
# remembering that convolution is defined as a backwards operator of the filter over the local pixels
# so the 2D convolution is like flipping the weights in both axes and applying directly
# backpropagation for convolution is just like with fully connected except the weights multiplication
# to get the current layer's gradient must be done with similarly (doubly) flipped weights from the subsequent layer,
# and with flipped activation flowing from the current layer
# dL/dW = layer_gradient * activation(flipped activation)
ce_softmax_delta = lambda labels,output: output - one_hot(labels, output.shape[0])
activations,features = forward(data, params_fcn, forward_only=False)
softmax_outputs = softmax(activations[-1])
loss = ce_loss(labels, activations[-1])
print('Single forward pass loss on initialized network: ',loss)

# like in the fc example, no sigmoid or other activation on last conv layer, just the softmax
gradients={'layers': [], 'weights': [], 'biases': []}
gradients['layers'].append(softmax_outputs - one_hot(labels, softmax_outputs.shape[0]))

# now work our way back
# this last layer is easy because its a 1x1 conv, similar to the fc example
curr_layer_back = 1
# note, the penultimate layer here is just a flatten() layer, so go back one more for the relevant activations
curr_layer_back += 1
prev_layer_back = curr_layer_back + 1
nch_in,nch_out = params_fcn[-curr_layer_back].weights.shape[:2]
# a bit more reshaping work to do here due to the shapes going in and out
gradients['weights'].append(np.dot(gradients['layers'][-1], activations[-prev_layer_back].reshape(nch_in,1).transpose()).T)
gradients['biases'].append(gradients['layers'][-1])

# propagate to the previous layer using the weights (still don't have to do anything special as it was a 1x1)
# simply dot the previous weights with the last layer's gradient and multiply by derivative of the activation of prev layer output
gradients['layers'].append(np.dot(params_fcn[-curr_layer_back].weights.reshape((nch_in,nch_out)), gradients['layers'][-1]) * relu_delta(features[-prev_layer_back]).reshape(nch_in,1))
curr_layer_back += 1
prev_layer_back = curr_layer_back + 1
gradients['biases'].append(gradients['layers'][-1])
nch_in,nch_out,filter_size = params_fcn[-curr_layer_back].weights.shape[:3]

# we have to rotate the activations and apply conv to get the correct backprop gradients to each weight
# flip the weights 180 before applying them (note, scipy's convolve2d handles this for us if we're using it)
ConvBack2D_1x1_to_NxN = lambda data, weights: np.asarray([[data[i] * weights[j, ::-1,::-1] for i in range(data.shape[0])] for j in range(weights.shape[0])])
grad_penultimate_weights = ConvBack2D_1x1_to_NxN(grad_penultimate_layer.flatten(), activations[-prev_layer_back])
# flip the grads back to match the weights
grad_penultimate_weights = grad_penultimate_weights[:,:,::-1,::-1]

# to get each layer gradient, convolve with the weights under 'valid' conv mode (padding if necessary)
layer gradient = convolve2d(padded_weights, flip(next_layer_grad), 'valid')
flipped_layer_gradient = flip(layer_gradient)
weights_gradient = flip(convolve2d(activations[prev_layer_back], flip(layer_gradient), 'valid'))

# previous layer is proper 5x5 convolution so we'll finally get to see the full backprop difference w.r.t. fc
# to get the gradient at the weights on this penultimate layer
# to get the grad of the next previous layer, we have to apply the 180-rotated weights here as well to get the layer's gradient
# signal.convolve2d handles the rotation properly when it computes convolution
ConvBack2DLayer = lambda data, weights: np.asarray([sum( [signal.convolve2d(data[i,:,:], weights[j,i,:,:].squeeze(), mode='full') for i in range(data.shape[0])] ) for j in range(weights.shape[0])])
# use 'full' convolution instead of the other options
grad_next_layer_back = ConvBack2DLayer(grad_penultimate_layer.reshape((nch_out, 1, 1)), params_fcn[-curr_layer_back].weights) * relu_delta(features[-prev_layer_back])
# shape is now 32,5,5 (output channels for this layer, input channels come from two layers up, before the maxpool)
curr_layer_back = 5
prev_layer_back = curr_layer_back + 2
grad_next_layer_back_biases = grad_next_layer_back
nch_in,nch_out,filter_size = params_fcn[-curr_layer_back].weights.shape[:3]
grad_next_layer_back_biases = grad_next_layer_back
# layer grad = Conv2D(next layer_grad, weights_next_layer) * delta_activation(activation[previous_layer])
# delta^l    = Conv2D(delta^l+1 * rot180{w^l+1}) * f-prime
# weights grad = Conv2D(layer grad * activation_previous_layer)
# dL/dw = Conv2D(activations[-previous_layer], layer gradient)
# dL/dAct = next layer gradient = FullConv2D(layer.weights[:,::-1,::-1], layer gradient)

# propagate grad for this layer (32x5x5) to 8x32x5x5 weights using the input activations (shape was 8x28x28)
print('incorrect...')
print('grad_next_layer_back: ',grad_next_layer_back.shape)
#grad_next_layer_back:  (32, 5, 5)
#activations[-prev_layer_back]:  (8, 28, 28)
print('activations[-prev_layer_back]: ',activations[-prev_layer_back].shape)
ConvBack2DWeights = lambda layergrad, activation: np.asarray([sum( [signal.convolve2d(layergrad[i,:,:], activation[j,:,:].squeeze(), mode='valid') for i in range(layergrad.shape[0])] ) for j in range(activation.shape[0])])
grad_next_layer_back_weights = ConvBack2DWeights(activations[-prev_layer_back], grad_next_layer_back) # produces 8x32x32 while we want 8x32
print(grad_next_layer_back_weights.shape)
# actually what we want is to look over the gradient map / feature map

# account for maxpool

# flip back
grad_next_layer_back_weights = grad_next_layer_back_weights[:,:,::-1,::-1]

# receptive field size calculation
# receptive field for a given activation pixel (typically the last image-like output or the last image-like output used in a particular inference task)
# is equal to the total width x height input pixels that end up affecting or could end up affecting (assume no weights are zeros) the given output pixel
# the first conv filter size times subsequent downsampling/conv filter sizes


#d. Common architectures and their unique features
# LeNet-5 in numpy: 32x32x1 input -> 6x1x5x5 conv -> sigmoid -> maxpool(2) -> 16x6x5x5 -> sigmoid -> maxpool(2) -> 400-120fc -> 120-84fc -> 84-10fc
# use convolve2d with 'same' padding mode
def forward(data):
    fmap_C1 = Conv2D(data, weights_C1, biases_C1)
    act_1   = Sigmoid(fmap_C1)
    fmap_S2 = MaxPool(act_1)
    fmap_C3 = Conv2D(data, weights_C3, biases_C3)
    act_2   = Sigmoid(fmap_C3)
    fmap_S4 = MaxPool(act_2)
    flatten = fmap_S4.flatten()
    fc_F5   = Dense(flatten, weights_F5, biases_F5)
    act_F5  = Sigmoid(fc_F5)
    fc_F6   = Dense(act_F5, weights_F6, biases_F6)
    act_F6  = Sigmoid(fc_F6)
    fc_F7   = Dense(act_F6, weights_F7, biases_F7)
    return fc_F7

