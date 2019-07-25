
import numpy as np

# Part 4: Optimization with loss feedback
#a. Loss with L1,L2 regression vs classification
l2_loss = lambda labels,output: -0.5*np.nanmean(np.power(output - labels, 2.0))
l1_loss = lambda labels,output: -np.nanmean(np.abs(output - labels))

one_hot = lambda labels, n_classes: np.eye(n_classes)[np.array(labels).reshape(-1)].transpose()
ce_loss = lambda labels,output: -np.nansum(np.log(softmax(output))* one_hot(labels, output.shape[0]), axis=0)
softmax = lambda x: np.exp(x)/np.nansum(np.exp(x), axis=0)

raw_outputs = np.random.randn(10)
maxind = np.argmax(softmax(raw_outputs), axis=0)

# image-based losses
loss_data = np.random.random((10,64,64))
#smoothness_loss
#ssim_loss
l2_image_loss = lambda labelmap,image: -0.5*nanmean(np.power(image - labelmap), 2.0)


# gradient at the loss has the magnitude of the label in it for all but L1 loss, which only has its polarity vs the output
# like warmer/colder versus a compass (but no map)
# L1 is robust to outliers but has major drawback of multiple solutions
# L1 just adds up the lengths in each of N dimensions which have many identical L1 norms
# while L2 takes the vector length across the dimensions and has only one solution
ce_softmax_delta = lambda labels,output: output - one_hot(labels, output.shape[0])
l2_loss_delta = lambda labels,output: -(output - labels)
l1_loss_delta = lambda labels,output: -np.where(output>labels, np.ones((len(labels),)), -np.ones((len(labels),)))



#b. Data splits for training, validation and testing
# often already handled by the platform but necessary to be aware of
# typical split: 70:20:10 for train/validation/test
# validation is used to determine how to modify training hyperparameters
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

# split into train, val and held-back test sets
index=np.arange(num_items)
split1=int(num_items*0.7)
split2=int(num_items*0.9)
index_train=index[:split1]
index_val=index[split1:split2]
index_test=index[split2:]
# randomize index (only for train set) and get data loader
np.random.shuffle(index_train)
train_data = zip([d for d in mnist_data[index_train,:,:,:]], [l for l in mnist_labels[index_train]])
val_data = zip([d for d in mnist_data[index_val,:,:,:]], [l for l in mnist_labels[index_val]])
test_data = zip([d for d in mnist_data[index_test,:,:,:]], [l for l in mnist_labels[index_test]])

# test should ideally ONLY be used to validate the finished network,
#  to reduce opportunity for network to learn how to fake you out
# The "unseen data" problem.
# The effect of using test sets for directing training can be negligible
# or it can be catastrophic if you don't have additional realistic test sets to confirm
# a much bigger effect however, is shift between the train/val/test dataset distributions
# and the real-world, production data distributions the network will encounter

# a bigger issue can be class imbalance. Say you have 90 examples of class 1 and 10 examples of class 2.
n_images=2000
labels=np.concatenate([np.ones((int(0.9*n_images),)), np.zeros((int(0.1*n_images),))])
data=np.random.random((n_images,784,))
trivial_inference = lambda data: np.ones(len(data),)
trivial_accuracy = lambda labels, outputs: sum(labels==outputs)/float(n_images)
outputs = trivial_inference(data)
accuracy = trivial_accuracy(labels, outputs)
print('Accuracy with an untrained, hand-crafted net: ', accuracy)

# many techniques to handle, roughly three paths:
# 1. rebalance the sampling of what the net is exposed to
# 2. rebalance the gradients by downweighting according to classes exposed
# 3. modify the loss to be self-balancing or less sensitive to imbalance (e.g. focal loss, sampling-sensitive loss, generalized regression loss)

# sampling considerations can be very subtle, for example, object detection requires
# learning from positive and negative examples. Typically, there will be far more potential
# negative examples (e.g. draw a box anywhere there isn't overlap with a label or use any label
# with the wrong class). Balancing negative/positive was necessary to get the first detectors working well.


#c. Optimization with SGD vs Adam
# stochastic gradient descent is a shortcut to computing the total sample gradient
# minibatches of data passed thru network and gradients accumulated, then used to adjust
# moving average with momentum
sgd_update_batch_grads = lambda grads, grads_mov_avg, momentum: [momentum * mov_grads + (1-momentum)*g for g,mov_grads in zip(grads, grads_mov_avg)]
sgd_update_weights = lambda params, batch_grads, lr: [param - lr * delta for param,delta in zip(params, batch_grads)]

# adaptive moment estimation, attempts to estimate the ideal gradient learning weight and momentum per parameter
# described in detail in https://arxiv.org/pdf/1412.6980.pdf
# initialize gradient mean/var as zero
#m = beta1 * m + (1 - beta1) * grad
#v = beta2 * v + (1 - beta2) * (grad**2)
#w = w - learning_rate * m / (sqrt(v) + epsilon)
adam_update_movavg = lambda grads_m, grads_v, grads, beta1, beta2: [[beta1*m + (1-beta1)*g,beta2*v + (1-beta2)*g*g] for m,v,g in zip(grads_m, grads_v, grads)]
adam_update_weights = lambda params, grads_m, grads_v, lr, time: [param - lr * delta for param,delta in zip(params, batch_grads)]
learning_rate=0.001
beta1=0.9
beta2=0.999
epsilon=1e-08
time=100
# the moments (m and v) must be rescaled according to the number of times they've been accumulated
#m = m / (1-beta1**time)  # where t=number of time steps
#v = v / (1-beta2**time)
# thus m and v start out larger (by 1-beta) and trend towards the real m and v (10-fold larger for m and 500-fold larger in the case of v)
# this can be combined with learning rate to set the effective learning rate as a function of time:
learning_rate = learning_rate * np.sqrt(1-beta2**time)/(1-beta1**time)


#d. Dropout and batchnorm
dropout = lambda data, mask: np.where(mask, data, np.zeros_like(data))
# essential to set the gradients to zero on each backward pass before we use them either to get the next layer back
# or when accumulating gradients, so the mask has to remain unchanged between forward and backward passes
dropout_backprop = lambda grads, mask: np.where(mask, grads, np.zeros_like(grads))

# batchnorm builds a moving-average mean/stddev from data passed thru it, this is then used
# to normalize the data, and we learn a de-normalization (gamma for new mean and beta for new stddev)
# backpropagation to learn the beta/gamma, its contributions are derivative of the output (f_bn) w.r.t. gamma/beta:
# df_bn/dbeta = 1
# df_bn/dgamm = data_norm (previous activation normalized by the moving average used to normalize)
def batchnorm(data, mov_avg_mu, mov_avg_sigma, gamma, beta, across_axis=0, eps=1e-5, momentum=0.9):
    mean = np.mean(data)
    var = np.sqrt(np.mean(np.mean(data - mean) ** 2))
    mov_avg_mu = momentum*mov_avg_mu + (1-momentum)*mean
    mov_avg_sigma = momentum*mov_avg_sigma + (1-momentum)*var
    data_norm = (data - mov_avg_mu) / mov_avg_sigma
    data_bn = (data_norm * gamma) + beta
    return (data_bn,mov_avg_mu,mov_avg_sigma,gamma,beta)

# batchnorm_delta returns data_norm to be used in a backward pass to multiply into the backward-working gradients
# for gamma, for delta treat it like we treated the bias in fc layer backprop
batchnorm_delta = lambda data, mov_avg_mu, mov_avg_sigma: (data - mov_avg_mu) / mov_avg_sigma

