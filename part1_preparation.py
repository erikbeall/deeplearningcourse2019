
import numpy as np

# Part 1: Preparation - Practical python for neural networks
from PIL import Image

# useful functions in Image(): .fromarray(), .resize(), .save()
# easily switch to numpy array with np.asarray() and back again with Image.fromarray()
# load image from disk and convert to np array
imgfile='digilabs.jpg'
img_pil = Image.open(imgfile)
img_np = np.asarray(img_pil)
print('Image shape: ', img_np.shape)
img_pil_again = Image.fromarray(img_np)

# resample to half the size of the original image
down_size = [shape//2 for shape in img_np.shape[:2]]
img_pil_halfsize = img_pil_again.resize(down_size, Image.NEAREST)
print('Halved image shape: ', img_pil_halfsize.size)
img_pil_halfsize.save(imgfile.replace('.jpg','_halved.jpg'), 'JPEG')
img_pil_224= img_pil_again.resize((224,224), Image.NEAREST)
img_pil_224.save(imgfile.replace('.jpg','_224.jpg'), 'JPEG')

# display min, mean, max of image data
print('min, mean, max original: ', [np.min(img_np), np.mean(img_np), np.max(img_np)])

# Display image, histogram, image stats
import matplotlib.pylab as plt
plt.ion()
plt.imshow(img_np)
plt.title('Original image')
plt.show()
plt.pause(4)
plt.close()

# scale image to 0-1 range (np.ptp is like range() in matlab - it is the max minus min)
img_np_scaled = (img_np - np.min(img_np))/np.ptp(img_np)
print('min, mean, max scaled: ', [np.min(img_np_scaled), np.mean(img_np_scaled), np.max(img_np_scaled)])
plt.imshow(img_np_scaled)
plt.title('Scaled image')
plt.show()
plt.pause(4)
plt.close()

# histogramming to get a quick sense of the data distribution(s)
# useful for finding problems and understanding what methods are more appropriate
plt.figure()
data=np.random.random((1000,1))
plt.hist(data,histtype='step')
data2=2*np.random.random((1000,1))
plt.hist(data2,histtype='step')
plt.title('Histogram of random data 0-1 and 0-2 overlain')
print('Histogram of random data 0-1 and 0-2 overlain')
plt.pause(4)
plt.clf()
(n,x,_)=plt.hist(data,histtype='step')
plt.hist(data2,x, histtype='step')
plt.title('Histogram of random data 0-1 and 0-2 using the same bins as first plot')
print('Histogram of random data 0-1 and 0-2 using the same bins as first plot')
plt.pause(4)
plt.close()


# Numpy basics, np.dot
data = np.random.random((28,28))
data = data.reshape((784,1))
# inner product dot(IxJ, JxK) -> IxK
weights = np.random.randn(10, 784)
offsets = np.random.randn(10, 1)
dot_prod = np.dot(weights,data) + offsets
# dot product projections
proj1 = lambda img: img @ np.ones((img.shape[1],img.shape[1])) * (1.0/img.shape[1])
proj2 = lambda img: ((img.T @ np.ones((img.shape[0],img.shape[0]))).T) * (1.0/img.shape[0])


# convolution
from scipy import signal
l1_filter = np.zeros((2,3,3))
l1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1,   1,  1],
                                [0,   0,  0],
                                [-1, -1, -1]]])
img=np.asarray(img_pil_224)
print('Filter shape: ', l1_filter.shape)
print('Image shape: ', img.shape)
# convert a 3-channel input to a 2-channel output by passing thru a 2-channel 3x3 filter
img_pad = np.pad(img,
                 pad_width=l1_filter.shape[2]//2,
                 mode='constant',
                 constant_values=0).astype(np.float64)
print('Padded image', img_pad.shape)
fmap = np.zeros((l1_filter.shape[0],img_pad.shape[0],img_pad.shape[1])).astype(np.float64)
print('Feature map shape: ', fmap.shape)

for c in range(l1_filter.shape[0]):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for in_channel in range(img.shape[2]):
                for m in range(l1_filter.shape[1]):
                    for n in range(l1_filter.shape[2]):
                        # note, convolution is applied in opposite direction as data
                        fmap[c, i, j] += img_pad[i + m, j + n, in_channel] * l1_filter[c, l1_filter.shape[1]-m-1, l1_filter.shape[2]-n-1]
                        # this will have implications for backpropagation

#fmap = (fmap-np.min(fmap))/np.ptp(fmap)

fmap_oneline = np.asarray([sum([signal.convolve2d(img[:,:,in_channel], filt) for in_channel in range(3)]) for filt in l1_filter])
# closer to fully-connected concept, filter is 4D, for 3,2-channels of 3x3 filters
#fmap = np.asarray([sum([signal.convolve2d(img[:,:,in_channel], weights[in_channel,out_channel,:,:], mode='valid') for in_channel in range(in_channels)]) for out_channel in range(out_channels)])
print('Feature map shape: ', fmap.shape)
print('Same, but oneline', fmap_oneline.shape)
plt.figure()
plt.subplot(2,2,1)
plt.imshow(fmap[0,:,:])
plt.title('first feat map array')
plt.subplot(2,2,2)
plt.imshow(fmap[1,:,:])
plt.title('second feat map array')
plt.show()
plt.subplot(2,2,3)
plt.imshow(fmap_oneline[0,:,:])
plt.title('first feat map')
plt.subplot(2,2,4)
plt.imshow(fmap_oneline[1,:,:])
plt.title('second feat map')
plt.pause(6)


# List (and other set) comprehension, plus zip, enumerate
nums = range(10)
print(nums)
squared_num_list = [x**2 for x in range(10)]
squared_num_dict = {x: x**2 for x in range(10)}
# sets exclude more than one of the same member, can be useful for one-liners that also exclude duplicates
squared_num_set = set(x**2 for x in range(10))

# flow control inside comprehension expression
squared_evens = [x**2 for x in range(10) if x%2 == 0]
# double loops (left loop is the outer loop)
doubled_list=[[0,1,2], [3], [4,5,6], [7,8], [9,10,11,12,13]]
doubled_list = [[i for i in range(x)] for x in range(5)]
flattened_list = [y for x in doubled_list for y in x]
flattened_set = set(y for x in doubled_list for y in x)

# These will be useful in writing concise, understandable
# operators on data. e.g. normalize a batch of N images:
'''
batch = [np.asarray(Image.open('%06d.jpg'%(imnum))) for imnum in range(0,batch_size)]
scaled_batch = [(img-np.min(img))/np.ptp(img) for img in batch]
# better yet, define a function or lambda with a unique, descriptive name and use that
# batch = [normalize_image(imread(imgfile)) for imgfile in img_files]
image_iterator,label_iterator = mnist_data_generator(batch_size, data_location)
batch_iterator = zip(image_iterator, label_iterator)
data_loader = enumerate(batch_iterator)
for batchnum, batch in data_loader:
    print('Batch ',batchnum)
    # unpack the batch
    images, labels = batch
    print('  Holding image with shape ', images.shape)
    print('  Holding label with shape ', labels.shape)
'''



# Lambdas for simple purposes
imread = lambda imgfile: np.array(Image.open(imgfile))
imwrite = lambda img,imgfile: Image.fromarray(img).save(imgfile,'JPEG')
# nan-safe version of normalize_image
normalize_image = lambda img: (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)) if np.nanmax(img) > 0.0 else img


