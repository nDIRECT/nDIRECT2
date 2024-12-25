#mxnet_test.py
import json
import time
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
from collections import namedtuple


#Loading the data
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
filename = "cat.png"
image = mx.image.imread(filename)
def transform(image):
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225]))
    # the network expect batches of the form (N,3,224,224)
    transposed = normalized.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    #batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    batchified = transposed
    return batchified
x=transform(image)
array = mx.nd.array(x)

x_temp = np.empty((48, 3, 224, 224))

x_new = mx.nd.array(x_temp) 

for i in range(48):
    x_new[i] = array

#print("array:", array)

#load the params for the resnet model
Batch = namedtuple('Batch', ['data'])

print("batch:", Batch)

ctx = mx.cpu()
sym, arg_params, aux_params = mx.model.load_checkpoint('vgg19', 0) 

mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (48,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
#run the model
repeat = 3
number = 3
mxnet_result = np.empty(repeat)
for r in range(repeat):
    since = time.time()
    for i in range(number):
        mod.forward(Batch([x_new]))
        prob = mod.get_outputs()[0].asnumpy()
  #m.run()
    end = time.time()
    mxnet_result[r] = (end-since)/number

for r in range(repeat):
    print("%d : %f"%(r, mxnet_result[r]))

prob = np.squeeze(prob[0])
a = np.argsort(prob)[::-1]

#print(a)
for i in a[0:1]:
    print('probability=%f, class=%s' %(prob[i], labels[i]))

# print the prediction result and time result
#prob = np.squeeze(prob)
#a = np.argsort(prob)[::-1]

#for i in a[0:1]:
#    print('probability=%f, class=%s' %(prob[i], labels[i]))
#print("mxnet_result:",mxnet_result)
