## PyCon Australia (20-minute video, 14-Aug-2017)  
- https://www.youtube.com/watch?v=ICMsWq7c5Z8

#### CPU
```python
# usage example
two_layer_nn = FC()
output = two_layer_nn(INPUT)
```

#### GPU - compatible
```python
# usage example
two_layer_nn = FC().cuda()
output = two_layer_nn(INPUT)
```

#### GPU - parallelize
```python
# usage example
two_layer_nn = nn.DataParallel(FC().cuda())
output = two_layer_nn(INPUT)
```
### Graph Computation

### DGC - Dynamic Graph Computation
- this is what PyTorch runs on
- define your network **by running it**

#### Why DGC
- easier to debug
  - can know which file and which line the network failed
- process inputs of variable sizes
- no sessions
- manipulate gradients during runtime


### SGC - Static Graph Computation
- define your network AND THEN run it
- TensorFlow
- Caffe
- Theano
- Torch

### Visualizing
- TensorFlow --->  has Tensorboard
- PyTorch ---> has visdom, it is platform agnostic (https://github.com/facebookresearch/visdom)
  - can communicate via HTTP REST

#### Custom Dataset Loader - Using Transforms
- Computer vision specific pre-processing library; there is also text and audio coming along 
- Contains the commonly using pre-processing and data augmentation such as scaling, converting PILLOW array to tensor, flipping image, randomly cropping it, randomly flipping it.  Can compose multiple transforms

```python
import torchvision.transforms as transforms
```

## Why Use PyTorch?
- it has a non-leaky API
  - easier to reason about your code, especially when it is a black box code 
- Commonly used features are baked into the framework, for your convenience
  - Ex:  converting from a tensor to a numpy, numpy to tensor
  - data augmentation, data pre-processing
  - multi-threading
- Vibrant and supportive community + Good docs
  - both on forums and slack 
