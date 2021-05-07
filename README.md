# MoViNet-pytorch
Pytorch implementation of [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/pdf/2103.11511.pdf). <br>
Authors: Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Mingxing Tan, Matthew Brown, Boqing Gong (Google Research) <br>
[[Authors' Implementation]](https://github.com/tensorflow/models/tree/master/official/vision) (available soon)<br>

## Stream Buffer
![stream buffer](https://github.com/Atze00/MoViNet-pytorch/blob/main/figures/Stream_buffer.png)

#### Clean stream buffer
It is required to clean the buffer after all the clips of the same video have been processed.
```python
model.clean_activation_buffers()
```
## Usage

#### Main Dependencies
` Python 3.8` <br>
`fvcore 0.1.5` <br>
`PyTorch 1.7.1` <br>

#### How to build a model

```python
from models import MoViNet
from config import _C

number_classes = 600

MoViNetA0 = MoViNet(_C.MODEL.MoViNetA0, number_classes, causal = True)
MoViNetA1 = MoViNet(_C.MODEL.MoViNetA1, number_classes, causal = True)
...
```

#### Training loop example
```python
def train_iter(model, optimz, data_load, n_clips = 5, n_clip_frames=8):
    """
    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    optimz.zero_grad()
    for i, data, target in enumerate(data_load):
        #backward pass for each clip
        for j in range(n_clips):
          out = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
          loss = F.nll_loss(out, target)/n_clips
          loss.backward()
        optimz.step()
        optimz.zero_grad()
        
        #clean the buffer of activations
        model.clean_activation_buffers()
```
#### Pretrained models
Currently the only available pretrained models are on tf hub: https://tfhub.dev/google/collections/movinet/1 <br>
Pytorch models with loaded TF weights don't seem to output the same tensor as the tf models. <br>
Some models on tf hub seem to have a slightly different architecture with respect to the one presented on the paper. <br>
The pretrained pytorch models will be released once the source code is available. <br>
