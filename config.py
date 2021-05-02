"""
Inspired by
https://github.com/PeizeSun/SparseR-CNN/blob/dff4c43a9526a6d0d2480abc833e78a7c29ddb1a/detectron2/config/defaults.py
"""
from fvcore.common.config import CfgNode as CN

def fill_SE_config(conf, input_channels, 
                    out_channels, 
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_avg,
                    norm_layer = None,
                    activation_layer = None):
    conf.expanded_channels =expanded_channels
    conf.padding_avg= padding_avg
    fill_conv(conf,input_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,
                norm_layer = None,
                activation_layer = None)

def fill_conv(conf, input_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,
                norm_layer = None,
                activation_layer = None):
    conf.input_channels = input_channels
    conf.out_channels = out_channels
    conf.kernel_size = kernel_size
    conf.stride = stride
    conf.padding = padding
    conf.norm_layer = norm_layer
    conf.activation_layer = activation_layer
   

    

_C = CN()

_C.MODEL = CN()


###################
#### MoViNetA0 ####
###################

_C.MODEL.MoViNetA0 = CN()
_C.MODEL.MoViNetA0.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA0.conv1, 3,8,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA0.blocks = [ [CN()],
        [CN(), CN(), CN()],
        [CN(), CN(), CN()],
        [CN(), CN(), CN(), CN()],
        [CN(), CN(), CN(), CN()]]#TODO fix

#Block2
fill_SE_config(_C.MODEL.MoViNetA0.blocks[0][0], 8, 8, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))

#block 3
fill_SE_config(_C.MODEL.MoViNetA0.blocks[1][0], 8, 32, 80, (5,3,3), (1,2,2), (0,1,1), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[1][1], 32, 32, 80, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[1][2], 32, 32, 80, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 4
fill_SE_config(_C.MODEL.MoViNetA0.blocks[2][0], 32, 56, 184, (5,3,3), (1,2,2), (0,1,1), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[2][1], 56, 56, 112, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[2][2], 56, 56, 184, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 5
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][0], 56, 56, 184, (5,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][1], 56, 56, 184, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][2], 56, 56, 184, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][3], 56, 56, 184, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 6
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][0], 56, 104, 344, (5,3,3), (1,2,2), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][1], 104, 104, 280, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][2], 104, 104, 280, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][3], 104, 104, 344, (1,5,5), (1,1,1), (0,2,2), (0,1,1))

_C.MODEL.MoViNetA0.conv7= CN()
fill_conv(_C.MODEL.MoViNetA0.conv7, 104,480,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA0.dense9= CN()
_C.MODEL.MoViNetA0.dense9.hidden_dim = 2048



###################
#### MoViNetA1 ####
###################

_C.MODEL.MoViNetA1 = CN()
_C.MODEL.MoViNetA1.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA1.conv1, 3, 16,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA1.blocks = [ [CN(),CN()],
        [CN(), CN(), CN(),CN()],
        [CN(), CN(), CN(),CN(),CN()],
        [CN(), CN(), CN(), CN(),CN(),CN()],
        [CN(), CN(), CN(), CN(),CN(),CN(),CN()]]#TODO fix

#Block2
fill_SE_config(_C.MODEL.MoViNetA1.blocks[0][0], 16, 16, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[0][1], 16, 16, 40, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 3
fill_SE_config(_C.MODEL.MoViNetA1.blocks[1][0], 16, 40, 96, (3,3,3), (1,2,2), (0,1,1), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[1][1], 40, 40, 120, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[1][2], 40, 40, 96, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[1][3], 40, 40, 96, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 4
fill_SE_config(_C.MODEL.MoViNetA1.blocks[2][0], 40, 64, 216, (5,3,3), (1,2,2), (0,1,1), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[2][1], 64, 64, 128, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[2][2], 64, 64, 216, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[2][3], 64, 64, 168, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[2][4], 64, 64, 216, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 5
fill_SE_config(_C.MODEL.MoViNetA1.blocks[3][0], 64, 64, 216, (5,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[3][1], 64, 64, 216, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[3][2], 64, 64, 216, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[3][3], 64, 64, 128, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[3][4], 64, 64, 128, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[3][5], 64, 64, 216, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 6
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][0], 64 , 136, 456, (5,3,3), (1,2,2), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][1], 136, 136, 360, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][2], 136, 136, 360, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][3], 136, 136, 360, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][4], 136, 136, 456, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][5], 136, 136, 456, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA1.blocks[4][6], 136, 136, 544, (1,3,3), (1,1,1), (0,1,1), (0,1,1))

_C.MODEL.MoViNetA1.conv7= CN()
fill_conv(_C.MODEL.MoViNetA1.conv7, 136,600,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA1.dense9= CN()
_C.MODEL.MoViNetA1.dense9.hidden_dim = 2048



###################
#### MoViNetA2 ####
###################

_C.MODEL.MoViNetA2 = CN()
_C.MODEL.MoViNetA2.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA2.conv1, 3,16,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA2.blocks = [ [CN(),CN(),CN()],
        [CN(), CN(), CN(),CN(),CN() ],
        [CN(), CN(), CN(),CN(),CN()],
        [CN(), CN(), CN(), CN(),CN(),CN()],
        [CN(), CN(), CN(), CN(),CN(),CN(),CN()]]#TODO fix

#Block2
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][0], 16, 16, 40, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][1], 16, 16, 40, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][2], 16, 16, 64, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 3
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][0], 16, 40, 96, (3,3,3), (1,2,2), (0,1,1), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][1], 40, 40, 120, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][2], 40, 40, 96, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][3], 40, 40, 96, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][4], 40, 40, 120, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 4
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][0], 40, 72, 240, (5,3,3), (1,2,2), (0,1,1), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][1], 72, 72, 160, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][2], 72, 72, 240, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][3], 72, 72, 192, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][4], 72, 72, 240, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 5
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][0], 72, 72, 240, (5,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][1], 72, 72, 240, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][2], 72, 72, 240, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][3], 72, 72, 240, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][4], 72, 72, 144, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][5], 72, 72, 240, (3,3,3), (1,1,1), (0,1,1), (0,1,1))

#block 6
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][0], 72 , 144, 480, (5,3,3), (1,2,2), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][1], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][2], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][3], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][4], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][5], 144, 144, 480, (3,3,3), (1,1,1), (0,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][6], 144, 144, 576, (1,3,3), (1,1,1), (0,1,1), (0,1,1))

_C.MODEL.MoViNetA2.conv7= CN()
fill_conv(_C.MODEL.MoViNetA2.conv7, 144,640,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA2.dense9= CN()
_C.MODEL.MoViNetA2.dense9.hidden_dim = 2048
