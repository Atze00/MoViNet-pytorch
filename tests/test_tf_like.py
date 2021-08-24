from movinets.models import tfAvgPool3D,ConvBlock3D
import tensorflow as tf
import torch
from torch import nn
from einops import rearrange
import unittest


class TestTfAvgPool3D(unittest.TestCase):
    def testTfAvgPool3D(self):
        """ Testing of the tfAvgPool"""
        for i in range(1,12):
            for j in range(1,12):
                for k in range(1,10):
                    for u in range(1,5):
                        for b in range(1,3):
                            a = tf.random.uniform([b, k, i, j, u], seed = 10)
                            
                            b = rearrange(torch.from_numpy(a.numpy()), "b t h w c-> b c t h w")
                            try:
                                output = tfAvgPool3D()(b)
                            except RuntimeError:
                                continue
                            output_tf = tf.keras.layers.AveragePooling3D(pool_size = (1,3,3), strides =(1,2,2), padding = "same")(a)
                            output_tf = rearrange(torch.from_numpy(output_tf.numpy()), "b t h w c-> b c t h w")
                            self.assertTrue(torch.allclose(output,output_tf))

class TestTfConv3D(unittest.TestCase):
    def testTfConv3D(self):
        """ Testing of the convolution with the tf_like behaviour,
            A lot of different values have to be tested because of the
            different convolution implementation in pytorch"""
        
        padding = (0,0,0)
        for i in range(1,5):
            for l in range(1,5):
                for j in range(1,5):
                    for t in range(1,3):
                        for n in [1,2,3]:
                            for m in [1,2,3]:
                                stride = (t,m,n)
                                kernel_size=(i,j,l)
                                for k in range(1,5):
                                    for u in range(1,8):
                                        for b in range(1,8):
                                            try:
                                                torch_conv = ConvBlock3D(2,2, causal = False, tf_like = True, kernel_size = kernel_size,
                                                                      padding = padding, stride = stride,
                                                                      conv_type = "3d",
                                                                      norm_layer = nn.Identity, activation_layer = nn.Identity)
                                            except ValueError:
                                                continue
                                            tf_conv = tf.keras.layers.Conv3D(
                                                                2, kernel_size,strides=stride, padding='same',
                                                                use_bias=False)
                                            a = tf.random.uniform([1, k, u, b, 2], seed = 10)
                                            output_tf = tf_conv(a)
                                            output_tf = rearrange(torch.from_numpy(output_tf.numpy()), "b t h w c-> b c t h w")
                                            #loading weights to pytorch conv
                                            w = tf_conv.weights[0]
                                            w = rearrange(w, "d h w c_in c_out -> c_out c_in d h w")
                                            w = torch.tensor(w.numpy())
                                            torch_conv.load_state_dict({"conv_1.conv3d.weight":w}, strict=True)

                                            b = rearrange(torch.from_numpy(a.numpy()), "b t h w c-> b c t h w")
                                            output = torch_conv(b)
                                            self.assertTrue(torch.allclose(output,output_tf,atol=1e-03))

    
if __name__ == '__main__':
    unittest.main()
