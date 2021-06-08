import sys
sys.path.append('.')
from models import tfAvgPool,ConvBNActivation
import tensorflow as tf
import torch
from torch import nn
from einops import rearrange
import unittest


class TestTfAvgPool(unittest.TestCase):
    def testTfAvgPool(self):
        for i in range(3,12):
            for k in range(1,10):
                for u in range(1,5):
                    for b in range(1,2):
                        a = tf.random.uniform([b, k, i, i, u], seed = 10)
                        b = rearrange(torch.from_numpy(a.numpy()), "b t h w c-> b c t h w")
                        output = tfAvgPool()(b)
                        output_tf = tf.keras.layers.AveragePooling3D(pool_size = (1,3,3), strides =(1,2,2), padding = "same")(a)
                        output_tf = rearrange(torch.from_numpy(output_tf.numpy()), "b t h w c-> b c t h w")
                        self.assertTrue(torch.allclose(output,output_tf))

class TestConv(unittest.TestCase):
    def testConv(self):
        padding = (0,0,0)
        for j in range(3,16):
            for i in range(1,7):
                for j in range(1,7):
                    for n in [1,2,3,4]:
                        stride = (1,n,n)
                        kernel_size=(1,i,j)
                        for k in range(1,2):
                            for u in range(1,2):
                                for b in range(1,2):
                                    tf_conv = tf.keras.layers.Conv3D(
                                                        u, kernel_size, strides=stride, padding='same',
                                                        use_bias=False)
                                    w = tf_conv.weights
                                    torch_conv = ConvBNActivation(u,u, causal = False, tf_like = True, kernel_size = kernel_size,
                                                              padding = padding, stride = stride,
                                                              norm_layer = nn.Identity, activation_layer = nn.Identity)

                                    a = tf.random.uniform([b, k, j, j, u], seed = 10)
                                    output_tf = tf_conv(a)
                                    output_tf = rearrange(torch.from_numpy(output_tf.numpy()), "b t h w c-> b c t h w")
                                    w = tf_conv.weights[0]
                                    w = rearrange(w, "d h w c_in c_out -> c_out c_in d h w")
                                    w = torch.tensor(w.numpy())
                                    torch_conv.load_state_dict({"conv3d.weight":w}, strict=True)
                                    self.assertTrue(torch.allclose(torch_conv.conv3d.weight,w))
                                    b = rearrange(torch.from_numpy(a.numpy()), "b t h w c-> b c t h w")
                                    output = torch_conv(b)
                                    self.assertTrue(torch.allclose(output,output_tf,atol=1e-03))

    
if __name__ == '__main__':
    unittest.main()
