import sys
import unittest
sys.path.append('.')
from models import MoViNet
from config import _C
from io import BytesIO
import tensorflow as tf
import numpy as np
from six.moves import urllib
from PIL import Image
from einops import rearrange
import torch
import tensorflow as tf
import tensorflow_hub as hub
movinets=[_C.MODEL.MoViNetA0,
        _C.MODEL.MoViNetA1,
        _C.MODEL.MoViNetA2,
        _C.MODEL.MoViNetA3,
        _C.MODEL.MoViNetA4,
        _C.MODEL.MoViNetA5,]


class TestPretrainedModels(unittest.TestCase):
    def testPretrainedModels(self):
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        image_height_l = [172,172,224,256,290,320]
        image_width_l  = [172,172,224,256,290,320]


        inputs = tf.keras.layers.Input(
            shape=[None, None, None, 3],
            dtype=tf.float32)


        f = open('/dev/null', 'w')
        sys.stderr = f

        for i in range(6):
            image_width=image_width_l[i]
            image_height=image_height_l[i]
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((image_height, image_width))
            video = tf.reshape(np.array(image), [1, 1, image_height, image_width, 3])
            video = tf.broadcast_to(video, [1, 2, image_height, image_width, 3])
            video = tf.cast(video, tf.float32) / 255.
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")
            encoder = hub.KerasLayer(
            f"https://tfhub.dev/tensorflow/movinet/a{i}/base/kinetics-600/classification/2")

            # Important: due to a bug in the tf.nn.conv3d CPU implementation, we must
            # compile with tf.function to enforce correct behavior. Otherwise, the output
            # on CPU may be incorrect.
            encoder.call = tf.function(encoder.call, experimental_compile=True)

            # [batch_size, 600]
            outputs = encoder(dict(image=inputs))

            model_tf = tf.keras.Model(inputs, outputs)
            output_tf = model_tf(video)
            del model_tf

            model = MoViNet(movinets[i], 600,causal = False, pretrained = True, tf_like = True )
            model.eval();
            with torch.no_grad():
                model.clean_activation_buffers()
                output = model(video_2)
            del model
            self.assertTrue(np.allclose(output.detach().numpy(),output_tf.numpy(),rtol=1e-06,atol=1e-4,))

if __name__ == '__main__':
    unittest.main()
