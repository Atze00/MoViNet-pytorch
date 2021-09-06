import sys
import unittest
from movinets import MoViNet
from movinets.config import _C
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
    def testBasePretrainedModels(self):
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
            video = tf.cast(video, tf.float32) / 255.
            video = tf.concat([video, video/2], axis=1)
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")
            encoder = hub.KerasLayer(
            f"https://tfhub.dev/tensorflow/movinet/a{i}/base/kinetics-600/classification/3")

            # Important: due to a bug in the tf.nn.conv3d CPU implementation, we must
            # compile with tf.function to enforce correct behavior. Otherwise, the output
            # on CPU may be incorrect.
            encoder.call = tf.function(encoder.call, experimental_compile=True)

            # [batch_size, 600]
            outputs = encoder(dict(image=inputs))

            model_tf = tf.keras.Model(inputs, outputs)
            output_tf = model_tf(video)
            del model_tf

            model = MoViNet(movinets[i],causal = False, pretrained = True )
            model.eval();
            with torch.no_grad():
                model.clean_activation_buffers()
                output = model(video_2)
            del model
            self.assertTrue(np.allclose(output.detach().numpy(),output_tf.numpy(),rtol=1e-06,atol=1e-4,))

    def testStreamPretrainedModels(self):
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        image_height_l = [172,172,224,256,290,320]
        image_width_l  = [172,172,224,256,290,320]


        inputs = tf.keras.layers.Input(
            shape=[None, None, None, 3],
            dtype=tf.float32)


        f = open('/dev/null', 'w')
        sys.stderr = f

        for i in range(3):
            image_width=image_width_l[i]
            image_height=image_height_l[i]
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((image_height, image_width))
            video = tf.reshape(np.array(image), [1, 1, image_height, image_width, 3])
            video = tf.cast(video, tf.float32) / 255.
            video = tf.concat([video, video/2, video/3], axis=1)
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")
            encoder = hub.KerasLayer(
            f"https://tfhub.dev/tensorflow/movinet/a{i}/stream/kinetics-600/classification/3")
            image_input = tf.keras.layers.Input(
                            shape=[None, None, None, 3],
                            dtype=tf.float32,
                            name='image')

            # Define the state inputs, which is a dict that maps state names to tensors.
            init_states_fn = encoder.resolved_object.signatures['init_states']
            state_shapes = {
                name: ([s if s > 0 else None for s in state.shape], state.dtype)
                for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
            }
            states_input = {
                name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
                for name, (shape, dtype) in state_shapes.items()
            }

            # The inputs to the model are the states and the video
            inputs = {**states_input, 'image': image_input}

            outputs = encoder(inputs)

            model_tf = tf.keras.Model(inputs, outputs, name='movinet')


            # Split the video into individual frames.
            # Note: we can also split into larger clips as well (e.g., 8-frame clips).
            # Running on larger clips will slightly reduce latency overhead, but
            # will consume more memory.
            frames = tf.split(video, video.shape[1], axis=1)

            # Initialize the dict of states. All state tensors are initially zeros.
            init_states = init_states_fn(tf.shape(video))

            # Run the model prediction by looping over each frame.
            states = init_states
            predictions = []
            for frame in frames:
              output, states = model_tf({**states, 'image': frame})
              predictions.append(output)

            # The video classification will simply be the last output of the model.
            output_tf = predictions[-1]

            del model_tf

            model = MoViNet(movinets[i], causal = True, pretrained = True)
            model.eval();
            with torch.no_grad():
                model.clean_activation_buffers()
                output = model(video_2)
                model.clean_activation_buffers()
                _ = model(video_2[:,:,:1])
                _ = model(video_2[:,:,1:2])
                output_2 = model(video_2[:,:,2:3])
            del model
            self.assertTrue(np.allclose(output.detach().numpy(),output_2.numpy(),rtol=1e-06,atol=1e-4,))
            self.assertTrue(np.allclose(output.detach().numpy(),output_tf.numpy(),rtol=1e-06,atol=1e-4,))
            
if __name__ == '__main__':
    unittest.main()
