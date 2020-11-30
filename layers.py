# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py
import tensorflow as tf
from tensorflow import keras


class AddSingletonDepth(keras.layers.Layer):

    def call(self, x, mask=None):
        x = keras.backend.expand_dims(x, -1)  # add a dimension of the right

        if keras.backend.ndim(x) == 4:
            return keras.backend.permute_dimensions(x, (0, 3, 1, 2))
        else:
            return x

    def get_output_shape_for(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Subtract(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0] - x[1]

    def get_output_shape_for(self, input_shape):
        return input_shape[0]


class BigFilter(keras.layers.Layer):
    def __init__(self, units=32,**kwargs):
        self.units = units

        super(BigFilter, self).__init__(**kwargs)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            # 'selector': self.selector,
            'units': self.units,
        })
        return config
    def build(self, input_shape):
        assert input_shape[-1] == 1
        w_init = tf.initializers.Ones()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-2], self.units), dtype=tf.float32),
            trainable=True,
        )
    def call(self, inputs):
        inputs_rep = tf.repeat(inputs,repeats=self.units,axis=-1)
        wn = tf.math.multiply(inputs_rep, self.w)

        return  wn

class Slice(keras.layers.Layer):

    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            # 'selector': self.selector,
            'desired_output_shape': self.desired_output_shape,
        })
        return config
    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = keras.backend.permute_dimensions(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = keras.backend.permute_dimensions(y, [0, 2, 1])

        return y


    def get_output_shape_for(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape
