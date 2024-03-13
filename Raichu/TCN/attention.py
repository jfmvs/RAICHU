from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import tensorflow as tf

class AttentionLayer(Layer):
    def __init__(self, attention_dim=50, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        #self.W = K.variable(self.init((input_shape[-1], 1)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self._trainable_weights = [self.W, self.b, self.u]
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())
            
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'attention_dim': self.attention_dim        
        })
        return config