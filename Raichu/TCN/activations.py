import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.backend import sigmoid
import numpy as np

def gelu(x):
	
	"""Gaussian Error Linear Unit (GELU). 
	It weights inputs by their magnitude, rather than gates inputs by their sign as in ReLUs. In adititon, 
	it randomly applies the identity or zero map to a neuron’s input [4]"""

	return 0.5*x*(1+K.tanh(np.sqrt(2/np.pi)*(x+0.044715*K.pow(x, 3))))


def gelu_t(x):
    """
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 
	For reference, see:
    1. https://arxiv.org/abs/1606.08415
	2. https://github.com/huggingface/transformers/blob/c89bdfbe720bc8f41c7dc6db5473a2cb0955f224/src/transformers/activations_tf.py#L6	
    """
    x = tf.convert_to_tensor(x)
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

    return x * cdf


def gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841
    Args:
        x: float Tensor to perform activation
    Returns:
        `x` with the GELU activation applied.
    """
    x = tf.convert_to_tensor(x)
    pi = tf.cast(math.pi, x.dtype)
    coeff = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))