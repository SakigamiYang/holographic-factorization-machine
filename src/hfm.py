# coding: utf-8
import tensorflow as tf
import tensorflow.keras.layers as L


class HFM(tf.keras.models.Model):
    """
    Holographic_Factorization_Machines
    https://ojs.aaai.org/index.php/AAAI/article/view/4448
    """

    def __init__(self, num_users, num_items, embedding_size):
        super().__init__()
        self.user_embeddings = L.Embedding(num_users, embedding_size)
        self.item_embeddings = L.Embedding(num_items, embedding_size)
        self.bias = tf.Variable(tf.constant_initializer(1e-6)(()))
        self.user_first_order_dense = L.Dense(1, name='ufo')
        self.item_first_order_dense = L.Dense(1, name='ifo')
        self.user_second_order_weights = tf.Variable(tf.random_uniform_initializer()((1, embedding_size)))
        self.item_second_order_weights = tf.Variable(tf.random_uniform_initializer()((1, embedding_size)))
        self.second_order_dense = L.Dense(1, name='so')

    def hrr1(self, a, b):
        a = tf.complex(a, .0 * a)
        b = tf.complex(b, .0 * b)
        return tf.math.real(tf.signal.ifft(tf.signal.fft(a) * tf.signal.fft(b)), name='hrr1')

    def hrr2(self, a, b):
        a = tf.complex(a, .0 * a)
        b = tf.complex(b, .0 * b)
        return tf.math.real(tf.signal.ifft(tf.math.conj(tf.signal.fft(a)) * tf.signal.fft(b)), name='hrr2')

    def second_order_features(self, user_embeddings_outputs, item_embeddings_outputs):
        outputs = tf.zeros((1,))
        for i in range(1):
            hrr1 = self.hrr1(self.user_second_order_weights[i: i + 1, :],
                             self.item_second_order_weights)
            hrr2 = self.hrr2(self.user_second_order_weights[i: i + 1, :],
                             self.item_second_order_weights)
            outputs += (hrr1 * tf.reduce_sum((user_embeddings_outputs[:, i: i + 1, :]
                                              * item_embeddings_outputs),
                                             axis=-1,
                                             keepdims=True)
                        + hrr2 * tf.reduce_sum((user_embeddings_outputs[:, i: i + 1, :]
                                                * item_embeddings_outputs),
                                               axis=-1,
                                               keepdims=True))
        return outputs

    def call(self, inputs, training=None, mask=None):
        user_inputs, item_inputs = inputs

        user_embeddings_outputs = self.user_embeddings(user_inputs)  # B * 1 * E
        item_embeddings_outputs = self.item_embeddings(item_inputs)  # B * 1 * E
        user_first_order_outputs = self.user_first_order_dense(user_embeddings_outputs)  # B * 1 * 1
        item_first_order_outputs = self.item_first_order_dense(item_embeddings_outputs)  # B * 1 * 1

        second_order_outputs = self.second_order_features(user_embeddings_outputs, item_embeddings_outputs)  # B * 1 * E
        second_order_outputs = self.second_order_dense(second_order_outputs)  # B * 1 * 1

        outputs = (self.bias
                   + user_first_order_outputs
                   + item_first_order_outputs
                   + second_order_outputs)
        outputs = tf.squeeze(outputs, axis=1)  # B * 1
        return outputs
