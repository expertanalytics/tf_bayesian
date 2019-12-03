import tensorflow as tf
import tensorflow_probability as tfp


class BayesianMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, model_obj):
        self.model = model_obj
        self.one_half = tf.constant(0.5, dtype=self.model.dtype)
        super(BayesianMeanSquaredError, self).__init__()

    @tf.function
    def call(self, y_true, pred):
        """
        Computes the loss for a model outputing yhat and log sigma^2
        in accordance with eq. 8 in https://arxiv.org/pdf/1703.04977.pdf
        """
        kld = tf.reduce_sum(self.model.losses)
        y_pred, log_var = tf.unstack(pred, num=2)
        var_m2 = tf.exp(-log_var)
        diff = tf.square(y_true - y_pred)
        # reduce_diff = tf.reduce_sum(diff, axis=1)
        reduce_diff = diff
        internal_prod = tf.math.multiply(var_m2, reduce_diff)
        loss_val = self.one_half * tf.reduce_sum(internal_prod + log_var)
        return kld + loss_val
