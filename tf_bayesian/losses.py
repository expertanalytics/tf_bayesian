import tensorflow as tf
import tensorflow_probability as tfp


class BayesianMeanSquaredError(tf.keras.losses.Loss):
    def __init__(
            self,
            ):
        super(BayesianMeanSquaredError, self).__init__()

    @tf.function
    def call(self, y_true, pred):
        """
        Computes the loss for a model outputing yhat and log sigma^2
        in accordance with eq. 8 in https://arxiv.org/pdf/1703.04977.pdf
        """
        y_pred, log_var = tf.unstack(pred, axis=0)
        diff = tf.square(y_true - y_pred)

        pred_dim = tf.size(tf.shape(y_pred))
        var_dim = tf.size(tf.shape(log_var))
        # reduce_dims = - tf.range(1, pred_dim)
        if tf.equal(pred_dim, var_dim):
            reduce_diff = diff
        else:
            reduce_diff = tf.reduce_sum(diff, axis=-1)

        oneover_var = tf.exp(-log_var)
        internal_prod = tf.math.multiply(oneover_var, reduce_diff)
        loss_val = tf.constant(0.5, dtype=y_pred.dtype) * \
            tf.reduce_sum(internal_prod + log_var)
        return loss_val / tf.dtypes.cast(tf.shape(y_true)[0], pred.dtype)

class BayesianStochasticLoss(tf.keras.losses.Loss):
    """
    Computes the loss for a model outputing f^w and sigma^2
    in accordance with eq. 12 in https://arxiv.org/pdf/1703.04977.pdf
    """
    def __init__(
            self,
            ):
        super(BayesianStochasticLoss, self).__init__()

    @tf.function
    def call(self, x_true, pred):
        f_w, var = tf.unstack(pred, axis=0)
        loss = []
        for i in f_w.shape[0]:
            norm_distr_diag =  tf.squeeze(
                    tf.linalg.diag(
                    tf.random.normal(
                    shape, mean=0, stddev=1)))
            x_hat = f_w + tf.math.multiply(var, norm_distr_diag)
            log_sum = tf.log(tf.reduce_sum(tf.exp(x_hat), axis=-1, keepdims=True))
            exp_sum = tf.reduce_sum(tf.exp(x_hat - log_sum), axis=-1)
            loss.append(tf.log(1/(tf.shape(x_true)[0]) * exp_sum))
    return tf.reduce_sum(loss, axis=0)


class EpistemicMeanSquaredError(tf.keras.losses.Loss):
    def __init__(
            self,
            model,
            ):
        self.model = model
        super(EpistemicMeanSquaredError, self).__init__()

    @tf.function
    def call(self, y_true, pred):
        """
        Computes the  MSE and sums up model lossses
        """
        y_pred, *_ = tf.unstack(pred, axis=0)
        diff = tf.square(y_true - y_pred)
        mse = tf.reduce_mean(diff)
        return mse
