import tensorflow as tf


def coef_determination(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_pred, _ = tf.unstack(y_pred)
    SS_res = tf.reduce_sum(tf.square(y_true-y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res/(SS_tot + tf.convert_to_tensor(1e-8, dtype=y_pred.dtype)))
