import tensorflow as tf
import numpy as np

from tf_bayesian.utils import BatchManager
# TODO: Datasets should not be args to logger classes. Make attr of model instead?


class MetricLogger(tf.keras.callbacks.Callback):
    """Logger callback for a set of metrics. 

    This callback monitors the given metrics for each epoch on the
    supplied datasets. The dataset is implied to be a list with 
    inputs as the first elements and targets as  the second.
    """

    def __init__(self, metrics, datasets):
        super(MetricLogger, self).__init__()
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = metrics
        self.datasets = datasets
        self.metrics_logs = []

    def on_train_begin(self, logs={}):
        self.metrics_logs = []

    def on_train_end(self, logs={}):
        self.metrics_logs = np.array(self.metrics_logs)

    def on_epoch_end(self, e, logs={}):
        bm = BatchManager(self.datasets[0].shape[1], 100, shuffle=False)
        tmp_metrics = np.zeros((len(bm), len(self.metrics)))

        for i, batch in enumerate(bm):
            xbatch = self.datasets[0][batch]
            ybatch = self.datasets[1][batch]
            batch_outs = self.model(xbatch)
            for j, metric in enumerate(self.metrics):
                measured = metric(ybatch, batch_outs)
                tmp_metrics[i, j] = measured
        self.metrics_logs.append(tmp_metrics.mean(0))


class StdLogger(tf.keras.callbacks.Callback):
    """Logger callback for the standard deviation of a bayesian network.

    Monitors the quality of the standard error predicitons of a
    bayesian neural net
    """

    def __init__(self, datasets):
        super(StdLogger, self).__init__()
        self.datasets = datasets
        self.std_log = None
        self.z_log = None

    def on_train_begin(self, logs={}):
        self.std_log = []
        self.z_log = []

    def on_epoch_end(self, e, logs={}):
        bm = BatchManager(self.datasets[0].shape[1], 100, shuffle=False)
        z_arr = np.zeros(
            [self.datasets[0].shape[0]] + list(self.datasets[1].shape[1:]))
        for batch in bm:
            xbatch = self.datasets[0][batch]
            ybatch = self.datasets[1][batch]

            stds = self.model.std(xbatch)
            means = self.model.predict_mean(xbatch)
            z = ((ybatch - means)/stds)
            z_arr[batch] = z
        self.z_log.append(z_arr)

class InferenceLogger(tf.keras.callbacks.Callback):
    """
    """
    
    def __init__(self):
        super(InferenceLogger, self).__init__()
        self.loss_logs = []
    
    def on_epoch_end(self, e, logs={}):
        to_log = tf.reduce_sum(self.model.losses).numpy()
        self.loss_logs.append(to_log)

    def on_train_end(self, logs={}):
        self.loss_logs = np.array(self.loss_logs)
