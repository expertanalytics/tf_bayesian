import tensorflow as tf 
import numpy as np

from tf_bayesian.utils import BatchManager

class MetricLogger(tf.keras.callbacks.Callback):

    def __init__(self, metrics, datasets):
        super(MetricLogger, self).__init__()
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = metrics
        self.datasets = datasets
        self.epoch_metric_logs = []

    def on_train_begin(self, logs={}):
        self.metrics_logs = []

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
        self.epoch_metric_logs.append(tmp_metrics.mean(0))
