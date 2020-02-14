import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.training_arrays import _get_num_samples_or_steps
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.utils.mode_keys import ModeKeys


def tfrecords_fit(
                 model,
                 input_directory,
                 targets=False,
                 sample_weights=None,
                 batch_size=1,
                 epochs=1,
                 verbose=1,
                 callbacks=None,
                 val_inputs=None,
                 val_targets=None,
                 val_sample_weights=None,
                 shuffle=True,
                 initial_epoch=0,
                 steps_per_epoch=None,
                 validation_steps=None,
                 validation_freq=1,
                 mode=ModeKeys.TRAIN,
                 validation_in_fit=False,
                 prepared_feed_values_from_dataset=False,
                 steps_name='steps',
                 **kwargs
                 ):

    # Currently not used
    do_validation = val_inputs is not None

    if "steps" in kwargs:
        steps_per_epoch = kwargs.pop("steps")
    elif kwargs:
        raise ValueError("Unknown argument {}".format(kwargs))

    TFR = TFRecordGenerator(input_directory, mode, batch_size, steps_per_epoch)
    data_generator = TFR.make_iterator()

    num_samples_or_steps = TFR.steps_per_epoch
    num_samples = TFR.num_examples


    callbacks = cbks.configure_callbacks(
        callbacks,
        model,
        do_validation=do_validation,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        samples=num_samples_or_steps,
        verbose=0,  # Handle ProgBarLogger separately in this loop.
        mode=mode
    )
    count_mode = "samples"
    progbar = training_utils.get_progbar(model, count_mode)
    progbar.params = callbacks.params
    progbar.params['verbose'] = verbose

    callbacks.model.stop_training = False
    callbacks._call_begin_hook(mode)
    progbar.on_train_begin()

    aggregator = training_utils.MetricsAggregator(
        True,
        num_samples=num_samples,
        steps=steps_per_epoch
        )

    for e in range(initial_epoch, epochs):
        epoch_logs = {}
        progbar.on_epoch_begin(e, epoch_logs)

        model.reset_metrics()
        if mode == ModeKeys.TRAIN:
            callbacks.on_epoch_begin(e, epoch_logs)

        iterable = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(iterable)

        posterior_weight = tf.constant(num_samples, model.dtype)

        for batch_index, batch in enumerate(iterable):
            batch_logs = {"batch": batch_index, "size": 1}
            progbar.on_batch_begin(batch_index, batch_logs)
            callbacks._call_batch_hook(mode, "begin", batch_index, batch_logs)
            # TODO: rewrite to not do boolean check each loop
            if targets:
                ybatch = data_generator[batch][1]
            else:
                ybatch = None
            xbatch = data_generator[batch][0]
            batch_outs = model.compute_grads(xbatch, ybatch, posterior_weight)
            if mode == ModeKeys.TRAIN:
                model.optimizer.apply_gradients(
                    zip(model.grads, model.trainable_variables))

            if not isinstance(batch_outs, list):
                batch_outs = [batch_outs]
            if batch_index == 0:
                aggregator.create(batch_outs)

            aggregator.aggregate(batch_outs)
            batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
            progbar.on_batch_end(batch_index, batch_logs)
            callbacks._call_batch_hook(mode, "end", batch_index, batch_logs)

        aggregator.finalize()
        results = aggregator.results
        epoch_logs = cbks.make_logs(model, epoch_logs, results, mode)
        progbar.on_epoch_end(e, epoch_logs)
        if mode == ModeKeys.TRAIN:
            # Epochs only apply to `fit`.
            callbacks.on_epoch_end(e, epoch_logs)

        if len(results) == 1:
            results = results[0]
    callbacks._call_end_hook(mode)
    print("")
    if mode == ModeKeys.TRAIN:
        return model.history
    return results


    # TODO: Make batch manager that can handle TF data as well as generator function. Should be able to handle both
    # TODO: as well as creating a superclass for all fitting methods. Will ease usage substantially.
    # TODO: Fitting for both ndarray, generator and TF-data should be combined to one function/class.


class TFRecordGenerator:
    """ Creating a generator obejct from TFRecords. Currently locked to a specific layout as
        dataset structure must be defined in _parse_examples_batch"""

    # TODO: Make _parse_examples_batch customizable by user.
    # TODO: _preprocess is data depedant and should be defined by user.

    def __init__(self, input_directory, mode, batch_size, steps_per_epoch):
        input_paths = glob.glob(os.path.join(input_directory, "*.tfrecords"))
        self.path_queue = tf.train.string_input_producer(input_paths, shuffle=self.mode == ModeKeys.TRAIN)

        self.mode = mode
        self.batch_size = batch_size
        self.num_examples = self._num_examples(self.path_queue)

        if steps_per_epoch is None:
            self.steps_per_epoch = np.ceil(self.num_examples / self.batch_size)

    def _build_dataset(self, path_queue):
        dataset = tf.data.Dataset.list_files(
            path_queue
        ).interleave(
            tf.TFRecordDataset
        ).shuffle(
            self.mode == ModeKeys.TRAIN
        ).batch(
            batch_size=self.batch_size,
            drop_remainder=True
        ).map(
            map_func=self._parse_examples_batch
        ).cache(
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        return dataset

    @staticmethod
    def _num_examples(filenames):
        c = 0
        for fn in filenames:
            for _ in tf.python_io.tf_record_iterator(fn):
                c += 1
        return c

    @staticmethod
    def _parse_examples_batch(examples):
        data_columns = {
            "features": tf.io.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
            "label": tf.io.FixedLenFeature((), tf.float32, -1)
        }
        return tf.io.parse_example(examples, data_columns)

    @staticmethod
    def _preprocess(data):
        # do some processing
        return data

    def make_iterator(self):
        dataset = self._build_dataset(self.path_queue)
        samples = dataset.make_one_shot_iterator().get_next()
        if self.mode == ModeKeys.TRAIN:
            tf.shuffle(samples)
        yield self._preprocess(samples)


