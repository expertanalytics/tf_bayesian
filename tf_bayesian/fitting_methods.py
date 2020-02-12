import tensorflow as tf
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.training_arrays import _get_num_samples_or_steps
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.utils.mode_keys import ModeKeys
import time
import glob
import os

import numpy as np

from tf_bayesian.utils import BatchManager




def tfrecords_fit(
                 model,
                 input_directory,
                 targets=None,
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

    input_paths = glob.glob(os.path.join(input_directory, "*.tfrecords"))
    path_queue = tf.train.string_input_producer(input_paths, shuffle=mode == ModeKeys.TRAIN)

    def _num_examples(filenames):
        c = 0
        for fn in filenames:
            for _ in tf.python_io.tf_record_iterator(fn):
                c += 1
        return c

    if not steps_per_epoch:
        steps_per_epoch = np.ceil(_num_examples(path_queue)/batch_size)
    num_samples_or_steps = steps_per_epoch

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

    def _build_dataset(path_queue, mode, batch_size, parse_examples_batch):
        dataset = tf.data.Dataset.list_files(
            path_queue
        ).interleave(
            tf.TFRecordDataset
        ).shuffle(
            mode == ModeKeys.TRAIN
        ).batch(
            batch_size=batch_size,
            drop_remainder=True
        ).map(
            map_func=parse_examples_batch
        ).cache(
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        return dataset

    def _parse_examples_batch(examples):
        data_columns = {
            "features": tf.io.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
            "label": tf.io.FixedLenFeature((), tf.float32, -1)
        }
        return tf.io.parse_example(examples, data_columns)

    def _preprocess(data):
        # do some processing
        return data

    def make_iterator(path_queue):
        dataset = _build_dataset(path_queue, mode, batch_size, _parse_examples_batch())
        samples = dataset.make_one_shot_iterator().get_next()
        yield _preprocess(samples)

    def fit_data(self):
        pass

    # TODO: Make batch manager that can handle TF data as well as generator function. Should be able to handle both
    # TODO: as well as creating a superclass for all fitting methods. Will ease usage substantially.
    # TODO: Fitting for both ndarray, generator and TF-data should be combined to one function/class.

    return


def ndarray_fit(
        model,
        inputs,
        targets=None,
        sample_weights=None,
        batch_size=None,
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
    """
    Fitting procedure for a model given numpy iterables
    """
    # TODO: build support for TF datasets
    do_validation = val_inputs is not None
    is_dataset = isinstance(
        inputs, (dataset_ops.DatasetV1, dataset_ops.DatasetV2))

    if "steps" in kwargs:
        steps_per_epoch = kwargs.pop("steps")
    elif kwargs:
        raise ValueError("Unknown argument {}".format(kwargs))
    if not is_dataset:
        num_samples_or_steps = np.ceil(inputs.shape[0] / batch_size)
    else:
        num_samples_or_steps = steps_per_epoch

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

    if targets is None:
        targets = []
    else:
        if not isinstance(targets, list):
            targets = [targets]

    aggregator = training_utils.MetricsAggregator(
        True,
        num_samples=None if steps_per_epoch else num_samples_or_steps,
        steps=steps_per_epoch
        )

    for e in range(initial_epoch, epochs):
        epoch_logs = {}
        progbar.on_epoch_begin(e, epoch_logs)

        model.reset_metrics()
        if mode == ModeKeys.TRAIN:
            callbacks.on_epoch_begin(e, epoch_logs)

        bm = BatchManager(inputs.shape[0], batch_size, shuffle=shuffle)
        posterior_weight = tf.constant(inputs.shape[0], model.dtype)

        for batch_index, batch in enumerate(bm):
            batch_logs = {"batch": batch_index, "size": 1}
            progbar.on_batch_begin(batch_index, batch_logs)
            callbacks._call_batch_hook(mode, "begin", batch_index, batch_logs)
            # TODO: rewrite to not do boolean check each loop
            if targets:
                ybatch = targets[0][batch]
            xbatch = inputs[batch]
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

