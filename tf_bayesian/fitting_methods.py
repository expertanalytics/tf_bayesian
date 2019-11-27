import tensorflow as tf
from tf_bayesian.utils import BatchManager


def ndarray_fit(
        model=None,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        **kwargs,
):
    """
    Fitting procedure for a model given numpy iterables
    """
    if y is None:
        outputs = []
    else:
        outputs = [y]

    for e in range(epochs):
        print("EPOCH ", e)
        bm = BatchManager(x.shape[0], batch_size, shuffle=True)
        for batch in bm:
            # TODO: rewrite to not do boolean check each loop
            if outputs:
                ybatch = outputs[0][batch]
            xbatch = x[batch]
            model.compute_grads(xbatch, ybatch)
            model.optimizer.apply_gradients(
                zip(model.grads, model.trainable_variables))
