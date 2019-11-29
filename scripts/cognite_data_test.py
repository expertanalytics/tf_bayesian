import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from cognite.client import CogniteClient
from sklearn.model_selection import train_test_split

from tf_bayesian.models import BayesianModel
from tf_bayesian.losses import BayesianMeanSquaredError

tf.keras.backend.set_floatx('float32')


class BayesianConvNet(BayesianModel):
    def __init__(self, out_nodes, activation=tf.keras.activations.relu):
        super(BayesianConvNet, self).__init__()
        self.conv1 = tfp.layers.Convolution1DFlipout(
            20, 5, padding="same", activation=activation)
        self.conv2 = tfp.layers.Convolution1DFlipout(
            20, 5, padding="same", activation=activation)
        self.conv3 = tfp.layers.Convolution1DFlipout(
            20, 5, padding="same", activation=activation)
        self.flatten = tf.keras.layers.Flatten()
        self.out_dense = tf.keras.layers.Dense(out_nodes)
        self.out_var = tf.keras.layers.Dense(out_nodes)
        self.loss_val = None
        self.grads = [None]

    def call(self, x):
        hidden = x
        for conv in [self.conv1, self.conv2, self.conv3]:
            hidden = conv(hidden)
        flat_tensor = self.flatten(hidden)
        out_pred = self.out_dense(flat_tensor)
        out_var = self.out_var(flat_tensor)
        return tf.stack([out_pred, out_var])


def construct_data(timeseries, window_length, future_steps, y_cols=[]):
    if timeseries.shape[0] < window_length:
        raise ValueError("Window cannot be larger than timeseries")
    N = timeseries.shape[0] - future_steps - window_length
    data = []
    targets = []
    for i in range(N):
        tmp = i + window_length
        data.append(timeseries[i: tmp])
        if y_cols:
            selected = timeseries[tmp-1: tmp + future_steps, y_cols]
            targets.append(selected)
        else:
            targets.append(timeseries[tmp-1: tmp + future_steps])
    return np.squeeze(np.array(data)), np.squeeze(np.array(targets))


WINDOW = 50
FUTURE = 2
TIME_FRAME = {"start": "4w-ago", "end": "1d-ago",
              "aggregates": ["average"], "granularity": "1h"}
KEYS = list(TIME_FRAME.keys())
KEYS.sort()
DATA_FN_ROOT = ""
for k in KEYS:
    DATA_FN_ROOT += k + "_" + str(TIME_FRAME[k]) + "_"
DATA_FN_ROOT += "window_" + str(WINDOW) + "_"
DATA_FN_ROOT += "future_" + str(FUTURE)
DATA_FN = DATA_FN_ROOT + "_data.npy"
TARGET_FN = DATA_FN_ROOT + "_targets.npy"
try:
    X = np.load("./data/"+DATA_FN,)
    Y = np.load("./data/"+TARGET_FN,)
    print("Loading data from Disk")
except FileNotFoundError:
    C = CogniteClient(project="publicdata", client_name="test")
    asset_ids = [
        786220428505816,
        4840206559741735,
        2814662602621825,
    ]
    TS = C.assets.retrieve_multiple(ids=asset_ids).time_series()
    TS_IDS = [ts.id for ts in TS]
    DATA = C.datapoints.retrieve(id=TS_IDS, **TIME_FRAME).to_pandas()
    DATA.fillna(method="bfill", inplace=True)
    DATA = DATA.values
    print("Loading data from Cognite API")
    X, Y = construct_data(DATA, WINDOW, FUTURE, y_cols=[0])
    np.save("./data/"+DATA_FN, X)
    np.save("./data/"+TARGET_FN, Y)
"""
-------- DATA PREPROC -------
"""

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, train_size=0.8)
print("Proc data shapes")
print("yshp", Ytr.shape)
print("xshp", Xtr.shape)
Xmu = Xtr.mean(axis=0, keepdims=True)
Xsigma = Xtr.std(axis=0, keepdims=True)
Xtr = (Xtr - Xmu)/Xsigma
Xte = (Xte - Xmu)/Xsigma
if len(Xtr.shape) == 1:
    Xtr = np.expand_dims(Xtr, axis=-1)
    Xte = np.expand_dims(Xte, axis=-1)

Ymu = Xmu[0, 0, 0]
Ysigma = Xsigma[0, 0, 0]
Ytr = (Ytr - Ymu)/Ysigma
Yte = (Yte - Ymu)/Ysigma


"""
-------- MODEL SPEC ---------
"""
ETA = 1e-3
BATCH_SIZE = 74
BUFFER_SIZE = 1024
EVALUATION_INTERVAL = 10
EPOCHS = 3

"same size batches"
# IN_TENSOR, OUT = model_constructor(Xtr, Ytr)
# MODEL_INST = tf.keras.models.Model(inputs=IN_TENSOR, outputs=OUT)
MODEL_INST = BayesianConvNet(Ytr.shape[1])
MODEL_INST.compile(
        loss=BayesianMeanSquaredError(MODEL_INST),
        optimizer=tf.keras.optimizers.Adam(),
        experimental_run_tf_function=False,
        )
retval = MODEL_INST.fit(Xtr, Ytr, batch_size=BATCH_SIZE, epochs=EPOCHS)
print(retval.history.keys())
