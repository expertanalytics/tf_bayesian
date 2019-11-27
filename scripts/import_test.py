from tf_bayesian.losses import BayesianMeanSquaredError
from tf_bayesian.models import BayesianConvNet
import numpy as np

model = BayesianConvNet(1)
BayesianMeanSquaredError(model)
