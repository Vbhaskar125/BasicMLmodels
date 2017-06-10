####################################################################################################
## A simple feed forward network using tensorflow and some of its visualization tools
##Architecture
## 2 hidden layers 1 input and 1 output layers
## input layer : 9 neurons corresponding to season, mnth,holiday,workingday, weathersit, temp, atemp, hum, windspeed
##hidden layers with 5 and 3 neurons respectively
##output neuron. This is a regression type of problem where the output value predicts the answer "cnt" in the dataset.
####################################################################################################

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
