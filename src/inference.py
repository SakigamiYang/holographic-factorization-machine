# coding: utf-8
from os import cpu_count

import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model, Model

from constant import DATA_DIR, MODEL_DIR

test_x = np.loadtxt(DATA_DIR / 'test_x', dtype=np.int, delimiter=',')
test_y = np.loadtxt(DATA_DIR / 'test_y', dtype=np.float, delimiter=',')

hfm: Model = load_model(MODEL_DIR / 'hfm')
y_pred = hfm.predict(
    x=[test_x[:, :1], test_x[:, 1:]],
    batch_size=1024,
    workers=cpu_count() - 2,
    use_multiprocessing=True
)

test_y = test_y.reshape((-1, 1))
y_diff = test_y - y_pred
result = np.hstack((test_y, y_pred, y_diff))
np.savetxt(DATA_DIR / 'result', result, fmt='%f', delimiter=',', header='real,pred,diff')
print(f'mse = {mean_absolute_error(test_y, y_pred)}')
