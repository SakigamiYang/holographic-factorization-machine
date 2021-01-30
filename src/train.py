# coding: utf-8
from os import cpu_count

import numpy as np

from constant import DATA_DIR, MODEL_DIR
from hfm import HFM

train_x = np.loadtxt(DATA_DIR / 'train_x', dtype=np.int, delimiter=',')
train_y = np.loadtxt(DATA_DIR / 'train_y', dtype=np.float, delimiter=',')
eval_x = np.loadtxt(DATA_DIR / 'eval_x', dtype=np.int, delimiter=',')
eval_y = np.loadtxt(DATA_DIR / 'eval_y', dtype=np.float, delimiter=',')

num_users = 73516
num_animes = 34519

hfm = HFM(num_users, num_animes, embedding_size=64)
hfm.compile(loss='mse', metrics=['mae', 'mse'])
hfm.fit(
    x=[train_x[:, :1], train_x[:, 1:]],
    y=train_y,
    batch_size=4096,
    epochs=20,
    verbose=2,
    validation_data=([eval_x[:, :1], eval_x[:, 1:]], eval_y),
    validation_batch_size=256,
    workers=cpu_count() - 2,
    use_multiprocessing=True
)
hfm.save(MODEL_DIR / 'hfm', save_format='tf')
