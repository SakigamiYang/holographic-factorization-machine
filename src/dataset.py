# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constant import DATA_DIR

data_file = DATA_DIR / 'rating.csv'
column_types = {'user_id': int, 'anime_id': int, 'rating': float}
df = pd.read_csv(data_file, delimiter=',').astype(column_types)

print(f"#users = {df['user_id'].max()}")
print(f"#animes = {df['anime_id'].max()}")

df = df[df['rating'] >= 0]
df['user_id'] = df['user_id'].apply(lambda x: x - 1)
df['anime_id'] = df['anime_id'].apply(lambda x: x - 1)
df['rating'] = df['rating'].apply(lambda x: x / 10.0)

print(f'#samples = {len(df)}')

train_set, test_set = train_test_split(df.to_numpy(), test_size=10000)
train_set, eval_set = train_test_split(train_set, test_size=10000)

print(f'#train = {len(train_set)}')
print(f'#eval = {len(eval_set)}')
print(f'#test = {len(test_set)}')

np.savetxt(DATA_DIR / 'train_x', train_set[:, :-1], fmt='%d', delimiter=',')
np.savetxt(DATA_DIR / 'train_y', train_set[:, -1], fmt='%f', delimiter=',')
np.savetxt(DATA_DIR / 'eval_x', eval_set[:, :-1], fmt='%d', delimiter=',')
np.savetxt(DATA_DIR / 'eval_y', eval_set[:, -1], fmt='%f', delimiter=',')
np.savetxt(DATA_DIR / 'test_x', test_set[:, :-1], fmt='%d', delimiter=',')
np.savetxt(DATA_DIR / 'test_y', test_set[:, -1], fmt='%f', delimiter=',')
