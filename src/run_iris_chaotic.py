import time, h5py
import numpy as np

import tensorflow_datasets as tfds

import iris_train

iris_ds = tfds.load('iris', split='train', batch_size=-1)

x = iris_ds['features'].numpy()
y = iris_ds['label'].numpy()

x_train, x_test = x[:120], x[120:]
y_train, y_test = y[:120], y[120:]

input_dim = 4
h_dim = 10
output_dim = 3
iho = (input_dim, h_dim, output_dim)

###
# Batch size 120, no permutation
###

epochs_eff = 10000

train_args = dict(
    batch_size = 120,
    n_sample_points = epochs_eff,     # Sample at every effective epoch
    return_loss = True,
    lr=5
)

train_args['xs'], train_args['ys'] = iris_train.get_nonpermuted_sequences(x_train, y_train, epochs_eff)

n_init = 20

tic = time.process_time()
toc = tic

with h5py.File('results/chaotic.h5', 'a') as f :
    attrs_dict = dict(**{k : v for k, v in train_args.items() if k not in ['xs', 'ys']}, n_init=n_init, epochs_eff=epochs_eff)
    for k, v in attrs_dict.items() : f.attrs[k] = v
    res_shape = (n_init, train_args['n_sample_points'])
    ws_shape = (input_dim+1 + h_dim+1), (h_dim + output_dim)
    weights_ds = f.require_dataset('weights', res_shape + ws_shape, dtype=np.float64)
    loss_ds = f.require_dataset('loss', res_shape, dtype=np.float64)

    for i_init in range(n_init) :
        w1_init = iris_train.get_random_init(input_dim, h_dim, output_dim)
        ws, ls = iris_train.train(initial_weights=w1_init, **train_args)
        weights_ds[i_init, ...] = np.squeeze(iris_train.weights_list_to_matrices([ws], *iho), axis=0)
        loss_ds[i_init, ...] = ls

        print(f"Finished init {i_init} in {time.process_time() - toc:.2f}s")
        toc = time.process_time()

toc = time.process_time()
print(f"Finished all runs in {toc - tic:.2f}s")