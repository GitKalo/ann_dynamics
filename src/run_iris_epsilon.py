import time, h5py
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

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

###
# Batch size 120, no permutation
###

epsilons = np.logspace(-18, 3, 22, base=10)
# n_init = 10
n_init = 1
# n_pert = 20
n_pert = 2
# epochs_eff = 5000
epochs_eff = 5

train_args = dict(
    batch_size = 120,
    n_sample_points = epochs_eff,     # Sample at every effective epoch
    return_loss = False
)

train_args['xs'], train_args['ys'] = iris_train.get_nonpermuted_sequences(x_train, y_train, epochs_eff)

# rng = np.random.default_rng()

tic = time.process_time()
toc = tic

with h5py.File('results/epsilon_full.h5', 'a') as f :
    attrs_dict = dict(**{k : v for k, v in train_args.items() if k not in ['xs', 'ys']}, n_init=n_init, n_pert=n_pert, epochs_eff=epochs_eff, epsilons=epsilons)
    for k, v in attrs_dict.items() : f.attrs[k] = v
    res_shape = (len(epsilons), n_init, 1+n_pert, train_args['n_sample_points'])
    ws_shape = (input_dim+1 + h_dim+1), (h_dim + output_dim)
    weights_ds = f.require_dataset('weights', res_shape + ws_shape, dtype=np.float64)
    # loss_ds = f.require_dataset('loss', res_shape, dtype=np.float64)

    for i_init in range(n_init) :
        w1_init = iris_train.get_random_init(input_dim, h_dim, output_dim)
        for i_eps, eps in enumerate(epsilons) :
            # ws, ls = iris_train.train_multiple_seq(n_pert, w1_init, eps, **train_args)
            ws = iris_train.train_multiple_seq(n_pert, w1_init, eps, **train_args)
            weights_ds[i_eps, i_init, ...] = iris_train.weights_list_to_matrices(ws, input_dim, h_dim, output_dim)
            # loss_ds[i_eps, i_init, ...] = np.array(ls)

        print(f"Finished init {i_init} in {time.process_time() - toc:.2f}s")
        toc = time.process_time()

toc = time.process_time()
print(f"Finished all runs in {toc - tic:.2f}s")