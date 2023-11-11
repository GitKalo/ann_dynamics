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

epochs_eff = 200

train_args = dict(
    batch_size = 120,
    n_sample_points = epochs_eff,     # Sample at every effective epoch
    return_loss=False,
    lr=1
)
train_args['xs'], train_args['ys'] = iris_train.get_nonpermuted_sequences(x_train, y_train, epochs_eff)

epsilons = np.logspace(-14, -1, 14, base=10)

n_init = 500
n_pert = 5

tic = time.time()
print(f"Starting...")

with h5py.File('results/eos_exponent_eps.h5', 'a') as f :
    attrs_dict = dict(**{k : v for k, v in train_args.items() if k not in ['xs', 'ys']}, n_init=n_init, n_pert=n_pert, epochs_eff=epochs_eff, epsilons=epsilons)
    for k, v in attrs_dict.items() : f.attrs[k] = v
    res_shape = (len(epsilons), n_init, 1+n_pert, train_args['n_sample_points'])
    ws_shape = (input_dim+1 + h_dim+1), (h_dim + output_dim)
    weights_ds = f.require_dataset('weights', res_shape + ws_shape, dtype=np.float64)
    # loss_ds = f.require_dataset('loss', res_shape, dtype=np.float64)

    for i_eps, eps in enumerate(epsilons) :
        for i_init in range(n_init) :
            w_init = iris_train.get_random_init(input_dim, h_dim, output_dim)
            ws = iris_train.train_multiple_seq(n_pert, w_init, eps, **train_args)
            weights_ds[i_eps, i_init, ...] = iris_train.weights_list_to_matrices(ws, *iho)

        toc = time.time()
        print(f"Finished {i_eps+1}/{len(epsilons)} eps after {toc-tic:.2f}s")

    # ls = iris_train.get_loss_from_weights_list(ws, model, x_train, y_train)
    # loss_ds[:] = np.concatenate([ls_prev, ls])

toc = time.time()
print(f"Finished in {toc - tic:.2f}s")