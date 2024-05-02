import h5py, time
import numpy as np
import tensorflow.keras as keras

from mnist_train import MNISTTrain

def pert_n_wi(ws_ref, n, eps) :
    # Assume w_ref in weights_list format
    ws_pert = [w.copy() for w in ws_ref]
    
    # ws_ref = ws_ref[::2]      # Will only perturb kernel weights
    
    ### TEMPORARY: only perturb (first) hidden layer weights
    ws = ws_pert[0]
    # print(ws.shape)
    original_shape = ws.shape
    ws = ws.flatten()
    # print(ws.shape)
    # Pick n indeces without replacement
    indeces = np.random.choice(range(ws.size), n, replace=False)
    pert_values = np.random.random_sample(n) * (2 * eps) - eps
    # print(pert_values.shape)
    for i_pert, ws_idx in enumerate(indeces) :
        ws[ws_idx] += pert_values[i_pert]
    ws = ws.reshape(original_shape)
    ws_pert[0] = ws
    ###

    return ws_pert

# Load and prepare training data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Use subset of data for training
train_subset_size = 5000
rng = np.random.default_rng()
perm_index = rng.permutation(x_train.shape[0])[:train_subset_size]    # Permute whole dataset and take first n as new index
x_train = x_train[perm_index,:]
y_train = y_train[perm_index]

train_args = dict(
    batch_size = x_train.shape[0],  # Full-batch, i.e. deterministic GD
    # epochs = 1000,
    epochs = 100,
    # n_sample_points = 10,
    x_train = x_train,
    y_train = y_train,
    x_test = x_test,
    y_test = y_test,
)

hidden_layers = [64]
mnist = MNISTTrain(
    h_dim=hidden_layers,
    **train_args
)

n_init = 20
eps = 10**-8

first, last = 0, 15
n_wi = np.logspace(first, last, last-first+1, base=2).astype(int)
n_pert = len(n_wi)

lrs = [0.01, 1, 2, 5]
n_lr = len(lrs)

output_filepath = './results/mnist_epsilon_limit.h5'

# Currently only saving at each epoch — n_sample_points to be implemented
with h5py.File(output_filepath, 'w') as outfile :
    # Write training and other parameters as attributes of HDF5 file
    mnist.write_params(outfile, n_init=n_init, n_pert=n_pert, eps=eps, lrs=lrs, n_wi=n_wi, train_size=train_subset_size)
    del outfile.attrs['lr']     # Since I am running for multiple lrs, no
                                # need to keep lr of dummy object

    ws_shape = mnist.get_ws_shape()
    weights_ds = outfile.require_dataset('weights', (n_lr, n_init, 1+n_pert, 2) + ws_shape, dtype=np.float64)
    dists_ds = outfile.require_dataset('dists', (n_lr, n_init, n_pert, train_args['epochs']), dtype=np.float64)
    res_shape = (n_lr, n_init, 1+n_pert, train_args['epochs'])
    loss_train_ds = outfile.require_dataset('loss_train', res_shape, dtype=np.float64)
    acc_train_ds = outfile.require_dataset('acc_train', res_shape, dtype=np.float64)
    loss_test_ds = outfile.require_dataset('loss_test', res_shape, dtype=np.float64)
    acc_test_ds = outfile.require_dataset('acc_test', res_shape, dtype=np.float64)

    print("Starting runs...")

    tic = time.time()
    toc = tic
    lr_toc = tic
    
    ### For each LR, train the same initial conditions
    for i_init in range(n_init) :
        # Generate perturbed initial weights for differen n_wi
        init_ref = mnist.get_random_init()
        perts_inits = []
        for n in n_wi :
            perts_inits.append(pert_n_wi(init_ref, n, eps))

        for i_lr, lr in enumerate(lrs) :
            train_args['lr'] = lr
            mnist = MNISTTrain(h_dim=hidden_layers, **train_args)
            
            dists, ws, (loss_train, acc_train), (loss_test, acc_test) = mnist.train_custom_perts(init_ref, perts_inits)
            
            # Record
            dists_ds[i_lr,i_init] = dists
            weights_ds[i_lr,i_init] = ws
            loss_train_ds[i_lr,i_init] = loss_train
            acc_train_ds[i_lr,i_init] = acc_train
            loss_test_ds[i_lr,i_init] = loss_test
            acc_test_ds[i_lr,i_init] = acc_test

            print(f"Finished LR ({i_lr}) = {lr} in {time.time() - lr_toc:.2f}s")
            lr_toc = time.time()
        print(f"Finished init {i_init} in {time.time() - toc:.2f}s")
        toc = time.time()
    print(f"Finished all runs in {time.time() - tic:.2f}s")
