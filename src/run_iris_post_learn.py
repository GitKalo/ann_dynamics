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
iho = (input_dim, h_dim, output_dim)

###
# Batch size 120, no permutation
###

with h5py.File('results/epsilon_full.h5', 'r') as f :
    weights_ds = f['weights']
    
    epsilons = f.attrs['epsilons']
    n_init = f.attrs['n_init']
    n_pert = f.attrs['n_pert']
    epochs_eff = f.attrs['epochs_eff']

    post_learn_epochs = 1000

    ref_traj = weights_ds[:,:,0,-post_learn_epochs:,...]   # Weights of reference traj for last post_learn_epochs
    # final_weights_mat = weights_ds[:,:,1:,epochs_eff-post_learn_epochs,...]

    # final_weights = []
    # for eps in final_weights_mat :
    #     inits = []
    #     for init in eps :
    #         perts = []
    #         for pert in init :
    #             perts.append(iris_train.matrix_to_weights(pert, *iho))
    #         inits.append(perts)
    #     final_weights.append(inits)

breakpoint()

train_args = dict(
    batch_size = 120,
    n_sample_points = post_learn_epochs,     # Sample at every effective epoch
    return_loss = False
)

with h5py.File('results/post_learn.h5', 'a') as f :
    attrs_dict = dict(**{k : v for k, v in train_args.items() if k not in ['xs', 'ys']}, n_init=n_init, n_pert=n_pert, epochs_eff=epochs_eff, post_learn_epochs=post_learn_epochs, epsilons=epsilons)
    for k, v in attrs_dict.items() : f.attrs[k] = v
    trans_shape = (n_init, train_args['n_sample_points'])
    res_shape = (len(epsilons), n_init, 1+n_pert, post_learn_epochs)
    ws_shape = (input_dim+1 + h_dim+1), (h_dim + output_dim)
    trans_ds = f.require_dataset('transient', trans_shape + ws_shape, dtype=np.float64)
    weights_ds = f.require_dataset('weights', res_shape + ws_shape, dtype=np.float64)
    weights_ds[:,:,0] = ref_traj

    train_args['xs'], train_args['ys'] = iris_train.get_nonpermuted_sequences(x_train, y_train, post_learn_epochs)

    tic = time.time()
    toc = tic

    for i_eps, eps in enumerate(epsilons) :
        for i_init in range(n_init) :
            for i_pert in range(n_pert) :
                pert_init = iris_train.pert_uniform(iris_train.matrix_to_weights(ref_traj[i_eps,i_init,0], *iho), eps)    # Perturb first weight of reference traj sequence
                ws = iris_train.train(initial_weights=pert_init, **train_args)
                weights_ds[i_eps,i_init,i_pert+1] = np.squeeze(iris_train.weights_list_to_matrices([ws], *iho), axis=0)
        
        print(f"Finished eps {i_eps+1:2d}/{len(epsilons)} in {time.time() - toc:.2f}s")
        toc = time.time()

toc = time.time()
print(f"Finished all runs in {toc - tic:.2f}s")