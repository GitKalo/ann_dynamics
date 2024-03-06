import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import Callback

tf.keras.backend.set_floatx('float64')

from multiprocessing import Pool
# from multiprocess import Pool
from keras_pickle_wrapper import KerasPickleWrapper

INPUT_DIM = 4
OUTPUT_DIM = 3

ACTIVATION_DEFAULT = 'sigmoid'
SGD_LR_DEFAULT = 0.01

class WeightsCallback(Callback) :
    # Only works if epochs=1 !
    def __init__(self, epochs, batch_size, train_size, batch_sample_period) :
        self.batch_sample_period = batch_sample_period
        self.n_samples = (epochs * train_size) // (batch_size * batch_sample_period)

        if batch_sample_period == 1 :
            self.on_batch_end = self.on_batch_end_simple
        else :
            self.batch_sample_points = set(range(0, train_size // batch_size, batch_sample_period))
            self.on_batch_end = self.on_batch_end_samples
        
        self.weights_list = [None] * self.n_samples
        # self.batch_loss = np.empty(self.n_samples)
    
    def on_batch_end_simple(self, batch, logs=None) :
        self.weights_list[batch] = self.model.get_weights()
        # self.batch_loss[sample_id] = logs['loss']
    def on_batch_end_samples(self, batch, logs=None) :
        if batch in self.batch_sample_points :
            sample_id = batch//self.batch_sample_period
            self.weights_list[sample_id] = self.model.get_weights()
            # self.batch_loss[sample_id] = logs['loss']

l1_norm = lambda m1, m2 : np.sum(np.abs(m1 - m2))

def get_random_init(i, h, o, sigma=1, rng=np.random.default_rng()) :
    dtype = np.float64
    return [
        rng.normal(0, sigma, (i, h)).astype(dtype),
        np.zeros(h).astype(dtype),
        rng.normal(0, sigma, (h, o)).astype(dtype),
        np.zeros(o).astype(dtype)
    ]

def pert_normal(w1, mean=0, std=0.01) :
    w2 = [w.copy() for w in w1]

    w2[0] += np.random.normal(mean, std, w2[0].shape)
    w2[2] += np.random.normal(mean, std, w2[2].shape)

    return w2

def pert_uniform(w1, eps) :
    """
    Perturb a set of weights `w1` by adding a random number
     uniformly distributed in the range (-`eps`, `eps`).
    """
    w2 = [w.copy() for w in w1]

    w2[0] += np.random.random_sample(w2[0].shape) * (2 * eps) - eps
    w2[2] += np.random.random_sample(w2[2].shape) * (2 * eps) - eps

    return w2

def flatten_weights(epochs_dict) :    
    '''
    Flatten a dict (epochs) of dicts (batches) of lists of arrays (weights) into a list of arrays.
    '''
    batch_weights_flat = []
    for batches in epochs_dict.values() :
        for batch_weights in batches.values() :
            batch_weights_flat.append(batch_weights)
    
    return batch_weights_flat

def get_dist(w1, w2) :
    # Sum of L1 norm of all matrices in each weights
    return np.sum([l1_norm(u, v) for u, v in zip(w1, w2)])

def get_dists_fromdicts(w1, w2s) :
    '''
    Input is dicts (epochs) of dicts (batches) of weights (i.e. lists of weight matrices).
    '''
    dists = []
    for w2 in w2s :
        dists.append([get_dist(ws1, ws2) for ws1, ws2 in zip(flatten_weights(w1), flatten_weights(w2))])

    return np.array(dists).T    # Convert to plot-friendly format before returning

def get_dists_fromlists(w1, w2s) :
    '''
    Input is list of weights (i.e. lists of weight matrices).
    '''
    dists = []
    for w2 in w2s :
        dists.append([get_dist(ws1, ws2) for ws1, ws2 in zip(w1, w2)])

    return np.array(dists).T    # Convert to plot-friendly format before returning

def get_dists_frommat(w1, w2s) :
    '''
    Input is lists of composite matrices.
    '''
    dists = []
    for w2 in w2s :
        dists.append([l1_norm(u, v) for u, v in zip(w1, w2)])
    
    return np.array(dists)

def get_dists_dataset_frommat(ws) :
    dists = np.empty((*ws.shape[:2], ws.shape[2]-1, ws.shape[3]))    # *(n_eps, n_init), n_pert, n_sample
    for i_eps in range(dists.shape[0]) :
        for i_init in range(dists.shape[1]) :
            dists[i_eps, i_init] = get_dists_frommat(ws[i_eps, i_init, 0], ws[i_eps, i_init, 1:])
    
    return dists

def get_permuted_sequences(x_train, y_train, epochs_eff, rng) :
    x_train_eff = np.repeat(x_train, epochs_eff, axis=0)
    y_train_eff = np.repeat(y_train, epochs_eff, axis=0)
    perm = rng.permutation(x_train_eff.shape[0])

    return x_train_eff[perm], y_train_eff[perm]

def get_nonpermuted_sequences(x_train, y_train, epochs_eff) :
    '''
    Do not permute sequences
    '''
    x_train_eff = np.concatenate([x_train] * epochs_eff)
    y_train_eff = np.concatenate([y_train] * epochs_eff)
    # perm = rng.permutation(x_train_eff.shape[0])

    return x_train_eff, y_train_eff

def train(
        xs, ys,
        initial_weights,
        h_dim=16,
        batch_size=1, epochs=1, batch_sample_period=1, n_sample_points=None,
        val_xy=None,
        activation=ACTIVATION_DEFAULT,
        verbose=0,
        shuffle=False,
        return_loss=True,
        return_model=False,
        lr=SGD_LR_DEFAULT,
        ) :
    
    if n_sample_points is not None :    # Sample points override sample period
        # print(epochs, xs.shape[0], batch_size, n_sample_points)
        batch_sample_period = epochs * xs.shape[0] // batch_size // n_sample_points
     
    # Infer h_dim from weights
    h_dim = initial_weights[1].size     # Length of first (hidden layer) bias vector

    w_const = keras.initializers.Constant

    model = keras.models.Sequential([
        keras.Input((INPUT_DIM,)),
        keras.layers.Dense(
            h_dim,
            kernel_initializer=w_const(initial_weights[0]),
            bias_initializer=w_const(initial_weights[1]),
            activation=activation
        ),
        keras.layers.Dense(
            OUTPUT_DIM,
            kernel_initializer=w_const(initial_weights[2]),
            bias_initializer=w_const(initial_weights[3]),
            activation=tf.keras.activations.softmax
        )
    ])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.legacy.SGD(learning_rate=lr),
        # optimizer='rmsprop',
        # optimizer='adam',
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # print(epochs, batch_size, xs.shape[0], batch_sample_period)
    weights_callback = WeightsCallback(epochs, batch_size, xs.shape[0], batch_sample_period)
    # batch_loss = []
    # batch_sample_points = set(range(0, xs.shape[0] // batch_size, batch_sample_period))
    # batch_loss_callback = keras.callbacks.LambdaCallback(
    #     on_batch_end=lambda batch, logs : batch_loss.append(logs['loss'])
    # )

    # model = KerasPickleWrapper(model)

    # history = model().fit(  # Calling the wrapped model to get original
    history = model.fit(
        xs, ys,
        validation_data=val_xy,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            weights_callback,
            # batch_loss_callback,
        ],
        shuffle=shuffle,
        verbose=verbose,
        use_multiprocessing=True
    )

    # Record inital weights and shift batch weights to correspond to loss
    ws = weights_callback.weights_list
    ws.insert(0, initial_weights)
    del ws[-1]

    out_list = [ws]

    if return_loss :    # Compute and return training loss at each snapshot
        loss = get_loss_from_weights_list(ws, model, xs, ys)
        out_list.append(loss)

    if return_model : out_list.append(model)

    if len(out_list) > 1 :      # If mutliple return as tuple
        return tuple(out_list)  
    else : return out_list[0]   # If only weights return as object

def train_multiple_seq(n_pert, w1_init, eps, pert_func=pert_uniform, **train_args) :
    '''
    Train an initial condition defined by `w1_init` and `n_pert` perturbations, and return the weights for all.
    '''
    # if 'return_loss' not in train_args :
    #     train_args['return_loss'] = True

    # Train 'reference' instance
    # w1, l1 = train(initial_weights=w1_init, **train_args)
    w1 = train(initial_weights=w1_init, **train_args)

    # Train perturbations
    w2s = []
    # l2s = []
    for i in range(n_pert) :
        # w2, l2 = train(initial_weights=pert_uniform(w1_init, eps), **train_args)
        w2 = train(initial_weights=pert_func(w1_init, eps), **train_args)
        w2s.append(w2)
        # l2s.append(l2)

    ws = [w1, *w2s]
    # ls = [l1, *l2s]

    # return ws, ls
    return ws

# Not working — Need to revise
def train_multiple(n_pert, pert_std, n_processes=None, **train_args) :
    xs = train_args.pop('xs')
    ys = train_args.pop('ys')

    h_dim = train_args.pop('h_dim')
    w1_init = get_random_init(INPUT_DIM, h_dim, OUTPUT_DIM)
    w1, bl1 = train(xs, ys, w1_init, **train_args)

    w2s = []
    # h2s = []
    bl2s = []
    with Pool(n_processes) as pool :
        w2_inits = [pert_normal(w1_init, 0, pert_std) for _ in range(n_pert)]
        args = [([xs, ys, w2_inits[i]], train_args) for i in range(n_pert)]
        # print(args)
        res = pool.starmap(train_wrapper, args)

        for w2, bl2 in res :
            w2s.append(w2)
            bl2s.append(bl2)

    # dists = []
    # for w2 in w2s :
    #     dists.append([np.sum([l1_norm(u, v) for u, v in zip(ws1, ws2)]) for ws1, ws2 in zip(flatten_weights(w1), flatten_weights(w2))])

    dists = get_dists_fromdicts(w1, w2s)

    return dists, [bl1, bl2s]

def train_wrapper(args, kwargs) :
    # print(args)
    # print(kwargs)
    return train(*args, **kwargs)

def weights_to_matrix(ws, i, h, o) :
    # ws is a list of 4 matrices
    ws_shape = (i+1 + h+1), (h + o)
    ws_mat = np.zeros(ws_shape)
    ws_mat[:i,:h] = ws[0]
    ws_mat[i,:h] = ws[1]
    ws_mat[i+1:-1,h:] = ws[2]
    ws_mat[-1,h:] = ws[3]

    return ws_mat

def matrix_to_weights(ws_mat, i, h, o) :
    ws = [[], [], [], []]
    ws[0] = ws_mat[:i,:h]
    ws[1] = ws_mat[i,:h]
    ws[2] = ws_mat[i+1:-1,h:]
    ws[3] = ws_mat[-1,h:]

    return ws

def weights_list_to_matrices(ws, i, h, o) :
    ws_shape = (i+1 + h+1), (h + o)
    n_pert = len(ws)
    n_samples = len(ws[0])
    ws_mat = np.zeros((n_pert, n_samples, *ws_shape))

    for i_pert in range(n_pert) :
        for i_pt in range(n_samples) :
            ws_mat[i_pert, i_pt, ...] = weights_to_matrix(ws[i_pert][i_pt], i, h, o)

    return ws_mat

def matrices_to_weights_list(ws_mat, i, h, o) :
    # ws_shape = (i+1 + h+1), (h + o)
    n_pert = ws_mat.shape[0]
    n_samples = ws_mat.shape[1]
    ws = []

    for i_pert in range(n_pert) :
        ws_pert = []
        for i_pt in range(n_samples) :
            ws_pert.append(matrix_to_weights(ws_mat[i_pert, i_pt, ...], i, h, o))
        ws.append(ws_pert)

    return ws

def get_loss_from_weights_list(ws, model, xs, ys) :
    """
    Return `model.loss()` for `xs` and `ys` from each set of weights in `ws`.

    Not a very time efficient solution, but to improve it I would need to
    implement tf training at a lower level...

    Or I could just define my model with a custom loss_tracker ?
    """
    loss = np.empty(len(ws))
    for i, w in enumerate(ws) :
        model.set_weights(w)
        loss[i] = model.loss(ys, model(xs)).numpy()
    
    return loss

def get_loss_from_matrices(ws, model, xs, ys) :
    """
    Return `model.loss()` for `xs` and `ys` from each weight matrix in `ws`.
    """
    h_dim = ws.shape[-1] - OUTPUT_DIM
    if ws.ndim > 3 :
        ws = matrices_to_weights_list(ws, INPUT_DIM, h_dim, OUTPUT_DIM)   # This could cause confusion if input is for multiple perturbations
    else :
        ws = matrices_to_weights_list(np.expand_dims(ws, axis=0), INPUT_DIM, h_dim, OUTPUT_DIM)[0]
    
    loss = get_loss_from_weights_list(ws, model, xs, ys)
    
    return loss

def get_empty_model(h_dim) :
    model = keras.models.Sequential([
        keras.Input((INPUT_DIM,)),
        keras.layers.Dense(
            h_dim,
            activation=ACTIVATION_DEFAULT
        ),
        keras.layers.Dense(
            OUTPUT_DIM,
            activation=tf.keras.activations.softmax
        )
    ])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.legacy.SGD(learning_rate=SGD_LR_DEFAULT),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model