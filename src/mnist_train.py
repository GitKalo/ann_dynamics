import h5py

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import Callback

tf.keras.backend.set_floatx('float64')

class MNISTTrain :
    """
    Class that encapsulates functions for training on MNIST dataset and analysis of trajectories.

    Ideally this would be a subclass of a base class common to different learning problems.
    """
    # Class constants
    INPUT_DIM = 784
    OUTPUT_DIM = 10

    # Class defaults
    ACTIVATION_DEFAULT = 'sigmoid'
    SGD_LR_DEFAULT = 0.01

    def __init__(self, h_dim=[16, 16], activation=ACTIVATION_DEFAULT, **train_args) :
        """
        Philosophy: one MNISTTrain object for each model architecture and training setup (incl. number of parameters, etc. encapsulated in train_args). So that we can train multiple models with same architecture and params, just different init.

        Perhaps ideally could separate out into a MNIST class (architecture) and MNISTTrain class (training instance)
        """
        # Parameters defining model architecture
        self.h_dim = h_dim
        self.L = len(h_dim) + 2     # Number of layers
        self.activation = activation

        # Parameters defining training process
        self.train_args = train_args

        if 'lr' not in self.train_args :
            self.train_args['lr'] = self.SGD_LR_DEFAULT

        # Should require some training params or set defaults, e.g. epochs...

    def train(self, xs, ys, w_init,
              lr=SGD_LR_DEFAULT,
              batch_size=1,
              epochs=10,
              epoch_sample_period=1, n_sample_points=None,  # Sometimes one or the other is more convenient
              val_xy=None,  # Tuple (x_val, y_val)
              verbose=0,
              shuffle=False,
              weights_arr=None,     # If arrays are None will not save, otherwise save resp. data
              train_loss_arr=None,
              test_loss_arr=None,
              return_model=False
    ) :
        """
        Core training function for MNIST dataset.

        xs, ys : Inputs and labels to perform training on — the training dataset.
        w_init : list of lists, first index corresponds to layer, second 
                 index corresponds to weights (0) and bias (1) weights.
        
        weights_arr, train_loss_arr, test_loss_arr : if either is provided, the respective
                                                     metrics will be recorded in the respective
                                                     array, based on given sampling period. If None,
                                                     metric will not be recorded. Arrays are modified
                                                     in-place, rather than returned.
        """
        if n_sample_points is not None :    # Sample points override sample period
            # print(epochs, xs.shape[0], batch_size, n_sample_points)
            epoch_sample_period = epochs // n_sample_points
        # epoch_sample_points = set(range(0, xs.shape[0] // batch_size, epoch_sample_period))
        epoch_sample_points = set(range(0, epochs, epoch_sample_period))

        model = self.get_empty_model()
        w_const = keras.initializers.Constant
        # BELOW WILL NO LONGER WORK WITH NEW WEIGHTS LIST FORMAT
        for i, layer in enumerate(model.layers) :
            layer.kernel_initializer = w_const(w_init[i][0])
            layer.bias_initializer = w_const(w_init[i][1])

        # Define loss and optimizer
        loss = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.legacy.SGD(learning_rate=lr)

        ##########################
        ### Main training loop ###
        ##########################
        for epoch in range(epochs) :
            ### Record metrics (before training)
            if epoch in epoch_sample_points :   # Only for sampling points
                if weights_arr is not None :
                    # Record weights as matrix
                    weights_arr[epoch] = self.weights_to_matrix(model.get_weights())
                if train_loss_arr is not None :
                    # Record training loss over whole dataset
                    train_loss = loss(ys, model(xs))  # Not very efficient — recalculating loss
                    train_loss_arr[epoch] = train_loss
                if test_loss_arr is not None and val_xy is not None :
                    # Record atest loss over validation / test set
                    test_loss = loss(val_xy[1], model(val_xy[0], training=False))
                    test_loss_arr[epoch] = test_loss

            ### Training
            for inputs, labels in batch_dataset(xs, ys, batch_size) :
                with tf.GradientTape() as tape :
                    predictions = model(inputs, training=True)
                    # regularization_loss = tf.math.add_n(model.losses)
                    # pred_loss = loss(labels, predictions)
                    # total_loss = pred_loss + regularization_loss
                    train_loss = loss(labels, predictions)
                
                # Optimization step
                # gradients = tape.gradient(total_loss, model.trainable_variables)
                gradients = tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f"Finished epoch {epoch}")

        if return_model :
            return model
    
    def get_training_generator(self, w_init, epochs) :
        """
        Generator function for network training on MNIST dataset, yields model at each epoch.

        Starting from w_init, so can be run from an arbitrary starting point.

        Assume w_init is a list containing all kernel and bias weights in top level (i.e. no nested tuples/lists).
        """
        # Generate model
        model = self.get_empty_model()

        # # Set initial weights
        # # TODO: could be simplified via layer.set_weights?
        # w_const = keras.initializers.Constant
        # for i, layer in enumerate(model.layers) :
        #     layer.kernel_initializer = w_const(w_init[2*i])
        #     layer.bias_initializer = w_const(w_init[2*i+1])

        model.set_weights(w_init)

        # Set loss function and optimizer
        model.loss = keras.losses.SparseCategoricalCrossentropy()
        model.optimizer = keras.optimizers.legacy.SGD(learning_rate=self.train_args['lr'])

        yield (model, 0)     # Return initial model at first call (epoch 0)
        
        for epoch in range(1, epochs) :  # Actually will do one epoch less than specified,
                                         # but OK (for ease of data storage/analysis reasons)
            
            # print(f"\nModel weights at epoch {epoch} (before training) : {model.get_weights()[0]}")

            ### Training step
            for inputs, labels in batch_dataset(self.train_args['x_train'], self.train_args['y_train'], self.train_args['batch_size']) :
                with tf.GradientTape() as tape :
                    predictions = model(inputs, training=True)
                    # regularization_loss = tf.math.add_n(model.losses)
                    # pred_loss = loss(labels, predictions)
                    # total_loss = pred_loss + regularization_loss
                    train_loss = model.loss(labels, predictions)
                    # print(train_loss)

                # Optimization step
                # gradients = tape.gradient(total_loss, model.trainable_variables)
                gradients = tape.gradient(train_loss, model.trainable_variables)
                # print(np.sum(gradients[0] != 0))
                # print(f"\nGradients at epoch {epoch} : {gradients[0]}")
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            yield (model, epoch)     # Return model at the end of epoch

    # TODO: can generalize this method to one used below for training with custom perts 
    def train_perts(self, eps, n_pert, w_init_ref=None) :
        """
        Train a reference and `n_pert` perturbed trajectories, returning distances at each epoch.
        Single epsilon, single initial condition.
        Returns distances and weights at initial and final epochs.
        """
        x_train, y_train = self.train_args['x_train'], self.train_args['y_train']
        x_test, y_test = self.train_args['x_test'], self.train_args['y_test']

        # Generate random initial weights for reference model if none specified
        if w_init_ref is None : w_init_ref = self.get_random_init()

        # Create data arrays (to be returned)
        ws = np.empty((n_pert+1, 2, *self.get_ws_shape())) # Weights at initial and final epoch
        loss_train = np.zeros((n_pert+1, self.train_args['epochs']))
        loss_test = np.zeros((n_pert+1, self.train_args['epochs']))
        accuracy_train = np.zeros((n_pert+1, self.train_args['epochs']))
        accuracy_test = np.zeros((n_pert+1, self.train_args['epochs']))
        dists = np.zeros((n_pert, self.train_args['epochs']))
        # Really in above shape we are discarding last epoch

        acc_metric = keras.metrics.SparseCategoricalAccuracy()

        # Create reference training generator
        ref_train = self.get_training_generator(w_init_ref, self.train_args['epochs'])
        model_ref, _ = next(ref_train)  # Get initial
        ws[0,0] = self.weights_to_matrix(w_init_ref)    # Record weight at init
        # Record train metrics
        loss_train[0,0], accuracy_train[0,0] = \
            get_loss_and_acc(model_ref, x_train, y_train, acc_metric)
        # Record test metrics
        loss_test[0,0], accuracy_test[0,0] = \
            get_loss_and_acc(model_ref, x_test, y_test, acc_metric)

        # Create perturbed training generators
        perts_train = []
        perts_models = []
        for i in range(n_pert) :
            w_init = self.pert_uniform(w_init_ref, eps)
            pert_train = self.get_training_generator(w_init, self.train_args['epochs']+1)    # One more epochs than ref, for practical, programmatic purposes (see except statement in main loop)
            model_pert, _ = next(pert_train)     # Get initial
            perts_train.append(pert_train)
            perts_models.append(model_pert)

            # Record initial weights
            ws[i+1,0] = self.weights_to_matrix(w_init)
            # Record train metrics
            loss_train[i+1,0], accuracy_train[i+1,0] = \
                get_loss_and_acc(model_pert, x_train, y_train, acc_metric)
            # Record test metrics
            loss_test[i+1,0], accuracy_test[i+1,0] = \
                get_loss_and_acc(model_pert, x_test, y_test, acc_metric)
            test_pred = model_pert(x_test)
            loss_test[i+1,0] = model_pert.loss(y_test, test_pred).numpy()
            acc_metric.update_state(y_test, test_pred)
            accuracy_test[i+1,0] = acc_metric.result().numpy()
            acc_metric.reset_state()
            # Record distances
            dists[i,0] = get_dist(w_init_ref, w_init)

        ### Main loop
        while True :
            try :
                # Iterate ref object
                model_ref, epoch = next(ref_train)
                # Record train metrics
                loss_train[0,epoch], accuracy_train[0,epoch] = \
                    get_loss_and_acc(model_ref, x_train, y_train, acc_metric)
                # Record test metrics
                loss_test[0,epoch], accuracy_test[0,epoch] = \
                    get_loss_and_acc(model_ref, x_test, y_test, acc_metric)
                
                # Loop over perturbed objects
                for i in range(n_pert) :
                    # Iterate perturbed object
                    model_pert, epoch = next(perts_train[i])
                    perts_models[i] = model_pert
                    # Record distances
                    dists[i,epoch] = get_dist(model_ref.get_weights(), model_pert.get_weights())
                    # Record train metrics
                    loss_train[i+1,epoch], accuracy_train[i+1,epoch] = \
                        get_loss_and_acc(model_pert, x_train, y_train, acc_metric)
                    # Record test metrics
                    loss_test[i+1,epoch], accuracy_test[i+1,epoch] = \
                        get_loss_and_acc(model_pert, x_test, y_test, acc_metric)
            except StopIteration :  # Have no more iterations of reference
                ws[0,1] = self.weights_to_matrix(model_ref.get_weights())
                for i in range(n_pert) :
                    ws[i+1,1] = self.weights_to_matrix(perts_models[i].get_weights())
                break 

        return dists, ws, (loss_train, accuracy_train), (loss_test, accuracy_test)

    def train_custom_perts(self, w_init_ref, w_init_perts) :
        """
        Given the initial conditions for a reference matrix and perturbations,
        train all and collect distances (to reference) and performance metrics.
        """
        # TODO: add below as lines in initialization (pointers are cheap!)
        x_train, y_train = self.train_args['x_train'], self.train_args['y_train']
        x_test, y_test = self.train_args['x_test'], self.train_args['y_test']

        # Create data arrays (to be returned)
        n_pert = len(w_init_perts)
        ws = np.empty((n_pert+1, 2, *self.get_ws_shape()))
        loss_train = np.zeros((n_pert+1, self.train_args['epochs']))
        loss_test = np.zeros((n_pert+1, self.train_args['epochs']))
        accuracy_train = np.zeros((n_pert+1, self.train_args['epochs']))
        accuracy_test = np.zeros((n_pert+1, self.train_args['epochs']))
        dists = np.zeros((n_pert, self.train_args['epochs']))
        # Really in above shape we are discarding last epoch...

        # TODO: add below as a class member
        acc_metric = keras.metrics.SparseCategoricalAccuracy()

        # Create reference training generator
        ref_train = self.get_training_generator(w_init_ref, self.train_args['epochs'])
        model_ref, _ = next(ref_train)  # Get initial
        ws[0,0] = self.weights_to_matrix(w_init_ref)    # Record weight at init
        # Record train metrics
        loss_train[0,0], accuracy_train[0,0] = \
            get_loss_and_acc(model_ref, x_train, y_train, acc_metric)
        # Record test metrics
        loss_test[0,0], accuracy_test[0,0] = \
            get_loss_and_acc(model_ref, x_test, y_test, acc_metric)

        # Create perturbed training generators
        perts_train = []
        perts_models = []
        for i in range(n_pert) :
            w_init = w_init_perts[i]
            pert_train = self.get_training_generator(w_init, self.train_args['epochs']+1)    # One more epochs than ref, for practical, programmatic purposes (see except statement in main loop)
            model_pert, _ = next(pert_train)     # Get initial
            perts_train.append(pert_train)
            perts_models.append(model_pert)

            # Record initial weights
            ws[i+1,0] = self.weights_to_matrix(w_init)
            # Record train metrics
            loss_train[i+1,0], accuracy_train[i+1,0] = \
                get_loss_and_acc(model_pert, x_train, y_train, acc_metric)
            # Record test metrics
            loss_test[i+1,0], accuracy_test[i+1,0] = \
                get_loss_and_acc(model_pert, x_test, y_test, acc_metric)
            test_pred = model_pert(x_test)
            loss_test[i+1,0] = model_pert.loss(y_test, test_pred).numpy()
            acc_metric.update_state(y_test, test_pred)
            accuracy_test[i+1,0] = acc_metric.result().numpy()
            acc_metric.reset_state()
            # Record distances
            dists[i,0] = get_dist(w_init_ref, w_init)

        ### Main loop
        while True :
            try :
                # Iterate ref object
                model_ref, epoch = next(ref_train)
                # Record train metrics
                loss_train[0,epoch], accuracy_train[0,epoch] = \
                    get_loss_and_acc(model_ref, x_train, y_train, acc_metric)
                # Record test metrics
                loss_test[0,epoch], accuracy_test[0,epoch] = \
                    get_loss_and_acc(model_ref, x_test, y_test, acc_metric)
                
                # Loop over perturbed objects
                for i in range(n_pert) :
                    # Iterate perturbed object
                    model_pert, epoch = next(perts_train[i])
                    perts_models[i] = model_pert
                    # Record distances
                    dists[i,epoch] = get_dist(model_ref.get_weights(), model_pert.get_weights())
                    # Record train metrics
                    loss_train[i+1,epoch], accuracy_train[i+1,epoch] = \
                        get_loss_and_acc(model_pert, x_train, y_train, acc_metric)
                    # Record test metrics
                    loss_test[i+1,epoch], accuracy_test[i+1,epoch] = \
                        get_loss_and_acc(model_pert, x_test, y_test, acc_metric)
            except StopIteration :  # Have no more iterations of reference
                ws[0,1] = self.weights_to_matrix(model_ref.get_weights())
                for i in range(n_pert) :
                    ws[i+1,1] = self.weights_to_matrix(perts_models[i].get_weights())
                break

        return dists, ws, (loss_train, accuracy_train), (loss_test, accuracy_test)

    def train_independent_inits(self, n_init) :
        """
        Train a number of independent initial conditions, recording distances,
        performance metrics, and weights at initialization and final epoch.
        """
        x_train, y_train = self.train_args['x_train'], self.train_args['y_train']
        x_test, y_test = self.train_args['x_test'], self.train_args['y_test']

        # Create data arrays (to be returned)
        ws = np.empty((n_init, 2, *self.get_ws_shape())) # Weights at initial and final epoch
        loss_train = np.zeros((n_init, self.train_args['epochs']))
        loss_test = np.zeros((n_init, self.train_args['epochs']))
        accuracy_train = np.zeros((n_init, self.train_args['epochs']))
        accuracy_test = np.zeros((n_init, self.train_args['epochs']))
        # Disances between each initial conditoin and all others
        dists = np.zeros((n_init, n_init-1, self.train_args['epochs']))
        # Really in above shape we are discarding last epoch

        acc_metric = keras.metrics.SparseCategoricalAccuracy()

        # Generate random initial weights for each independent initial condition
        w_inits = []
        for i in range(n_init) :
            w_inits.append(self.get_random_init())

        inits_train = []
        for i in range(n_init) :
            # Record initial weights
            ws[i,0] = self.weights_to_matrix(w_inits[i])
            # Create training generators
            inits_train.append(self.get_training_generator(w_inits[i], self.train_args['epochs']))

        inits_models = [0] * n_init     # Create dummy list to hold models
        ### Main loop
        while True :
            try :
                # Iterate over init training objects
                for i in range(n_init) :
                    model, epoch = next(inits_train[i])
                    inits_models[i] = model

                    # Record performance metrics
                    loss_train[i,epoch], accuracy_train[i,epoch] = \
                        get_loss_and_acc(model, x_train, y_train, acc_metric)
                    # Record test metrics
                    loss_test[i,epoch], accuracy_test[i,epoch] = \
                        get_loss_and_acc(model, x_test, y_test, acc_metric)
                    
                # Record all-to-all distances
                for i in range(n_init) :
                    model_ref = inits_models[i]
                    counter = 0     # Used to keep track of other inits 
                    for j, model in enumerate(inits_models) :
                        if i != j :
                            dists[i,counter,epoch] = get_dist(model_ref.get_weights(), model.get_weights())
                            counter += 1
            except StopIteration :  # Have no more training iterations
                for i in range(n_init) :
                    ws[i,1] = self.weights_to_matrix(inits_models[i].get_weights())
                break

        return dists, ws, (loss_train, accuracy_train), (loss_test, accuracy_test)

    def get_empty_model(self) :
        '''
        Get skeleton model, i.e. with correct shape but uninitialized weights.

        TODO: should think whether this empty model should include loss and optimizer (perhaps yes if it is set for the class anyway/)
        '''
        dense_layers = [
            keras.layers.Dense(d, activation=self.activation) 
            for d in self.h_dim
        ]

        model = keras.models.Sequential([
            # Input layer
            keras.Input((self.INPUT_DIM,)),
            # Hidden layers
            *dense_layers,
            # Output layer with Softmax activation
            keras.layers.Dense(
                self.OUTPUT_DIM,
                activation='softmax'
            )
        ])

        return model
    
    def get_random_init(self, sigma=1, rng=np.random.default_rng()) :
        """
        Return random weights and bias set to zero, in flat list of 
        numpy arrays (format expected by `model.set_weights`).
        """
        dtype = np.float64
        w_init = []
        # Input layer
        w_init += [rng.normal(0, sigma, (self.INPUT_DIM, self.h_dim[0])).astype(dtype), np.zeros(self.h_dim[0]).astype(dtype)]
        # Hidden layers
        for h1, h2 in zip(self.h_dim[:-1], self.h_dim[1:]) :
            w_init += [rng.normal(0, sigma, (h1, h2)).astype(dtype), np.zeros(h2).astype(dtype)]
        # Output layer
        w_init += [rng.normal(0, sigma, (self.h_dim[-1], self.OUTPUT_DIM)).astype(dtype), np.zeros(self.OUTPUT_DIM).astype(dtype)]

        return w_init
    
    def pert_uniform(self, w1, eps) :
        """
        Perturb a set of weights `w1` by adding a random number uniformly
        distributed in the range (-`eps`, `eps`) to all (kernel) weights.
        """
        # Create copy
        w2 = [w.copy() for w in w1]
        # Add perturbation (to kernel weights only)
        for i in range(0, len(w1), 2) :
            w2[i] += np.random.random_sample(w2[i].shape) * (2 * eps) - eps
        return w2
    
    def weights_to_matrix(self, ws) :
        '''
        Convert a list of model weights to a single composite matrix, essentially aligned along the diagonal.

        ws : list of weights, either as flat list or as list of (kernel, bias) weights tuples.
        '''
        ws_shape = self.get_ws_shape()
        ws_mat = np.zeros(ws_shape)

        i, j = 0, 0
        if len(ws) == 2*(self.L-1) : ws = self.ws_tuples(ws)
        for k, b in ws :    # Iterate over kernel and bias weight tuples
            ki, kj = i + k.shape[0], j + k.shape[1]
            ws_mat[i:ki, j:kj] = k
            i = ki
            ws_mat[i, j:kj] = b
            i += 1
            j = kj

        return ws_mat
    
    def matrix_to_weights(self, ws_mat) :
        """
        
        """
        ws = [[]] * ((self.L-1) * 2)

        # Input weights
        ws[0] = ws_mat[:self.INPUT_DIM, :self.h_dim[0]]
        ws[1] = ws_mat[self.INPUT_DIM, :self.h_dim[0]]
        i, j = self.INPUT_DIM + 1, self.h_dim[0] + 1
        # Hidden weights
        for ih in range(len(self.h_dim)-1) :
            # Kernel
            ws[2 + 2*ih] = ws_mat[i:self.h_dim[ih], j:self.h_dim[ih+1]]
            i += self.h_dim[ih]
            # Bias
            ws[2 + 2*ih+1] = ws_mat[i, j:self.h_dim[ih+1]]
            i += 1
            j += self.h_dim[ih+1]
        # Output weights
        ws[-2] = ws_mat[i:-1, self.h_dim[-1]:]
        ws[-1] = ws_mat[-1, self.h_dim[-1]:]

        return ws

    def ws_tuples(self, ws) :
        '''
        Convert a list of [kernel, bias, ...] weights to a list of (kernel, bias) weights tuples. `ws` is of length 2(L-1), while the output is of length L-1.
        '''
        return list(zip(ws[0::2], ws[1::2]))  # Not great memory-wise for large models, but for this purpose should be fine
        
    def get_ws_shape(self) :
        return (self.INPUT_DIM+1) + (sum(self.h_dim)+len(self.h_dim)), sum(self.h_dim) + self.OUTPUT_DIM
    
    def write_params(self, h5_file, **kwargs) :
        """
        Add model training params (in self.train_args dict) and params in 
        kwargs dict to attributes of `h5_file` (HDF5 file).
        """
        for k, v in self.train_args.items() :
            if k not in ['x_train', 'y_train', 'x_test', 'y_test'] :
                h5_file.attrs[k] = v
        for k, v in kwargs.items() :
            h5_file.attrs[k] = v
    
def batch_dataset(xs, ys, batch_size) :
    """
    Convert arrays of inputs (xs) and labels (ys) to list of (batch_input, batch_labels) tuples, with `batch_size` elements in each tuple element. Assume xs and ys have samples along axis 0.

    Currently no shuffling of data.
    """
    if ys.ndim == 1 : ys = np.expand_dims(ys, 1)
    batched_dataset = []
    n_samples = xs.shape[0]
    counter = 0
    while counter < n_samples :
        batched_dataset.append((xs[counter:counter+batch_size,:], ys[counter:counter+batch_size,:]))
        counter += batch_size
    if counter > n_samples :
        batched_dataset.append((xs[counter-batch_size:,:], ys[counter-batch_size:,:]))
    return batched_dataset

def l1_norm(m1, m2) :
    """Return L1-norm of difference between two matrices."""
    return np.sum(np.abs(m1 - m2))
    
def get_dist(w1, w2) :
    """
    w1 and w2 are weight lists (as passed to tf models)
    """
    return np.sum([l1_norm(u, v) for u, v in zip(w1, w2)])

def get_loss_and_acc(model, xs, ys, m_acc) :
    """
    Resets metric m_acc state before computing.
    """
    pred = model(xs)
    loss = model.loss(ys, pred).numpy()
    m_acc.reset_state()
    m_acc.update_state(ys, pred)
    acc = m_acc.result().numpy()
    return loss, acc

def write_params(self, h5_file, train_args, **kwargs) :
    """
    Add model training params (in self.train_args dict) and params in 
    kwargs dict to attributes of `h5_file` (HDF5 file).
    """
    for k, v in train_args.items() :
        if k not in ['x_train', 'y_train', 'x_test', 'y_test'] :
            h5_file.attrs[k] = v
    for k, v in kwargs.items() :
        h5_file.attrs[k] = v