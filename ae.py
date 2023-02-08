from keras.layers import Dense, Input
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import tensorflow as tf

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []    # stitch indices
    part = []   # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model(train_x, training=True), train_y)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


class AE:
    def __init__(self, input_len, X_train, label):
        self._input_len = input_len
        self._X_train = X_train
        self._label = label

    def train_autoencoder(self):
        input_features = Input(shape=(self._input_len,))
        # encoder
        encoded = Dense(units=self._input_len, activation='sigmoid')(input_features)
        encoded = Dense(units=self._input_len/2, activation='sigmoid')(encoded)
        encoded = Dense(units=self._input_len/4, activation='sigmoid')(encoded)
        # latent-space
        encoded = Dense(units=self._input_len/8, activation='sigmoid')(encoded)
        # decoded
        decoded = Dense(units=self._input_len/4, activation='sigmoid')(encoded)
        decoded = Dense(units=self._input_len/2, activation='sigmoid')(decoded)
        decoded = Dense(units=self._input_len, activation='sigmoid')(decoded)

        autoencoder = Model(input_features, decoded)
        autoencoder.summary()
        '''
        loss_fun = tf.keras.losses.MeanSquaredError()
        func = function_factory(autoencoder, loss_fun, self._X_train, self._X_train)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, autoencoder.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=100,tolerance=1e-06,num_correction_pairs=10)
        func.assign_new_model_parameters(results.position)
        '''
        opt = Adam(learning_rate=0.03)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])
        autoencoder.fit(self._X_train, self._X_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping, lr_reducer])

        autoencoder.save("model/autoencoder"+self._label+".h5")
