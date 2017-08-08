import torch

from collections import OrderedDict
from functools import partial
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD

from .utils import add_metrics_to_log, get_loader, log_to_message, ProgressBar


DEFAULT_LOSS = CrossEntropyLoss()
DEFAULT_OPTIMIZER = partial(SGD, lr=0.001, momentum=0.9)


class FitModule(Module):

    def fit(self,
            X,
            y,
            batch_size=32,
            epochs=1,
            verbose=1,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            initial_epoch=0,
            seed=None,
            loss=DEFAULT_LOSS,
            optimizer=DEFAULT_OPTIMIZER,
            metrics=None):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            X: training data Tensor.
            y: target data Tensor.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: (x_val, y_val) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors

        # Returns
            list of OrderedDicts with training metrics
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)
        # Prepare validation data
        if validation_data:
            X_val, y_val = validation_data
        elif validation_split and 0. < validation_split < 1.:
            split = int(X.size()[0] * (1. - validation_split))
            X, X_val = X[:split], X[split:]
            y, y_val = y[:split], y[split:]
        else:
            X_val, y_val = None, None
        # Build DataLoaders
        train_data = get_loader(X, y, batch_size, shuffle)
        # Compile optimizer
        opt = optimizer(self.parameters())
        # Run training loop
        logs = []
        self.train()
        for t in range(initial_epoch, epochs):
            if verbose:
                print("Epoch {0} / {1}".format(t+1, epochs))
            # Setup logger
            if verbose:
                pb = ProgressBar(len(train_data))
            log = OrderedDict()
            epoch_loss = 0.0
            # Run batches
            for batch_i, batch_data in enumerate(train_data):
                # Get batch data
                X_batch = Variable(batch_data[0])
                y_batch = Variable(batch_data[1])
                # Backprop
                opt.zero_grad()
                y_batch_pred = self(X_batch)
                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                # Update status
                epoch_loss += batch_loss.data[0]
                log['loss'] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_to_message(log))
            # Run metrics
            if metrics:
                y_train_pred = self.predict(X, batch_size)
                add_metrics_to_log(log, metrics, y, y_train_pred)
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val, batch_size)
                val_loss = loss(Variable(y_val_pred), Variable(y_val))
                log['val_loss'] = val_loss.data[0]
                if metrics:
                    add_metrics_to_log(log, metrics, y_val, y_val_pred, 'val_')
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        return logs

    def predict(self, X, batch_size=32):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        # Build DataLoader
        data = get_loader(X, batch_size=batch_size)
        # Batch prediction
        self.eval()
        r, n = 0, X.size()[0]
        for batch_data in data:
            # Predict on batch
            X_batch = Variable(batch_data[0])
            y_batch_pred = self(X_batch).data
            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
            # Add to prediction tensor
            y_pred[r : min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred
