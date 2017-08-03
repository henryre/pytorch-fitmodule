import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_fitmodule import FitModule
from sklearn.datasets import make_multilabel_classification


SEED = 1701


def print_title(s):
    print("\n\n{0}\n{1}\n{0}".format("="*len(s), s))


##### Generate training set #####
print_title("Generating data set")

n_feats, n_classes = 200, 5
X, Y = make_multilabel_classification(
    n_samples=10000, n_features=n_feats, n_classes=n_classes, n_labels=0.01,
    length=50, allow_unlabeled=False, sparse=False, return_indicator='dense',
    return_distributions=False, random_state=SEED
)
Y = np.argmax(Y, axis=1)
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).long()


##### Define model #####
print_title("Building model")

class MLP(FitModule):
    def __init__(self, n_feats, n_classes, hidden_size=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

f = MLP(n_feats, n_classes)


##### Train model #####
print_title("Training model")

def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))

f.fit(
    X, Y, epochs=5, validation_split=0.1, seed=SEED, metrics=[accuracy]
)
