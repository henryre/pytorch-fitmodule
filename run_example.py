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
X, y = make_multilabel_classification(
    n_samples=10000, n_features=n_feats, n_classes=n_classes, n_labels=0.01,
    length=50, allow_unlabeled=False, sparse=False, return_indicator='dense',
    return_distributions=False, random_state=SEED
)
y = np.argmax(y, axis=1)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()


##### Define model #####
print_title("Building model")

class MLP(FitModule):
    def __init__(self, n_feats, n_classes, hidden_size=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
    def forward(self, x):
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))))

f = MLP(n_feats, n_classes)


##### Train model #####
print_title("Training model")

def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))

f.fit(
    X, y, epochs=10, validation_split=0.3, seed=SEED, metrics=[accuracy]
)
