# A super simple `fit` method for PyTorch `Module`s

Ever wanted a pretty, Keras-like `fit` method for your PyTorch `Module`s?
Here's one. It lacks some of the advanced functionality, but it's easy to use:

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_fitmodule import FitModule

X, Y, n_classes = torch.get_me_some_data()

class MLP(FitModule):
    def __init__(self, n_feats, n_classes, hidden_size=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
    def forward(self, x):
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))))

f = MLP(X.size()[1], n_classes)

def n_correct(y_true, y_pred):
    return (y_true == torch.max(y_pred, 1)[1]).sum()

f.fit(X, Y, epochs=5, validation_split=0.3, metrics=[n_correct])
```


## Installation

Just clone this repo and add it to your Python path. You'll need
* [PyTorch](http://pytorch.org)
* [NumPy](http://numpy.org/)
* [Scikit-Learn](http://scikit-learn.org/) (just for the example)

all of which are available via [Anaconda](https://www.continuum.io/downloads).

## Example

Try out a simple example with the included script:

```bash
python run_example.py
```

```bash
Epoch 1 / 10
[========================================] 100%	train_loss: 0.0416    accuracy: 0.6251    val_accuracy: 0.6287

Epoch 2 / 10
[========================================] 100%	train_loss: 0.0247    accuracy: 0.8834    val_accuracy: 0.8860

Epoch 3 / 10
[========================================] 100%	train_loss: 0.0144    accuracy: 0.8990    val_accuracy: 0.8977

Epoch 4 / 10
[========================================] 100%	train_loss: 0.0099    accuracy: 0.9141    val_accuracy: 0.9163

Epoch 5 / 10
[========================================] 100%	train_loss: 0.0075    accuracy: 0.9414    val_accuracy: 0.9413

Epoch 6 / 10
[========================================] 100%	train_loss: 0.0059    accuracy: 0.9631    val_accuracy: 0.9620

Epoch 7 / 10
[========================================] 100%	train_loss: 0.0048    accuracy: 0.9773    val_accuracy: 0.9730

Epoch 8 / 10
[========================================] 100%	train_loss: 0.0039    accuracy: 0.9837    val_accuracy: 0.9800

Epoch 9 / 10
[========================================] 100%	train_loss: 0.0033    accuracy: 0.9880    val_accuracy: 0.9840

Epoch 10 / 10
[========================================] 100%	train_loss: 0.0028    accuracy: 0.9903    val_accuracy: 0.9860
```