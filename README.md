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
Epoch 1 / 5
[========================================] 100%	train_loss: 0.0432    accuracy: 0.4312    val_accuracy: 0.4360

Epoch 2 / 5
[========================================] 100%	train_loss: 0.0283    accuracy: 0.8830    val_accuracy: 0.8950

Epoch 3 / 5
[========================================] 100%	train_loss: 0.0148    accuracy: 0.8998    val_accuracy: 0.9050

Epoch 4 / 5
[========================================] 100%	train_loss: 0.0098    accuracy: 0.9017    val_accuracy: 0.9050

Epoch 5 / 5
[========================================] 100%	train_loss: 0.0077    accuracy: 0.9164    val_accuracy: 0.9170
```