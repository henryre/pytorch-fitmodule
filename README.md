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
[========================================] 100%	loss: 1.3285    accuracy: 0.5676    val_loss: 1.0450    val_accuracy: 0.5693

Epoch 2 / 10
[========================================] 100%	loss: 0.8004    accuracy: 0.8900    val_loss: 0.5804    val_accuracy: 0.8900

Epoch 3 / 10
[========================================] 100%	loss: 0.4638    accuracy: 0.8981    val_loss: 0.3845    val_accuracy: 0.8983

Epoch 4 / 10
[========================================] 100%	loss: 0.3357    accuracy: 0.9033    val_loss: 0.2998    val_accuracy: 0.9043

Epoch 5 / 10
[========================================] 100%	loss: 0.2684    accuracy: 0.9196    val_loss: 0.2462    val_accuracy: 0.9213

Epoch 6 / 10
[========================================] 100%	loss: 0.2215    accuracy: 0.9374    val_loss: 0.2061    val_accuracy: 0.9423

Epoch 7 / 10
[========================================] 100%	loss: 0.1841    accuracy: 0.9586    val_loss: 0.1738    val_accuracy: 0.9590

Epoch 8 / 10
[========================================] 100%	loss: 0.1543    accuracy: 0.9704    val_loss: 0.1478    val_accuracy: 0.9673

Epoch 9 / 10
[========================================] 100%	loss: 0.1298    accuracy: 0.9806    val_loss: 0.1266    val_accuracy: 0.9747

Epoch 10 / 10
[========================================] 100%	loss: 0.1099    accuracy: 0.9861    val_loss: 0.1094    val_accuracy: 0.9800
```