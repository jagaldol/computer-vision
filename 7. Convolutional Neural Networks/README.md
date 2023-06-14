# Convolutional Neural Networks

## Introduction

Pytorch로  Convolutional  Neural  Network(CNN)를  구축하여  CIFAR-10  이미지  셋을  분류하고  성능을  평가해본다.

## Contents

1. Preparation
2. Barebones PyTorch
3. PyTorch Module API
4. PyTorch Sequential API
5. CIFAR-10 open-ended challenge

## Process

My final model is as follows.

```python
channel_1 = 32
channel_2 = 64
learning_rate = 1e-2

model = nn.Sequential(
    nn.Conv2d(3, channel_1, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2)),
    nn.Conv2d(channel_1, channel_2, 3, padding=1),
    nn.MaxPool2d((2, 2), (2, 2)),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 8 * 8, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)
```

The final validation score  I got is 73.5%.

and the final **test score I got is 71.29%**.

> If you want to know the full details of the process, please read `Convolutional Neural Network.ipynb`.