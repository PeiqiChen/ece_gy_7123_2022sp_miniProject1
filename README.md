### This is repository for ECE GY 7123 2022sp mini Project1

Code structure is shown below.

```
project1_model.py: contains ResNet class and main code for training.
```

|               | Best Parameters           |
| ------------- | ------------------------- |
| N             | 4                         |
| B             | 3643                      |
| C             | 32                        |
| F             | 3                         |
| K             | 3                         |
| P             | 4                         |
| b_s           | 128/100                   |
| l_r           | 0.1                       |
| Learning Rate | cosine annealing + LR 0.1 |
| epoch         | 200                       |
| optimizer     | SGD+L2 Regularization     |
| parameter     | 4.88M                     |
| loss          | 0.20                      |
| accuracy      | 95.01                     |
