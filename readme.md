# Identifying Causal Structure in Dynamical Systems

This repository is the official implementation of the [paper](https://arxiv.org/abs/2006.03906) "Identifying Causal Structure in Dynamical Systems" by Dominik Baumann, Friedrich Solowjow, Karl Henrik Johansson, and Sebastian Trimpe. 

## Requirements

The code was developed using Python 3.6.9. caus_id_linear was also tested with Python 2.7. The following libraries are required:

* numpy
* scipy
* copy
* GPy

If you want to use parallel computing for estimating the GP models, the following libraries are additionally required:

* multiprocessing
* itertools

## Execution

To execute the code, run the command:

```
python caus_id_linear.py
```
for the linear synthetic example and

```
python caus_id_GP.py

```

for the quadruple tank system.

The algorithm after each iteration prints a "causality matrix", i.e., a matrix, indicating which variables have a causal influence on one another. In the beginning, all entries are 0 since our null hypothesis is that variables do not cause each other. If a variable is found to cause some other variables, this 0 is replaced by a 1. A 1 in line 2, column 1, for instance, indicates that variable x<sub>2</sub> causes variable x<sub>1</sub>.

## Results

After finishing, the final causality matrix the algorithm should be

| | x<sub>1</sub> | x<sub>2</sub> | x<sub>3</sub> | u<sub>1</sub> | u<sub>2</sub> | u<sub>3</sub>|
| --- | --- | --- | --- | --- | --- | --- |
| **x<sub>1</sub>** | 1 | 1 | 1 | 1 | 1 | 1|
| **x<sub>2</sub>** | 0 | 1 | 1 | 1 | 1 | 1|
| **x<sub>3</sub>** | 0 | 0 | 1 | 1 | 0 | 1|

for the synthetic example and 

| | x<sub>1</sub> | x<sub>2</sub> | x<sub>3</sub> | x<sub>4</sub> | u<sub>1</sub> | u<sub>2</sub>|
| --- | --- | --- | --- | --- | --- | --- |
| **x<sub>1</sub>** | 1 | 0 | 1 | 0 | 1 | 1|
| **x<sub>2</sub>** | 0 | 1 | 0 | 1 | 1 | 1|
| **x<sub>3</sub>** | 0 | 0 | 1 | 0 | 0 | 1|
| **x<sub>4</sub>** | 0 | 0 | 0 | 1 | 1 | 0|

for the quadruple tank system
