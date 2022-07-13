# Identifying Causal Structure in Dynamical Systems

This repository contains an algorithm for automatically identifying the causal structure of a dynamical system. The algorithm is implemented for a synthetic, linear example and a nonlinear quadruple tank system. A description of the method can be found in:

* Dominik Baumann, Friedrich Solowjow, Karl Henrik Johansson, and Sebastian Trimpe, "Identifying causal structure in dynamical systems," Transactions on Machine Learning Research, 2022, [arXiv](https://arxiv.org/abs/2006.03906).

## Requirements

The code was developed using Python 3.8.10. The following libraries are required:

* numpy (tested with version 1.23.1)
* scipy (tested with version 1.8.0)
* GPy (tested with version 1.10.0)

## Examples

We provide to examples: a synthetic, linear example, and a nonlinear quadruple tank system.

### Synthetic linear example

The synthetic linear example serves as a proof of concept at low compute time. We estimate a linear state-space model for the system using least-squares and then subsequently identify its causal structure. For each experiment, we steer the system to its initial condition instead of just setting them in the code to be more realistic.

### Quadruple tank system

For the quadruple tank system, we estimate a Gaussian process (GP) model. For the paper, this was achieved using parallel computing. To simplify running the algorithm and enhance reproducibility, we here provide an implementation without parallel computing and less data used to learn the GP model, which reduces the computational demand of GP inference. This leads to slightly higher test statistics as reported in the paper, but the qualitative results are the same.

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
