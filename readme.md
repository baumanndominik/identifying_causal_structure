# Identifying Causal Structure in Dynamical Systems

This repository is the official implementation of the paper "Identifying Causal Structure in Dynamical Systems" by Dominik Baumann, Friedrich Solowjow, Karl Henrik Johansson, and Sebastian Trimpe. 

## Requirements

The code was developed for and tested in Python 3, but should also work with Python 2. The following libraries are required:

* numpy
* scipy
* copy

## Execution

To execute the code, run the command:

```
python caus_id.py
```

The algorithm after each iteration prints a "causality matrix", i.e., a matrix, indicating which variables have a causal influence on one another. In the beginning, all entries are 0 since our null hypothesis is that variables do not cause each other. If a variable is found to cause some other variables, this 0 is replaced by a 1. A 1 in line 2, column 1, for instance, indicates that variable x<sub>2</sub> causes variable x<sub>1</sub>.

## Results

After finishing, the final causality matrix the algorithm will be

$$
\begin{pmatrix}
1 & 1 & 1 & 1 & 1\\
0 & 1 & 1 & 1 & 1\\
0 & 0 & 1 & 0 & 1
\end{pmatrix}
$$

| | x<sub>1</sub> | x<sub>2</sub> | x<sub>3</sub> | u<sub>1</sub> | u<sub>2</sub> | u<sub>3</sub>|
| --- | --- | --- | --- | --- | --- | --- |
| **x<sub>1</sub>** | 1 | 1 | 1 | 1 | 1 | 1|
| **x<sub>2</sub>** | 0 | 1 | 1 | 1 | 1 | 1|
| **x<sub>1</sub>** | 0 | 0 | 1 | 1 | 0 | 1|

## Copyright

Copyright (c) 2020 Max Planck Gesellschaft
