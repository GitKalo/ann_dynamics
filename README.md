# ANNs through the lens of dynamical systems

This work was a part of a Master's thesis titled *Artificial neural networks through the lens of dynamical systems theory* for the Masters in Physics of Complex Systems at IFISC (UIB-CSIC), academic year 2022/23.

Broadly, the approach is to record the training trajectory of ANNs through weight space (under GD) and use dynamical systems concepts and methods to analyze the trajectories.

This code representes a part of the numerical work done for the thesis. Mainly, it consists of:
- the core implementation script `src/iris_train.py`, which contains the TensorFlow-based implementation of the training function and a set of functions used for manipulating network trajectories and data analysis;
- four example scripts for training networks and saving results in HDF5 format: `run_iris_epsilon.py`, `run_iris_post_learn.py`, `run_iris_eos_epsilon.py`, (for the "edge of stability" regime) and `run_iris_chaotic.py` (for the "fully-chaotic" regime);
- and a Jupyter Notebook `report_figures.ipynb` with the data analysis used to generate many of the figures shown in the report.

Work was done under the supervision of Lucas Lacasa and Miguel Soriano (IFISC).

The work was followed by a publication in Frontiers of Complex Systems (under review as of Jan 2024).