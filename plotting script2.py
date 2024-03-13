#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 07:26:18 2024

@author: mohammadrahman
"""
import numpy as np
import matplotlib.pyplot as plt

vectorised_steps = np.array([2.144, 4.260, 6.363, 8.729, 10.600, 12.762, 14.912, 16.889, 19.345, 21.356, 23.371, 25.496,  27.752, 30.311, 31.937])
numba_steps = np.array([0.570, 0.624, 0.690, 0.726, 0.770, 0.826, 0.873, 0.930, 0.966, 1.019, 1.096, 1.116, 1.173, 1.220, 1.260])
cython_steps = np.array([0.134, 0.268, 0.400, 0.537, 0.672, 0.788, 0.945, 1.058, 1.187, 1.344, 1.453, 1.599, 1.745, 1.833, 2.037])

# Generate the MCS values for the x-axis
MCS_values = np.arange(0, 1500, 100)

# Plotting the execution times for vectorised, numba, and cython techniques
plt.figure(figsize=(10, 6))
plt.plot(MCS_values, vectorised_steps, label='Vectorised', color='blue', marker='o')
plt.plot(MCS_values, numba_steps, label='Numba', color='green', marker='x')
plt.plot(MCS_values, cython_steps, label='Cython', color='red', marker='^')

# Setting up the labels and title
plt.xlabel('MCS')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


