#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:12:43 2024

@author: mohammadrahman
"""

import pandas as pd
import matplotlib.pyplot as plt
    

def read_data(file_path):

    df = pd.read_csv(file_path, delim_whitespace=True, header=9)
    df.columns = ['MC step', 'Ratio', 'Energy', 'Order']
    return df

def plot_order_parameter(df):

    plt.figure(figsize=(10, 6))
    plt.plot(df['MC step'], df['Order'], marker='o', linestyle='-', linewidth=1.25, markersize=1.25)
    plt.title('Reduced Temperature, $T^* = 0.65$')
    plt.xlabel('MCS')
    plt.ylabel('Order Parameter, S')
    plt.grid(True)
    plt.show()
    
def plot_energy(df):

    plt.figure(figsize=(10, 6))
    plt.plot(df['MC step'], df['Energy'], marker='o', linestyle='-', linewidth=1.25, markersize=1.25, color='red')
    plt.title('Reduced Temperature, $T^* = 0.65$')
    plt.xlabel('MCS')
    plt.ylabel(r'Reduced Energy, $U/\varepsilon$') 
    plt.grid(True)
    plt.show()
    
def main():
    file_path = '/Users/MohammadRahman/Desktop/LebwohlLasher7/LL-Output-Wed-13-Mar-2024-at-12-52-11AM.txt'
    df = read_data(file_path)
    
    plot_order_parameter(df)
    plot_energy(df)
    
if __name__ == '__main__':
    main()









