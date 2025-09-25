import csv
import os
import numpy as np

def load_dataset(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)

load_dataset('realworld_experiments/data_colletion/ball_dataset.csv')