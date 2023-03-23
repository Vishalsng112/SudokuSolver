#train a transformer to solve a sudoku puzzle
import torch
import csv
import numpy as np
from model import SudokuTransformer
    
def main():
    model = SudokuTransformer()
    # model.train()
    print(model.eval())
    
    #load data
    inputs = np.load('data/sudoku_inputs.npz')
    inputs = inputs.f.arr_0
    outputs = np.load('data/sudoku_outputs.npz')
    outputs = outputs.f.arr_0
    
    print(inputs.shape, outputs.shape)
    print(inputs[0])
    #split inputs[0] into 1x81
    inn = [int(i) for i in str(inputs[0])]
    inn = torch.from_numpy(np.array(inn))
    print(inn.shape)
    out = model(inn.reshape(1,-1))
    
main()