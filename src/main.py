#train a transformer to solve a sudoku puzzle
import torch
import csv
import numpy as np
from model import SudokuTransformer, SudokuTransformerED
from model import train
import torch.nn.functional as F


def load_data(inputs, outputs):
    INs = []
    OUTs = []
    # for index in range(len(inputs)):
    dp = 100
    for index in range(dp):
        INs.append([ int(i) for i in str(inputs[index])] )
        OUTs.append([ int(i) for i in str(outputs[index])])
    #convert to numpy array
    INs = np.array(INs, dtype = np.float32)
    OUTs = np.array(OUTs, dtype = np.float32)

    #reshape
    INs = INs.reshape(dp, 1, -1)
    OUTs = OUTs.reshape(dp, 1, -1)
    #convert to tensor
    INs = torch.from_numpy(INs)
    OUTs = torch.from_numpy(OUTs)
    #return
    return INs, OUTs

def main():
    model = SudokuTransformer()
    # model = SudokuTransformerED()
    model.train()
    # print(model.eval())
    
    #load data
    inputs = np.load('data/sudoku_inputs.npz')
    inputs = inputs.f.arr_0
    outputs = np.load('data/sudoku_outputs.npz')
    outputs = outputs.f.arr_0
    
    # print(inputs.shape, outputs.shape)
    # print(inputs[0])
    # #split inputs[0] into 1x81
    # inn = [int(i) for i in str(inputs[0])]
    # inn = torch.from_numpy(np.array(inn))
    # print(inn.shape)
    # out = model(inn.reshape(1,-1))

    # print(inputs.shape, outputs.shape)
    # print(load_data(inputs=inputs, outputs=outputs)[0].shape)
    inputs, outputs =load_data(inputs=inputs, outputs=outputs)
    # create tensor dataset
    train_data = torch.utils.data.TensorDataset(inputs, outputs)
    print(inputs[0])
    model2 = train(model = model, train_data= train_data, epochs=100, lr = 0.01)
    
    #get prediction
    model2.eval()

    inn = inputs[0].reshape(1,-1)  
    pred = model2(inn)
    pred = torch.argmax(pred, dim=-1)
    print(inputs[0].reshape(1,-1))
    print(pred)
    print(pred[0] - inn)
main()