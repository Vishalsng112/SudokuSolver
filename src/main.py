#train a transformer to solve a sudoku puzzle
import torch
import csv
import numpy as np
from model import SudokuTransformer, SudokuTransformerED, BiDirectionalSudokuTransformer
from model import train
import torch.nn.functional as F
import gc
import pickle 
import time 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_data(inputs, outputs):
    INs = []
    OUTs = []

    # dp = len(inputs)
    # for index in range(len(inputs)):
    dp = 500000
    for index in range(dp):
        print('{}/{}'.format(index, dp))
        INs.append([ int(i) for i in str(inputs[index])] )
        OUTs.append([ int(i) for i in str(outputs[index])])
    #convert to numpy array
    INs = np.array(INs, dtype = np.float32)
    OUTs = np.array(OUTs, dtype = np.float32)

    #reshape
    INs = INs.reshape(dp, -1)
    OUTs = OUTs.reshape(dp, -1)
    #convert to tensor
    INs = torch.from_numpy(INs)
    OUTs = torch.from_numpy(OUTs)

    #dump numpy array to a file in np.z format
    np.savez_compressed('data/INs.npz', INs)
    np.savez_compressed('data/OUTs.npz', OUTs)
    #return
    return INs, OUTs



def RearrangeData(inputs, outputs):
    inputs = inputs.numpy()
    outputs = outputs.numpy()
    newInputs = []
    newOutputs = []
    # print(inputs.shape, outputs.shape)
    for inp, out in zip(inputs, outputs):
        #map first row of the output from 1 to 9
        Map = {}
        inp = inp.reshape(9,9)
        out = out.reshape(9,9)
        # print(inp)
        # print(out)
        for index, value in enumerate(out[0]):
            Map[value] = index + 1
        # print(Map)
        #remap the output
        newOut = []
        for row in out:
            newRow = []
            for value in row:
                newRow.append(Map[value])
            newOut.append(newRow)
        newOut = np.array(newOut)
        newOutputs.append(newOut)

        #remap the input
        newIn = []
        for row in inp:
            newRow = []
            for value in row:
                if value == 0:
                    newRow.append(0)
                else:
                    newRow.append(Map[value])
            newIn.append(newRow)
        newIn = np.array(newIn)
        newInputs.append(newIn)
    newInputs = np.array(newInputs)
    newOutputs = np.array(newOutputs)
    #create tensor
    newInputs = torch.from_numpy(newInputs)
    newOutputs = torch.from_numpy(newOutputs)
    #change type to float
    newInputs = newInputs.type(torch.FloatTensor)
    newOutputs = newOutputs.type(torch.FloatTensor)
    #reshape
    newInputs = newInputs.reshape(-1,81)
    newOutputs = newOutputs.reshape(-1, 81)
    return newInputs, newOutputs

def main():
    model = SudokuTransformer()
    # model = SudokuTransformerED()
    # model = BiDirectionalSudokuTransformer()
    model.train()
    
    #load data
    inputs = np.load('data/sudoku_inputs.npz')
    inputs = inputs.f.arr_0
    outputs = np.load('data/sudoku_outputs.npz')
    outputs = outputs.f.arr_0
    
    inputs, outputs =load_data(inputs=inputs, outputs=outputs)
    print('inputs shape: {}'.format(inputs.shape))
    print('outputs shape: {}'.format(outputs.shape))

    #rearrange data
    inputs, outputs = RearrangeData(inputs, outputs)
    print(inputs[0])
    print(outputs[0])
    # print(1/0)

    inputs, outputs = pretraining()
    # create tensor dataset
    train_data = torch.utils.data.TensorDataset(inputs, outputs)

    #create dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    print(inputs[0])
    print(outputs[0])
    del inputs, outputs
    gc.collect()

    model2 = train(model = model, train_data= train_loader, epochs=20, lr = 0.001)

    #if model folder does not exist, create it
    import os
    if not os.path.exists('model'):
        os.makedirs('model')
    pickle.dump(model2, open('model/sudoku_transformer.pkl', 'wb'))

    #id model folder does not exist, create it
    import os
    if not os.path.exists('model'):
        os.makedirs('model')

    #save model
    torch.save(model2.state_dict(), 'model/sudoku_transformer.pth')

    #get prediction
    model2.eval()

    # inn = inputs[0].reshape(1,-1)  
    # pred = model2(inn)
    # pred = torch.argmax(pred, dim=-1)
    # print(inputs[0].reshape(1,-1))
    # print(pred)
    # print(pred[0] - inn)

def test():
    start = time.time()
    if False:
        #load INs and OUTs
        INs = np.load('data/INs.npz')
        INs = INs.f.arr_0
        OUTs = np.load('data/OUTs.npz')
        OUTs = OUTs.f.arr_0
    else:
        #load INs and OUTs
        INs = np.load('data/sudoku_inputs_pretrain.npz')
        INs = INs.f.arr_0
        OUTs = np.load('data/sudoku_outputs_pretrain.npz')
        OUTs = OUTs.f.arr_0
    #load model
    # model = SudokuTransformer()
    # model.load_state_dict(torch.load('model/sudoku_transformer.pth'))
    # model.eval()
    model = pickle.load(open('model/sudoku_transformer.pkl', 'rb'))
    model.eval()
    print(model.eval())
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params: {}'.format(pytorch_total_params))
    # print(1/0)

    #create tensor object of INs and OUTs
    INs = torch.from_numpy(INs)
    OUTs = torch.from_numpy(OUTs)
    # get accuracy
    correct =  0
    total = len(INs)

    accuracies = []
    # for index in range(1000):
    for index in range(INs.shape[0]):
        #prepare attention mask
        attention_mask = INs[index].reshape(1,-1) != 0
        pred = model(INs[index].reshape(1,-1), attention_mask = attention_mask)
        pred = torch.argmax(pred, dim=-1)
        input = INs[index].reshape(-1)
        pred = pred.reshape(-1)
        target = OUTs[index].reshape(-1)
        # print(pred.shape, target.shape)
        assert(pred.shape == target.shape)
        #if prediction is correct
        if (pred == target).all():
            #increase accuracy
            correct += 1
        # else:
        #     print(input.reshape(9,9))
        #     print(pred.reshape(9,9))
        #     print(target.reshape(9,9))
        #     print(1*(input.reshape(9,9) == pred.reshape(9,9)))
        #     print(1/0)

        #get indices where input is 0
        indices = torch.where(input == 0)[0]
        #check how many predictions are correct
        # count = (pred[indices] == target[indices]).sum().item()
        # accuracies.append(count/(indices.shape[0]))
        
        count = (pred[indices] == target[indices]).sum().item()
        accuracies.append(count/(indices.shape[0]))

        #convert input to type int
        input = input.type(torch.IntTensor)
        #convert pred to type int
        pred = pred.type(torch.IntTensor)
        #convert target to type int
        target = target.type(torch.IntTensor)
        # print(input.reshape(9,9))
        # print(pred.reshape(9,9))
        # print(target.reshape(9,9))
        # print(1*(input.reshape(9,9) != pred.reshape(9,9)))
        indices = torch.where(input != 0)[0]
        # print(indices.shape)
        # print(input[indices])
        # print(pred[indices])

        #create output if not exists
        # path = 'output/train/{}/'.format(index)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # import os 
        # if not os.path.exists('output'):
        #     os.makedirs('output')
        # # save input output and prediction as csv text file
        # np.savetxt('output/{}_input.csv'.format(index), input.reshape(9,9), delimiter=',', fmt='%d')
        # np.savetxt('output/{}_prediction.csv'.format(index), pred.reshape(9,9), delimiter=',', fmt='%d')
        # np.savetxt('output/{}_target.csv'.format(index), target.reshape(9,9), delimiter=',', fmt='%d')


        # assert((pred[indices] == input[indices]).all())
        # print(pred.shape)
    print(accuracies)
    print('Accuracy: {}'.format(correct/total))  
    print(np.mean(accuracies))
    end = time.time()
    print('Time: {}'.format(end-start))
          
# main()
# test()

def pretraining():
    import copy
    #load inputs and outputs
    inputs = np.load('data/sudoku_inputs.npz')
    inputs = inputs.f.arr_0
    outputs = np.load('data/sudoku_outputs.npz')
    outputs = outputs.f.arr_0
    inputs, outputs =load_data(inputs=inputs, outputs=outputs)
    inputs,outputs = RearrangeData(inputs, outputs)
    
    inputs = copy.deepcopy(outputs)
    print(outputs.shape)
    total = len(outputs)
    rng = np.random.RandomState(42)
    for i in range(total):
        #fill 10% outputs with 0
        #get random indices
        indices = rng.choice(81, 10, replace=False)
        # print(indices)
        #fill outputs with 0
        inputs[i][indices] = 0
    
    #save inputs and outputs
    np.savez_compressed('data/sudoku_inputs_pretrain.npz', inputs)
    np.savez_compressed('data/sudoku_outputs_pretrain.npz', outputs)

    for i in range(2):
        print(inputs[i])
        print(outputs[i])
        print("------------------")
    #return inputs, outputs as LONG TENSOR
    return torch.from_numpy(inputs.numpy()).long(), torch.from_numpy(outputs.numpy()).long()
    # return inputs, outputs


def create_train_test_data(inputs, outputs):
    #split data into train and test
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    #convert to tensor
    train_inputs = torch.from_numpy(train_inputs.numpy()).long()
    train_outputs = torch.from_numpy(train_outputs.numpy()).long()
    test_inputs = torch.from_numpy(test_inputs.numpy()).long()
    test_outputs = torch.from_numpy(test_outputs.numpy()).long()
    #return train and test data
    return train_inputs, test_inputs, train_outputs, test_outputs


def test_new():
    
# pretraining()
# main()
test()
