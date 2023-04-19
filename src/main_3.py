#train a transformer to solve a sudoku puzzle
import torch
import csv
import numpy as np
from model import SudokuTransformer, SudokuTransformerED, BiDirectionalSudokuTransformer, SudokuTransformerATTENTION
from model import train
import torch.nn.functional as F
import gc
import pickle 
import time 

class Sudoku:
    def __init__(self):
        pass
    def load_data(self, inputs, outputs):
        """
        loads the whole csv data and returns a numpy array also save its compressed version in npz format
        """
        INs = []
        OUTs = []

        dp = len(inputs)
        for index in range(len(inputs)):
        # dp = 500000
        # for index in range(dp):
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



    def RearrangeData(self, inputs, outputs):
        """
        Remapping data here.
        Since the digits itself has no meaning into it, we need to map the first row from 1 to 9
        Rearrange the data to make it more suitable for the model
        returns the new inputs and outputs
        """
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


    def pretraining(self, inputs, outputs):
        """
        pretraining means here we are creating our own data for the model to learn

        """
        import copy
        # #load inputs and outputs
        # inputs = np.load('data/sudoku_inputs.npz')
        # inputs = inputs.f.arr_0
        # outputs = np.load('data/sudoku_outputs.npz')
        # outputs = outputs.f.arr_0
        # inputs, outputs =load_data(inputs=inputs, outputs=outputs)
        # inputs,outputs = RearrangeData(inputs, outputs)
        
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
        
        # #save inputs and outputs
        # np.savez_compressed('data/sudoku_inputs_pretrain.npz', inputs)
        # np.savez_compressed('data/sudoku_outputs_pretrain.npz', outputs)

        #return inputs, outputs as LONG TENSOR
        return torch.from_numpy(inputs.numpy()).long(), torch.from_numpy(outputs.numpy()).long()
        # return inputs, outputs
    
    def create_train_val_test_data(self):
        #load data
        inputs = np.load('data/sudoku_inputs.npz')
        inputs = inputs.f.arr_0
        outputs = np.load('data/sudoku_outputs.npz')
        outputs = outputs.f.arr_0
        
        inputs, outputs =self.load_data(inputs=inputs, outputs=outputs)
        print('inputs shape: {}'.format(inputs.shape))
        print('outputs shape: {}'.format(outputs.shape))
        print('data loaded')

        #rearrange data i.e remapping the data
        inputs, outputs = self.RearrangeData(inputs, outputs)
        print('data rearranged')
        #creating our own dataset for training
        inputs, outputs = self.pretraining(inputs=inputs, outputs=outputs)
        print('data pretraining is done')

        #dump the data
        np.savez_compressed('data/sudoku_inputs_remapped.npz', inputs)
        np.savez_compressed('data/sudoku_outputs_remapped.npz', outputs)
        print('data dumped')

        #use sklearn to split the data
        from sklearn.model_selection import train_test_split
        #split the data into train and test
        train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
        #split the train data into train and validation
        train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=42)
        print('splitting is done, saving the data')
        #save the data in npz format
        np.savez_compressed('data/train_inputs.npz', train_inputs)
        np.savez_compressed('data/train_outputs.npz', train_outputs)
        np.savez_compressed('data/val_inputs.npz', val_inputs)
        np.savez_compressed('data/val_outputs.npz', val_outputs)
        np.savez_compressed('data/test_inputs.npz', test_inputs)
        np.savez_compressed('data/test_outputs.npz', test_outputs)



    def load_train(self):
        train_inputs = np.load('data/train_inputs.npz')
        train_inputs = train_inputs.f.arr_0
        train_outputs = np.load('data/train_outputs.npz')
        train_outputs = train_outputs.f.arr_0
        return train_inputs, train_outputs

    def load_val(self):
        val_inputs = np.load('data/val_inputs.npz')
        val_inputs = val_inputs.f.arr_0
        val_outputs = np.load('data/val_outputs.npz')
        val_outputs = val_outputs.f.arr_0
        return val_inputs, val_outputs

    def load_test(self):
        test_inputs = np.load('data/test_inputs.npz')
        test_inputs = test_inputs.f.arr_0
        test_outputs = np.load('data/test_outputs.npz')
        test_outputs = test_outputs.f.arr_0
        return test_inputs, test_outputs

    def trainModel(self):
        # model = SudokuTransformer()
        model = SudokuTransformerATTENTION()
        model.train()
        

        inputs, outputs = self.load_train()
        print(inputs.shape)
        print(outputs.shape)

        #use only 1000 samples for training
        inputs = inputs[:1000]
        outputs = outputs[:1000]


        #convert to tensor
        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)

        # create tensor dataset
        train_data = torch.utils.data.TensorDataset(inputs, outputs)

        #create dataloader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
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

    def test(self):
        start = time.time()
        # if False:
        #     #load INs and OUTs
        #     INs = np.load('data/INs.npz')
        #     INs = INs.f.arr_0
        #     OUTs = np.load('data/OUTs.npz')
        #     OUTs = OUTs.f.arr_0
        # else:
        #     #load INs and OUTs
        #     INs = np.load('data/sudoku_inputs_pretrain.npz')
        #     INs = INs.f.arr_0
        #     OUTs = np.load('data/sudoku_outputs_pretrain.npz')
        #     OUTs = OUTs.f.arr_0

        INs = np.load('data/train_inputs.npz')
        INs = INs.f.arr_0
        OUTs = np.load('data/train_outputs.npz')
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

        import os
        if not os.path.exists('output'):
            os.makedirs('output')
        
        accuracies = []
        # for index in range(1000):
        for index in range(10):
            #prepare attention mask
            attention_mask = INs[index].reshape(1,-1) != 0
            with torch.no_grad():
                pred, attention_scores_list = model(INs[index].reshape(1,-1), attention_mask = attention_mask, test = True)
            # print(attention_scores)
            # print([scrs.reshape(-1) for scrs in attention_scores])

            temp_outpath = 'output/data_{}/'.format(index)
            if not os.path.exists(temp_outpath):
                os.makedirs(temp_outpath)
            
            #get the attention of last layer
            attention_scores = attention_scores_list[-1]
            attention_scores  = attention_scores.reshape(81, 81)

            #for each unfilled cells, extract the attention scores
            unfilled_cells = np.where(INs[index].reshape(1,-1) == 0)[1]


            #create a hetmap plot of each empty cell
            for i_empty, cell in enumerate(unfilled_cells):
                scores = attention_scores[cell]
                scores = scores.reshape(9,9)
                import matplotlib.pyplot as plt
                
                #create a figure
                fig, ax = plt.subplots()

                #create a heatmap using seaborn
                import seaborn as sns
                sns.heatmap(scores, annot=True, ax = ax, cmap='gray', fmt='.2f', annot_kws={"size": 8}) # font size
                #save the figure
                plt.savefig(temp_outpath + 'heatmap_{}_{}.png'.format(cell //9, cell % 9), bbox_inches='tight')
                plt.close()

            
            
            # #for each attentions score create a heatmap
            # for i, attention_scores in enumerate(attention_scores_list):
            #     print(attention_scores.reshape(-1).numpy().tolist())
            #     print(attention_scores.shape)


            #     # get the attention scores sum of each row
            #     attention_scores = attention_scores.reshape(81,81)
            #     attention_scores = attention_scores.detach().numpy()

            #     #for each empty cell extract the attention scores
            #     empty_cells = np.where(INs[index].reshape(1,-1) == 0)[1]

            #     if i == 9:
            #         #create a hetmap plot of each empty cell
            #         for i_empty, cell in enumerate(empty_cells):
            #             scores = attention_scores[cell]
            #             scores = scores.reshape(9,9)
            #             import matplotlib.pyplot as plt
            #             # create a figure
            #             fig, ax = plt.subplots(figsize=(9,9))
            #             # plot heatmap use color map gray: use seaborn
            #             import seaborn as sns
            #             sns.heatmap(scores, cmap='gray', ax=ax)
            #             # save the figure
            #             filename = temp_outpath + 'attention_scores_{}_{}.png'.format(cell // 9, cell % 9)
            #             fig.savefig(filename, bbox_inches='tight')
            #             # close the figure
            #             plt.close(fig)





            #     #for rach row get the sum of the attention scores
            #     attention_scores_mean = np.sum(attention_scores, axis=0)

            #     print(attention_scores_mean)

            #     #create a heatmp of the attention scores
            #     attention_scores_mean = attention_scores_mean.reshape(9,9)

            #     #get a list of indices where there attention scores are in descending order
            #     attention_scores_indices_sorted = np.argsort(attention_scores_mean, axis=None)[::-1]
                
            #     #convert 1D indices to 2D indices
            #     attention_scores_indices_sorted = np.unravel_index(attention_scores_indices_sorted, attention_scores_mean.shape)

            #     #convert to list
            #     attention_scores_indices_sorted = list(attention_scores_indices_sorted)
            #     x = [attention_scores_indices_sorted[0], attention_scores_indices_sorted[1]]
            #     x = np.array(x).T.tolist()
            #     attention_scores_indices_sorted = x

            #     #write it into a file
            #     with open('{}attention_scores_indices_sorted_{}.txt'.format(temp_outpath, i), 'w') as f:
            #         writer = csv.writer(f)
            #         writer.writerows(attention_scores_indices_sorted)
            
            #     #plot as heatmap
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt

            #     #plot as heatmap and add input as label for each cell in heatmap and use gray scale colorbar
            #     sns.heatmap(attention_scores_mean, annot=INs[index].reshape(9,9), fmt='d', cmap='gray')

            #     # sns.heatmap(attention_scores_mean, annot=INs[index].reshape(9,9), fmt='d')
            #     # sns.heatmap(attention_scores_mean)

            #     #add title
            #     plt.title('Attention Scores, Layer: {}'.format(i+1))

            #     #make x ticks and y ticks hidden
            #     plt.xticks([])
            #     plt.yticks([])


            #     #save plot
            #     plt.savefig('{}attention_scores_{}.png'.format(temp_outpath, i), bbox_inches='tight')
            #     plt.close()

            #     #write attention scores to file
            #     with open('{}attention_scores_{}.txt'.format(temp_outpath, i), 'w') as f:
            #         writer = csv.writer(f)
            #         writer.writerows(attention_scores_mean)

            #     # #get index of max attention score
            #     # max_index = np.argmax(attention_scores)
            #     # print(max_index)
            #     # # print max index in 2D format
            #     # print('max index: {}'.format((max_index//9, max_index%9)))



            # # attention_scores = np.sum(attention_scores, axis=1)

            # # print(attention_scores)


            # # #create a heatmeap of the attention scores
            # # attention_scores = attention_scores[-1].reshape(81,81)
            # # attention_scores = attention_scores.detach().numpy()
            # # #plot as heatmap
            # # import seaborn as sns
            # # import matplotlib.pyplot as plt
            # # # sns.heatmap(attention_scores)
            # # # #save plot
            # # # plt.savefig('attention_scores.png')

            # # #for each attention score, create a heatmap
            # # for i in range(len(attention_scores)):
            # #     fig, ax = plt.subplots(1,1)
            # #     #plot as heatmap
            # #     sns.heatmap(attention_scores[i].reshape(81,81).detach().numpy())
            # #     #save plot
            # #     plt.savefig('attention_scores_{}.png'.format(i))
            # #     plt.close()
            # # #create a hemap plot of input sudoku and label the numbers
            print(INs[index].reshape(9,9))


            pred = torch.argmax(pred, dim=-1)
            input = INs[index].reshape(-1)
            pred = pred.reshape(-1)
            print(pred.reshape(9,9))
            target = OUTs[index].reshape(-1)
            # print(pred.shape, target.shape)
            assert(pred.shape == target.shape)

            #write input, pred and target to file
            with open('output/data_{}/input.txt'.format(index), 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(INs[index].reshape(9,9).numpy().tolist())
            with open('output/data_{}/pred.txt'.format(index), 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(pred.reshape(9,9).numpy().tolist())
            with open('output/data_{}/target.txt'.format(index), 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(target.reshape(9,9).numpy().tolist())

            #dump input as list string so that it can be parse using ast.literal_eval later
            with open('output/data_{}/input_eval.txt'.format(index), 'w') as f:
               f.write(str(INs[index].reshape(9,9).numpy().tolist()))
                

            #if prediction is correct
            if (pred == target).all():
                #increase accuracy
                correct += 1
                if correct > 2: 
                    print(1/0)

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
          
if __name__ == '__main__':
    sudoku = Sudoku()
    print(sudoku)
    # sudoku.create_train_val_test_data()
    print('data creation and dump done')

    # sudoku.trainModel()
    sudoku.test()

