import csv
import numpy as np
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self) -> None:
        pass
    
    def preprocess(self, file_path: str , outfile) -> list:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]
            #convert to numpy array
            data = np.array(data)
            #save the data as a csv file
            csv_writer = csv.writer(open(outfile, 'w'))
            csv_writer.writerows(data)
            
            #dump numpy array to a file in np.z format
            np.savez_compressed('data/sudoku_preprocessed.npz', data)
            np.save('data/sudoku_preprocessed.npy', data)
            
            inputs, outputs = data[:,0], data[:,1]
            
            #save the inputs and outputs as numpy compressed files
            np.savez_compressed('data/sudoku_inputs.npz', inputs)
            np.savez_compressed('data/sudoku_outputs.npz', outputs)
            
    def display_sudoku(self, data: np.array):
        fig, axs = plt.subplots(1,1)
        
        
    
def main():
    data_handler = DataHandler()
    data_handler.preprocess(file_path = 'data/sudoku.csv', outfile= 'data/sudoku_preprocessed.csv')
main()