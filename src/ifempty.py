import os
import numpy as np

#read folder name in output folder
path = 'output/'
folders = os.listdir(path)
print(folders)
folders.sort()
#for each folder read filenames having "unsat_core" in it
for folder in folders:
    # read filenames inside this folde
    filenames = os.listdir(path+folder)
    # for each filename
    for filename in filenames:
        if  "unsat_core" in filename :
            # unsat_core_0_1_81.txt
            #replacplce ".txt" with ""
            filename = filename.replace(".txt","")
            #split filename with "_"
            filename = filename.split("_")
            #get the last element
            filename = filename[-1]
            #convert it to integer
            filename = int(filename)
            # if it less then 81 then print file name
            if filename < 81:
                print(filename)
                print(1/0)
            else:
                print("error")
                
