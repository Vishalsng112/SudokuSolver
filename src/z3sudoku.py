#import z3
from z3 import *
import numpy as np

class z3Sudoku:
    def __init__(self, board):
        self.board = board
        self.solver = Solver()
        #create 9x9 matrix of integers
        self.matrix = [[Int("x_%s_%s" % (i, j)) for j in range(9)] for i in range(9)]
        #add constraints
        self.current_clause_count = 0
        self.constraintsMap = {}
        self.constraintsReverseMap = {}
        self.addConstraints()

        
    def addConstraints(self):
        #add constraints for each row
        for i in range(9):
            # self.solver.add(Distinct(self.matrix[i]))
            self.solver.assert_and_track(Distinct(self.matrix[i]), 'c_{}'.format(self.current_clause_count))
            self.constraintsMap['c_{}'.format(self.current_clause_count)] = Distinct(self.matrix[i])
            self.current_clause_count += 1

        #add constraints for each column
        for j in range(9):
            # self.solver.add(Distinct([self.matrix[i][j] for i in range(9)]))
            self.solver.assert_and_track(Distinct([self.matrix[i][j] for i in range(9)]), 'c_{}'.format(self.current_clause_count))
            self.constraintsMap['c_{}'.format(self.current_clause_count)] = Distinct([self.matrix[i][j] for i in range(9)])
            self.current_clause_count += 1

        #add constraints for each 3x3 square
        for i in range(3):
            for j in range(3):
                # self.solver.add(Distinct([self.matrix[3*i+k][3*j+l] for k in range(3) for l in range(3)]))
                self.solver.assert_and_track(Distinct([self.matrix[3*i+k][3*j+l] for k in range(3) for l in range(3)]), 'c_{}'.format(self.current_clause_count))
                self.constraintsMap['c_{}'.format(self.current_clause_count)] = Distinct([self.matrix[3*i+k][3*j+l] for k in range(3) for l in range(3)])
                self.current_clause_count += 1


        #add constraints for each cell
        for i in range(9):
            for j in range(9):
                # self.solver.add(And(self.matrix[i][j] >= 1, self.matrix[i][j] <= 9))
                self.solver.assert_and_track(And(self.matrix[i][j] >= 1, self.matrix[i][j] <= 9), 'c_{}'.format(self.current_clause_count))
                self.constraintsMap['c_{}'.format(self.current_clause_count)] = And(self.matrix[i][j] >= 1, self.matrix[i][j] <= 9)
                self.current_clause_count += 1
    
    def addKnownValues(self, knownValues):
        #add constraints for known values
        for (i, j, val) in knownValues:
            # print(type(i), type(j), type(val))
            # self.solver.add(self.matrix[i][j] == int(val))
            self.solver.assert_and_track(self.matrix[i][j] == int(val), 'c_{}'.format(self.current_clause_count))
            self.constraintsMap['c_{}'.format(self.current_clause_count)] = self.matrix[i][j] == int(val)
            self.current_clause_count += 1


    def addKnowValuesWithNot(self, knownValues):
        #add constraints for known values
        for (i, j, val) in knownValues:
            # print(type(i), type(j), type(val))
            # self.solver.add(self.matrix[i][j] != int(val))
            self.solver.assert_and_track(self.matrix[i][j] != int(val), 'c_{}'.format(self.current_clause_count))
            self.constraintsMap['c_{}'.format(self.current_clause_count)] = self.matrix[i][j] != int(val)
            self.current_clause_count += 1    

    def solve(self):
        #check if there is a solution
        if self.solver.check() == sat:
            #get the model
            m = self.solver.model()
            #print the solution
            for i in range(9):
                print([m.evaluate(self.matrix[i][j]) for j in range(9)])
                return True
        else:
            print("No solution found")
            return False
    
    def printMatrix(self):
        #print the matrix
        for i in range(9):
            print([self.matrix[i][j] for j in range(9)])
    
    def printSolver(self):
        #print the solver
        print(self.solver)
    
    def printModel(self):
        #print the model
        print(self.solver.model())
    
    def printConstraints(self):
        #print the constraints
        print(self.solver.assertions())
    
    def printCheck(self):
        #print the check
        print(self.solver.check())
    
    def printUnsatCore(self):
        #print the unsat core
        print(self.solver.unsat_core())
    
    def printStatistics(self):
        #print the statistics
        print(self.solver.statistics())
    
    def printAll(self):
        #print everything
        self.printMatrix()
        self.printSolver()
        self.printModel()
        self.printConstraints()
        self.printCheck()
        self.printUnsatCore()
        self.printStatistics()
    def isMultpleSolutions(self):
        #check if there are multiple solutions
        if self.solver.check() == sat:
            # print('print the first solution')
            # print(self.solver.model())
            #extract the solution
            m = self.solver.model()
            # ADD CONSTRAINT THAT THE SOLUTION IS NOT THE SAME AS THE FIRST ONE
            # self.solver.push()

            # print(Not(And([self.matrix[i][j] == m.evaluate(self.matrix[i][j]) for i in range(9) for j in range(9)])))
            # print(1/0)
            #add the constraint that the solution is not the same as the first one
            self.solver.add(Not(And([self.matrix[i][j] == m.evaluate(self.matrix[i][j]) for i in range(9) for j in range(9)])))

            # self.solver.add(Not(self.solver.model() == self.solver.model()))
            if self.solver.check() == sat:
                return True
            else:
                return False
        else:
            # raise RuntimeError('It does not have a solution.')
            print('It does not have a solution.')
            return False
        
    def computeAllSolutionCounts(self):
        #compute the number of solutions
        count = 0
        while self.solver.check() == sat:
            count += 1
            print(count)
            m = self.solver.model()
            self.solver.add(Not(And([self.matrix[i][j] == m.evaluate(self.matrix[i][j]) for i in range(9) for j in range(9)])))
        return count

    
if __name__ == '__main__':
    # board = [[0, 0, 0, 4, 5, 6, 0, 0, 0],
    #         [0, 0, 0, 7, 2, 3, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [9, 0, 7, 5, 4, 2, 1, 6, 3],
    #         [3, 1, 0, 8, 7, 9, 4, 5, 2],
    #         [5, 4, 2, 3, 6, 1, 9, 7, 8],
    #         [2, 3, 9, 0, 8, 4, 5, 1, 7],
    #         [4, 7, 8, 2, 1, 0, 3, 9, 0],
    #         [6, 5, 1, 9, 0, 7, 8, 2, 4]]


    path = 'output/data_0/'
    import csv

    with open(path + 'input.txt', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        board = list(reader)
        board = [[int(x) for x in row] for row in board]
        board = np.array(board)
        print(board)

    #read pred.txt file 
    with open(path + 'pred.txt', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        pred = list(reader)
        pred = [[int(x) for x in row] for row in pred]
        pred = np.array(pred)
        print(pred)

    attention_file_index = 9
    with open(path + 'attention_scores_indices_sorted_{}.txt'.format(attention_file_index), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        attention_list = list(reader)
        attention_list = [[int(x) for x in row] for row in attention_list]
    print(attention_list)




    # board = [[0, 0, 3, 4, 5, 6, 7, 8, 9], 
    #          [8, 6, 4, 7, 3, 0, 2, 1, 5], 
    #          [7, 9, 5, 0, 2, 8, 6, 4, 3], 
    #          [4, 8, 7, 9, 6, 0, 3, 5, 2], 
    #          [0, 0, 6, 8, 7, 5, 1, 0, 4],
    #          [5, 1, 9, 2, 4, 0, 8, 0, 0],
    #          [9, 4, 8, 3, 1, 0, 0, 2, 6],
    #          [0, 0, 1, 5, 9, 2, 4, 3, 8], 
    #          [3, 5, 2, 6, 8, 4, 9, 7, 1]]

    # board = [[1., 0., 0., 4., 0., 0., 7., 0., 9.],
    #         [0., 0., 6., 0., 0., 0., 0., 2., 5.],
    #         [0., 7., 4., 0., 0., 8., 3., 0., 0.],
    #         [0., 0., 7., 3., 8., 9., 6., 1., 0.],
    #         [0., 0., 1., 0., 0., 0., 0., 9., 4.],
    #         [2., 8., 0., 0., 0., 0., 5., 0., 0.],
    #         [6., 0., 0., 9., 7., 2., 0., 0., 0.],
    #         [0., 3., 5., 0., 6., 0., 2., 0., 0.],
    #         [0., 1., 0., 8., 3., 0., 0., 0., 0.]]
    import copy 

    actual_board = copy.deepcopy(board)
    board = actual_board
    for i in range(len(attention_list)):
        row, col = attention_list[i]
        # board = np.array(actual_board)
        board[row][col] = 0
        # print(board)
        # print(row, col)
        s = z3Sudoku(board)
        knownValues = [(i, j, board[i][j]) for i in range(9) for j in range(9) if actual_board[i][j] != 0]
        s.addKnownValues(knownValues=knownValues)
        s.addKnowValuesWithNot(knownValues = [(row,col, pred[row,col])])
        
        if s.isMultpleSolutions():
            print('multiple solutions')
            board[row][col] = actual_board[row][col]
            print(i)
            break
        else:
            print(i)
        # print(i, s.isMultpleSolutions())


    # knownValues = [(i, j, board[i][j]) for i in range(9) for j in range(9) if board[i][j] != 0]
    # print(board)
    # print(knownValues)

    # s = z3Sudoku(board)
    # s.addKnownValues(knownValues)
    # # s.solve()
    # print(s.isMultpleSolutions())
    # # print(s.computeAllSolutionCounts())


