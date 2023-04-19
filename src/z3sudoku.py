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
        self.addConstraints()
    
    def addConstraints(self):
        #add constraints for each row
        for i in range(9):
            self.solver.add(Distinct(self.matrix[i]))
        #add constraints for each column
        for j in range(9):
            self.solver.add(Distinct([self.matrix[i][j] for i in range(9)]))
        #add constraints for each 3x3 square
        for i in range(3):
            for j in range(3):
                self.solver.add(Distinct([self.matrix[3*i+k][3*j+l] for k in range(3) for l in range(3)]))
        #add constraints for each cell
        for i in range(9):
            for j in range(9):
                self.solver.add(And(self.matrix[i][j] >= 1, self.matrix[i][j] <= 9))
    
    def addKnownValues(self, knownValues):
        #add constraints for known values
        for (i, j, val) in knownValues:
            # print(type(i), type(j), type(val))
            self.solver.add(self.matrix[i][j] == int(val))
    
    def solve(self):
        #check if there is a solution
        if self.solver.check() == sat:
            #get the model
            m = self.solver.model()
            #print the solution
            for i in range(9):
                print([m.evaluate(self.matrix[i][j]) for j in range(9)])
        else:
            print("No solution found")
    
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
            self.solver.push()
            # ADD CONSTRAINT THAT THE SOLUTION IS NOT THE SAME AS THE FIRST ONE
            print(self.solver.model())

            #extract the solution
            m = self.solver.model()

            #add the constraint that the solution is not the same as the first one
            self.solver.add(Not([self.matrix[i][j] == m.evaluate(self.matrix[i][j]) for i in range(9) for j in range(9)]))

            # self.solver.add(Not(self.solver.model() == self.solver.model()))
            if self.solver.check() == sat:
                return True
            else:
                return False
        else:
            return False

    

board = [[1, 0, 3, 4, 5, 6, 7, 8, 9],
        [0, 9, 5, 7, 2, 3, 6, 0, 1],
        [0, 6, 4, 1, 9, 8, 2, 3, 5],
        [9, 0, 7, 5, 4, 2, 1, 6, 3],
        [3, 1, 0, 8, 7, 9, 4, 5, 2],
        [5, 4, 2, 3, 6, 1, 9, 7, 8],
        [2, 3, 9, 0, 8, 4, 5, 1, 7],
        [4, 7, 8, 2, 1, 0, 3, 9, 0],
        [6, 5, 1, 9, 0, 7, 8, 2, 4]]

board = np.array(board)
knownValues = [(i, j, board[i][j]) for i in range(9) for j in range(9) if board[i][j] != 0]
print(board)
print(knownValues)

s = z3Sudoku(board)
s.addKnownValues(knownValues)
# s.solve()
print(s.isMultpleSolutions())

