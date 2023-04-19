#import z3
from z3 import *

class z3Sudoku:
    def __init__(self):
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
            self.solver.add(self.matrix[i][j] == val)
    
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
    