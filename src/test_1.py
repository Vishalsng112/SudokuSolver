from z3 import *

def solve_sudoku(sudoku):
    solver = Solver()
    cells = [[Int("cell_%s_%s" % (i, j)) for j in range(9)] for i in range(9)]

    def add_constraints(cells, solver):
        # Add constraints for rows, columns, and 3x3 boxes
        for i in range(9):
            solver.add(Distinct(cells[i]))  # Rows
            solver.add(Distinct([cells[j][i] for j in range(9)]))  # Columns

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                solver.add(Distinct([cells[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]))  # Boxes
        
        for i in range(9):
            for j in range(9):
                solver.add(And(cells[i][j] >=1,cells[i][j] <=9 ))
        return solver
    
    solver = add_constraints(cells= cells, solver= solver)
    solver.push()
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0 :
                solver.add(cells[i][j] == sudoku[i][j])
    # Check for unique solution
    result = solver.check()
    if result == sat:
        model = solver.model()
        solution = [[model.evaluate(cells[i][j]).as_long() for j in range(9)] for i in range(9)]
        print(solution)
        #create constraints
        constr = []
        for i in range(9):
            for j in range(9):
                if sudoku[i][j] == 0 :
                    constr.append(cells[i][j] == solution[i][j])
        solver.push()
        solver.add(Not(And(constr)))

        if solver.check() ==sat:
            print('It has multiple solution')
            model = solver.model()
            solution = [[model.evaluate(cells[i][j]).as_long() for j in range(9)] for i in range(9)]
            print(solution)
            return False
        else:
            print('it has unique solution')
            return True
    elif result == unsat:
        raise RuntimeError('It does not have any solution.')

# Example Sudoku puzzle
sudoku = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solution = solve_sudoku(sudoku)
print(solution)
# if solution is not None:
#     print("The Sudoku puzzle has a unique solution.")
#     print("Solution:")
#     for row in solution:
#         print(row)
# else:
#     print("The Sudoku puzzle does not have a unique solution.")
