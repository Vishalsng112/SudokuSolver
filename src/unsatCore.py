from z3 import *

#create three boolean varibles a,b,c
a, b, c = Bools('a b c')

#create a solver
s = Solver()
# s.set(unsat_core=True)

#add constraints
s.assert_and_track(a, 'p1')
# add ~a ^b
s.assert_and_track(And(Not(a), a), 'p2')
# # add ~a v ~b
# s.assert_and_track(Or(Not(a), Not(b)), 'p3')
# # add b v c
# s.assert_and_track(Or(b, c), 'p4')

#print the solver
print(s)

print(s.check())

#if unsat then print the unsat core
if s.check() == unsat:
    print(s.unsat_core())
    