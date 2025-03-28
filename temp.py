from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
a = np.array([[1,0,1,0],
              [2,97,1,1],
              [0,2,0,1],
              [0,1,1,2]])

print(a[:,1].sum())
# ind = linear_assignment(a.max()-a)
# print(ind)
# ind = np.vstack(ind).T
# print(ind)
