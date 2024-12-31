import numpy as np
row = 22
start = (row+1)*128 + 1
end = start+128


array = np.arange(start, end).reshape(1, -1) 
column_vector = np.arange(1, 129).reshape(-1, 1)
result = np.dot(array, column_vector)
print(column_vector)
print(len(column_vector[0]))
print(result)