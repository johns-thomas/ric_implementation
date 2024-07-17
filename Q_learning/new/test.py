import numpy as np
import os
q_table_file_path = 'q_table-r1.npy'
if os.path.exists(q_table_file_path):
    q_table= np.load(q_table_file_path)

    non_zero_indices = np.argwhere(q_table != 0)

# Extract unique column indices where non-zero values exist
    non_zero_columns = np.unique(non_zero_indices[:, 1])
    print(non_zero_columns)