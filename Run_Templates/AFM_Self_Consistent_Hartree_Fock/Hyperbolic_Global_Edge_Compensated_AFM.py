import numpy as np
from Axial_Magnetic_Field.Order_Parameter import magnetic_order_parameters_localgen
import matplotlib.pyplot as plt

# System parameters
pval = 10
num_generations = 4
hubbard_U = 1.0

# This code presumes calculated and saved raw data (from Hyperbolic_SCHF_AFM_With_Axial_Field.py in this directory)
raw_data_path = 'path_to_saved_raw_data_npy_file'

# generate generation-averaged AFM and FM order parameters for each U value in raw_data_path file
gen_local_afm_order, gen_local_fm_order = magnetic_order_parameters_localgen(raw_data_path, pval, num_generations)

# Here I am assuming only one U value is sought for analysis (may need to adjust line for
# desired floating point tolerance)
relevant_row = np.where(np.abs(hubbard_U - gen_local_afm_order[:, 0]) < 0.001)[0]
relevant_gen_local_afm_order = gen_local_afm_order[relevant_row, :]
relevant_gen_local_fm_order = gen_local_fm_order[relevant_row, :]

# Plot AFM and FM order parameters as a function of generation number
plt.scatter(range(num_generations), gen_local_afm_order[0, 1:])
plt.show()
plt.scatter(range(num_generations), gen_local_fm_order[0, 1:])
plt.show()



