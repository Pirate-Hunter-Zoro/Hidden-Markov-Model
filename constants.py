import numpy as np

# Initial state probabilitys
# TODO - find out if this should be the steady state distribution instead
Initial_P = np.array([[0.7],[0.3]])

# State transition numpy array
Transformation_Matrix = np.array([[0.8,0.2],[0.3,0.7]])
# Row 0 -> prior state is sleeping enough
# Row 1 -> prior state is not sleeping enough
# Row 2 -> next state is sleeping enough
# Row 3 -> next state is not sleeping enough

# Evidence probability numpy array for eye color
O_Red_Eyes = np.array([[0.2, 0],[0, 0.7]])
# Row 0 -> state is sleeping enough
# Row 1 -> state is not sleeping enough
# Non-zero entry is probability of having red eyes
O_No_Red_Eyes = np.array([[0.8, 0],[0, 0.3]])

# Evidence probability numpy array for sleeping in class
O_Sleep_In_Class = np.array([[0.1, 0],[0, 0.3]])
# Row 0 -> state is sleeping enough
# Row 1 -> state is not sleeping enough
# Non-zero entry is probability of sleeping in class
O_No_Sleep_In_Class = np.array([[0.9, 0],[0, 0.7]])

# Now we combine those two variables into one variable with four distinct values
# Independent evidence event probabilities can be multiplied
O_R_S = O_Red_Eyes * O_Sleep_In_Class
O_R_NS = O_Red_Eyes * O_No_Sleep_In_Class
O_NR_S = O_No_Red_Eyes * O_Sleep_In_Class
O_NR_NS = O_No_Red_Eyes * O_No_Sleep_In_Class
# For convenient calculations, we will store these in a map
Observation_Matrix_Map = {0: O_R_S, 1: O_R_NS, 2: O_NR_S, 3: O_NR_NS}