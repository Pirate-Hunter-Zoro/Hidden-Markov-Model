import numpy as np
from constants import *

def next_forward(observation: int, prev_forward: np.array) -> np.array:
    """Helper method to calculate the next state probability distribution given the previous probability distribution

    Args:
        observation (int): key into the map for our observation matrices
        prev_forward (np.array): previous probability distribution of states

    Returns:
        np.array: next state probability distribution
    """
    f_next = np.matmul(Observation_Map[observation],np.matmul(Transformation_Matrix.T,prev_forward))
    return f_next/np.sum(f_next)

def filter(observations: list[int]) -> np.array:
    """Use the forward algorithm to find the state probability distribution of the day corresponding with the last observed evidence

    Args:
        observations (list[int]): list of discrete observation values - each identified with an integer

    Returns:
        np.array: resulting state probability distribution
    """
    current = Initial_P
    for i in range(1, len(observations)):
        prev_observation = observations[i-1]

    return current