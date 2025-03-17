import math
import numpy as np
from constants import *
import matplotlib.pyplot as plt

######################################################################################################################################################################################################

# Smoothing algorithm for Probability Distribution of Past Evidence

def next_forward(observation: int, prev_forward: np.array) -> np.array:
    """Helper method to calculate the next state probability distribution given the previous probability distribution

    Args:
        observation (int): key into the map for our observation matrices
        prev_forward (np.array): previous probability distribution of states

    Returns:
        np.array: next state probability distribution
    """
    O = Observation_Matrix_Map[observation]
    Transf_T = Transformation_Matrix.T
    f_next = np.matmul(O, np.matmul(Transf_T, prev_forward))
    return f_next/np.sum(f_next)

def filter(observations: list[int]) -> np.array:
    """Use the forward algorithm to find the state probability distribution of the day corresponding with the last observed evidence

    Args:
        observations (list[int]): list of discrete observation values - each identified with an integer

    Returns:
        np.array: resulting state probability distribution
    """
    current = Initial_P
    for observation in observations:
        current =  next_forward(observation=observation, prev_forward=current)

    return current

def prev_forward(observation: int, last_forward: np.array) -> np.array:
    """Calculate the last forward filter probability given the current observation and the previous forward filter probability

    Args:
        observation (int): Current observation
        last_forward (np.array): Previous forward filter probability distribution from the next state

    Returns:
        np.array: probability distribution for the prior state directly preceding the input state
    """
    Transf = Transformation_Matrix
    O = Observation_Matrix_Map[observation]
    res = np.matmul(
        np.linalg.inv(Transf.T),
        np.matmul(np.linalg.inv(O), last_forward)
    )
    return res / np.sum(res)

def next_backward(observation: int, prior_b: np.array) -> np.array:
    """Helper method to calculate the next (previous chronologically) evidence distribution given the previous

    Args:
        observation (int): key into the map for our observation matrices
        prior_b (np.array): resulting evidence probability distribution

    Returns:
        np.array: (un-normalized) probability distribution for evidence_{k+1:t} given X_k
    """
    O = Observation_Matrix_Map[observation]
    Transf = Transformation_Matrix
    return np.matmul(Transf, np.matmul(O, prior_b))

def country_dance_smoothing(observations: list[int]) -> list[np.array]:
    """Use the country dance smoothing algorithm to compute the state probability distribution for each day given the evidence

    Args:
        observations (list[int]): List of observations by the day

    Returns:
        list[np.array]: Probability distribution of the hidden state for each day (including the day prior to the first seen evidence)
    """
    f = filter(observations)
    b = np.ones(shape=(2,1))
    res = [f] # technically f*b, but since b is the 1-vector, that makes no difference here
    for i in range(len(observations)-1,-1,-1):
        prior_obs = observations[i]
        b = next_backward(observation=prior_obs, prior_b=b)
        f = prev_forward(observation=prior_obs, last_forward=f)
        prob_state_distribution = f*b
        res.insert(0, prob_state_distribution/np.sum(prob_state_distribution))
    return res

def fixed_lag_smoothing(observations: list[int], offset: int) -> list[np.array]:
    """Given the evidence up to time step t, and the offset, return the probability distribution of all states using the country-dance-esque algorithm by only going forward 'offset' steps

    Args:
        observations (list[int]): all observations from 1-t
        offset (int): fixed lag of our smoothing - how far we look ahead to generate a probability distribution for the hidden state at a given time

    Returns:
        list[np.array]: probability distribution for all states
    """
    probability_distributions = []
    for i in range(len(observations)):
        look_ahead_posn = min(i+offset, len(observations)-1)
        f = filter(observations[:look_ahead_posn+1])
        b = np.ones(shape=(2,1))
        for j in range(look_ahead_posn, i, -1):
            prior_obs = observations[j]
            b = next_backward(observation=prior_obs, prior_b=b)
            f = prev_forward(observation=prior_obs, last_forward=f)
        prob_state_distribution = f*b
        probability_distributions.append(prob_state_distribution/np.sum(prob_state_distribution))
    return probability_distributions

######################################################################################################################################################################################################

# Viterbi Algorithm for Most Likely Explanation
# argmin_{X_{1:t}}(-log(P(x_0))+sum_t(-log(P(x_t|x_{t-1})))-log(P(e_t|x_t)))
# For each state at time t, keep track of the maximum probability of any path to it

def viterbi(evidence: list[int]) -> list[bool]:
    """Given a list of evidence for each time step, return the most likely sequence of state values that could have caused said evidence.

    Args:
        evidence (list[int]): list of evidence occurrences

    Returns:
        list[bool]: resulting sequence of state values
    """
    # First we must define a Trellis Graph - at each time step, for each value of the state, keep track of its -log(probability) value and the last state that led to it
    trellis = [{} for _ in range(len(evidence)+1)]
    trellis[0][True] = (-math.log(Initial_P[0][0]), None)
    trellis[0][False] = (-math.log(Initial_P[1][0]), None)

    # Look at all the evidence to fill out our trellis graph
    for i, e in enumerate(evidence):
        # Find the maximum probability of arriving at the True (enough sleep) state for this current time
        # Consider coming from the True (enough sleep) state in the previous evidence
        from_true_to_true = trellis[i][True][0] - math.log(Transformation_Matrix[0][0]) - math.log(Observation_Matrix_Map[e][0][0])
        from_false_to_true = trellis[i][False][0] - math.log(Transformation_Matrix[1][0]) - math.log(Observation_Matrix_Map[e][0][0])
        if from_true_to_true <= from_false_to_true:
            # It was more likely to come from the True state to be True in this state
            trellis[i+1][True] = (from_true_to_true, True)
        else:
            # It was more likely to come from the False state to be True in this state
            trellis[i+1][True] = (from_false_to_true, False)

        # Find the maximum probability of arriving at the False (not enough sleep) state for this current time
        # Once again consider coming from the both previous states
        from_true_to_false = trellis[i][True][0] - math.log(Transformation_Matrix[0][1]) - math.log(Observation_Matrix_Map[e][1][1])
        from_false_to_false = trellis[i][False][0] - math.log(Transformation_Matrix[1][1]) - math.log(Observation_Matrix_Map[e][1][1])
        if from_true_to_false <= from_false_to_false:
            # It was more likely to come from the True state to be False in this state
            trellis[i+1][False] = (from_true_to_false, True)
        else:
            # It was more likely to come from the False state to be False in this state
            trellis[i+1][False] = (from_false_to_false, False)
    
    # Now return the most likely sequence of states (an array of booleans) starting from the end
    most_likely = [trellis[len(trellis)-1][True][0] <= trellis[len(trellis)-1][False][0]]
    posn = (len(trellis)-1, most_likely[0])
    for i in range(len(trellis)-1, 0, -1):
        last_boolean = posn[1]
        prev_boolean = trellis[i][last_boolean][1]
        most_likely.insert(0, prev_boolean)
        posn = (i-1, prev_boolean)
    
    return most_likely

######################################################################################################################################################################################################

# Graphing Helper
def graph_probabilities(observations: list[int], probabilities: list[np.array], title: str) -> None:
    """Graph the probability distributions of the hidden states over time

    Args:
        probabilities (list[np.array]): list of probability distributions for the hidden states
        title (str): title of the graph
        ylabel (str): label for the y-axis
    """
    observations = [-1] + observations
    _, ax = plt.subplots()
    # Map each observation to the indices where it occurs
    obs_map = {}
    for i, obs in enumerate(observations):
        if obs not in obs_map:
            obs_map[obs] = []
        obs_map[obs].append(i)
    for obs, indices in obs_map.items():
        x = indices
        y = [probabilities[i][0][0] for i in indices]
        label = Observation_String_Map[obs]
        ax.scatter(x, y, label=label)

    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Probability of Having Slept Enough")
    plt.show()

# Viterbi Graphing Helper
def graph_viterbi(evidence: list[int], state_sequence: list[bool], title: str) -> None:
    """Graph the Viterbi state sequence

    Args:
        evidence (list[int]): list of evidence occurrences
        state_sequence (list[bool]): resulting sequence of state values
        title (str): title of the graph
    """
    _, ax = plt.subplots()
    # Map each observation to the indices where it occurs
    obs_map = {}
    evidence = [-1] + evidence
    for i, obs in enumerate(evidence):
        if obs not in obs_map:
            obs_map[obs] = []
        obs_map[obs].append(i)
    for obs, indices in obs_map.items():
        x = indices
        y = [state_sequence[i] for i in indices]
        label = Observation_String_Map[obs]
        ax.scatter(x, y, label=label)

    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.show()