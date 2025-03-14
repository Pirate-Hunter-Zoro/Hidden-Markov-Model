import random
import numpy as np
import localization_env as le

headings_list = [h for h in le.Headings]

def generate_probabilities(particles: np.array, env: le.Environment) -> tuple[np.array, list[list[float]]]:
    """Use particle filtering to generate a probability map of the environment - for every cell we want a probability that the robot is in said cell

    Args:
        particles (np.array): particles representing our current particle filtering sampling
        env (le.LocalizationEnv): robot environment

    Returns:
        tuple[np.array, list[list[float]]]: new array of particles and probability map (of the robot being in said cell) for each cell location
    """
    moved_particles = np.zeros_like(particles)
    observation_table = env.create_observation_tables()
    
    # This tells us what the robot currently sees in terms of walls to the north, south, east, and west
    walls = env.observe()
    heading_walls = {}
    i = 0
    for direction in le.Directions:
        if direction != le.Directions.St:
            heading_walls[direction] = walls[i]
            i += 1

    # For each particle, we want to move it and then update the probability of it being in that location
    weights = np.zeros(len(particles))
    for i in range(len(particles)):
        particle = particles[i]
        particle_x = particle[0]
        particle_y = particle[1]
        particle_heading = headings_list[particle[2]]
        # Move the particle - stealing the preceding code from localization_env.py
        probs = env.location_transitions[particle_x][
            particle_y
        ][particle_heading]
        location_transition = env.random_dictionary_sample(probs)

        particle[0] = particle_x + location_transition.value[0]
        particle[1] = particle_y + location_transition.value[1]

         # Get the new heading
        h_probs = env.headings_transitions[particle_x][
            particle_y
        ][particle_heading]
        
        new_direction = headings_list.index(env.random_dictionary_sample(h_probs))
        particle[2] = new_direction
        moved_particles[i] = particle

        # Now we need to find the probability of the robot seeing the walls that it sees given we are in this particle's location
        # TODO - modify this probability calculation
        observations = [
                    0
                    if env.traversable(
                        particle[0], particle[1], direction
                    )
                    else 1
                    for direction in le.Directions
                    if direction != le.Directions.St
                ]
        # We know which directions we can see walls in, so we can use this to get the probability of the robot seeing these walls at this location
        weights[i] = observation_table[particle[0]][particle[1]][tuple(observations)]
    
    # Normalize the weights
    weights = weights / np.sum(weights)
    
    # Now we need to sample particles based on their weights
    indices = np.random.choice(len(particles), len(particles), p=weights)
    new_particles = np.array([moved_particles[i] for i in indices])

    prob_map = list()
    for i in env.map:
        a = [0 for _ in i]
        prob_map.append(a)

    # Add to the count of new particles
    for i in range(len(new_particles)):
        particle = new_particles[i]
        prob_map[particle[0]][particle[1]] += 1
    
    for i in range(len(prob_map)):
        for j in range(len(prob_map[i])):
            prob_map[i][j] = prob_map[i][j] / len(new_particles)

    return new_particles, prob_map