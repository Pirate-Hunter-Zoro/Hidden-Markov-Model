import numpy as np
import localization_env as le

def generate_probabilities(particles: list[list], env: le.Environment) -> tuple[list[list], list[list[float]]]:
    """Use particle filtering to generate a probability map of the environment - for every cell we want a probability that the robot is in said cell

    Args:
        particles list[list]: particles representing our current particle filtering sampling
        env (le.LocalizationEnv): robot environment

    Returns:
        tuple[list[list], list[list[float]]]: new array of particles and probability map (of the robot being in said cell) for each cell location
    """
    headings_table = env.headings_transitions

    # For each particle, we want to move it and then update the probability of it being in that location
    weights = np.zeros(len(particles))
    for i in range(len(particles)):
        particle = particles[i]
        particle_x = particle[0]
        particle_y = particle[1]
        particle_heading = particle[2]

        probs = env.location_transitions[particle_x][
            particle_y
        ][particle_heading]

        direction = env.random_dictionary_sample(probs)

        new_particle_location = (
            particle_x + direction.value[0],
            particle_y + direction.value[1],
        )

        # Get the new heading
        h_probs = env.headings_transitions[new_particle_location[0]][
            new_particle_location[1]
        ][particle_heading]
        new_particle_heading = env.random_dictionary_sample(h_probs)

        particle = [new_particle_location[0], new_particle_location[1], new_particle_heading]
        particles[i] = particle

        # We have the direction we tried to go in, and the direction we went in.
        # What are the odds the robot would have followed suit from its location?
        headings_dict = headings_table[env.robot_location[0]][env.robot_location[1]]
        if particle_heading not in headings_dict.keys() or new_particle_heading not in headings_dict[particle_heading].keys():
            weights[i] = 0
        else:
            weights[i] = headings_dict[particle_heading][new_particle_heading]
    
    # Normalize the weights
    weights = weights / np.sum(weights)
    
    # Now we need to sample particles based on their weights
    indices = np.random.choice(len(particles), len(particles), p=weights)
    particles = [particles[i] for i in indices]

    prob_map = list()
    for i in env.map:
        a = [0 for _ in i]
        prob_map.append(a)

    # Add to the count of new particles
    for i in range(len(particles)):
        particle = particles[i]
        prob_map[particle[0]][particle[1]] += 1
    
    prob_sum = 0
    for i in range(len(prob_map)):
        for j in range(len(prob_map[i])):
            prob_map[i][j] = prob_map[i][j] / len(particles)
            prob_sum += prob_map[i][j]

    return particles, prob_map