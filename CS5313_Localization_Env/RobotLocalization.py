"""
    Skeleton Code for Visualization by James Hale
    Edits by Robert Geraghty
"""
from robot_localization_helper import generate_probabilities

try:
    from CS5313_Localization_Env import localization_env as le
except:
    print(
        'Problem finding CS5313_Localization_Env.localization_env... Trying to "import localization_env" only...'
    )
    try:
        import localization_env as le

        print("Successfully imported localization_env")
    except Exception as ex:
        print("Could not import localization_env")
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
        print(ex)
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
import pygame
import random
import numpy as np

num_particles = 10000

class Game:
    def init_pygame(self, window_size):
        pygame.init()
        pygame.display.set_caption("ROBOT MAP")
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.particles_initialized = False
        return self.screen, self.clock

    def update(
        self, env_map, robot_loc, robot_heading, prob_map, heading_probs,
    ):
        x_dir = self.window_size[0] / len(env_map)
        y_dir = self.window_size[1] / len(env_map[0])
        self.screen.fill((0, 0, 0))
        a = np.argmax(prob_map)
        most_prob = [a // len(prob_map[0]), a % len(prob_map[0])]
        most_prob_value = prob_map[most_prob[0]][most_prob[1]]
        for i in range(len(env_map)):
            for j in range(len(env_map[i])):
                if env_map[i][j] == 1:
                    color = (0, 0, 0)
                else:
                    # The farther away we are from the most probable, the more red we are
                    color = (
                        min(100*(most_prob_value - prob_map[i][j]) * 255,255),
                        255,
                        0,
                    )
                pygame.draw.rect(
                    self.screen, color, [i * x_dir, j * y_dir, x_dir, y_dir]
                )
            pygame.draw.circle(
                self.screen,
                (255, 100, 100),
                (
                    int(robot_loc[0] * x_dir) + int(x_dir / 2),
                    int(robot_loc[1] * y_dir) + int(y_dir / 2),
                ),
                int(x_dir / 2.5),
            )
        # Draw lines
        for heading in le.Headings:
            color = (255, 255, 255) if heading.name == robot_heading.name else (0, 0, 0)
            width = 3 if heading.name == robot_heading.name else 1
            if heading == le.Headings.S:  # DOWN
                pygame.draw.line(
                    self.screen,
                    color,
                    (
                        int(robot_loc[0] * x_dir) + int(x_dir / 2),
                        int(robot_loc[1] * y_dir) + int(y_dir / 2),
                    ),
                    (
                        int(robot_loc[0] * x_dir) + int(x_dir / 2),
                        int(robot_loc[1] * y_dir)
                        + int(y_dir / 2)
                        + heading_probs[heading] * y_dir / 2.5,
                    ),
                    width
                )
            elif heading == le.Headings.N:  # UP
                pygame.draw.line(
                    self.screen,
                    color,
                    (
                        int(robot_loc[0] * x_dir) + int(x_dir / 2),
                        int(robot_loc[1] * y_dir) + int(y_dir / 2),
                    ),
                    (
                        int(robot_loc[0] * x_dir) + int(x_dir / 2),
                        int(robot_loc[1] * y_dir)
                        + int(y_dir / 2)
                        - heading_probs[heading] * y_dir / 2.5,
                    ),
                    width
                )
            elif heading == le.Headings.E:  # Right
                pygame.draw.line(
                    self.screen,
                    color,
                    (
                        int(robot_loc[0] * x_dir) + int(x_dir / 2),
                        int(robot_loc[1] * y_dir) + int(y_dir / 2),
                    ),
                    (
                        int(robot_loc[0] * x_dir)
                        + int(x_dir / 2)
                        + heading_probs[heading] * x_dir / 2.5,
                        int(robot_loc[1] * y_dir) + int(y_dir / 2),
                    ),
                    width
                )
            else:  # LEFT
                pygame.draw.line(
                    self.screen,
                    color,
                    (
                        int(robot_loc[0] * x_dir) + int(x_dir / 2),
                        int(robot_loc[1] * y_dir) + int(y_dir / 2),
                    ),
                    (
                        int(robot_loc[0] * x_dir)
                        + int(x_dir / 2)
                        - heading_probs[heading] * x_dir / 2.5,
                        int(robot_loc[1] * y_dir) + int(y_dir / 2),
                    ),
                    width
                )

    def quit(self):
        pygame.quit()

    def display(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        self.clock.tick(60)
        pygame.display.flip()
        return True

    def generate_possibilities(self, env: le.Environment):
        # Call to my OWN implementation to generate probabilities
        if not self.particles_initialized:
            open_positions = [(i, j) for i in range(len(env.map)) for j in range(len(env.map)) if env.map[i][j] != 1]
            particle_indices = np.random.choice(np.array([i for i in range(len(open_positions))]), size=num_particles, replace=True)
            self.particles = [[open_positions[i][0], open_positions[i][1], random.choice(list(le.Headings))] for i in particle_indices]
            self.particles_initialized = True
        self.particles, probabilities = generate_probabilities(self.particles, env)
        return probabilities

    def generate_heading_possibilities(self):
        probs = {}
        prob_sum = 0
        for h in le.Headings:
            num = random.random()
            probs[h] = num
            prob_sum += num
        for h in le.Headings:
            probs[h] /= prob_sum
        return probs


def main():
    seed = 10
    speed = 10  # The higher, the lower
    random.seed(seed)
    window_size = [750, 750]
    env = le.Environment(0.1, 0.1, 0.1, (20, 20), seed=seed)
    game = Game()
    _, clock = game.init_pygame(window_size)
    done = False
    i = 0
    while not done:
        if i == 100:
            i = 0
        if i % speed == 0:
            env.move()
            prob_array = np.random.rand(4)
            prob_array /= np.sum(prob_array)
            game.update(
                env.map,
                env.robot_location,
                env.robot_heading,
                prob_map=game.generate_possibilities(env),
                heading_probs={le.Headings.N: prob_array[0], le.Headings.E: prob_array[1], le.Headings.S: prob_array[2], le.Headings.W: prob_array[3]},
            )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        clock.tick(6)
        pygame.display.flip()
        i += 1


if __name__ == "__main__":
    main()
