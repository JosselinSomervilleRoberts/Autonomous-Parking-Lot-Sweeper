import pygame
import numpy as np


class SweeperGame:
    def __init__(self, grid, sweeper, width=1600, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.carImg = pygame.transform.scale(pygame.image.load('assets/car.png'), sweeper.size)
        screen_size = np.array(self.screen.get_size())
        self.screen_center = screen_size / 2
        self.grid = pygame.surfarray.make_surface(grid)
        self.scale = screen_size / np.array(self.grid.get_size())
        print(self.grid.get_size())
        self.grid = pygame.transform.scale(self.grid, (width, height))

    def fill_shapely_outline(self, input, color=(0,0,0)):
        x, y = input.xy
        outline = [[x[i], y[i]] for i in range(len(x))]
        pygame.draw.polygon(self.screen, color, self.scale * np.array(outline))

    def render(self, sweeper_pos, sweeper_angle, path, patch):
        # Fills the screen with white
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.grid, (0, 0))

        # Draw the sweeper's area
        self.fill_shapely_outline(patch.exterior, (0,255,0))
        for interior in patch.interiors:
            self.fill_shapely_outline(interior, (0,0,0))

        # Draw the sweeper's path
        pygame.draw.lines(self.screen, (0,0,255), False, self.scale * np.array(path), width=2)

        # Draw the sweeper
        carImg_temp = pygame.transform.rotate(self.carImg, -sweeper_angle)
        carImg_rotated_rect = carImg_temp.get_rect()
        carImg_rotated_rect.center = sweeper_pos * self.scale
        self.screen.blit(carImg_temp, carImg_rotated_rect)
        # Updates the screen
        pygame.display.flip()
