import pygame
import numpy as np


class SweeperGame:
    def __init__(self, grid, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.carImg = pygame.transform.scale(pygame.image.load('assets/car.png'), (40, 20))
        screen_size = np.array(self.screen.get_size())
        self.screen_center = screen_size / 2
        self.grid = pygame.surfarray.make_surface(grid)
        self.screen.blit(self.grid, (0, 0))

    def fill_shapely_outline(self, input, color=(0,0,0)):
        x, y = input.xy
        outline = [[x[i], y[i]] for i in range(len(x))]
        pygame.draw.polygon(self.screen, color, np.array(outline))

    def render(self, sweeper_pos, sweeper_angle, path, patch):
        # Fills the screen with white
        # self.screen.fill((255, 255, 255))

        # Draw the sweeper's path and the cleaned area
        pygame.draw.lines(self.screen, (0,0,255), False, np.array(path), width=2)
        # TODO: Fix. This ends up drawing over the above line.
        self.fill_shapely_outline(patch.exterior, (0,255,0))

        for interior in patch.interiors:
            self.fill_shapely_outline(interior, (255,255,255))

        # TODO: Commented out because draws one instance of the sweeper for each call to render.
        # Draw the sweeper
        # carImg_temp = pygame.transform.rotate(self.carImg, -sweeper_angle)
        # carImg_rotated_rect = carImg_temp.get_rect()
        # carImg_rotated_rect.center = sweeper_pos
        # self.screen.blit(carImg_temp, carImg_rotated_rect)
        # Updates the screen
        pygame.display.flip()
