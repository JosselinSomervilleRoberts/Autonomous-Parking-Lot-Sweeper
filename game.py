import pygame
import numpy as np
import shapely.geometry as geom
from map import Map

class SweeperGame:

    def __init__(self, width, height, sweeper, cell_size):
        # Init pygame
        pygame.init()
        pygame.display.set_caption('Sweeper Environment')
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        self.clock = pygame.time.Clock()
        screen_size = np.array(self.screen.get_size())
        self.screen_center = screen_size / 2

        # Init map
        self.map = Map(width, height)

        # Init sweeper
        self.carImg = pygame.transform.scale(pygame.image.load('assets/car.png'), (sweeper.size[0] * cell_size, sweeper.size[1] * cell_size))
        self.sweeper = sweeper

    def fill_shapely_outline(self, input, color=(0,0,0)):
        x, y = input.xy
        outline = [[x[i], y[i]] for i in range(len(x))]
        pygame.draw.polygon(self.screen, color, self.cell_size * np.array(outline))

    def render(self, path=None):
        # Render map
        self.map.display(self.screen, self.cell_size)

        # Draw the sweeper's path
        if path is not None:
            pygame.draw.lines(self.screen, (0,0,255), False, self.cell_size * np.array(path), width=2)

        # Draw the sweeper (seeper_pos is the center of the sweeper)
        sweeper_pos = self.sweeper.position
        sweeper_angle = self.sweeper.angle
        carImg_temp = pygame.transform.rotate(self.carImg, -sweeper_angle)
        carImg_rotated_rect = carImg_temp.get_rect()
        carImg_rotated_rect.center = sweeper_pos * self.cell_size
        self.screen.blit(carImg_temp, carImg_rotated_rect)

        # Draw bounding box of sweeper (a np.array of 4 points)
        sweeper_bbox = self.sweeper.get_bounding_box()
        pygame.draw.lines(self.screen, (255,0,0), True, self.cell_size * sweeper_bbox, width=2)

        # Display a circle around the sweeper's center
        pygame.draw.circle(self.screen, (0,255,0), self.cell_size * sweeper_pos, 5)


        # Updates the screen
        pygame.display.flip()
