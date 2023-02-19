import pygame
import numpy as np

class SweeperGame:

    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.carImg = pygame.transform.scale(pygame.image.load('car.png'), (40, 20))

    def render(self, sweeper_pos, sweeper_angle, path):
        self.screen.fill((255, 255, 255))
        screen_size = np.array(self.screen.get_size())
        screen_center = screen_size / 2
        
        # Draw Sweeper as a rotated rectangle. (0,0) is at the center of the screen
        # sweeper = pygame.Surface((20, 10))
        # sweeper.fill((0, 0, 0))
        # sweeper_temp = pygame.transform.rotate(sweeper, sweeper_angle)
        # sweeper_rotated = pygame.transform.rotate(sweeper, sweeper_angle)
        # sweeper_rotated_rect = sweeper_rotated.get_rect()
        # sweeper_rotated_rect.center = sweeper_pos + screen_center
        # self.screen.blit(sweeper_temp, sweeper_rotated_rect)
        
        # Draws the carImg rotated around its center by sweeper_angle
        # The position of the sweeper if offecter by the center of the screen
        carImg_temp = pygame.transform.rotate(self.carImg, -sweeper_angle)
        carImg_rotated_rect = carImg_temp.get_rect()
        carImg_rotated_rect.center = sweeper_pos + screen_center
        self.screen.blit(carImg_temp, carImg_rotated_rect)


        # pygame.draw.line(self.screen, (0, 0, 0), sweeper_pos, sweeper_pos + 20 * np.array([np.cos(sweeper_angle), np.sin(sweeper_angle)]))
        #pygame.draw.lines(self.screen, (0, 0, 0), False, path)
        pygame.display.flip()
