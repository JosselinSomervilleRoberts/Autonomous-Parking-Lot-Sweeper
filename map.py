# Defines the Map class, which is used to represent the map of the world.
# The map is a 2D array of integers, where each integer represents a type of tile:
# 0: Empty
# 1: Obstacle
# 2: Cleaned
# Initially, there are no cleaned tiles.

# The Mpa class provides a generate radom map function, which generates a random map
# The map is generated by choosing random outlines for the border and the obstacles
# with the correct tiles

# The Map also supports a collision detection function, which checks if a given position
# is colliding with an obstacle or the border of the map. This takes a rotated rectangle
# as input, which is used to represent the sweeper. A first pass is done to check if the
# axis aligned bounding box of the rectangle is colliding with the map. If it is, then
# a second pass is done to check if the rotated rectangle is colliding with the map.

from typing import Tuple, List
from shapely.geometry import Polygon, Point, LineString
from random_shape import get_random_shapely_outline
from metrics import get_patch_of_line
import numpy as np
import random
import pygame
import os
from config import RenderOptions

class Map:

    CELL_TYPE = np.uint8

    CELL_EMPTY = 0
    CELL_OBSTACLE = 255
    CELL_CLEANED = 64
    CELL_FRONT_SWEEPER = 128
    CELL_BACK_SWEEPER = 192

    def __init__(self, width: int, height: int, render_options: RenderOptions):
        self.render_options = render_options
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=Map.CELL_TYPE)
        self.cleaning_path = []
        self.cleaned_cells_to_display = []

    def init_random(self):
        # print("Generating random map...")
        # Call generate_random with random parameters
        self.generate_random(
            nb_obstacles=random.randint(5, 10),
            avg_size_obstacles=0.2 * self.width,
            var_size_obstacles=0.1 * self.width
        )

        # Fill the border with obstacles
        for x in range(self.width):
            self.grid[x, 0] = Map.CELL_OBSTACLE
            self.grid[x, self.height - 1] = Map.CELL_OBSTACLE
        for y in range(self.height):
            self.grid[0, y] = Map.CELL_OBSTACLE
            self.grid[self.width - 1, y] = Map.CELL_OBSTACLE

        # Needs to regenerate the image
        self.render_options.first_render = True

    def clear(self):
        # Keeps the obstacles but clears the cleaned tiles
        self.grid[self.grid == Map.CELL_CLEANED] = Map.CELL_EMPTY
        self.cleaning_path = []
        self.cleaned_cells_to_display = []

        # Needs to regenerate the image
        self.render_options.first_render = True
        

    def generate_image(self):
        """Generates an image of the map"""
        # Create image
        print("Generating map image...")
        self.image = pygame.Surface(self.grid.shape)
        self.image.fill(self.render_options.cell_empty_color)

        # Fill in obstacles
        cleaned_color = self.render_options.cell_cleaned_color if self.render_options.show_area else self.render_options.cell_empty_color
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Map.CELL_OBSTACLE:
                    self.image.set_at((x, y), self.render_options.cell_obstacle_color)
                if self.grid[x, y] == Map.CELL_CLEANED:
                    self.image.set_at((x, y), cleaned_color)

        # Scale image
        self.image = pygame.transform.scale(self.image, (self.width * self.render_options.cell_size, self.height * self.render_options.cell_size))


    def generate_random(self, nb_obstacles, avg_size_obstacles, var_size_obstacles):
        # First fill the map with obstacles
        self.grid = np.ones((self.width, self.height), dtype=Map.CELL_TYPE)

        # Generate random border
        border = get_random_shapely_outline(n=8, center=(self.width / 2, self.height / 2), min_width=self.width * 0.75,
                                            max_width=self.width * 1.25, min_height=self.height * 0.75, max_height=self.height * 1.25, edgy=0.5)
        self.fill_polygon(border, Map.CELL_EMPTY)

        # Generate random obstacles
        for i in range(nb_obstacles):
            # Choose random position
            empty_tiles = self.get_empty_tiles()
            pos = empty_tiles[random.randint(0, len(empty_tiles) - 1)]

            # Generate random obstacle
            avg_width = max(0.5*avg_size_obstacles, min(2*avg_size_obstacles, random.normalvariate(avg_size_obstacles, var_size_obstacles)))
            avg_height = max(0.5*avg_size_obstacles, min(2*avg_size_obstacles, random.normalvariate(avg_size_obstacles, var_size_obstacles)))
            min_width = 0.8 * avg_width
            max_width = 1.2 * avg_width
            min_height = 0.8 * avg_height
            max_height = 1.2 * avg_height
            obstacle = get_random_shapely_outline(n=8, center=(pos[0], pos[1]), min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height, edgy=0.5)
            self.fill_polygon(obstacle, Map.CELL_OBSTACLE)

    def apply_cleaning(self, cur_pos, width=1.0, resolution=1.0):
        # Get the line between the two positions
        self.cleaning_path.append(cur_pos)
        if len(self.cleaning_path) >= 2:
            patch = get_patch_of_line(self.cleaning_path, width=width, resolution=16)
            # If polygon is not empty, fill it
            if not patch.is_empty:
                count = self.fill_polygon_for_cleaning(patch, Map.CELL_CLEANED)
                self.cleaning_path = [self.cleaning_path[-1]]
                return count / resolution**2
        return 0

    def fill_polygon_for_cleaning(self, polygon: Polygon, value: int, grid: np.ndarray = None):
        if grid is None: grid = self.grid
        # Get bounding box
        min_x = max(0, int(polygon.bounds[0]))
        min_y = max(0, int(polygon.bounds[1]))
        max_x = min(self.width, int(polygon.bounds[2]))
        max_y = min(self.height, int(polygon.bounds[3]))

        # Fill in polygon
        count = 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if self.grid[x, y] == Map.CELL_EMPTY and polygon.contains(Point(x, y)):
                    self.grid[x, y] = value
                    self.cleaned_cells_to_display.append((x, y))
                    count += 1
        return count

    def fill_polygon(self, polygon: Polygon, value: int, grid: np.ndarray = None):
        if grid is None: grid = self.grid
        # Get bounding box
        min_x = max(0, int(polygon.bounds[0]))
        min_y = max(0, int(polygon.bounds[1]))
        max_x = min(self.width, int(polygon.bounds[2]))
        max_y = min(self.height, int(polygon.bounds[3]))

        # Fill in polygon
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if self.grid[x, y] != value and polygon.contains(Point(x, y)):
                    self.grid[x, y] = value

    def fill_rectangle(self, rectangle: np.ndarray, value: int, grid: np.ndarray = None):
        if grid is None: grid = self.grid
        polygon = Polygon(rectangle)
        self.fill_polygon(polygon, value, grid=grid)

    def get_reshaped_grid_with_sweeper(self, sweeper) -> np.ndarray:
        g = self.grid.reshape(self.width, self.height, 1)
        self.fill_rectangle(sweeper.get_lower_bounding_box(), Map.CELL_BACK_SWEEPER, grid=g)
        self.fill_rectangle(sweeper.get_upper_bounding_box(), Map.CELL_FRONT_SWEEPER, grid=g)
        return g

    def check_collision(self, rectangle: np.ndarray) -> bool:
        # Convert the coordinates to integers
        x_min, y_min = np.floor(rectangle.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(rectangle.max(axis=0)).astype(int)
        
        # Check if the rectangle is completely outside the grid
        if x_max < 0 or y_max < 0 or x_min >= self.grid.shape[0] or y_min >= self.grid.shape[1]:
            return False
        
        # Check if the rectangle is completely inside the grid
        if np.all(self.grid[x_min:x_max+1, y_min:y_max+1] == 0):
            return False 
        
        # Check if any obstacle cell is inside the rectangle (slow)
        # The rectangle is a numpy array of shape (4, 2)
        poly_rect = Polygon(rectangle)
        for x in range(max(0,x_min), min(self.width, x_max+1)):
            for y in range(max(0,y_min), min(self.height, y_max+1)):
                if self.grid[x, y] == Map.CELL_OBSTACLE:
                    # Check if the cell is inside the rectangle
                    if Point(x, y).within(poly_rect):
                        return True

        # No collision found
        return False

    def get_cleaned_tiles(self) -> List[Tuple[int, int]]:
        return np.argwhere(self.grid == Map.CELL_CLEANED)

    def get_obstacle_tiles(self) -> List[Tuple[int, int]]:
        return np.argwhere(self.grid == Map.CELL_OBSTACLE)

    def get_empty_tiles(self) -> List[Tuple[int, int]]:
        return np.argwhere(self.grid == Map.CELL_EMPTY)

    def save(self, filename: str = None):
        self.clear()

        # If the filename is not specified, map_i is used
        # where i is the next number available in maps/
        if filename is None:
            i = 0
            while os.path.isfile(f"maps/map_{i}.npy"):
                i += 1
            filename = f"maps/map_{i}"
         
        # Save numpy array
        np.save(filename, self.grid)

    def load(self, filename: str):
        self.grid = np.load(filename).astype(Map.CELL_TYPE)
        self.clear()

    def load_random(self):
        len_maps = len(os.listdir("maps/"))
        if len_maps == 0:
            print("No maps found in maps/ folder")
            print("Creating a random map")
            self.init_random()
            return
        i = random.randint(0, len_maps - 1)
        # print(f"Loading map {i}...")
        self.load(f"maps/map_{i}.npy")

    def display(self, sweeper, screen, rerender=False):
        if self.render_options.first_render:
            self.generate_image()

        tile_size = self.render_options.cell_size
        # Redraw everything
        if rerender:
            if self.render_options.show_area:
                # Draw on the image
                for x, y in self.cleaned_cells_to_display:
                    pygame.draw.rect(self.image, self.render_options.cell_cleaned_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))

            # Displays self.image
            screen.blit(self.image, (0, 0))

        else:
            # Update only around the sweeper
            # Get bounding box
            SWEEPER_SIZE_FACTOR = 1.8 * max(1., self.render_options.simulation_speed)
            if self.render_options.show_velocity:
                SWEEPER_SIZE_FACTOR *= 1 + abs(sweeper.speed * 0.04)
            bounding_box = sweeper.get_bounding_box(factor=SWEEPER_SIZE_FACTOR)
            x_min, y_min = np.floor(bounding_box.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(bounding_box.max(axis=0)).astype(int)
            cleaned_color = self.render_options.cell_cleaned_color if self.render_options.show_area else self.render_options.cell_empty_color
            for x in range(max(0,x_min), min(self.width, x_max+1)):
                for y in range(max(0,y_min), min(self.height, y_max+1)):
                    if self.grid[x, y] == Map.CELL_EMPTY:
                        pygame.draw.rect(screen, self.render_options.cell_empty_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))
                    elif self.grid[x, y] == Map.CELL_OBSTACLE:
                        pygame.draw.rect(screen, self.render_options.cell_obstacle_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))
                    elif self.grid[x, y] == Map.CELL_CLEANED:
                        pygame.draw.rect(screen, cleaned_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))


    def clean(self, rect: Polygon):
        # Get bounding box
        min_x = int(rect.bounds[0])
        min_y = int(rect.bounds[1])
        max_x = int(rect.bounds[2])
        max_y = int(rect.bounds[3])

        # Clean tiles
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if rect.contains(Point(x, y)):
                    self.grid[x, y] = Map.CELL_CLEANED

    def set_cell_value(self, x, y, value):
        self.grid[x, y] = value
        self.render_options.first_render = True

        # Update the display
        if self.render_options.render:
            tile_size = self.render_options.cell_size
            if value == Map.CELL_EMPTY:
                pygame.draw.rect(self.image, self.render_options.cell_empty_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))
            elif value == Map.CELL_OBSTACLE:
                pygame.draw.rect(self.image, self.render_options.cell_obstacle_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))
            elif value == Map.CELL_CLEANED:
                pygame.draw.rect(self.image, self.render_options.cell_cleaned_color, pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size))

    def get_cleaned_area(self, resolution=1.0) -> float:
        return len(self.get_cleaned_tiles()) / resolution

    def get_empty_area(self, resolution=1.0) -> float:
        return len(self.get_empty_tiles()) / resolution

    def compute_distance_to_closest_obstacle(self, pos, rad_angle, max_distance=1000):
        """Returns the distance to the closest obstacle in the given direction"""
        # Compute the direction vector
        direction = np.array([np.cos(rad_angle), np.sin(rad_angle)], dtype=float)

        # Compute the distance to the closest obstacle using a step by step approach
        d = max_distance
        for distance in range(1, max_distance):
            # Check if the next cell is an obstacle
            next_cell = pos + distance * direction
            
            # Round and check if within bounds
            rounded_cell = [int(next_cell[0]), int(next_cell[1])]
            if rounded_cell[0] < 0 or rounded_cell[0] >= self.width or rounded_cell[1] < 0 or rounded_cell[1] >= self.height:
                d = distance-1
                break
            if self.grid[rounded_cell[0], rounded_cell[1]] == Map.CELL_OBSTACLE:
                d = distance - 1
                break
        
        # If the distance is not max_distance, then we found an obstacle
        # We refine the value with a binary search
        if d != max_distance:
            # Binary search
            min_distance = d-1
            max_distance = d+1
            while max_distance - min_distance > 0.1:
                d = (min_distance + max_distance) / 2.
                next_cell = pos + d * direction
                # Checks if the point is within bounds
                if int(next_cell[0]) < 0 or int(next_cell[0]) >= self.width or int(next_cell[1]) < 0 or int(next_cell[1]) >= self.height:
                    max_distance = d
                # Checks if the point is within an obstacle (without rounding)
                elif self.grid[int(next_cell[0]), int(next_cell[1])] == Map.CELL_OBSTACLE:
                    max_distance = d
                else:
                    min_distance = d

        return d
