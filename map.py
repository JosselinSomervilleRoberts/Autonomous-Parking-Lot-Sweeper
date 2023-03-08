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

from typing import Tuple, List, Callable
from shapely.geometry import Polygon, Point, LineString
from random_shape import get_random_shapely_outline
from metrics import get_patch_of_line
import numpy as np
import random
import pygame
import os
from scipy import signal
from config import RenderOptions
from user_utils import warn


def subtract_array_at(big_array, small_array, x, y, r = None):
    if r is None: r = (small_array.shape[0] - 1) // 2
    xmin, xmax = max(x - r, 0), min(x + r + 1, big_array.shape[0])
    ymin, ymax = max(y - r, 0), min(y + r + 1, big_array.shape[1])
    x_offset, y_offset = -(x - r - xmin), -(y - r - ymin)
    big_array[xmin:xmax, ymin:ymax] -= small_array[x_offset:x_offset + (xmax - xmin), y_offset:y_offset + (ymax - ymin)]
    return big_array

def add_array_at(big_array, small_array, x, y, r = None):
    if r is None: r = (small_array.shape[0] - 1) // 2
    xmin, xmax = max(x - r, 0), min(x + r + 1, big_array.shape[0])
    ymin, ymax = max(y - r, 0), min(y + r + 1, big_array.shape[1])
    x_offset, y_offset = -(x - r - xmin), -(y - r - ymin)
    big_array[xmin:xmax, ymin:ymax] += small_array[x_offset:x_offset + (xmax - xmin), y_offset:y_offset + (ymax - ymin)]
    return big_array



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

    def compute_matrix_cum(self, max_radius: int):
        """Computes a Matrix cum of shape (3, max_radius+1, width, height) such that
        self.cum[self.i_cum[val], r, x, y] is the number of cells in a radius r
        around (x, y) that are equal to val."""
        
        # Create self.cum a matrix of shape (3, max_radius+1, width, height) filled with zeros
        # Warning we use np.uint8 to store the number of cells in a radius
        # This means that the maximum number of cells in a radius is 255
        # which corresponds to a radius of pi * r^2 = 255 -> r = 9
        self.i_cum = {Map.CELL_EMPTY: 0, Map.CELL_OBSTACLE: 1, Map.CELL_CLEANED: 2}
        self.cum = np.zeros((3, max_radius+1, self.width, self.height), dtype=np.uint8)
        self.nb_element_in_radius = np.zeros(max_radius+1, dtype=np.uint8)

        # Create a list of kernels
        self.kernels = []
        # For each radius r
        for r in range(max_radius+1):
            # Create a kernel of shape (2r+1, 2r+1) filled with zeros
            kernel = np.zeros((2*r+1, 2*r+1), dtype=np.uint8)
            self.kernels.append(kernel)
            # Fill the kernel with ones in a circle of radius r
            for x in range(2*r+1):
                for y in range(2*r+1):
                    if (x-r)**2 + (y-r)**2 <= r**2:
                        kernel[x, y] = 1
                        self.nb_element_in_radius[r] += 1

        # Compule the matrix cum
        for cell_value in self.i_cum.keys():
            idx = self.i_cum[cell_value]

            # Gets an array is_equal_to_value of shape (width, height) such that
            # is_equal_to_value[x, y] is True if the cell (x, y) is equal to cell_value
            is_equal_to_value = self.grid == cell_value

            # For each radius r
            for r in range(max_radius+1):
                # Apply the kernel to the is_empty is_equal_to_value to get the number of celles
                # equal to cell_value in a radius r around each cell
                self.cum[idx, r] = np.uint8(
                    signal.convolve2d(is_equal_to_value, self.kernels[r], mode="same", boundary="fill", fillvalue=0)
                )

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
                    for r in range(len(self.kernels)):
                        subtract_array_at(self.cum[self.i_cum[Map.CELL_EMPTY],r], self.kernels[r], x=x, y=y, r=r)
                        add_array_at(self.cum[self.i_cum[Map.CELL_CLEANED],r], self.kernels[r], x=x, y=y, r=r)
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
        
        # Check if the rectangle is outside the grid
        if x_min < 0 or y_min < 0 or x_max >= self.grid.shape[0] or y_max >= self.grid.shape[1]:
            return True
        
        # Check if the rectangle is completely inside the grid
        if np.all(self.grid[x_min:x_max+1, y_min:y_max+1] != Map.CELL_OBSTACLE):
            return False 
        
        # Check if any obstacle cell is inside the rectangle (slow)
        # The rectangle is a numpy array of shape (4, 2)
        poly_rect = Polygon(rectangle)
        for x in range(max(0,x_min), min(self.width, x_max+1)):
            for y in range(max(0,y_min), min(self.height, y_max+1)):
                if self.grid[x, y] == Map.CELL_OBSTACLE:
                    # Check if the cell is inside the rectangle
                    if Point(x+0.5, y+0.5).within(poly_rect):
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
        return len(self.get_cleaned_tiles()) / resolution**2

    def get_empty_area(self, resolution=1.0) -> float:
        return len(self.get_empty_tiles()) / resolution**2

    def compute_distance_to_closest_obstacle(self, pos, rad_angle, max_distance=1000):
        return self.compute_distance_to_closest_cell_of_value(self, pos, rad_angle, value=Map.CELL_OBSTACLE, max_distance=max_distance)

    def check_zone_value(self, x_center: float, y_center: float, radius: float, value: int, min_ratio: float = 0.5) -> bool:
        """Check if a zone is made of a certain value by a certain ratio.
        WARNING: This function is slow, use it only for debugging.
        Otherwise try to precompute the zone and store it like self.cum"""
        warn("This function is slow, use it only for debugging. Otherwise try to precompute the zone and store it like self.cum")
        # Smart nested for loop that only checks the cells that are in the circle
        # Compute the bounding box
        x_min = max(0, int(x_center - radius))
        y_min = max(0, int(y_center - radius))
        x_max = min(self.width, int(x_center + radius))
        y_max = min(self.height, int(y_center + radius))

        # Count the number of cells that are in the circle
        nb_cells_in_circle = 0
        nb_cells_in_circle_with_value = 0
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Check if the cell is in the circle
                if (x - x_center)**2 + (y - y_center)**2 < radius**2:
                    nb_cells_in_circle += 1
                    if self.grid[x, y] == value:
                        nb_cells_in_circle_with_value += 1

        # Check if the ratio is above the threshold
        return nb_cells_in_circle_with_value / (np.pi * radius**2) > min_ratio


    def compute_distance_to_closest_match(self, pos: Tuple[float, float], rad_angle: float, match_fn: Callable[[int,int], bool], step: float = 1., max_distance: float = 1000, set_minus_one_on_out_of_bounds: bool = True, precision: float = 0.1, min_distance: float = 0.0) -> float:
        """Returns the distance to the closest point that matches the given function"""
        # Compute the direction vector
        direction = np.array([np.cos(rad_angle), np.sin(rad_angle)], dtype=float)

        # Compute the distance to the closest obstacle using a step by step approach
        d = max_distance
        found = False
        for distance in np.arange(max(step, min_distance), max_distance + step, step):
            # Check if the next cell is an obstacle
            next_cell = pos + distance * direction
            
            # Round and check if within bounds
            rounded_cell = [int(next_cell[0]), int(next_cell[1])]
            if rounded_cell[0] < 0 or rounded_cell[0] >= self.width or rounded_cell[1] < 0 or rounded_cell[1] >= self.height:
                if set_minus_one_on_out_of_bounds: return -1
                d = distance-step
                found = True
                break
            elif match_fn(rounded_cell[0], rounded_cell[1]):
                d = distance-step
                found = True
                break
        if not found: return -1

        # If the distance is not max_distance, then we found a cell with the right value
        # We refine the value with a binary search
        if d < max_distance:
            # Binary search
            d_min = max(min_distance, d - step)
            d_max = d + step
            while d_max - d_min > precision:
                d_mid = (d_min + d_max) / 2
                next_cell = pos + d_mid * direction
                rounded_cell = [int(next_cell[0]), int(next_cell[1])]
                if rounded_cell[0] < 0 or rounded_cell[0] >= self.width or rounded_cell[1] < 0 or rounded_cell[1] >= self.height:
                    d_max = d_mid
                elif match_fn(rounded_cell[0], rounded_cell[1]):
                    d_max = d_mid
                else:
                    d_min = d_mid

            d = d_min
        return min(d, max_distance)


    def compute_distance_to_closest_zone_of_value(self, pos, radius:int, rad_angle: float, value:int, max_distance: float = 1000, min_ratio: float = 0.8, set_minus_one_on_out_of_bounds: bool = True, min_distance: float = 0.0) -> float:
        """Returns the distance to the closest obstacle in the given direction using th precomputed self.C"""
        def match_fn(x, y):
            return self.cum[self.i_cum[value], radius, x, y] / self.nb_element_in_radius[radius] > min_ratio
        precision = 0.1 * (radius + 1)
        step = max(0.4, radius)
        return self.compute_distance_to_closest_match(pos, rad_angle, match_fn, step=step, max_distance=max_distance, set_minus_one_on_out_of_bounds=set_minus_one_on_out_of_bounds, precision=precision, min_distance=min_distance)


    def compute_distance_to_closest_zone_of_value_multi_radius_multi_value(self, pos, max_radius: int, rad_angle: float, values: List[int], max_distances: np.array, min_ratio: float = 0.8, precision: float = 0.2, resolution: int = 1) -> np.array:
        # Compute the direction vector
        if min_ratio <= 0.5:
            warn("The min_ratio is too low, it should be at least 0.5 to give good results")

        direction = np.array([np.cos(rad_angle), np.sin(rad_angle)], dtype=float)

        step = 0.4
        max_dist = np.max(max_distances)

        distances = -np.ones((len(values), max_radius + 1), dtype=float) * resolution
        min_distances = [3.0 if value == Map.CELL_CLEANED else 0.0 for value in values]
            
        d = step
        r_found = - np.ones(len(values), dtype=int)
        while d < max_dist:
            cell = pos + d * direction
            rounded_cell = [int(cell[0]), int(cell[1])]

            # Out of bounds
            if rounded_cell[0] < 0 or rounded_cell[0] >= self.width or rounded_cell[1] < 0 or rounded_cell[1] >= self.height:
                # Set obstacles to current distance, let the other at -1
                for i, value in enumerate(values):
                    if value == Map.CELL_OBSTACLE:
                        distances[i, r_found[i] + 1:] = d - step
                        r_found[i] = max_radius
                break
            
            # Check if there is a circle of radius r_found[i] + 1 of each value i
            for i, value in enumerate(values):
                # # If we are already at max distance for the given radius, skip and set -1 (already initialized at -1, so nothing to do)
                while (r_found[i] < max_radius and max_distances[i, r_found[i] + 1] < d):
                    r_found[i] += 1

                # If we are at at least the min distance and the ratio is good, then we found a circle
                found = False
                while (r_found[i] < max_radius and min_distances[i] <= d \
                    and self.cum[i, r_found[i] + 1, rounded_cell[0], rounded_cell[1]] / self.nb_element_in_radius[r_found[i] + 1] > min_ratio):
                        found = True
                        distances[i, r_found[i] + 1] = d - step
                        r_found[i] += 1

                        # W need to check again for the max distance
                        while (r_found[i] < max_radius and max_distances[i, r_found[i] + 1] < d):
                            r_found[i] += 1
                
                # TODO
                if found:
                    break
            
            # Compute step and increment d
            min_r_found = np.min(r_found)
            if min_r_found == max_radius:
                break
            step = max(0.5, min_r_found)
            d += step

        # Refine with binary search
        for i, value in enumerate(values):
            for r in range(r_found[i] + 1):
                initial_step = max(0.5, r)
                if distances[i, r] > 0: # If we found a circle, then refine
                    d_min = max(distances[i, r] - initial_step, min_distances[i])
                    d_max = distances[i, r] + 0.1 * initial_step
                    while d_max - d_min > precision:
                        d_mid = (d_min + d_max) / 2
                        cell = pos + d_mid * direction
                        rounded_cell = [int(cell[0]), int(cell[1])]
                        if rounded_cell[0] < 0 or rounded_cell[0] >= self.width or rounded_cell[1] < 0 or rounded_cell[1] >= self.height:
                            d_max = d_mid
                        elif self.cum[self.i_cum[value], r, rounded_cell[0], rounded_cell[1]] / self.nb_element_in_radius[r] > min_ratio:
                            d_max = d_mid
                        else:
                            d_min = d_mid
                    distances[i, r] = d_min

        return distances / resolution

            


    def compute_distance_to_closest_zone_of_value_slow(self, pos, radius:float, rad_angle: float, value: int, max_distance: float = 1000, min_ratio: float = 0.5) -> float:
        """Returns the distance to the closest obstacle in the given direction
        WARNING: This function is slow, use it only for debugging.
        Otherwise try to precompute the zone and store it like self.cum"""
        warn("This function is slow, use it only for debugging. Otherwise try to precompute the zone and store it like self.cum")
        # Compute the direction vector
        direction = np.array([np.cos(rad_angle), np.sin(rad_angle)], dtype=float)

        # Compute the distance to the closest obstacle using a step by step approach
        d = max_distance
        for distance in range(1, max_distance, int(radius)):
            # Check if the next cell is an obstacle
            next_cell = pos + distance * direction
            
            # Round and check if within bounds
            rounded_cell = [int(next_cell[0]), int(next_cell[1])]
            if rounded_cell[0] < 0 or rounded_cell[0] >= self.width or rounded_cell[1] < 0 or rounded_cell[1] >= self.height:
                d = distance-radius
                break
            if self.check_zone_value(rounded_cell[0], rounded_cell[1], radius, value, min_ratio):
                d = distance-radius
                break

        # If the distance is not max_distance, then we found a cell with the right value
        # We refine the value with a binary search
        if d != max_distance:
            # Binary search
            d_min = d - radius
            d_max = d + radius
            while d_max - d_min > 0.1:
                d_mid = (d_min + d_max) / 2
                next_cell = pos + d_mid * direction
                rounded_cell = [int(next_cell[0]), int(next_cell[1])]
                if self.check_zone_value(rounded_cell[0], rounded_cell[1], radius, value, min_ratio):
                    d_max = d_mid
                else:
                    d_min = d_mid

            d = d_min
        return d
    
    def compute_distance_to_closest_cell_of_value(self, pos, rad_angle, value, max_distance=1000):
        """Returns the distance to the closest obstacle in the given direction"""
        # Compute the direction vector
        warn("This function is deprecated, use compute_distance_to_closest_zone_of_value with a radius of 0 instead")
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
            if self.grid[rounded_cell[0], rounded_cell[1]] == value:
                d = distance - 1
                break
        
        # If the distance is not max_distance, then we found a cell with the right value
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
                elif self.grid[int(next_cell[0]), int(next_cell[1])] == value:
                    max_distance = d
                else:
                    min_distance = d

        return d
