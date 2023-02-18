import matplotlib.pyplot as plt
from shapely.geometry import LineString
from metrics import get_patch_of_line


def plot_patch(patch, color="blue", alpha=0.5):
    plt.plot(*patch.exterior.xy, color=color)
    plt.fill(*patch.exterior.xy, color=color, alpha=alpha)
    for interior in patch.interiors:
        plt.plot(*interior.xy, color=color)
        # Could erase what is inside the interior
        if alpha> 0:
            plt.fill(*interior.xy, color="white", alpha=1)
    print("Area of the patch: ", patch.area)

def plot_line(line, color="red"):
    plt.plot(*line.xy, color=color)


def plot_area_with_line(line, patch, color_line="red", color_area="blue", alpha=0.5):
    plot_line(line, color=color_line)
    plot_patch(patch, color_area, alpha)


def main():
    # Creates a line
    line = LineString([(3, 3), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0), (1, 4)])

    # Creates a patch from the line
    patch = get_patch_of_line(line)

    # Plot the interior and exterior of the patch
    plot_area_with_line(line, patch)
    plt.show()


if __name__ == '__main__':
    main()
    print("Done")