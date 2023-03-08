import numpy as np

def subtract_array_at(big_array, small_array, x, y):
    r = (small_array.shape[0] - 1) // 2
    xmin, xmax = max(x - r, 0), min(x + r + 1, big_array.shape[0])
    ymin, ymax = max(y - r, 0), min(y + r + 1, big_array.shape[1])
    x_offset, y_offset = -(x - r - xmin), -(y - r - ymin)
    print(xmin, xmax, ymin, ymax, x_offset, y_offset)
    big_array[xmin:xmax, ymin:ymax] -= small_array[x_offset:x_offset + (xmax - xmin), y_offset:y_offset + (ymax - ymin)]
    return big_array

a = np.ones((20, 20))
r = 0
b = np.zeros((2*r+1, 2*r+1))
for x in range(2*r+1):
    for y in range(2*r+1):
        if (x - r) ** 2 + (y - r) ** 2 <= r ** 2:
            b[x, y] = 1

def print_arr(arr, pos=None):
    # Print a 2d array with spaces if the value is 0, and █ if the value is 1
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if pos is not None and x == pos[0] and y == pos[1]:
                print("X", end="")
            elif arr[x, y] == 0:
                print(" ", end="")
            else:
                print("█", end="")
        print()

print("a before:")
print_arr(a)
print("\nb:")
print_arr(b)
print("\na after:")
subtract_array_at(a, b, 8, 10)
print_arr(a)#, pos=(8, 10))