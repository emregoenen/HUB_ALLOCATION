import numpy as np
from itertools import combinations

colorspace = np.array(
    ["orange", "purple", "brown", "gray", "cyan", "magenta", "green", "blue", "yellow", "pink"])

# for i, color in enumerate(colorspace):
#     print(colorspace[i % len(colorspace)])


for i in range(20):
    print(colorspace[i % len(colorspace)])