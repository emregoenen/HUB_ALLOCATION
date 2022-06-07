import numpy as np
from itertools import combinations

x = np.array([[24512.513179, 38835.826089], [27520.840478, 46921.515986], [36067.877874, 44894.490748], [19345.710923, 51972.04094 ], [33635.749233,49663.479404]])
print(x)
x = np.append(x,[[5000,5000]],axis=0)
print(x)