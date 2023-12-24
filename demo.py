import numpy as np
import time

z = np.loadtxt('z.txt')
mask = np.loadtxt('mask.txt')
mask = np.array(mask, dtype=bool)

print("z:", len(z), z)
print("mask:", len(mask), mask)

start_time = time.time()
z[~mask] = -1
end_time = time.time()
print("!!! -1:",end_time-start_time, "s")
print("z:", len(z), z)
print("mask:", len(mask), mask)