import numpy as np
sizes=[1,2,3]
we=[np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print(we)
