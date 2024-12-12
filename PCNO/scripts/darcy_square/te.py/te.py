import numpy as np

a=np.full((10,2),np.NaN)


e=np.array([1,2,1])
e=e[e>=0]
print(a[e,0])