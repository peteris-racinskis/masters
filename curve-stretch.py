import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,100,1, dtype=np.float64)
x2 = np.arange(20,100,1, dtype=np.float64)
y1 = x ** 2
y2 = (x2 - 20) ** 2
offs = 20
step = offs / len(x2)
for i in range(len(x2)):
    x2[i] -= step
    step += offs / len(x2)

plt.plot(y1, x)
plt.plot(y2, x2)
plt.show()