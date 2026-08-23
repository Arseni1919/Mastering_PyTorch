import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# A single line from (0,0,0) to (1,2,3)
ax.plot([0, 1], [0, 2], [0, 3], color='red', linewidth=2)

# A parametric curve
t = np.linspace(0, 6*np.pi, 300)
ax.plot(np.cos(t), np.sin(t), t/10, label='helix')

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend()
plt.show()