---
layout: post
title: "Second Post"
author: "Alexander Song"
background: '/img/posts/03.jpg'
---

# Here is some Python code:

{% highlight python %}
import matplotlib.pyplot as plt
print("Hello world")
{% endhighlight %}

```python
x = 10
```

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

# Construct a mesh over domain Omega = [-phi, phi] x [-1, 1].
phi = (1 + np.sqrt(5))/2
a, b, c, d = -1, 1, -phi, phi
mesh_size = 1/500    # Approximate mesh size.
n, m = math.ceil((b-a) / mesh_size), math.ceil((d-c) / mesh_size)
hx, hy = (b-a)/(n+1), (d-c)/(m+1)
k = 1/250    # Time step.

# Omega is partitioned by a degree 3 polynomial p and has characteristic function chi0.
def p(x): return x * (x - 3 * phi / 4) * (x + 3 * phi / 4)
def x(i): return a + i * hx
def y(j): return c + j * hy
grid = np.indices((n+2, m+2))
ind = x(grid[0,:,:]) < (3/4) * p(y(grid[1,:,:]))
chi0 = np.zeros((n+2, m+2))
chi0[ind] = 1
            
# Define a plotting function.
def plot(chi, ax, colorbar=True, T_cool=0, T_hot=1):
    ax.tick_params(axis='both',      # Changes apply to the x-axis.
                   which='both',     # Both major and minor ticks are affected.
                   bottom=False,     # Turn off ticks along edges.
                   top=False,
                   left=False,
                   right=False,
                   labelleft=False,  # Turn off labels.
                   labelbottom=False)
    im = ax.imshow(chi[30:-30,30:-30], cmap=plt.get_cmap('hot'), vmin=T_cool, vmax=T_hot)    
    # Create a color axis on the right side of ax.
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['0', '0.5', '1'])
    return

# Plot the figure.
f1, ax1 = plt.subplots(figsize=(15,30))
plot(chi0, ax1)
plt.show()
```

Please check back soon for new content!
