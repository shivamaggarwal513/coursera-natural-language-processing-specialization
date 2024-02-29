import numpy as np
import matplotlib.pyplot as plt


# Procedure to plot and arrows that represents vectors with pyplot
def plot_vectors(vectors, colors=['k', 'b', 'r', 'm', 'c'], axes=None, fname='image.svg', ax=None):
    scale = 1
    scale_units = 'x'
    x_dir = []
    y_dir = []
    
    for i, vec in enumerate(vectors):
        if vec.shape[0] == 1:     # row vector
            x_dir.append(vec[0][0])
            y_dir.append(vec[0][1])
        elif vec.shape[1] == 1:   # column vector
            x_dir.append(vec[0][0])
            y_dir.append(vec[1][0])
    
    if ax == None:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax
      
    if axes == None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]
        
    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])
        
    for i, vec in enumerate(vectors):
        if vec.shape[0] == 1:     # row vector
            ax2.arrow(0, 0, vec[0][0], vec[0][1], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
        elif vec.shape[1] == 1:   # column vector
            ax2.arrow(0, 0, vec[0][0], vec[1][0], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
    
    if ax == None:
        plt.show()
        fig.savefig(f'images/{fname}')
