import numpy as np
import matplotlib.pyplot as plt
import flopy
from cmcrameri import cm
from matplotlib.animation import FuncAnimation

def movie(gwf, diff = False, bc=False, contour = False):
    
    heads = np.load('Virtual_Reality/model_data/head_ref.npy')
    if diff:
        heads = heads - heads[0,:,:]

    vmin = np.min(heads)
    vmax = np.max(heads)
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    h  = mm.plot_array(heads[0, 0, :], cmap=cm.devon_r, alpha=1, vmin = vmin, vmax = vmax)
    if contour:
        mm.contour_array(heads[0, 0, :], vmin = vmin, vmax = vmax)
    plt.colorbar(h, ax=ax, label='Head [m]')  
    ax.set_aspect('equal')
     
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()  # Clear the previous plot
        ax.set_aspect('equal')
        mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
        mm.plot_array(heads[frame], cmap=cm.devon_r, alpha=1, vmin = vmin, vmax = vmax)  # Update the plot for the current frame
        if contour:
            mm.contour_array(heads[frame, 0, :], vmin = vmin, vmax = vmax)
        # cbar.update_normal(h)  # Update colorbar
        ax.set_title(f'Time: {(frame*0.25):.2f} days')  # Optional: Add a title
        # Add any other customization you need
        
    # Create the animation
    animation = FuncAnimation(fig, update, frames=np.shape(heads)[0], interval=500, repeat=False)

    plt.close(fig)
    # Save the animation as a GIF using ffmpeg
    animation.save("Transient.gif", writer="ffmpeg", fps=36)