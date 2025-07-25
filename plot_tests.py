import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

def create_static_plot():
    # Generate random data
    x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
    y = np.random.randn(100)  # 100 random numbers

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Random Data')
    plt.title('Simple Random Number Plot')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def create_interactive_plot():
    # Create the main figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Make room for the slider

    # Initial data
    x = np.linspace(0, 10, 100)
    y = np.random.randn(100)
    line, = ax.plot(x, y, 'b-', label='Random Data')

    # Customize the plot
    ax.set_title('Interactive Random Number Plot')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.grid(True)
    ax.legend()

    # Create slider axis
    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    num_points_slider = Slider(
        ax=slider_ax,
        label='Number of Points',
        valmin=10,
        valmax=200,
        valinit=100,
        color='blue'
    )

    def update(val):
        # Get the current number of points from the slider
        num_points = int(num_points_slider.val)
        
        # Generate new data
        x = np.linspace(0, 10, num_points)
        y = np.random.randn(num_points)
        
        # Update the line data
        line.set_data(x, y)
        fig.canvas.draw()

    # Connect the slider to the update function
    num_points_slider.on_changed(update)

    # Show the plot
    plt.show()

if __name__ == "__main__":

    create_static_plot()
    # create_interactive_plot()
