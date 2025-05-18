import numpy as np
import matplotlib.pyplot as plt
import math as m

# Constants
wind_speed = 10.0
sensor_distance = 10.0
run_time = 10.0 # seconds
power = 10
time_period = 10.0 # seconds
frequency = 1.0 / time_period  # Frequency in Hz (cycles per second)

def fan_base_position(t, omega=1.0, waveform="sinusoidal"):
    """
    Returns the position of the fan base at time t for different waveforms.
    omega: angular frequency (default 1.0)
    waveform: type of waveform (sinusoidal, triangular, sawtooth, square)
    """
    if waveform == "sinusoidal":
        return 5 * np.sin(omega * t) + 5
    elif waveform == "triangular":
        return 5 * (2 * np.abs((omega * t / np.pi) % 2 - 1) - 1) + 5
    elif waveform == "sawtooth":
        return 5 * ((omega * t / np.pi) % 2)
    elif waveform == "square":
        return 5 * (np.sign(np.sin(omega * t)) + 1)
    else:
        raise ValueError("Invalid waveform type")

# Simulation parameters
num_sensors = 1000
x_sensors = np.linspace(0, 10, num_sensors)
y_sensor = sensor_distance
dt = 0.01
times = np.arange(0, run_time, dt)
omega = 2 * np.pi * frequency # Calculate omega from frequency
waveforms = ["sinusoidal", "triangular", "sawtooth", "square"]

# Initialize accumulators for each sensor
accumulators = np.zeros(num_sensors)
wind_vector = np.array([0, wind_speed])

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(5, 2)
    
    # Sensor accumulation plot (left column, spans all rows)
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Create 4 subplots for fan base positions (right column)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[3, 1])
    
    fan_axes = [ax2, ax3, ax4, ax5]
    
    for i, waveform in enumerate(waveforms):
        # Initialize accumulators for each sensor
        accumulators = np.zeros(num_sensors)

        for t_step in times: # Renamed t to t_step to avoid conflict with fan_base_position's t parameter
            fan_x = fan_base_position(t_step, omega=omega, waveform=waveform)
            fan_pos = np.array([fan_x, 0])

            # Vectorized computation for all sensors at once
            sensor_positions = np.stack((x_sensors, np.full_like(x_sensors, y_sensor)), axis=1)
            r = sensor_positions - fan_pos
            # Compute distance from fan to each sensor
            r_norms = np.linalg.norm(r, axis=1)
            # Compute projection of wind on the direction of r (angle effect)
            dot_products = (wind_vector[1] * r[:, 1]) / r_norms
            # Accumulate (dot_product ** power) * dt for each sensor
            accumulators += (dot_products ** power) * dt

        # Normalize the accumulators (by max value for now)
        current_max_abs = np.max(np.abs(accumulators))
        if current_max_abs > 1e-9: # Avoid division by zero or very small numbers
            accumulators /= current_max_abs
        else:
            accumulators = np.zeros_like(accumulators) # Set to zero if all values are tiny
            
        ax1.plot(x_sensors, accumulators, label=f"{waveform}")
        
        # Plot fan base position in its own subplot
        t_plot = np.linspace(0, run_time, 1000)
        fan_positions = fan_base_position(t_plot, omega=omega, waveform=waveform)
        fan_axes[i].plot(t_plot, fan_positions)
        fan_axes[i].set_title(f"{waveform}")
        fan_axes[i].set_ylim(0, 10)
        fan_axes[i].grid()
        
        # Only show xlabel on bottom plot
        if i == len(waveforms) - 1:
            fan_axes[i].set_xlabel("Time (s)")
        else:
            fan_axes[i].set_xticklabels([])
        
        # Show ylabel only on first plot
        if i == 0:
            fan_axes[i].set_ylabel("Position (x)")

    ax1.set_title("Normalized Sensor Accumulation vs Sensor Position")
    ax1.set_xlabel("Sensor x position")
    ax1.set_ylabel("Normalized Accumulation")
    ax1.grid()
    ax1.legend()

    plt.tight_layout()
    plt.savefig("maximum_wind_problem/sim.png")