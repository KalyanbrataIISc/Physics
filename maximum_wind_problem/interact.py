import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m

# Constants (can be adjusted)
wind_speed = 10.0
sensor_distance = 10.0
power = 10  # Example power, can be changed
run_time = 50.0  # Total simulation time for animation
dt = 0.05  # Time step for animation, affects smoothness and speed

time_period_interactive = 10.0 # seconds
frequency_interactive = 1.0 / time_period_interactive # Frequency in Hz for interactive mode

# Fan motion parameters (fixed for this interactive script)
omega_interactive = 2 * np.pi * frequency_interactive # Calculate omega from frequency
waveform_interactive = "sinusoidal"

def fan_base_position(t, omega=1.0, waveform="sinusoidal"):
    """
    Returns the position of the fan base at time t for different waveforms.
    omega: angular frequency (default 1.0)
    waveform: type of waveform (sinusoidal, triangular, sawtooth, square)
    """
    # Ensure t is a scalar for these calculations if issues arise with array ops
    # For these simple functions, numpy handles array t correctly.
    if waveform == "sinusoidal":
        return 5 * np.sin(omega * t) + 5
    elif waveform == "triangular":
        # Scaled and shifted triangular wave: period 2*pi/omega, range [0, 10]
        # Normalized time within one period: (omega * t / (2 * np.pi)) % 1.0
        # Or simpler with modulo on omega*t/pi for half-period symmetry
        return 5 * (2 * np.abs((omega * t / np.pi) % 2 - 1) -1) + 5 if omega != 0 else 5.0
    elif waveform == "sawtooth":
        # Scaled and shifted sawtooth wave: period 2*pi/omega, range [0, 10]
        return 5 * (((omega * t / np.pi)) % 2) if omega !=0 else 0.0
    elif waveform == "square":
        # Scaled and shifted square wave: period 2*pi/omega, range [0, 10]
        return 5 * (np.sign(np.sin(omega * t)) + 1) if omega != 0 else 5.0
    else:
        raise ValueError("Invalid waveform type")

# Simulation parameters
num_sensors = 500  # Reduced for smoother animation
x_sensors = np.linspace(0, 10, num_sensors)
y_sensor = sensor_distance
times = np.arange(0, run_time, dt)

# Initialize accumulators for each sensor
accumulators = np.zeros(num_sensors)
wind_vector = np.array([0, wind_speed])

# Set up the figure and subplots
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

# --- Fan position subplot (top left) ---
ax_fan = fig.add_subplot(gs[0, 0])
ax_fan.set_xlim(0, 10)
ax_fan.set_ylim(-1, 1)
ax_fan.set_yticks([])
ax_fan.set_title(f"Fan Base Position (Live) - {waveform_interactive} @ {frequency_interactive:.2f} Hz", fontsize=10)
ax_fan.set_xlabel("Fan X Position", fontsize=9)
fan_marker, = ax_fan.plot([], [], 'ro', markersize=8) # Fan marker
ax_fan.plot([0, 10], [0, 0], 'k-', lw=1)

# --- Accumulator subplot (bottom left) ---
ax_accum = fig.add_subplot(gs[1, 0])
ax_accum.set_xlim(0, 10)
ax_accum.set_ylim(0, 1)
ax_accum.set_title("Sensor Accumulation (Live)", fontsize=10)
ax_accum.set_xlabel("Sensor x position", fontsize=9)
ax_accum.set_ylabel("Normalized Accumulation (Approx.)", fontsize=9)
ax_accum.grid(True)
line_accum, = ax_accum.plot(x_sensors, accumulators, 'b-', linewidth=1)

# --- Instantaneous dot product subplot (right column) ---
ax_dot = fig.add_subplot(gs[:, 1])  # Spans both rows
ax_dot.set_xlim(0, 10)
ax_dot.set_ylim(0, 1)  # Normalized range
ax_dot.set_title(f"Instantaneous Wind Power", fontsize=10)
ax_dot.set_xlabel("Sensor x position", fontsize=9)
ax_dot.set_ylabel("Normalized Value", fontsize=9)
ax_dot.grid(True)
line_dot, = ax_dot.plot([], [], 'g-', linewidth=1)

# Initialization function for the animation
def init():
    global accumulators
    accumulators = np.zeros(num_sensors)
    line_accum.set_ydata(accumulators)
    fan_marker.set_data([], [])
    line_dot.set_data([], [])
    return line_accum, fan_marker, line_dot

# Animation update function
def update(frame_time):
    global accumulators
    t = frame_time

    fan_x = fan_base_position(t, omega=omega_interactive, waveform=waveform_interactive)
    fan_pos = np.array([fan_x, 0])
    fan_marker.set_data([fan_x], [0])

    sensor_positions = np.stack((x_sensors, np.full_like(x_sensors, y_sensor)), axis=1)
    r = sensor_positions - fan_pos
    
    r_norms = np.linalg.norm(r, axis=1)
    epsilon = 1e-9
    dot_products = np.zeros_like(r_norms)
    valid_indices = r_norms > epsilon
    
    dot_products[valid_indices] = (wind_vector[1] * r[valid_indices, 1]) / r_norms[valid_indices]
    
    # Calculate powered and normalized dot products for display
    powered_dot = np.abs(dot_products) ** power
    if np.max(powered_dot) > 1e-9:
        normalized_dot = powered_dot / np.max(powered_dot)
    else:
        normalized_dot = np.zeros_like(powered_dot)
    
    # Update dot product plot
    line_dot.set_data(x_sensors, normalized_dot)
    
    accumulators += powered_dot * dt

    # Normalize accumulators
    current_max_abs = np.max(np.abs(accumulators))
    if current_max_abs > 1e-9:
        line_accum.set_ydata(accumulators / current_max_abs)
    else:
        line_accum.set_ydata(np.zeros_like(accumulators))

    return line_accum, fan_marker, line_dot

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=times,
                              init_func=init, blit=True, interval=dt*1000, repeat=False)

plt.tight_layout()
plt.show() 