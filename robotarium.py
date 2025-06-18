# This section of the code simulates the movement of an ant searching for food and returning to its nest.
# It initializes the robotarium environment with a single robot, sets up controllers for movement and collision avoidance,
# and defines points A (the nest) and B (the food source) within a specified area.
# The robot's initial position is set to the nest, and a zigzag pattern of exploration points is created to facilitate efficient search.
# The robot's movement is guided by a detection radius, which triggers a change in its goal from exploration to moving towards the food source when detected.
# Once the food source is reached, the robot's goal changes to returning to the nest, following a direction calculated based on its current position and the nest's position.
# The robot's speed is adjusted during its return journey to simulate a more efficient path back to the nest.
# The simulation includes visualizations of the robot's movement, the nest, the food source, and a detection circle that follows the robot.
# Created by Gabriel Jean Vermeille, in hight shool named Jean Vilar in France. For any problem, please send me an e-mail at g.jeanvermeille@gmail.com .
# Thank you to do this big job for us.


import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
import numpy as np
import matplotlib.patches as patches

# Initial configuration
N = 1
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

speed_factor = 2.0

# Create controllers
si_barrier_cert = create_single_integrator_barrier_certificate()
si_position_controller = create_si_position_controller()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Define points and parameters
point_A = np.array([[0.0], [0.0]])
point_B = np.array([[np.random.uniform(-1.5, 1.5)], [np.random.uniform(-0.9, 0.9)]])
detection_radius = 0.4  # Increased for more efficient search

# Initial position
initial_conditions = np.array([[point_A[0, 0]], [point_A[1, 0]], [0.0]])
r = robotarium.Robotarium(number_of_robots=1, show_figure=True, initial_conditions=initial_conditions)

# Create a smarter grid of points to explore
x_points = np.linspace(-1.5, 1.5, 8)  # More points for better coverage
y_points = np.linspace(-0.9, 0.9, 5)
exploration_points = []

# Create a zigzag grid for more efficient exploration
for i, x in enumerate(x_points):
    y_line = y_points if i % 2 == 0 else y_points[::-1]
    for y in y_line:
        exploration_points.append(np.array([[x], [y]]))

# Display configuration
r.figure.axes[0].plot(point_A[0], point_A[1], 'go', markersize=15, label='nest')
r.figure.axes[0].plot(point_B[0], point_B[1], 'ro', markersize=15, label='food')
r.figure.axes[0].legend(
    loc='upper right',
    bbox_to_anchor=(0.95, 0.95),
    fontsize=10,
    framealpha=0.8
)

# Add detection circle that will follow the robot
detection_circle = patches.Circle((0, 0), detection_radius, fill=False, color='g', alpha=0.3)
r.figure.axes[0].add_patch(detection_circle)

# Add arrow to indicate direction to point A
arrow = patches.FancyArrowPatch((0, 0), (0, 0), color='b', arrowstyle='->', mutation_scale=15)
r.figure.axes[0].add_patch(arrow)

# State variables
point_B_found = False
going_to_B = False
returning_to_A = False
current_target_idx = 0
x_goal = exploration_points[0]


# Function to calculate direction to point A following the arrow
def get_direction_to_A(current_pos, point_A):
    direction = point_A - current_pos
    return direction / np.linalg.norm(direction)


# Main loop
while True:
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Update detection circle position
    detection_circle.center = (x_si[0, 0], x_si[1, 0])

    # Update arrow to point A
    arrow.set_positions((x_si[0, 0], x_si[1, 0]), (point_A[0, 0], point_A[1, 0]))

    # Check if point B is detected
    current_pos = np.array([[x_si[0, 0]], [x_si[1, 0]]])
    distance_to_B = np.linalg.norm(point_B - x_si)
    distance_to_A = np.linalg.norm(point_A - current_pos)

    if distance_to_B < detection_radius and not point_B_found:
        print("Point B detected!")
        point_B_found = True
        going_to_B = True
        x_goal = point_B

    # If we're at point B, start returning to A
    if going_to_B and distance_to_B < 0.05:
        print("Point B reached!")
        going_to_B = False
        returning_to_A = True

    direction_to_A = get_direction_to_A(current_pos, point_A)

    # Define target point
    if returning_to_A:
        # Use a short distance point in the arrow direction
        step_distance = 0.2
        x_goal = current_pos + direction_to_A * step_distance

        # Check if we reached point A
        if distance_to_A < 0.05:
            print("Successfully returned to point A!")
            break
    elif not going_to_B:
        # Normal exploration if B hasn't been found
        if np.linalg.norm(x_goal - x_si) < 0.08:
            current_target_idx = (current_target_idx + 1) % len(exploration_points)
            x_goal = exploration_points[current_target_idx]

    # Calculate and apply controls
    dxi = si_position_controller(x_si, x_goal)
    dxi = si_barrier_cert(dxi, x_si)

    # Adjust speed when returning to point A
    if returning_to_A:
        dxi = dxi * 1.5  # Increased speed for return

    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

# End simulation
r.call_at_scripts_end()