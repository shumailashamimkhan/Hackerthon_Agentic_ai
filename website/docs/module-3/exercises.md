---
title: Isaac Sim Exercises
sidebar_position: 6
---

# Isaac Sim Exercises

## Exercise Set 1: Basic Isaac Sim Usage

### Exercise 1.1: Environment Setup and Basic Simulation

**Objective**: Set up Isaac Sim and run a basic simulation with a mobile robot.

**Tasks**:
1. Install Isaac Sim following the official installation guide
2. Launch Isaac Sim and load a simple scene
3. Add a robot model (e.g., Carter or similar differential drive robot)
4. Verify the robot appears correctly with proper physics properties
5. Run the simulation and observe robot behavior
6. Take screenshots of your working environment

**Expected Outcome**: 
- Isaac Sim launches without errors
- Robot model loads and simulates properly
- Physics behave realistically

### Exercise 1.2: Sensor Integration

**Objective**: Add sensors to a robot and verify sensor data publication.

**Tasks**:
1. Add a camera sensor to your robot model
2. Configure the camera with appropriate parameters (resolution, FOV, etc.)
3. Verify the camera topic is being published (use `ros2 topic echo`)
4. Add an IMU sensor to the robot
5. Verify IMU data is being published
6. Add a LIDAR sensor to the robot
7. Verify LIDAR data is being published

**Expected Outcome**:
- All sensors publish data on appropriate ROS topics
- Sensor data is accurate and consistent with simulation state

**Code Template**:
```python
# Example sensor verification code
import rclpy
from sensor_msgs.msg import Image, Imu, LaserScan

def verify_camera_data(msg):
    print(f"Received camera image: {msg.width}x{msg.height}")
    
def verify_imu_data(msg):
    print(f"Received IMU data: {msg.linear_acceleration.x:.2f} m/s²")
    
def verify_lidar_data(msg):
    print(f"Received LIDAR data: {len(msg.ranges)} range values")

def main():
    rclpy.init()
    node = rclpy.create_node('sensor_verifier')
    
    # Subscribe to sensor topics
    cam_sub = node.create_subscription(Image, '/camera/image_raw', verify_camera_data, 10)
    imu_sub = node.create_subscription(Imu, '/imu/data', verify_imu_data, 10) 
    lidar_sub = node.create_subscription(LaserScan, '/scan', verify_lidar_data, 10)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### Exercise 1.3: Robot Control Interface

**Objective**: Implement basic robot control through ROS interfaces.

**Tasks**:
1. Publish velocity commands to move the robot
2. Verify the robot responds to velocity commands
3. Implement a simple navigation behavior (square pattern, circle, etc.)
4. Add obstacle avoidance using sensor feedback
5. Document the control behavior and any limitations observed

**Expected Outcome**:
- Robot responds to velocity commands appropriately
- Navigation behavior executes as expected
- Obstacle avoidance works in simulation

**Implementation Hints**:
```python
import rclpy
from geometry_msgs.msg import Twist
import math

class SimpleController:
    def __init__(self):
        self.node = rclpy.create_node('simple_controller')
        self.pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.node.create_timer(0.1, self.navigate)
        self.state = 0  # For square pattern navigation
        self.state_timer = 0
        
    def navigate(self):
        cmd = Twist()
        
        # Simple square pattern
        if self.state == 0:  # Going forward
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
        elif self.state == 1:  # Turning
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
            
        self.state_timer += 1
        if self.state_timer > 50:  # Change state every 5 seconds
            self.state = (self.state + 1) % 2
            self.state_timer = 0
            
        self.pub.publish(cmd)
```

## Exercise Set 2: Isaac Sim for Humanoid Applications

### Exercise 2.1: Humanoid Model Import and Setup

**Objective**: Import and configure a humanoid robot model in Isaac Sim.

**Tasks**:
1. Choose a humanoid robot model (e.g., use a basic human-like model or robot)
2. Import the model into Isaac Sim using appropriate methods
3. Verify joint limits and ranges are correctly configured
4. Test actuator properties and ensure they match physical capabilities
5. Set up proper collision meshes for all links
6. Validate the model's physical properties (mass, inertia)

**Expected Outcome**:
- Robot model loads without errors
- All joints function within correct limits
- Physics properties are realistic

### Exercise 2.2: Balance Control Simulation

**Objective**: Implement a basic balance controller for a humanoid robot.

**Background**: Humanoid robots require constant balance control to prevent falling. The inverted pendulum model is commonly used to represent this challenge.

**Tasks**:
1. Implement a simplified inverted pendulum model for balance control
2. Calculate Center of Mass (CoM) position from joint states
3. Implement a PD controller to maintain balance
4. Test the controller's response to external forces
5. Evaluate stability margins and adjust parameters as needed

**Implementation Requirements**:
```python
import numpy as np
import math

class BalanceController:
    def __init__(self, com_height=0.8):
        self.com_height = com_height  # Height of center of mass
        self.gravity = 9.81
        
        # Controller gains
        self.kp = 50.0  # Proportional gain
        self.kd = 10.0  # Derivative gain
        
        # Previous CoM position for velocity estimation
        self.prev_com_pos = np.array([0.0, 0.0])
        self.prev_time = 0.0
    
    def compute_balance_torques(self, com_pos, current_time, target_pos=np.array([0.0, 0.0])):
        """Compute balance control torques based on CoM position error"""
        dt = current_time - self.prev_time if self.prev_time > 0 else 0.01
        
        # Calculate CoM velocity
        if dt > 0:
            com_vel = (com_pos[:2] - self.prev_com_pos) / dt
        else:
            com_vel = np.array([0.0, 0.0])
        
        # Calculate error
        pos_error = com_pos[:2] - target_pos
        vel_error = com_vel
        
        # Simple inverted pendulum control law
        # u = -Kp * error - Kd * error_dot
        control_effort = -self.kp * pos_error - self.kd * vel_error
        
        # Convert to required joint torques (simplified - in practice, use full kinematics)
        # This would require inverse kinematics and whole-body control in a real implementation
        required_torques = self.map_control_to_joints(control_effort)
        
        # Update for next iteration
        self.prev_com_pos = com_pos[:2]
        self.prev_time = current_time
        
        return required_torques, pos_error, vel_error
    
    def map_control_to_joints(self, control_effort):
        """Map CoM control effort to joint torques (simplified mapping)"""
        # In a real implementation, this would involve:
        # 1. Whole-body inverse kinematics
        # 2. Task-space control
        # 3. Joint limit checking
        # 4. Singularity handling
        
        # Simplified mapping for demonstration
        # Map x-direction control to hip pitch joints
        # Map y-direction control to hip roll joints
        joint_torques = np.zeros(12)  # Assuming 6 joints per leg
        
        joint_torques[2] = control_effort[0] * 0.5  # Left hip pitch
        joint_torques[8] = control_effort[0] * 0.5  # Right hip pitch
        joint_torques[1] = control_effort[1] * 0.5  # Left hip roll
        joint_torques[7] = control_effort[1] * 0.5  # Right hip roll
        
        return joint_torques

# Example usage
def main():
    controller = BalanceController()
    
    # Simulated CoM position (would come from robot state in real implementation)
    com_pos = np.array([0.01, -0.02, 0.8])  # Slightly off balanced
    current_time = 1.0
    
    torques, pos_error, vel_error = controller.compute_balance_torques(com_pos, current_time)
    
    print(f"Computed torques: {torques}")
    print(f"Position error: {pos_error}")
    print(f"Velocity error: {vel_error}")
```

### Exercise 2.3: Gait Generation for Bipedal Locomotion

**Objective**: Implement a basic gait pattern generator for bipedal walking.

**Background**: Creating stable walking patterns for bipedal robots requires careful coordination of footstep placement, timing, and balance control.

**Tasks**:
1. Implement a simple gait pattern generator
2. Define basic walking parameters (step length, step width, step height)
3. Generate appropriate foot trajectories
4. Coordinate with balance controller to maintain stability
5. Test gait patterns at different speeds

**Implementation Requirements**:
```python
import numpy as np
import math

class GaitPatternGenerator:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05, 
                 step_duration=0.5, com_height=0.8):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.step_duration = step_duration
        self.com_height = com_height
        
        # Gait timing parameters
        self.cycle_duration = 2 * step_duration  # For alternating feet
        self.current_phase = 0.0
        
        # Support state tracking
        self.left_support = True  # Start with left foot in support
        self.step_in_progress = False
        self.swing_foot = "right"  # The foot that will swing next
    
    def generate_gait_pattern(self, time_now, desired_speed=0.5):
        """Generate gait pattern based on current time and desired speed"""
        # Calculate gait phase (0.0 to 1.0 cycling)
        cycle_phase = (time_now % self.cycle_duration) / self.cycle_duration
        step_phase = (time_now % self.step_duration) / self.step_duration
        
        # Determine which foot is stance vs swing
        left_swing_phase = self.calculate_swing_phase(cycle_phase, 0.0, 0.5, self.left_support)
        right_swing_phase = self.calculate_swing_phase(cycle_phase, 0.5, 1.0, not self.left_support)
        
        # Generate foot positions for both feet
        left_foot_pos = self.generate_foot_trajectory(left_swing_phase, "left", desired_speed)
        right_foot_pos = self.generate_foot_trajectory(right_swing_phase, "right", desired_speed)
        
        return [
            ("left_foot", left_foot_pos),
            ("right_foot", right_foot_pos)
        ]
    
    def calculate_swing_phase(self, cycle_phase, start, end, is_stance):
        """Calculate the phase of a foot in swing or stance mode"""
        if is_stance:
            return 1.1  # Value > 1.0 indicates stance phase
        else:
            # Normalize to [0, 1] for swing phase
            normalized_phase = (cycle_phase - start) / (end - start)
            return max(0.0, min(1.0, normalized_phase))
    
    def generate_foot_trajectory(self, swing_phase, foot_name, speed_scale=1.0):
        """Generate foot trajectory for swing phase"""
        if swing_phase >= 1.0:
            # Foot is in stance phase
            x_offset = self.step_length / 2 if "left" in foot_name else -self.step_length / 2
            y_offset = self.step_width / 2 if "left" in foot_name else -self.step_width / 2
            return np.array([0, y_offset, 0])  # Foot is flat on ground under body
        
        # Foot is in swing phase
        # Forward progression
        x_progress = swing_phase * self.step_length * speed_scale
        
        # Lateral position (maintain step width)
        y_pos = self.step_width / 2 if foot_name == "left" else -self.step_width / 2
        
        # Vertical trajectory (parabolic lift)
        parabolic_factor = 4 * swing_phase * (1 - swing_phase)  # From 0->1->0
        z_lift = parabolic_factor * self.step_height
        
        return np.array([x_progress, y_pos, z_lift])
    
    def update_gait_parameters(self, new_step_length=None, new_step_width=None, 
                             new_step_height=None, new_duration=None):
        """Update gait parameters during operation"""
        if new_step_length is not None:
            self.step_length = new_step_length
        if new_step_width is not None:
            self.step_width = new_step_width
        if new_step_height is not None:
            self.step_height = new_step_height
        if new_duration is not None:
            self.step_duration = new_duration
            self.cycle_duration = 2 * new_duration

# Test the gait generator
def test_gait_generator():
    gait_gen = GaitPatternGenerator()
    
    print("Gait Pattern Generator Test:")
    print("Time\tLeft Foot\tRight Foot")
    
    for t in [i * 0.1 for i in range(0, 50)]:  # 5 seconds of gait
        patterns = gait_gen.generate_gait_pattern(t)
        left_pos = patterns[0][1]
        right_pos = patterns[1][1]
        print(f"{t:.1f}\t[{left_pos[0]:.2f}, {left_pos[1]:.2f}]\t[{right_pos[0]:.2f}, {right_pos[1]:.2f}]")

if __name__ == "__main__":
    test_gait_generator()
```

## Exercise Set 3: Perception and Navigation in Isaac Sim

### Exercise 3.1: Perception Pipeline Implementation

**Objective**: Create a perception pipeline that processes data from Isaac Sim sensors for humanoid navigation.

**Tasks**:
1. Implement a camera-based object detection pipeline
2. Integrate LIDAR-based obstacle detection
3. Fuse sensor data for environment understanding
4. Create occupancy grid or topological map from sensor data
5. Validate perception accuracy against ground truth

**Implementation Structure**:
```python
import numpy as np
import cv2
from scipy.ndimage import binary_dilation
import math

class PerceptionPipeline:
    def __init__(self, camera_resolution=(640, 480), lidar_range=10.0):
        self.camera_width, self.camera_height = camera_resolution
        self.lidar_range = lidar_range
        
        # Initialize internal state
        self.latest_image = None
        self.latest_scan = None
        self.occupancy_grid = np.zeros((100, 100))  # 100x100 grid, 10cm resolution
        self.grid_origin = (-5.0, -5.0)  # Bottom-left corner in meters
        self.grid_resolution = 0.1  # 10cm per grid cell
        
    def process_camera_image(self, image_msg):
        """Process camera image to detect objects and features"""
        # Convert ROS image to OpenCV format (simplified)
        # In practice, you'd use cv2 bridge to convert image_msg
        img = self.ros_img_to_cv2(image_msg) if image_msg else self.latest_image
        
        if img is None:
            return []
        
        # Example: Use color-based detection to identify key objects
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common objects (red obstacles, green targets)
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        green_lower = np.array([50, 100, 100])
        green_upper = np.array([70, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Find contours for detected objects
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate object centroids and sizes
        detected_objects = []
        for contour in red_contours:
            if cv2.contourArea(contour) > 100:  # Filter small detections
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert pixel coordinates to world coordinates
                    # This requires knowledge of camera pose and intrinsics
                    world_x, world_y = self.pixel_to_world(cx, cy, distance=2.0)  # Estimated distance
                    
                    detected_objects.append({
                        'class': 'obstacle',
                        'center': (world_x, world_y),
                        'size': cv2.contourArea(contour),
                        'confidence': 0.8
                    })
        
        for contour in green_contours:
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    world_x, world_y = self.pixel_to_world(cx, cy, distance=2.0)
                    
                    detected_objects.append({
                        'class': 'target',
                        'center': (world_x, world_y),
                        'size': cv2.contourArea(contour),
                        'confidence': 0.8
                    })
        
        return detected_objects
    
    def process_lidar_scan(self, scan_msg):
        """Process LIDAR scan to detect obstacles"""
        if scan_msg is None:
            return []
        
        # Process LIDAR ranges into obstacle positions
        obstacles = []
        
        angle_increment = scan_msg.angle_increment
        current_angle = scan_msg.angle_min
        
        for i, range_val in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                # Calculate world coordinates
                x = range_val * math.cos(current_angle)
                y = range_val * math.sin(current_angle)
                
                # Add to obstacles if under distance threshold (likely to be obstacle)
                if range_val < 2.0:  # Consider anything closer than 2m as obstacle
                    obstacles.append((x, y))
            
            current_angle += angle_increment
        
        return obstacles
    
    def fuse_perception_data(self, camera_objects, lidar_obstacles):
        """Fuse camera and lidar perception data"""
        # Create a comprehensive perception result
        perception_result = {
            'static_objects': [],
            'dynamic_objects': [],  # This would require temporal analysis
            'obstacles': lidar_obstacles,
            'targets': [obj for obj in camera_objects if obj['class'] == 'target'],
            'hazards': [obj for obj in camera_objects if obj['class'] == 'obstacle']
        }
        
        # Update occupancy grid with LIDAR data
        self.update_occupancy_grid(lidar_obstacles)
        
        return perception_result
    
    def update_occupancy_grid(self, lidar_obstacles):
        """Update occupancy grid with LIDAR measurements"""
        for x, y in lidar_obstacles:
            # Convert world coordinates to grid coordinates
            grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
            grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)
            
            # Check bounds
            if (0 <= grid_x < self.occupancy_grid.shape[1] and 
                0 <= grid_y < self.occupancy_grid.shape[0]):
                # Mark as occupied (value 100)
                self.occupancy_grid[grid_y, grid_x] = 100
    
    def pixel_to_world(self, pixel_x, pixel_y, distance):
        """Convert pixel coordinates to world coordinates using known distance"""
        # This is a simplified projection model
        # In practice, you'd use camera intrinsic matrix and actual depth
        center_x = self.camera_width / 2
        center_y = self.camera_height / 2
        
        # Assume horizontal and vertical fields of view of 60 degrees each
        fov_x = math.radians(60)
        fov_y = math.radians(45)  # Usually narrower for rectilinear cameras
        
        # Calculate angles from center
        angle_x = (pixel_x - center_x) * fov_x / self.camera_width
        angle_y = (pixel_y - center_y) * fov_y / self.camera_height
        
        # Calculate world coordinates using simple trigonometry
        world_x = distance * math.tan(angle_x)
        world_y = distance * math.tan(angle_y)
        
        return world_x, world_y

    def ros_img_to_cv2(self, img_msg):
        """Convert ROS Image message to OpenCV image (simplified)"""
        # This is a simplified implementation
        # In practice, use cv2_bridge
        if hasattr(img_msg, 'data'):
            # Convert from ROS format to numpy array
            # This is a simplified conversion - actual implementation needs cv2_bridge
            pass
        return np.random.rand(480, 640, 3)  # Placeholder
```

### Exercise 3.2: Navigation Stack Configuration

**Objective**: Configure and test the ROS2 navigation stack with Isaac Sim for humanoid robot navigation.

**Tasks**:
1. Configure the Navigation2 stack for humanoid-specific navigation
2. Set up costmaps appropriately for bipedal navigation
3. Implement or configure a path planner that accounts for humanoid constraints
4. Test navigation in various simulated environments
5. Evaluate performance with different obstacle configurations

## Exercise Set 4: Advanced Isaac Sim Applications

### Exercise 4.1: Synthetic Data Generation

**Objective**: Use Isaac Sim's Replicator framework to generate synthetic training data for Physical AI applications.

**Tasks**:
1. Set up Isaac Replicator for data generation
2. Configure domain randomization parameters
3. Generate synthetic datasets for perception tasks
4. Validate data quality and diversity
5. Train a simple model on synthetic data and test on simulation

**Implementation Example**:
```python
import omni.replicator.core as rep

# Configure Replicator for data generation
with rep.new_layer():
    # Define environment
    # Add ground plane
    ground = rep.create.plane(
        semantic_label="floor",
        position=(0, 0, 0),
        scale=(10, 10, 1)
    )
    
    # Add background objects with randomization
    def randomize_background():
        # Randomly place cubes
        cubes = rep.create.cube(
            position=rep.distribution.uniform((-5, -5, 0.5), (5, 5, 0.5)),
            scale=rep.distribution.uniform((0.1, 0.1, 0.1), (0.8, 0.8, 0.8)),
            semantics=rep.utils.semantics.annotate_semantic_label("obstacle")
        )
        
        # Randomly place cylinders
        cylinders = rep.create.cylinder(
            position=rep.distribution.uniform((-5, -5, 0.5), (5, 5, 0.5)),
            radius=rep.distribution.uniform(0.1, 0.3),
            height=rep.distribution.uniform(0.2, 1.0),
            semantics=rep.utils.semantics.annotate_semantic_label("pillar")
        )
        
        return cubes, cylinders
    
    # Define camera randomization
    def camera_look_at_target():
        cam = rep.create.camera()
        cam.set_position(rep.distribution.uniform((-3, -3, 1), (3, 3, 3)))
        cam.look_at(
            position=rep.distribution.uniform((-2, -2, 0), (2, 2, 0)),
            look_at=rep.distribution.uniform((-1, -1, 0), (1, 1, 0)),
            up_axis="y",
            look_direction="z"
        )
        return cam.node
    
    # Register the camera generator
    rep.register_camera_generator(camera_look_at_target)
    
    # Register the background randomizer
    rep.randomizer.register(randomize_background)
    
    # Annotators to generate ground truth data
    with rep.trigger.on_frame(num_frames=1000):  # Generate 1000 frames
        # Randomize background
        rep.randomizer.randomize_background()
        
        # Generate annotations
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(rep.GetCameraX())
        
        seg_annotator = rep.AnnotatorRegistry.get_annotator("class_segmentation")
        seg_annotator.attach(rep.GetCameraX())
        
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        depth_annotator.attach(rep.GetCameraX())

# Execute the data generation
rep.orchestrator.run()
```

### Exercise 4.2: Learning-Enabled Control

**Objective**: Implement a learning-enabled control system using Isaac Sim for training.

**Tasks**:
1. Set up Isaac Sim for Reinforcement Learning training
2. Define a suitable reward function for humanoid tasks
3. Implement a simple RL agent in Isaac Sim
4. Train the agent to perform basic humanoid tasks
5. Evaluate the trained model's performance

**Implementation Example**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class HumanoidActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(HumanoidActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))  # Actions in [-1, 1] range
        return action

class HumanoidCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(HumanoidCriticNetwork, self).__init__()
        
        # Q-network that takes state and action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output single Q-value
        self.relu = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = HumanoidActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = HumanoidActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = HumanoidCriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = HumanoidCriticNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize target networks to match main networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Hyperparameters
        self.tau = tau  # Soft update parameter
        
    def select_action(self, state, noise_scale=0.1):
        """Select action with optional noise for exploration"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add noise for exploration
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update(self, replay_buffer, batch_size=100, gamma=0.99):
        """Update the actor and critic networks"""
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).to(self.device).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            # Get next action from target actor
            next_action = self.actor_target(next_state)
            # Compute next Q-value from target critic
            next_Q = self.critic_target(next_state, next_action)
            # Compute target Q-value
            target_Q = reward + (gamma * next_Q * (1 - done))
        
        # Current Q-value
        current_Q = self.critic(state, action)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Example Training Loop (Conceptual)
def train_humanoid_agent():
    env = HumanoidEnvironment()  # From Isaac Sim
    agent = DDPGAgent(state_dim=env.state_space.shape[0], 
                      action_dim=env.action_space.shape[0])
    
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            if len(buffer) > 1000:  # Start training after some experience
                agent.update(buffer)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode}: Total Reward = {total_reward}")
```

## Exercise Set 5: Integration Challenges

### Exercise 5.1: Isaac Sim to Real-Robot Transfer

**Objective**: Investigate and implement techniques to reduce the sim-to-real gap for humanoid robots.

**Tasks**:
1. Implement domain randomization techniques in Isaac Sim
2. Apply system identification to your simulated robot
3. Implement adaptive control methods
4. Test the transfer of learned behaviors to real robots (or more accurate simulations)
5. Document the effectiveness of different techniques

### Exercise 5.2: Multi-Modal Perception Integration

**Objective**: Integrate multiple perception modalities in Isaac Sim to improve humanoid robot capabilities.

**Tasks**:
1. Combine camera, LIDAR, and IMU data in a single perception pipeline
2. Implement sensor fusion techniques
3. Test the robustness of the system under various conditions
4. Evaluate the contribution of each sensor modality

## Self-Assessment Checklist

After completing these exercises, verify:

### Basic Isaac Sim Skills:
- [ ] Successfully launched and configured Isaac Sim
- [ ] Created and simulated a robot model
- [ ] Integrated sensors and verified data output
- [ ] Controlled robot motion through ROS interfaces

### Humanoid-Specific Skills:
- [ ] Implemented balance control algorithms
- [ ] Generated stable gait patterns
- [ ] Processed multi-modal sensor data
- [ ] Configured navigation for humanoid-specific constraints

### Advanced Techniques:
- [ ] Generated synthetic training data with Replicator
- [ ] Implemented learning-enabled control systems
- [ ] Evaluated sim-to-real transfer techniques
- [ ] Integrated perception and navigation systems

### Physical AI Concepts:
- [ ] Applied Physical AI principles in simulation
- [ ] Explored the digital-twin concept for humanoid robotics
- [ ] Validated control algorithms in simulated environments
- [ ] Prepared systems for real-world deployment

## Project Challenge: Complete Humanoid Task

As a comprehensive challenge, implement a complete humanoid task:

1. Set up a humanoid robot model in Isaac Sim
2. Implement perception system to detect targets and obstacles
3. Design balance control to maintain stability during motion
4. Implement gait generation for locomotion
5. Create a navigation system to move to target locations
6. Execute a pick-and-place task with the humanoid
7. Document the system's performance and limitations

This challenge will integrate all the concepts covered in these exercises and provide a comprehensive understanding of using Isaac Sim for Physical AI applications in humanoid robotics.