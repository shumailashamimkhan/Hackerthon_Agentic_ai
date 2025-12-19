---
title: Isaac Sim Examples
sidebar_position: 5
---

# Isaac Sim Examples

## Basic Robot Simulation Example

This example demonstrates how to create a simple robot simulation in Isaac Sim that can be used for Physical AI development. We'll walk through creating a basic mobile robot and controlling it via ROS2.

### Setting Up a Basic Scene

```python
import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.range_sensor import _range_sensor
import carb
import numpy as np
import sys

# Initialize simulation application
config = {
    "headless": False,  # Set to True for headless server applications
    "window_width": 1280,
    "window_height": 720,
    "clear_color": (0.098, 0.098, 0.098, 1.0)
}

simulation_app = SimulationApp(config_dict=config)

# Import components after application initialization
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot

# Get the world instance
world = World(stage_units_in_meters=1.0)

# Set up the camera view
viewport = omni.kit.viewport.utility.get_viewport_window()
viewport.set_active_camera("/OmniverseKit_Persp")
set_camera_view(eye=np.array([2, 2, 2]), target=np.array([0, 0, 0]))

# Load a simple robot model
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Could not locate assets root path")
else:
    # Add a simple ground plane
    add_reference_to_stage(
        usd_path=f"{assets_root_path}/Isaac/Props/Grid/default_ground_plane.usda",
        prim_path="/World/defaultGroundPlane"
    )
    
    # Add a simple robot (using Carter as example)
    add_reference_to_stage(
        usd_path=f"{assets_root_path}/Isaac/Robots/Carter/carter_v2.usd",
        prim_path="/World/Carter"
    )

# Reset the world to apply changes
world.reset()

# Define control parameters
linear_velocity = 0.5  # m/s
angular_velocity = 0.5  # rad/s

# Simulation loop
while simulation_app.is_running():
    # Perform simulation step
    world.step(render=True)
    
    # Simple control logic (for demonstration)
    if world.current_time_step_index % 100 == 0:
        print(f"Simulation time: {world.current_time_step_index * world.get_physics_dt():.3f}s")

simulation_app.close()
```

### Creating a Humanoid Robot Controller

For humanoid robotics specifically, we might have a more complex controller:

```python
import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import Articulation
import numpy as np
import carb

# Initialize simulation
config = {"headless": False}
simulation_app = SimulationApp(config)

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()

if assets_root_path is not None:
    # Add ground plane
    add_reference_to_stage(
        usd_path=f"{assets_root_path}/Isaac/Props/Grid/default_ground_plane.usda",
        prim_path="/World/defaultGroundPlane"
    )
    
    # Add a humanoid robot model
    # For this example, we'll use a simple jointed figure
    # In practice, you'd use a specific humanoid model like a NAO or similar
    add_reference_to_stage(
        usd_path=f"{assets_root_path}/Isaac/Robots/Franka/franka.usd",
        prim_path="/World/Robot"
    )

world.reset()

# Get the robot as an Articulation object
robot = world.scene.get_object("Robot")

# Define joint names for humanoid-like robot
joint_names = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
]

def move_to_position(joint_position_targets):
    """Function to move robot to specified joint positions"""
    # Apply joint position targets
    robot.set_joint_position_targets(
        joint_positions=joint_position_targets,
        joint_indices=[robot.get_joint_index(joint_name) for joint_name in joint_names]
    )

# Example of moving through a sequence of poses
pose_sequence = [
    [0, -1.0, 0, -2.0, 0, 1.0, 0],  # Neutral pose
    [0.5, -1.0, 0.5, -2.0, 0.5, 1.0, 0.5],  # Arms raised slightly
    [0, -0.5, 0, -1.5, 0, 0.5, 0],  # Arms forward
]

current_pose_idx = 0
pose_hold_steps = 100  # Hold each pose for 100 simulation steps

while simulation_app.is_running():
    world.step(render=True)
    
    if world.current_time_step_index % pose_hold_steps == 0:
        # Move to next pose in sequence
        if current_pose_idx < len(pose_sequence):
            target_pose = pose_sequence[current_pose_idx]
            move_to_position(target_pose)
            current_pose_idx = (current_pose_idx + 1) % len(pose_sequence)

simulation_app.close()
```

### Using Isaac Sim for Synthetic Data Generation

One of the main benefits of Isaac Sim is generating synthetic data for AI training:

```python
import omni
from omni.isaac.kit import SimulationApp
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import cv2
import os

# Initialize Isaac Sim with replicator
config = {"headless": False}
simulation_app = SimulationApp(config)

# Enable Replicator
rep.orchestrator._orchestrator = None  # Reset if needed

# Create a scene with objects for synthetic data generation
with rep.new_layer():
    # Add ground plane
    rep.create.plane(semantics=[("class", "floor")], position=(0, 0, 0), scale=(10, 10, 1))
    
    # Create objects to be detected
    def spawn_objects():
        # Randomly scatter cubes
        cubes = rep.create.cube(
            position=rep.distribution.uniform((-4, -4, 0.5), (4, 4, 0.5)),
            scale=rep.distribution.uniform((0.1, 0.1, 0.1), (0.5, 0.5, 0.5)),
            semantics=rep.utils.semantics.annotate_semantic_label("cube")
        )
        
        # Randomly place cylinders
        cylinders = rep.create.cylinder(
            position=rep.distribution.uniform((-4, -4, 0.5), (4, 4, 0.5)),
            scale=rep.distribution.uniform((0.1, 0.1, 0.2), (0.3, 0.3, 0.6)),
            semantics=rep.utils.semantics.annotate_semantic_label("cylinder")
        )
        
        return cubes, cylinders

    # Generate objects
    spawn_objects()

    # Create camera with randomized position
    camera = rep.create.camera(position=(-3, -3, 2), look_at=(0, 0, 0))

    # Randomize camera position and look-at point
    with rep.trigger.on_frame(num_frames=1000):
        # Randomize object positions and properties
        with rep.randomizer.on_replicate():
            rep.modify.visibility(rep.get.prims(), visibility=rep.distribution.choice([True, True, True, False], weight=[0.7, 0.1, 0.1, 0.1]))
        
        # Annotate with RGB, depth, and semantic segmentation
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([camera])
        
        depth_annotator = rep.AnnotatorRegistry.get_annotator("depth")
        depth_annotator.attach([camera])
        
        seg_annotator = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        seg_annotator.attach([camera])
        
        # Configure semantic segmentation
        seg_annotator.config.colorize_instance_segmentation = True

# Run the generation
rep.orchestrator.run()
simulation_app.close()
```

### ROS2 Integration Example

Here's how to connect Isaac Sim with ROS2 for Physical AI development:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
import numpy as np

class IsaacSimROSController(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_controller')
        
        # ROS2 publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        # Timer to publish joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        
        # Initialize Isaac Sim
        self.simulation_app = SimulationApp({"headless": False})
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        self.setup_simulation()
    
    def setup_simulation(self):
        """Initialize the Isaac Sim environment"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is not None:
            # Add ground
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Props/Grid/default_ground_plane.usda",
                prim_path="/World/defaultGroundPlane"
            )
            
            # Add robot
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Robots/Carter/carter_v2.usd",
                prim_path="/World/Carter"
            )
        
        self.world.reset()
        self.robot = self.world.scene.get_object("Carter")
    
    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS2"""
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z
        
        # Apply control to the robot in simulation
        # This is a simplified example - in practice, you'd use proper control
        if self.robot is not None:
            # Get current pose
            current_pose = self.robot.get_world_pose()
            current_orientation = current_pose[1]
            
            # Calculate new pose based on velocity commands
            dt = 0.1  # Time step - should match your publishing rate
            
            # Update position based on current orientation and velocities
            # This is a simplified kinematic model
            new_x = current_pose[0][0] + self.linear_vel * np.cos(current_orientation[2]) * dt
            new_y = current_pose[0][1] + self.linear_vel * np.sin(current_orientation[2]) * dt
            new_theta = current_orientation[2] + self.angular_vel * dt
            
            # Set new pose for the robot
            self.robot.set_world_pose(position=np.array([new_x, new_y, current_pose[0][2]]), 
                                    orientation=orientation_from_euler(0, 0, new_theta))
    
    def publish_joint_states(self):
        """Publish joint state information"""
        if self.world.is_playing() and self.robot is not None:
            # Get current joint states
            joint_positions = self.robot.get_joint_positions()
            joint_velocities = self.robot.get_joint_velocities()
            
            # Create and publish joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            # In this example, we assume the robot has predefined joint names
            # In practice, these would be retrieved from the robot description
            msg.name = [f"joint_{i}" for i in range(len(joint_positions))]
            msg.position = joint_positions
            msg.velocity = joint_velocities
            
            self.joint_state_pub.publish(msg)
    
    def step_simulation(self):
        """Step the simulation"""
        if self.simulation_app.is_running():
            self.world.step(render=True)
            return True
        return False

def orientation_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])

def main(args=None):
    rclpy.init(args=args)
    
    # Initialize Isaac Sim controller
    controller = IsaacSimROSController()
    
    # Simulation loop
    while True:
        if not controller.step_simulation():
            break
        
        # Spin ROS to process callbacks
        rclpy.spin_once(controller, timeout_sec=0.01)
    
    controller.simulation_app.close()
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Sim for Humanoid Robotics Research

### Implementing Balance Control

For humanoid robotics, Isaac Sim can simulate various balance control algorithms:

```python
import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import Articulation
import numpy as np
import carb

class BalanceController:
    def __init__(self):
        self.simulation_app = SimulationApp({"headless": False})
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.com_offset = np.array([0.0, 0.0, 0.0])  # Center of mass offset
        self.com_height = 0.8  # m - approximate CoM height
        
        self.setup_environment()
    
    def setup_environment(self):
        """Set up the simulation environment with a humanoid robot"""
        assets_root_path = get_assets_root_path()
        if assets_root_path:
            # Add ground plane
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Props/Grid/default_ground_plane.usda",
                prim_path="/World/defaultGroundPlane"
            )
            
            # Add a humanoid robot (for this example, using a simplified articulated figure)
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Robots/Franka/franka.usd",
                prim_path="/World/Humanoid"
            )
        
        self.world.reset()
        self.robot = self.world.scene.get_object("Humanoid")
        
        # Initialize balance controller parameters
        self.p_gain = 10.0  # Proportional gain for balance control
        self.d_gain = 1.0   # Derivative gain for balance control
        
    def compute_balance_control(self, com_pos, com_vel, target_pos=np.array([0.0, 0.0])):
        """
        Compute balance control torques using inverted pendulum model
        """
        # Simplified inverted pendulum model: x_ddot = g/h * x + u/h
        # Where g is gravity, h is COM height, x is COM position error, u is control input
        g = 9.81  # Gravity constant
        
        # Calculate COM position error from target
        pos_error = com_pos[:2] - target_pos  # Only x, y position error
        
        # Calculate COM velocity error
        vel_error = com_vel[:2]  # Only x, y velocity error
        
        # Compute control using PD controller (simplified inverted pendulum control)
        control_output = -self.p_gain * pos_error - self.d_gain * vel_error
        
        # Convert to required torques (simplified model)
        # In practice, this would involve full kinematic and dynamic models
        required_torques = control_output * (g / self.com_height)
        
        return required_torques
    
    def get_com_state(self):
        """Get current center of mass position and velocity"""
        if self.robot is not None:
            # In practice, COM calculation would be more complex for articulated robots
            # For this example, we'll approximate using the base link position
            base_pos = self.robot.get_world_pose()[0]
            base_vel = self.robot.get_linear_velocity()
            return base_pos, base_vel
        
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    def run_simulation(self):
        """Run the balance control simulation"""
        while self.simulation_app.is_running():
            self.world.step(render=True)
            
            if self.robot is not None:
                # Get current state
                com_pos, com_vel = self.get_com_state()
                
                # Compute balance control
                balance_torques = self.compute_balance_control(com_pos, com_vel)
                
                # Apply control to the robot (simplified)
                # In practice, this would map torques to joint controls
                # For now, we'll just print the computed torques
                
                if self.world.current_time_step_index % 10 == 0:  # Print every 10 steps
                    print(f"Balance torques: {balance_torques}, COM pos: {com_pos[:2]}")
        
        self.simulation_app.close()

def main():
    controller = BalanceController()
    controller.run_simulation()

if __name__ == "__main__":
    main()
```

## Advanced Isaac Sim Features for Physical AI

### Domain Randomization

For improving sim-to-real transfer, domain randomization can be implemented:

```python
import omni.replicator.core as rep
import numpy as np

def setup_domain_randomization():
    """Configure domain randomization for robust physical AI training"""
    
    with rep.new_layer():
        # Randomize lighting conditions
        lights = rep.create.light(
            position=rep.distribution.uniform((-10, -10, 10), (10, 10, 15)),
            intensity=rep.distribution.log_uniform(100, 5000),
            color=rep.distribution.uniform((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        )
        
        # Randomize material properties
        materials = rep.get.material()
        with materials:
            # Randomize roughness and metallic properties
            materials.roughness = rep.distribution.uniform(0.1, 0.9)
            materials.metallic = rep.distribution.uniform(0.0, 0.8)
            
            # Randomize colors (for objects in the scene)
            materials.diffuse_color = rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0))
        
        # Randomize environmental properties
        def randomize_env():
            # Randomize floor appearance
            floor = rep.create.from_usd(
                usd_path="path/to/ground_plates.usd",  # Placeholder path
                position=(0, 0, 0),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 3.14)),
                scale=rep.distribution.uniform((0.5, 0.5, 0.5), (2.0, 2.0, 2.0))
            )
            
            # Randomize object placement
            objects = rep.create.cube(
                position=rep.distribution.uniform((-5, -5, 0.5), (5, 5, 2.0)),
                scale=rep.distribution.uniform((0.1, 0.1, 0.1), (0.5, 0.5, 0.5)),
                semantics=rep.utils.semantics.annotate_semantic_label("obstacle")
            )
            
            return floor, objects
        
        rep.randomizer.register(randomize_env)
        
        # Randomize camera intrinsics
        camera = rep.create.camera()
        camera.set_focal_length(rep.distribution.uniform(18, 55))  # For a DSLR-like range
        camera.set_resolution(rep.distribution.choice([(640, 480), (1280, 720), (1920, 1080)]))
        
        # Trigger randomization on every frame
        with rep.trigger.on_frame(num_frames=10000):
            rep.randomizer.randomize_env()

# Execute domain randomization setup
setup_domain_randomization()
```

These examples provide a foundation for using Isaac Sim in Physical AI and humanoid robotics development. In the next sections, we'll explore more advanced topics including sensor simulation, physics tuning, and integration with AI training pipelines.