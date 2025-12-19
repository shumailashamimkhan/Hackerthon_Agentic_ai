---
title: ROS 2 Exercises
sidebar_position: 6
---

# ROS 2 Exercises

## Exercise Set 1: Basic ROS 2 Concepts

### Exercise 1.1: Publisher/Subscriber Practice

Create a publisher node that publishes temperature readings on a topic called `temperature_data`. The publisher should:

1. Publish Float64 messages containing randomly generated temperatures every 2 seconds
2. Log the published values to the console
3. Create a separate subscriber node that logs received temperature values to the console

**Solution Outline:**
```python
# Publisher
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import random

class TemperaturePublisher(Node):
    def __init__(self):
        super().__init__('temperature_publisher')
        self.publisher_ = self.create_publisher(Float64, 'temperature_data', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float64()
        msg.data = random.uniform(10.0, 40.0)  # Random temperature between 10-40°C
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing temperature: {msg.data:.2f}°C')

# Similar approach for subscriber
```

### Exercise 1.2: Topic Analysis

Using the command line tools, perform the following tasks:

1. Create a running publisher for `temperature_data` from Exercise 1.1
2. Use `ros2 topic list` to verify the topic exists
3. Use `ros2 topic info temperature_data` to view topic details
4. Use `ros2 topic echo temperature_data` to view messages in real-time
5. Use `ros2 topic bw /temperature_data` to check bandwidth (if available)

### Exercise 1.3: Service Implementation

Create a service that calculates the distance between two points in 2D space:

1. Define a service called `CalculateDistance` with two Point messages (x, y) as request and a Float64 as response
2. Implement the service server that calculates Euclidean distance
3. Create a service client that sends different points and prints the calculated distances

**Service Definition (in srv/CalculateDistance.srv):**
```
geometry_msgs/Point point1
geometry_msgs/Point point2
---
std_msgs/Float64 distance
```

## Exercise Set 2: Advanced ROS 2 Concepts

### Exercise 2.1: Action Server for Trajectory Execution

Implement an action server that simulates executing a trajectory for a robotic manipulator:

1. Define an action that takes a trajectory (sequence of joint positions) and returns execution status
2. The action should publish feedback on progress through the trajectory
3. Include cancellation capability
4. Create a client that sends a trajectory and monitors progress

### Exercise 2.2: Launch File Configuration

Create a launch file that starts multiple nodes:

1. A node that publishes joint states
2. A node that subscribes to joint states and processes the data
3. A node that publishes transformations
4. Configure the launch file with parameters that can be overridden at runtime

**Launch file outline:**
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_pkg',
            executable='joint_publisher',
            name='joint_publisher',
            parameters=[
                {'publish_rate': 50},
                {'num_joints': 6}
            ]
        ),
        Node(
            package='my_robot_pkg',
            executable='joint_processor',
            name='joint_processor',
        ),
    ])
```

### Exercise 2.3: TF Transformations

Create a system that demonstrates TF concepts:

1. Set up a robot with multiple links (base, link1, link2, end_effector)
2. Broadcast transforms between these frames
3. Create a node that uses tf2 to query transformations
4. Demonstrate looking up transforms at different timestamps

## Exercise Set 3: Physical AI Application

### Exercise 3.1: Humanoid Perception Pipeline

Create a simple perception pipeline for a humanoid robot:

1. Implement a node that simulates sensor data (IMU, joint encoders, possibly camera)
2. Create a fusion node that combines these data streams
3. Use the fused data to estimate the robot's center of mass (COM)
4. Publish COM estimates for further processing

**Hints:**
- Use sensor_msgs/Imu for IMU data
- Use sensor_msgs/JointState for encoder data
- Publish geometry_msgs/PointStamped for COM location
- Consider using tf2 for coordinate transformations

### Exercise 3.2: Behavior Arbitration

Implement a simple behavior arbitration system:

1. Create two competing behaviors: balance controller and reaching controller
2. Implement a higher-level arbitrator that can switch between behaviors
3. Use latching topics or services for behavior selection
4. Ensure smooth transitions between behaviors

### Exercise 3.3: Learning-Enabled Component

Create a component that learns to improve its performance over time:

1. Implement a simple neural network controller (you can use pytorch)
2. Allow the controller to adjust its parameters based on feedback
3. Store learned parameters for persistence across sessions
4. Evaluate improvement over time

**Example architecture:**
```python
import rclpy
import torch
import numpy as np
from rclpy.node import Node

class LearningController(Node):
    def __init__(self):
        super().__init__('learning_controller')
        # Simple neural network
        self.network = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 6)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters())
        
    def update_weights(self, state, action, reward):
        # Simple REINFORCE update
        loss = -(reward * torch.log(action))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Exercise Set 4: Integration and Testing

### Exercise 4.1: Integration Test

Create a test that verifies the integration between several nodes:

1. Use launch_testing to start multiple nodes
2. Verify that messages are correctly passed between nodes
3. Check that expected transformations exist
4. Validate that services respond appropriately

### Exercise 4.2: Performance Analysis

Analyze the performance of your ROS 2 implementation:

1. Use `ros2 topic hz` to measure message frequency
2. Profile your nodes using `ros2 doctor` or other tools
3. Measure CPU and memory usage
4. Test how performance scales with multiple nodes

### Exercise 4.3: Fault Tolerance

Test the resilience of your system:

1. Implement timeouts in service clients
2. Add error handling for missing transforms
3. Test graceful degradation when nodes fail
4. Implement a watchdog that detects node failures

## Challenge Exercise: Complete Physical AI Task

### Exercise 5.1: Humanoid Balancing Controller

Combine all concepts to create a humanoid balancing controller:

1. **Hardware Interface**: Subscribe to IMU and joint state data
2. **State Estimation**: Estimate robot state (orientation, joint angles, velocities)
3. **Control**: Implement a balance controller (PID, LQR, or learning-based)
4. **Actuation**: Publish joint commands to maintain balance
5. **Monitoring**: Publish diagnostic information
6. **Safety**: Include emergency stop conditions

**Implementation requirements:**
- Use proper ROS 2 node structure with parameters
- Publish TF transforms for all robot links
- Implement service for controller configuration
- Use latching topics for critical states
- Include diagnostic aggregator for monitoring

**Testing:**
1. Test in simulation first
2. Verify stability under small disturbances
3. Measure controller response times
4. Analyze energy efficiency

## Exercises in Simulation Environment

For each exercise, consider implementing and testing in simulation before considering real-world deployment:

1. **Gazebo Simulation**: Create models that match your URDF description
2. **Isaac Sim**: For more advanced simulation and synthetic data generation
3. **PyBullet**: For physics simulation and reinforcement learning

## Solutions and Verification

### Self-Assessment

For each exercise, verify your implementation by:

1. Checking that nodes start correctly: `ros2 run package_name node_name`
2. Verifying topic connectivity: `ros2 topic echo /topic_name`
3. Confirming service availability: `ros2 service list`
4. Testing action functionality: `ros2 action send_goal`
5. Ensuring parameter configuration works as expected

### Verification Checklist

- [ ] Nodes start without errors
- [ ] Topics publish and subscribe correctly
- [ ] Services respond appropriately
- [ ] Actions execute with proper feedback
- [ ] Transforms are broadcast correctly
- [ ] Parameters can be loaded/changed
- [ ] Launch files work as expected
- [ ] Error handling is implemented
- [ ] Performance is acceptable
- [ ] Logging is informative

Complete all exercises to gain proficiency in ROS 2 for Physical AI applications. For each exercise, consider how the concepts would apply to humanoid robotics challenges, and think about scaling and real-time requirements.