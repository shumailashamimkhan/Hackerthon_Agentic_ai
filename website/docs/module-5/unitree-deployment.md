---
sidebar_position: 8
title: "Deployment on Unitree Robots"
---

# Deployment on Unitree Robots

This section covers deploying the Physical AI system on Unitree humanoid robots, specifically focusing on the H1 model. Unitree robots provide advanced bipedal platforms for testing and deploying humanoid robotics applications.

## Unitree H1 Platform Overview

The Unitree H1 is a lightweight, high-performance humanoid robot designed for research and development:

- **Height**: 172 cm (adjustable)
- **Weight**: 43 kg
- **Degrees of Freedom**: 23 (19 for body, 4 for hands)
- **Battery Life**: Up to 2 hours depending on activity
- **Payload**: 5 kg distributed across the robot
- **Maximum Speed**: 1.5 m/s walking, 2.8 m/s running

### Hardware Specifications

Key hardware components of the Unitree H1:

- **Actuators**: High-torque, precise servo actuators at each joint
- **Sensors**: IMU, encoders, force/torque sensors, cameras, LiDAR
- **Computing**: Edge computing module with NVIDIA Jetson AGX Orin
- **Communication**: Ethernet, Wi-Fi, Bluetooth, CAN bus

## Unitree SDK Integration

### Environment Setup

To develop for Unitree robots:

1. Install Unitree ROS 2 packages
2. Set up communication with the robot
3. Configure control parameters
4. Test basic communication

### Basic Communication

```cpp
// Example C++ code for Unitree robot communication
#include <unitree/idl/h1/H1_JointState.h>
#include <unitree/robot/channel/channel_publisher.h>
#include <unitree/robot/channel/channel_subscriber.h>
#include <unitree/robot/client.h>

class UnitreeController {
public:
    UnitreeController() {
        joint_state_pub = std::make_shared<unitree::robot::ChannelPublisher<unitree::idl::msg::H1_JointState>>(H1_JOINT_STATE_CHANNEL_NAME);
        joint_state_pub->Init();
    }

    void sendJointCommand(const std::vector<float>& positions) {
        unitree::idl::msg::H1_JointState joint_cmd;
        // Set desired joint positions
        for (int i = 0; i < positions.size(); i++) {
            joint_cmd.position[i] = positions[i];
        }
        
        joint_state_pub->Write(joint_cmd);
    }

private:
    std::shared_ptr<unitree::robot::ChannelPublisher<unitree::idl::msg::H1_JointState>> joint_state_pub;
};
```

### ROS 2 Bridge

For integration with our ROS 2 system, we need to bridge the Unitree SDK with ROS 2:

```python
# Example ROS 2 node for Unitree communication
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from unitree_h1_msgs.msg import ServoCmd, ServoData  # Hypothetical messages

class UnitreeBridgeNode(Node):
    def __init__(self):
        super().__init__('unitree_bridge')
        
        # Subscribe to ROS 2 JointState commands
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            'joint_commands',
            self.joint_cmd_callback,
            10
        )
        
        # Subscribe to velocity commands
        self.vel_cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.vel_cmd_callback,
            10
        )
        
        # Publish joint states from robot
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        
        # Initialize Unitree SDK connection
        self.unitree_client = self.initialize_unitree_connection()
        
    def initialize_unitree_connection(self):
        # Initialize Unitree SDK connection
        # This would involve setting up channels and establishing communication
        pass
        
    def joint_cmd_callback(self, msg):
        # Convert ROS JointState to Unitree command
        unitree_cmd = self.convert_ros_to_unitree(msg)
        # Send to Unitree robot
        self.send_to_unitree(unitree_cmd)
        
    def vel_cmd_callback(self, msg):
        # Handle velocity commands for base movement
        self.send_velocity_command(msg.linear.x, msg.angular.z)
        
    def convert_ros_to_unitree(self, ros_joint_state):
        # Convert ROS JointState to Unitree format
        # Implementation depends on exact Unitree SDK
        pass
        
    def send_to_unitree(self, cmd):
        # Send command through Unitree SDK
        pass
        
    def send_velocity_command(self, linear_vel, angular_vel):
        # Send velocity command for base movement
        pass

def main(args=None):
    rclpy.init(args=args)
    node = UnitreeBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()