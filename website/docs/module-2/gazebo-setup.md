---
title: Gazebo Environment Setup
sidebar_position: 2
---

# Gazebo Environment Setup

## Introduction to Gazebo

Gazebo is a powerful physics simulator that is widely used in the robotics community for developing, testing, and validating robotic algorithms. For Physical AI and humanoid robotics applications, Gazebo provides realistic simulation of physics, sensors, and environments that enables safe and efficient development of complex robotic behaviors.

### Gazebo Versions and Ecosystem

Gazebo has evolved over the years:
- **Classic Gazebo**: The original Gazebo platform (versions 1-11), still actively maintained
- **Gazebo Garden**: The next-generation Gazebo platform (formerly Ignition Gazebo)
- **Gazebo Harmonic**: The latest release with enhanced features and performance

For humanoid robotics and Physical AI applications, the choice may depend on:
- **ROS Integration**: Gazebo Classic has mature ROS 1/2 integration
- **Performance**: Gazebo Garden offers improved performance and modularity
- **Features**: Newer versions may have advanced rendering and physics capabilities

## Installing Gazebo

### System Requirements

For humanoid robotics simulation in Gazebo:
- **OS**: Ubuntu 20.04/22.04, macOS, or Windows WSL2
- **CPU**: Multi-core processor (recommended 4+ cores)
- **RAM**: 8GB minimum, 16GB+ recommended for complex scenes
- **GPU**: Dedicated GPU recommended for realistic rendering
- **Disk**: 2GB+ for basic installation, more for models and worlds

### Installation Methods

#### Installation via Package Manager (Recommended)

For Ubuntu/Debian systems:
```bash
# For Gazebo Garden
sudo apt-get update
sudo apt-get install gazebo
# Or for Classic Gazebo (e.g., Fortress):
sudo apt-get install gazebo-classic
```

#### Build from Source

For the latest features or development:
```bash
# Clone the repository
git clone https://github.com/gazebosim/gazebo
cd gazebo
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### Verification

After installation, verify Gazebo is properly configured:
```bash
gazebo --version
gz --version  # for Gazebo Garden/Harmonic
```

## Basic Gazebo Concepts for Physical AI

### Worlds

A world file defines the simulation environment:
- **Environment**: Terrain, objects, lighting
- **Physics**: Gravity, engine parameters
- **Entities**: Robots, obstacles, decorations
- **Plugins**: Custom simulation components

Example world file structure:
```xml
<sdf version='1.7'>
  <world name='basic_world'>
    <!-- Physics properties -->
    <physics type='ode'>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <light type='directional' name='sun'>
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.8</direction>
    </light>

    <!-- Robot -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Models

Models represent objects in the simulation:
- **Robot Models**: Typically URDF or SDF representations
- **Environment Models**: Static objects, buildings, furniture
- **Sensor Models**: Cameras, LIDAR, IMU devices

### SDF vs URDF

For humanoid robotics in Gazebo:
- **URDF**: Used primarily for robot descriptions, integrates with ROS/ROS2
- **SDF**: Gazebo's native format, supports more features (plugins, sensors, etc.)

Conversion tools exist to go between URDF and SDF when needed.

## Configuring Gazebo for Humanoid Robotics

### Physics Engine Settings

For humanoid robotics simulation, proper physics settings are critical:

```xml
<physics type="ode" name="default_physics">
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>1000</iters>
      <sor>1.3</sor>
      <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>0.1</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

Key parameters for humanoid applications:
- **Max Step Size**: 0.001 or smaller for stable control
- **Real Time Factor**: 1.0 for real-time simulation
- **Update Rate**: 1000Hz to match typical control rates
- **ERP/CFM**: Tune for proper contact behavior

### Model Configuration for Humanoids

Optimized SDF for humanoid robots:
```xml
<model name="humanoid_robot">
  <static>false</static>
  <self_collide>false</self_collide>
  <enable_wind>false</enable_wind>
  <pose>0 0 1 0 0 0</pose>

  <!-- Links with inertial properties -->
  <link name="base_link">
    <pose>0 0 0 0 0 0</pose>
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>0.4</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.4</iyy>
        <iyz>0</iyz>
        <izz>0.2</izz>
      </inertia>
    </inertial>

    <!-- Visual representation -->
    <visual name="visual">
      <geometry>
        <mesh><uri>model://humanoid/meshes/base_link.dae</uri></mesh>
      </geometry>
    </visual>

    <!-- Collision geometry -->
    <collision name="collision">
      <geometry>
        <mesh><uri>model://humanoid/meshes/base_collision.stl</uri></mesh>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
            <slip1>0.0</slip1>
            <slip2>0.0</slip2>
          </ode>
        </friction>
        <bounce>
          <restitution_coefficient>0.0</restitution_coefficient>
          <threshold>100000</threshold>
        </bounce>
      </surface>
    </collision>
  </link>

  <!-- Joints for humanoid articulation -->
  <joint name="hip_yaw_joint" type="revolute">
    <parent>base_link</parent>
    <child>left_hip</child>
    <axis>
      <xyz>0 0 1</xyz>
      <limit>
        <lower>-1.57</lower>
        <upper>1.57</upper>
        <effort>100</effort>
        <velocity>5</velocity>
      </limit>
      <dynamics>
        <damping>1.0</damping>
        <friction>0.1</friction>
      </dynamics>
    </axis>
  </joint>

  <!-- Plugins for control and sensors -->
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>~/out:=joint_states</remapping>
    </ros>
    <update_rate>30</update_rate>
  </plugin>
</model>
```

### Sensor Configuration

For Physical AI applications, proper sensor configuration is essential:

#### Camera Sensor
```xml
<sensor name="camera" type="camera">
  <update_rate>30</update_rate>
  <camera name="head_camera">
    <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/humanoid</namespace>
      <remapping>image_raw:=camera/image_raw</remapping>
      <remapping>camera_info:=camera/camera_info</remapping>
    </ros>
  </plugin>
</sensor>
```

#### IMU Sensor
```xml
<sensor name="imu_sensor" type="imu">
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.02</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.02</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.02</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Gazebo GUI and Visualization

### Running Gazebo with GUI

For development and debugging:
```bash
# Start Gazebo with GUI
gazebo worlds/empty.world

# Or with a specific world file
gazebo worlds/humanoid_challenge.world

# For Gazebo Garden/Harmonic
gz sim -g worlds/empty.sdf
```

### GUI Controls

Essential Gazebo GUI controls for Physical AI development:
- **Spacebar**: Pause/resume simulation
- **Right-click + drag**: Rotate camera view
- **Middle-click + drag**: Pan camera
- **Scroll wheel**: Zoom in/out
- **Ctrl+click + drag**: Translate objects (in edit mode)

### Visualization Tips

For effective simulation visualization:
- **Show Contacts**: Toggle to see contact points during debugging
- **Show Axes**: Visualize coordinate systems
- **Show Grid**: Aid in spatial reasoning
- **Adjust Transparency**: Better view internal structures

## ROS/ROS2 Integration

### Gazebo Plugins

For integration with ROS/ROS2, Gazebo uses plugins:

#### Joint Control Plugin
```xml
<plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
  <robotNamespace>/humanoid</robotNamespace>
  <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
</plugin>
```

#### Controller Configuration
Controllers are typically configured via YAML files:

```yaml
# controllers.yaml
joint_state_controller:
  type: joint_state_controller/JointStateController
  publish_rate: 50

position_controllers:
  type: position_controllers/JointGroupPositionController
  joints:
    - hip_yaw_joint
    - hip_roll_joint
    - hip_pitch_joint
    # ... more joints
  gains:
    hip_yaw_joint: {p: 100.0, i: 0.01, d: 10.0}
    # ... more gains
```

### Launch Files for Simulation

Example launch file to start Gazebo with a robot:

```python
# Launch Gazebo simulation with robot
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_path = get_package_share_directory('humanoid_robot_description')

    return LaunchDescription([
        # Start Gazebo
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'humanoid_robot',
                      '-file', os.path.join(pkg_path, 'models/humanoid_robot.sdf'),
                      '-x', '0', '-y', '0', '-z', '1'],
            output='screen'
        )
    ])
```

## Performance Optimization

### Simulation Configuration for Efficiency

For complex humanoid simulations:
- **Use Simplified Collisions**: Reduce collision mesh complexity
- **Optimize Update Rates**: Match sensor update rates to actual requirements
- **Limit Physics Substeps**: Balance between accuracy and performance
- **Use Fixed Joints**: When applicable, use `<joint type="fixed">` for permanently connected parts

### GPU Acceleration

Enable GPU acceleration for rendering:
```bash
# Set environment variables for GPU acceleration
export GAZEBO_RENDERING_LIBRARY=ogre
export OGRE_RESOURCE_PATH=/usr/lib/x86_64-linux-gnu/OGRE/Media/
```

## Troubleshooting Common Issues

### Common Problems and Solutions

1. **Simulation Runs Too Slow**: Reduce physics update rate or simplify models
2. **Robot Falls Through Floor**: Check collision properties and physics parameters
3. **Joints Behaving Unexpectedly**: Verify joint limits and dynamics parameters
4. **Sensors Not Publishing**: Check plugin configuration and namespaces
5. **Gravity Issues**: Verify gravity settings in world file

### Debugging Tips

- Use `gz topic -l` to see available topics
- Use `gz topic -i` to inspect topic information
- Enable verbose logging with `--verbose` flag
- Check Gazebo server logs for error messages

## Advanced Gazebo Features for Physical AI

### Plugins and Extensions

Custom plugins for advanced Physical AI functionality:
- **Terrain Generation**: Dynamic terrain modification
- **Weather Simulation**: Wind, rain, and other environmental effects
- **Crowd Simulation**: Multiple humanoids interacting in the same environment
- **Learning Environments**: Plugins for reinforcement learning tasks

### Multi-Robot Simulation

Simulating multiple humanoid robots:
```xml
<!-- World file with multiple robots -->
<model name="humanoid_1">
  <pose>0 0 1 0 0 0</pose>
  <!-- Model definition -->
</model>

<model name="humanoid_2">
  <pose>2 0 1 0 0 0</pose>
  <!-- Model definition -->
</model>
```

### Scenario Generation

Automating the generation of testing scenarios:
- Random object placement
- Variable terrain conditions
- Dynamic obstacles
- Environmental perturbations

## Summary

Setting up Gazebo for humanoid robotics requires careful configuration of physics parameters, robot models, sensors, and ROS/ROS2 integration. Proper setup enables efficient development and testing of Physical AI algorithms in a safe and controlled environment. The combination of realistic physics simulation and accurate sensor modeling makes Gazebo an invaluable tool for the development of intelligent humanoid systems.

In the next section, we'll explore how to connect Gazebo simulation with Unity for advanced visualization and interactive development environments.