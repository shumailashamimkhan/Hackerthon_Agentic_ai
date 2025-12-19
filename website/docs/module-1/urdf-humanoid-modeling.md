---
title: URDF for Humanoid Modeling
sidebar_position: 4
---

# URDF for Humanoid Modeling

## Understanding URDF in Physical AI

URDF (Unified Robot Description Format) is the native XML-based format for representing robot models in ROS. In the context of Physical AI and humanoid robotics, URDF serves as the blueprint that defines the physical properties, kinematic structure, and visual appearance of humanoid robots. It's essential for simulation, control, and perception in Physical AI applications.

### The Role of URDF in Physical AI

In Physical AI, URDF models serve multiple purposes:

- **Simulation**: Physics engines use URDF to simulate robot behavior
- **Perception**: Computer vision algorithms use URDF for robot state estimation
- **Control**: Kinematics solvers use URDF for motion planning and control
- **Visualization**: RViz and other tools render robots using URDF definitions
- **Hardware Integration**: Real robot drivers often mirror URDF structure

## URDF Fundamentals for Humanoid Robots

### Basic URDF Structure

A humanoid robot URDF consists of:

- **Links**: Rigid bodies representing physical parts (torso, head, arms, legs, etc.)
- **Joints**: Connections between links with specific kinematic properties
- **Materials**: Visual properties for rendering
- **Transmissions**: Mapping between actuators and joints
- **Gazebo Plugins**: Simulation-specific extensions

### Humanoid-Specific Considerations

Humanoid robots have unique characteristics in their URDF:

- **Symmetry**: Arms and legs are typically mirrored left/right
- **Degrees of Freedom**: Usually 30-40+ joints for whole-body control
- **Balance Requirements**: Center of mass considerations
- **Collision Detection**: Complex multi-point contact scenarios

## URDF Link Elements

### Link Definition

Each link represents a rigid body part of the robot:

```xml
<link name="base_link">
  <inertial>
    <mass value="10.0"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <inertia ixx="0.4" ixy="0" ixz="0" iyy="0.4" iyz="0" izz="0.2"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_robot_description/meshes/base_link.stl"/>
    </geometry>
    <material name="grey">
      <color rgba="0.5 0.5 0.5 1.0"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_robot_description/meshes/base_collision.stl"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

For Physical AI applications, accurate inertial properties are critical:

- **Mass**: Total mass of the link
- **Center of Mass**: Origin offset in the link frame
- **Inertia Matrix**: Moments of inertia about the center of mass

For humanoid robots:
- Lower body segments typically have higher mass
- Careful calculation of inertia tensors for stable locomotion
- Consider adding point masses for motor weights

### Visual vs. Collision Geometry

Humanoid robots require special attention to geometry differences:

**Visual**: Detailed meshes for rendering
**Collision**: Simplified geometries for physics simulation

Best practices:
- Use convex hulls or primitive shapes for collision
- Separate collision mesh optimized for simulation
- Different LOD models for different use cases

## URDF Joint Elements

### Joint Types in Humanoids

Humanoid robots typically use these joint types:

1. **Revolute**: Rotational joints with limits (elbows, knees, neck)
2. **Continuous**: Unbounded rotational joints (waist, shoulders)
3. **Fixed**: Rigid connections (head to camera mount)
4. **Prismatic**: Linear sliding joints (rarely used)

### Joint Specification

```xml
<joint name="left_hip_yaw" type="revolute">
  <parent link="pelvis"/>
  <child link="left_thigh"/>
  <origin xyz="0.0 0.1 0.0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="2.0"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>
```

### Humanoid-Specific Joint Parameters

For Physical AI and balance control:

- **Effort Limits**: Reflect actual actuator capabilities
- **Velocity Limits**: Prevent damage during rapid motions
- **Damping**: Helps stabilize simulation
- **Friction**: Accounts for mechanical friction in joints

### Joint Control in Physical AI

Humanoid joint specifications must match controller expectations:

```xml
<transmission name="left_hip_yaw_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_hip_yaw">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_yaw_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Humanoid Model Architecture

### Kinematic Tree Structure

A typical humanoid follows this kinematic structure:

```
base_link (world or pelvis)
├── torso
│   ├── head
│   │   ├── camera_link
│   │   └── lidar_mount
│   ├── left_shoulder
│   │   ├── left_upper_arm
│   │   ├── left_lower_arm
│   │   └── left_hand
│   ├── right_shoulder
│   │   ├── right_upper_arm
│   │   ├── right_lower_arm
│   │   └── right_hand
│   └── waist
│       ├── left_pelvis
│       │   ├── left_thigh
│       │   ├── left_shin
│       │   └── left_foot
│       └── right_pelvis
│           ├── right_thigh
│           ├── right_shin
│           └── right_foot
```

### Floating Base Considerations

For Physical AI applications, the floating base is often represented as a fixed joint to world initially:

```xml
<joint name="floating_base" type="fixed">
  <parent link="world"/>
  <child link="pelvis"/>
  <origin xyz="0 0 1.0" rpy="0 0 0"/>
</joint>
```

Later, this can be switched to a planar or floating joint for more complex movements.

## Advanced URDF for Physical AI

### Gazebo-Specific Tags

For simulation in Gazebo, include these elements:

```xml
<gazebo reference="left_foot">
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <fdir1>1 0 0</fdir1>
  <maxVel>1.0</maxVel>
  <minDepth>0.001</minDepth>
</gazebo>
```

For humanoid walking, friction properties are especially important on feet and palms.

### Sensor Integration

Humanoid robots have many sensors that must be properly positioned:

```xml
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <origin xyz="0 0 0"/>
    <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/> <!-- Mount in torso center of mass -->
</joint>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        ...
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        ...
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Transmission Specifications for Physical AI

Different control strategies require different transmission interfaces:

```xml
<!-- For torque control (preferred for Physical AI) -->
<transmission name="left_knee_trans">
  <type>transmission_interface/EffortJointInterface</type>
  <joint name="left_knee">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_knee_actuator">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- For position control -->
<transmission name="head_pan_trans">
  <type>transmission_interface/PositionJointInterface</type>
  <joint name="head_pan">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="head_pan_actuator">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## URDF Organization for Complex Humanoids

### Package Structure

Organize URDF files in a standard structure:

```
robot_description/
├── urdf/
│   ├── robot.urdf.xacro
│   ├── head.xacro
│   ├── arm.xacro
│   ├── leg.xacro
│   └── sensors.xacro
├── meshes/
│   ├── visual/
│   └── collision/
├── launch/
└── config/
    └── control.yaml
```

### Using Xacro for Complex Models

Xacro simplifies complex humanoid definitions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Import common macros -->
  <xacro:include filename="$(find robot_description)/urdf/macros.xacro"/>

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Define a limb macro -->
  <xacro:macro name="limb" params="name side prefix *origin *axis">
    <xacro:macro name="joint_block_${prefix}">
      <joint name="${side}_${name}_joint" type="revolute">
        <xacro:insert_block name="origin"/>
        <parent link="torso"/>
        <child link="${side}_${name}_link"/>
        <axis xyz="0 1 0"/>
        <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="2.0"/>
      </joint>
    </xacro:macro>
  </xacro:macro>

  <!-- Instantiate limbs -->
  <xacro:limb name="arm" side="left" prefix="shoulder">
    <origin xyz="0.2 0.1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </xacro:limb>

  <xacro:limb name="arm" side="right" prefix="shoulder">
    <origin xyz="0.2 -0.1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </xacro:limb>

</robot>
```

## Simulation-Specific Considerations

### Physics Parameters for Humanoid Stability

Humanoid robots are inherently unstable and require careful physics tuning:

- **Damping**: Higher values for stable simulation
- **Friction**: Adequate for foot-ground interactions
- **Contact Parameters**: Proper kp/kd values for contact stiffness
- **Solver Parameters**: Use appropriate solver for stiff systems

### Realism vs. Stability Trade-offs

In Physical AI applications, there's often a trade-off:

- **Accurate physics** requires detailed modeling but may be unstable
- **Stable simulation** may sacrifice some realism but enables AI training
- **Hybrid approaches** use simplified models during training and detailed during validation

## Generating URDF Programmatically

For Physical AI research, you may need to generate URDF programmatically:

```python
import xml.etree.ElementTree as ET

def create_link_element(name, mass, com, inertia_matrix, visual_mesh=None):
    """Programmatically create a URDF link element"""
    link = ET.Element('link', {'name': name})
    
    # Add inertial element
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', {'value': str(mass)})
    ET.SubElement(inertial, 'origin', {
        'xyz': f"{com['x']} {com['y']} {com['z']}",
        'rpy': f"{com['roll']} {com['pitch']} {com['yaw']}"
    })
    
    # Add inertia matrix
    ET.SubElement(inertial, 'inertia', {
        'ixx': str(inertia_matrix[0]), 'ixy': str(inertia_matrix[1]), 'ixz': str(inertia_matrix[2]),
        'iyy': str(inertia_matrix[3]), 'iyz': str(inertia_matrix[4]), 'izz': str(inertia_matrix[5])
    })
    
    if visual_mesh:
        # Add visual element
        visual = ET.SubElement(link, 'visual')
        geom = ET.SubElement(visual, 'geometry')
        ET.SubElement(geom, 'mesh', {'filename': visual_mesh})
        
    return link
```

## URDF Validation and Testing

### Validation Tools

Several ROS tools help validate URDFs:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# View robot model
urdf_to_graphiz /path/to/robot.urdf
```

### Robot State Publishing

Ensure proper TF trees for Physical AI applications:

```xml
<node pkg="robot_state_publisher" type="robot_state_publisher" name="robot_state_publisher">
  <param name="publish_frequency" type="double" value="50.0"/>
</node>

<node pkg="joint_state_publisher" type="joint_state_publisher" name="joint_state_publisher"/>
```

## URDF for AI Training Pipelines

### Simulation-Reality Transfer

When using URDF for training AI agents:

- Include realistic noise models for sensors
- Match physical parameters between simulation and reality
- Use domain randomization techniques
- Account for model inaccuracies in training

### Morphology Experiments

URDF flexibility allows for:
- Changing limb lengths for gait optimization
- Adding/removing sensors during design
- Modifying mass distributions for stability studies
- Testing different actuator configurations

## Best Practices for Physical AI

### Maintainability
- Use consistent naming conventions
- Organize URDF into modular Xacro files
- Document joint limits and control parameters
- Version control for URDF files

### Performance
- Simplify collision geometry where possible
- Use appropriate mesh resolutions
- Minimize unnecessary joints
- Optimize for real-time simulation

### Compatibility
- Ensure URDF works with different kinematics solvers
- Test with multiple simulators (Gazebo, MuJoCo, PyBullet)
- Verify compatibility with planning frameworks (MoveIt!)

## Summary

URDF is fundamental to Physical AI applications, providing the essential link between abstract AI algorithms and real-world robot morphology. For humanoid robotics specifically, careful attention to:

- Accurate inertial properties for stable simulation
- Proper joint limits and control interfaces
- Realistic friction and contact models
- Modular organization for maintainability
- Validation for physics accuracy

Enables AI agents to learn and transfer skills between simulation and reality. Well-crafted URDF models are essential for successful Physical AI implementations, serving as the foundation upon which intelligent robot behaviors are built.