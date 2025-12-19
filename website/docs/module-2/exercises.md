---
title: Simulation Exercises
sidebar_position: 6
---

# Simulation Exercises

## Exercise Set 1: Gazebo Fundamentals

### Exercise 1.1: Creating a Simple Robot Model

**Objective**: Create a basic robot model in Gazebo with one movable joint and one sensor.

**Requirements**:
1. Create a URDF file for a simple robot with:
   - A base link (cube shape)
   - One movable joint (revolute)
   - One child link (cylinder shape)
   - One sensor (camera, IMU, or LIDAR)

2. Spawn the robot in Gazebo

3. Verify that the robot:
   - Appears correctly in the simulation
   - Has proper physics properties
   - Shows sensor data (if using camera, verify image topic exists)

**Steps**:
1. Create a new ROS 2 package: `simulation_exercises_description`
2. Create a URDF file with the specified robot structure
3. Launch Gazebo with your robot model
4. Use `ros2 topic echo` to verify sensor data

**Hint**:
```xml
<!-- Example structure -->
<link name="base_link">
  <inertial>...</inertial>
  <visual>...</visual>
  <collision>...</collision>
</link>

<joint name="joint1" type="revolute">
  <parent link="base_link"/>
  <child link="child_link"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

### Exercise 1.2: Joint Control and Movement

**Objective**: Implement and test a simple joint controller for your robot.

**Requirements**:
1. Create a ROS 2 controller that sends position commands to your robot's joint
2. Move the joint in a predictable pattern (oscillating, sine wave, etc.)
3. Visualize the joint movement in Gazebo
4. Log the actual vs. commanded joint positions

**Implementation**:
1. Use `rclpy` to create a ROS 2 node
2. Publish to the appropriate joint command topic
3. Subscribe to joint states to compare actual vs. desired positions
4. Use matplotlib to plot the results

**Evaluation**:
- The robot joint moves as commanded
- No physics errors or instabilities in simulation
- Smooth, controlled motion

### Exercise 1.3: Sensor Data Analysis

**Objective**: Analyze and visualize sensor data from your simulated robot.

**Requirements**:
1. If using a camera: capture and display images from the simulated camera
2. If using LIDAR: plot a 2D scan of the environment
3. If using IMU: log and analyze the acceleration and angular velocity data
4. Create visualizations of the sensor data

**Implementation**:
1. Create a ROS 2 node that subscribes to your robot's sensor topic
2. Process the sensor data to extract meaningful information
3. Create plots or visualizations of the data
4. Compare simulated data to expected real-world behavior

## Exercise Set 2: Unity Visualization

### Exercise 2.1: URDF Import and Visualization

**Objective**: Import your Gazebo robot model into Unity and visualize it correctly.

**Requirements**:
1. Use the URDF Importer to import your robot model
2. Verify that all links and joints appear correctly
3. Ensure the coordinate systems match between Gazebo and Unity
4. Configure materials and appearances appropriately

**Steps**:
1. Install URDF Importer package in Unity
2. Import your URDF file
3. Troubleshoot any import issues (coordinate systems, scaling, etc.)
4. Create a simple test scene with the robot

**Evaluation**:
- Robot model imports without errors
- Links and joints are correctly positioned and oriented
- Visual elements (meshes, materials) are properly displayed

### Exercise 2.2: Joint State Synchronization

**Objective**: Connect your Unity visualization to the Gazebo simulation and synchronize joint states.

**Requirements**:
1. Set up ROS-TCP-Connector in Unity
2. Subscribe to joint states topic from Gazebo
3. Update Unity robot model based on joint angles from Gazebo
4. Verify the Unity robot mirrors movements from Gazebo

**Implementation**:
1. Add ROS-TCP-Connector to Unity scene
2. Create a C# script to subscribe to joint states
3. Map joint names to GameObjects in Unity
4. Update joint rotations based on received joint angles

```csharp
// Example structure for joint synchronization
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class JointStateSynchronizer : MonoBehaviour
{
    public string jointStatesTopic = "/robot_name/joint_states";
    private Dictionary<string, Transform> jointDict = new Dictionary<string, Transform>();
    
    void Start()
    {
        ROSTCPConnector.instance.Subscribe<JointStateMsg>(jointStatesTopic, UpdateJoints);
        // Populate jointDict with robot joint transforms
    }
    
    void UpdateJoints(JointStateMsg jointStateMsg)
    {
        for (int i = 0; i < jointStateMsg.name.Count; i++)
        {
            string jointName = jointStateMsg.name[i];
            double jointAngle = jointStateMsg.position[i];
            
            if (jointDict.ContainsKey(jointName))
            {
                // Update the joint transform based on jointAngle
                jointDict[jointName].rotation = Quaternion.AngleAxis(
                    (float)(jointAngle * Mathf.Rad2Deg), 
                    Vector3.up
                );
            }
        }
    }
}
```

### Exercise 2.3: Camera Feed Visualization

**Objective**: Display the simulated camera feed in Unity as a texture on a screen or augmented reality overlay.

**Requirements**:
1. Subscribe to the robot's camera feed in Unity
2. Convert the ROS Image message to a Unity texture
3. Display the texture on a 3D object (screen, tablet, etc.)
4. Verify that camera movements are reflected in the feed

## Exercise Set 3: Digital Twin Integration

### Exercise 3.1: Closed-Loop Control with Visualization

**Objective**: Create a complete system where visualizations in Unity influence control decisions for the Gazebo robot.

**Requirements**:
1. Implement a Unity interface that allows users to set robot goals/targets
2. Send these goals to a ROS 2 node
3. Control the Gazebo robot based on these goals
4. Visualize the robot state in Unity in real-time
5. Close the loop: Unity interface → ROS → Gazebo → ROS → Unity visualization

**Implementation**:
1. Create Unity UI elements for setting goals
2. Implement ROS message publishing from Unity
3. Create a ROS 2 node that receives goals and commands the robot
4. Ensure the Unity visualization updates based on the robot's actual state in Gazebo

### Exercise 3.2: Multi-Sensor Fusion Visualization

**Objective**: Combine data from multiple simulated sensors to create a comprehensive visualization.

**Requirements**:
1. Use at least two different sensor types (e.g., camera AND LIDAR, OR IMU AND force sensors)
2. Visualize the sensor data in Unity in a meaningful way
3. Implement basic sensor fusion to combine the data
4. Create visual representations of the fused understanding

**Examples**:
- Overlay LIDAR points on a camera view
- Show IMU data as a virtual horizon indicator
- Combine joint encoders and IMU for more accurate pose estimation
- Display force sensor information with visual indicators

### Exercise 3.3: Humanoid-specific Simulation Challenge

**Objective**: Create a simulation scenario specific to humanoid robotics that demonstrates the power of Digital Twin technology.

**Requirements**:
1. Create a humanoid robot model in Gazebo (or use an existing one)
2. Implement a balance controller that keeps the robot upright
3. Visualize the robot in Unity with detailed mesh representations
4. Add environmental obstacles in Gazebo that are also visualized in Unity
5. Implement a teleoperation interface in Unity to control the robot

**Implementation**:
1. Use a humanoid model (like the sample NAO or similar)
2. Implement a simple balance/pelvis stabilization controller
3. Create a visualization that shows the robot's state (COM, ZMP, etc.)
4. Add interaction elements in Unity for user control
5. Test the complete system with various scenarios

## Exercise Set 4: Performance and Optimization

### Exercise 4.1: Simulation Performance Analysis

**Objective**: Analyze the performance of your simulation and identify bottlenecks.

**Requirements**:
1. Profile the Gazebo simulation for FPS and resource usage
2. Profile the Unity application for FPS and resource usage
3. Measure the latency between command and response
4. Identify the main performance bottlenecks
5. Suggest optimizations for each identified bottleneck

**Tools**:
- Gazebo's built-in performance metrics
- Unity Profiler
- ROS 2 tools like `ros2 topic hz` to measure message rates
- System monitoring tools

### Exercise 4.2: Simulation Quality vs. Performance Trade-offs

**Objective**: Investigate the relationship between simulation quality and performance.

**Requirements**:
1. Test your system with different physics parameters (step size, solver iterations)
2. Test with different visual qualities in Unity (rendering resolution, effects)
3. Document the impact of each parameter on:
   - Physics accuracy
   - Visual quality
   - Real-time performance
   - Simulation stability
4. Find optimal settings for your specific use case

### Exercise 4.3: Network and Communication Optimization

**Objective**: Optimize the communication between components to minimize latency and maximize bandwidth.

**Requirements**:
1. Measure communication latencies between components
2. Optimize message rates for your specific use case
3. Implement compression if needed for high-bandwidth data (images)
4. Consider the trade-offs between message frequency and network load
5. Test performance under network stress

## Exercise Set 5: Advanced Integration

### Exercise 5.1: AI Training Environment

**Objective**: Use your Gazebo-Unity Digital Twin as an environment for training AI agents.

**Requirements**:
1. Create a task/environment in Gazebo suitable for learning
2. Configure sensor data to be accessible for AI training (observations)
3. Define rewards for the AI agents
4. Optionally, use Unity for visualization of the training process
5. Implement a simple learning algorithm (RL, imitation, etc.) to solve the task

### Exercise 5.2: Simulation-to-Reality Transfer

**Objective**: Implement techniques to reduce the "reality gap" between simulation and real-world behavior.

**Requirements**:
1. Implement domain randomization techniques
2. Add realistic noise to sensors
3. Introduce parameter variations in the simulation
4. Document the transfer of learned behaviors/parameters to the real robot
5. Evaluate the effectiveness of your techniques

**Techniques to Consider**:
- Randomize physical parameters (mass, friction, etc.)
- Add camera noise and distortion
- Vary lighting conditions
- Add delays and packet drop to communication

### Exercise 5.3: Multi-Robot Simulation

**Objective**: Extend your simulation to support multiple robots in the same environment.

**Requirements**:
1. Spawn multiple humanoid robots in Gazebo
2. Implement individual control for each robot
3. Create a unified visualization in Unity showing all robots
4. Implement basic multi-robot coordination or interaction
5. Analyze the scalability of your solution

## Self-Assessment Checklist

For each exercise, verify:

**Gazebo Components**:
- [ ] Robot model loads without errors
- [ ] Physics behaves realistically
- [ ] Sensors produce expected data
- [ ] Controllers work as expected
- [ ] Simulation runs in real-time (RTF ≈ 1.0)

**Unity Components**:
- [ ] Robot model imports correctly
- [ ] Joint states synchronize properly
- [ ] Camera feeds display correctly
- [ ] Visualization updates in real-time
- [ ] UI elements respond appropriately

**Integration**:
- [ ] Data flows correctly between components
- [ ] Latency is acceptable for use case
- [ ] System remains stable under load
- [ ] Error handling is implemented
- [ ] System recovers from communication loss

**Performance**:
- [ ] Gazebo simulation runs at acceptable speed
- [ ] Unity runs at sufficient frame rate (>30fps)
- [ ] Communication overhead is minimized
- [ ] Resource usage is within acceptable limits

Complete these exercises to gain proficiency in Digital Twin technology for Physical AI applications. For each exercise, document your approach, challenges encountered, and lessons learned.