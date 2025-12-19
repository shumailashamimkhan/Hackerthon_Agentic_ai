---
title: Gazebo and Unity Examples
sidebar_position: 5
---

# Gazebo and Unity Examples

This page contains practical examples and code samples for working with Gazebo and Unity for Digital Twin applications in Physical AI and humanoid robotics.

## Gazebo Examples

### 1. Basic Robot Model in Gazebo

This example shows a minimal robot model that can be spawned in Gazebo with basic sensors.

#### Robot Model (SDF format)

```xml title="humanoid_model.sdf"
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_humanoid">
    <link name="base_link">
      <pose>0 0 1 0 0 0</pose>
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
          <box>
            <size>0.5 0.5 0.5</size>
          </box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.8 1</ambient>
          <diffuse>0.2 0.2 1.0 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
        </material>
      </visual>
      
      <!-- Collision representation -->
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.5</size>
          </box>
        </geometry>
      </collision>
    </link>
    
    <!-- Hip joint -->
    <joint name="hip_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_leg</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
    </joint>
    
    <!-- Leg link -->
    <link name="left_leg">
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.05</izz>
        </inertia>
      </inertial>
      <visual name="leg_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="leg_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
    
    <!-- Camera sensor -->
    <sensor name="camera" type="camera">
      <pose>0.25 0 0.1 0 0 0</pose>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
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
        <frame_name>camera_frame</frame_name>
        <topic_name>camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </model>
</sdf>
```

### 2. Gazebo World with Environment

This world file creates an environment for testing humanoid robots:

```xml title="humanoid_test_world.world"
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>1000</iters>
          <sor>1.3</sor>
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
    </physics>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Simple obstacle -->
    <model name="obstacle">
      <pose>2 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.3 0.3 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.3 0.3 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.1 0.1 1</ambient>
            <diffuse>1.0 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Spawn location marker -->
    <model name="spawn_marker">
      <pose>0 0 0.05 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.1 0.8 0.1 0.5</ambient>
            <diffuse>0.2 1.0 0.2 0.5</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### 3. ROS 2 Controller for Gazebo

A simple ROS 2 controller to command the humanoid robot:

```python title="simple_controller.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import math

class SimpleHumanoidController(Node):
    def __init__(self):
        super().__init__('simple_humanoid_controller')
        
        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray, 
            '/simple_humanoid/joint_group_position_controller/commands', 
            10
        )
        
        # Subscriber for sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/simple_humanoid/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50Hz
        
        # Storage for joint states
        self.joint_positions = {}
        self.joint_velocities = {}
        
        # Initialize trajectory parameters
        self.time = 0.0
        
    def joint_state_callback(self, msg):
        """Update internal joint state representation"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def control_loop(self):
        """Main control loop"""
        # Simple periodic trajectory for hip joint
        hip_pos = math.sin(self.time) * 0.5
        
        # Create command message
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [hip_pos]  # Sending position command for hip_joint
        
        # Publish command
        self.joint_cmd_publisher.publish(cmd_msg)
        
        # Update time
        self.time += 0.02

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleHumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4. Launch File for Gazebo Simulation

A launch file to bring up the simulation with controller:

```python title="launch/humanoid_simulation_launch.py"
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    ld = LaunchDescription()
    
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=[PathJoinSubstitution([FindPackageShare('my_robot_gazebo'), 'worlds', 'humanoid_test_world.world'])],
        description='Choose one of the world files from `/my_robot_gazebo/worlds`'
    )
    
    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'gui': 'true',
            'verbose': 'false',
        }.items()
    )
    
    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_humanoid',
            '-file', PathJoinSubstitution([FindPackageShare('my_robot_description'), 'models', 'humanoid_model.sdf']),
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )
    
    # Launch controller
    controller = Node(
        package='my_robot_control',
        executable='simple_controller',
        name='simple_humanoid_controller',
        output='screen'
    )
    
    # Add actions to launch description
    ld.add_action(world_arg)
    ld.add_action(gazebo)
    ld.add_action(spawn_robot)
    ld.add_action(controller)
    
    return ld
```

## Unity Examples

### 1. URDF Import and Setup

Example code to import and configure a humanoid robot in Unity:

```csharp title="Scripts/RobotSetup.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class RobotSetup : MonoBehaviour
{
    [Tooltip("Path to URDF file")]
    public string urdfPath;
    
    [Tooltip("Robot's ROS namespace")]
    public string robotNamespace = "/simple_humanoid";
    
    private Dictionary<string, Transform> jointTransforms;
    private ROSTCPConnector rosConnector;
    
    void Start()
    {
        InitializeROSConnection();
        InitializeRobot();
    }
    
    void InitializeROSConnection()
    {
        rosConnector = ROSTCPConnector.instance;
        if (rosConnector == null)
        {
            Debug.LogError("Could not find ROSTCPConnector in scene!");
            return;
        }
        
        // Subscribe to joint states
        rosConnector.Subscribe<JointStateMsg>(
            robotNamespace + "/joint_states", 
            OnJointStateReceived
        );
    }
    
    void InitializeRobot()
    {
        // This would typically happen through URDF Import functionality
        // For this example, we assume a robot is already loaded in the scene
        
        // Create dictionary of joint transforms
        jointTransforms = new Dictionary<string, Transform>();
        
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            // Convention: joints in URDF typically have a suffix indicating their function
            if (child.name.EndsWith("_joint") || child.name.Contains("hip") || 
                child.name.Contains("knee") || child.name.Contains("ankle"))
            {
                jointTransforms[child.name] = child;
            }
        }
        
        Debug.Log($"Found {jointTransforms.Count} joints in robot");
    }
    
    void OnJointStateReceived(JointStateMsg msg)
    {
        if (jointTransforms == null) return;
        
        for (int i = 0; i < msg.name.Count; i++)
        {
            string jointName = msg.name[i];
            float jointPosition = (float)msg.position[i];
            
            if (jointTransforms.ContainsKey(jointName))
            {
                // Update joint rotation (assuming rotation around Z-axis for this example)
                jointTransforms[jointName].localRotation = 
                    Quaternion.Euler(0, 0, jointPosition * Mathf.Rad2Deg);
            }
        }
    }
    
    void Update()
    {
        // Update at 60 FPS for smooth visualization
        // Actual joint updates happen in OnJointStateReceived
    }
}
```

### 2. Camera Sensor Visualization

Example to visualize camera feeds from the simulated robot:

```csharp title="Scripts/CameraVisualizer.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using System.Threading.Tasks;

public class CameraVisualizer : MonoBehaviour
{
    public string cameraTopic = "/simple_humanoid/camera/image_raw";
    public Renderer screenRenderer;  // Mesh renderer for the screen
    private Texture2D cameraTexture;
    private ROSTCPConnector ros;
    
    void Start()
    {
        ros = ROSTCPConnector.instance;
        
        // Initialize texture to default size
        cameraTexture = new Texture2D(640, 480, TextureFormat.RGB24, false);
        if (screenRenderer != null)
            screenRenderer.material.mainTexture = cameraTexture;
        
        // Subscribe to camera topic
        ros.Subscribe<ImageMsg>(cameraTopic, OnCameraReceived);
    }
    
    void OnCameraReceived(ImageMsg msg)
    {
        if (msg.encoding == "rgb8" && msg.data != null)
        {
            // Update texture with camera data
            UpdateTextureFromMsg(msg);
        }
    }
    
    void UpdateTextureFromMsg(ImageMsg msg)
    {
        // Resize texture if dimensions changed
        if (cameraTexture.width != msg.width || cameraTexture.height != msg.height)
        {
            cameraTexture.Resize((int)msg.width, (int)msg.height);
        }
        
        // Convert ROS image data to Unity Texture2D
        Color32[] colors = new Color32[msg.data.Count/3];
        for (int i = 0; i < msg.data.Count; i += 3)
        {
            byte r = msg.data[i];
            byte g = msg.data[i + 1]; 
            byte b = msg.data[i + 2];
            colors[i/3] = new Color32(r, g, b, 255);
        }
        
        cameraTexture.SetPixels32(colors);
        cameraTexture.Apply();
    }
}
```

### 3. Unity-ROS Bridge for Control

Sending commands from Unity back to the robot simulation:

```csharp title="Scripts/UnityToRobotControl.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using System.Collections.Generic;

public class UnityToRobotControl : MonoBehaviour
{
    [SerializeField] private string cmdVelTopic = "/simple_humanoid/cmd_vel";
    [SerializeField] private float linearSpeed = 1.0f;
    [SerializeField] private float angularSpeed = 1.0f;
    
    private ROSTCPConnector ros;
    
    void Start()
    {
        ros = ROSTCPConnector.instance;
    }
    
    void Update()
    {
        // Get input from keyboard/controller
        float linear = Input.GetAxis("Vertical");   // W/S or arrow keys
        float angular = Input.GetAxis("Horizontal"); // A/D or arrow keys
        
        // Create and send twist command
        if (Mathf.Abs(linear) > 0.1f || Mathf.Abs(angular) > 0.1f)
        {
            var twist = new TwistMsg();
            twist.linear = new Vector3Msg(linear * linearSpeed, 0, 0);
            twist.angular = new Vector3Msg(0, 0, angular * angularSpeed);
            
            ros.Publish(cmdVelTopic, twist);
        }
        
        // Additional commands could be triggered by other inputs
        if (Input.GetButtonDown("Fire1")) // Left mouse button
        {
            // Send custom command
            SendCustomCommand();
        }
    }
    
    void SendCustomCommand()
    {
        // Example: Send a custom command to the robot
        var stringCmd = new StringMsg("custom_command_triggered");
        ros.Publish(robotNamespace + "/custom_command", stringCmd);
    }
}
```

### 4. Scene Setup Script

A script to initialize the entire Unity scene with robotics components:

```csharp title="Scripts/SceneInitializer.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class SceneInitializer : MonoBehaviour
{
    [Tooltip("IP address of ROS system")]
    public string rosIpAddress = "127.0.0.1";
    
    [Tooltip("Port for ROS TCP connection")]
    public int rosPort = 10000;
    
    [Tooltip("Whether to attempt ROS connection on startup")]
    public bool connectToRosOnStart = true;
    
    void Start()
    {
        if (connectToRosOnStart)
        {
            ConnectToRos();
        }
    }
    
    public void ConnectToRos()
    {
        ROSTCPConnector ros = ROSTCPConnector.instance;
        if (ros != null)
        {
            ros.Initialize(rosIpAddress, rosPort);
            Debug.Log($"Connected to ROS at {rosIpAddress}:{rosPort}");
        }
        else
        {
            Debug.LogError("ROSTCPConnector not found in scene!");
        }
    }
    
    public void DisconnectFromRos()
    {
        ROSTCPConnector ros = ROSTCPConnector.instance;
        if (ros != null)
        {
            ros.Disconnect();
            Debug.Log("Disconnected from ROS");
        }
    }
}
```

## Running the Examples

### Gazebo Examples

To run the Gazebo examples:

1. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-joint-state-publisher ros-humble-robot-state-publisher
   ```

2. **Prepare Robot Description:**
   - Create a ROS 2 package called `my_robot_description`
   - Place the SDF file in `my_robot_description/models/`
   - Create URDF files if needed

3. **Create Controller Package:**
   - Create a ROS 2 package called `my_robot_control`
   - Place the Python controller in `my_robot_control/my_robot_control/`
   - Create the Python executable

4. **Launch Simulation:**
   ```bash
   ros2 launch my_robot_gazebo humanoid_simulation_launch.py
   ```

5. **Run Controller:**
   ```bash
   ros2 run my_robot_control simple_controller
   ```

### Unity Examples

To run the Unity examples:

1. **Install Unity Robotics Tools:**
   - Add the Unity Robotics Hub from Unity Package Manager
   - Import ROS-TCP-Connector and URDF-Importer packages

2. **Setup Scene:**
   - Create a new scene
   - Add the `SceneInitializer` script to a GameObject
   - Add the `RobotSetup` script to your imported robot model
   - Add `CameraVisualizer` to a screen object
   - Add `UnityToRobotControl` to a player controller object

3. **Configure Connection:**
   - Set the ROS IP address and port in `SceneInitializer`
   - Connect to ROS through the UI or automatically

4. **Run Simulation:**
   - Press Play in Unity
   - The robot should receive joint state updates from Gazebo
   - Camera feeds should display on screens in the Unity scene
   - Robot should respond to Unity commands

## Integration Between Gazebo and Unity

### Architecture for Combined Use

For a complete Digital Twin setup, you might run:

1. **Gazebo**: Physics simulation and sensor simulation
2. **Unity**: Visualization and user interface
3. **ROS/ROS2**: Communication broker between systems

Both environments subscribe and publish on the same ROS topics, staying synchronized.

This architecture allows leveraging Gazebo's physics strengths and Unity's visualization capabilities simultaneously, creating a powerful Digital Twin environment for Physical AI development.