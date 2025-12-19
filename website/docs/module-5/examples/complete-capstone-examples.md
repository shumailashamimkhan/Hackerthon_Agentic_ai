---
sidebar_position: 9
title: "Complete Capstone Examples"
---

# Complete Capstone Examples

This section provides complete, integrated examples that demonstrate the full Physical AI system combining all modules. These examples show how ROS 2, Digital Twin simulation, NVIDIA Isaac, Vision-Language-Action capabilities, and deployment work together.

## Example 1: Autonomous Room Navigation and Object Interaction

A complete example where the humanoid robot receives a voice command to navigate to a specific room, recognize a target object, and perform an action.

### System Architecture

```
[Voice Command] -> [NLP Processing] -> [Task Planning] -> [Navigation] -> [Object Recognition] -> [Manipulation] -> [Action Execution]
```

### Complete Implementation

```python
# complete_capstone_example.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import Trigger

class CompleteCapstoneNode(Node):
    def __init__(self):
        super().__init__('complete_capstone')
        
        # Subscribers for different inputs
        self.voice_sub = self.create_subscription(
            String,
            'voice_commands',
            self.voice_command_callback,
            10
        )
        
        # Publishers for different systems
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            'navigation/goal',
            10
        )
        
        self.object_detection_sub = self.create_subscription(
            Detection2DArray,
            'object_detections',
            self.object_detection_callback,
            10
        )
        
        self.manipulation_cmd_pub = self.create_publisher(
            String,
            'manipulation/commands',
            10
        )
        
        # Service clients for different subsystems
        self.nav_client = self.create_client(Trigger, 'navigation/start')
        self.manip_client = self.create_client(Trigger, 'manipulation/start')
        
        # Internal state
        self.current_task = None
        self.target_object = None
        self.robot_pose = None
        
    def voice_command_callback(self, msg):
        """Process voice command and initiate appropriate action"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")
        
        # Use LLM to parse command and determine task
        task = self.parse_command_with_llm(command)
        
        if task['action'] == 'navigate_to_object':
            self.current_task = task
            self.target_object = task['object']
            
            # Start navigation to search for object
            self.begin_navigation_to_search_area()
            
        elif task['action'] == 'grasp_and_place':
            self.current_task = task
            self.target_object = task['object']
            self.destination = task['destination']
            
            # Start navigation to object, then manipulation, then to destination
            self.begin_grasp_and_place_task()
    
    def parse_command_with_llm(self, command):
        """Use LLM to parse command into structured task"""
        # This would interface with an LLM API to parse the command
        # For this example, we'll simulate the LLM response
        if "bring me the red cup" in command.lower():
            return {
                'action': 'grasp_and_place',
                'object': 'red cup',
                'destination': 'kitchen counter'
            }
        elif "go to the living room" in command.lower():
            return {
                'action': 'navigate_to_object',
                'object': 'living room',
                'destination': 'living room'
            }
        else:
            return {
                'action': 'unknown',
                'object': None,
                'destination': None
            }
    
    def begin_navigation_to_search_area(self):
        """Navigate to area where target object is likely to be found"""
        # Determine approximate location of target object
        location = self.get_approximate_location_of_object(self.target_object)
        
        # Create navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = location['x']
        goal_msg.pose.position.y = location['y']
        goal_msg.pose.position.z = 0.0
        # Orientation (facing in direction of movement)
        goal_msg.pose.orientation.w = 1.0
        
        # Send navigation goal
        self.navigation_goal_pub.publish(goal_msg)
        
        # Wait for navigation to complete
        self.wait_for_navigation_completion()
    
    def object_detection_callback(self, msg):
        """Process detected objects and take appropriate action"""
        for detection in msg.detections:
            # Check if this is our target object
            if detection.results[0].hypothesis.class_id == self.target_object:
                self.get_logger().info(f"Found target object: {self.target_object}")
                
                # If in navigation mode, switch to manipulation mode
                if self.current_task['action'] == 'grasp_and_place':
                    self.initiate_grasp_sequence(detection)
                break
    
    def initiate_grasp_sequence(self, detection):
        """Initiate the sequence to grasp the detected object"""
        # Calculate grasp position from detection
        grasp_pose = self.calculate_grasp_pose(detection)
        
        # Send manipulation command
        cmd_msg = String()
        cmd_msg.data = f"grasp_object_at:{grasp_pose.position.x},{grasp_pose.position.y},{grasp_pose.position.z}"
        self.manipulation_cmd_pub.publish(cmd_msg)
        
        # Wait for grasp completion
        self.wait_for_grasp_completion()
        
        # Navigate to destination
        self.navigate_to_destination()
    
    def navigate_to_destination(self):
        """Navigate to the destination for placing the object"""
        # Determine destination location
        location = self.get_destination_location(self.current_task['destination'])
        
        # Create navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = location['x']
        goal_msg.pose.position.y = location['y']
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        # Send navigation goal
        self.navigation_goal_pub.publish(goal_msg)
        
        # Wait for navigation to destination
        self.wait_for_navigation_completion()
        
        # Place object
        place_cmd = String()
        place_cmd.data = f"place_object_at:{location['x']},{location['y']},{location['z']}"
        self.manipulation_cmd_pub.publish(place_cmd)
    
    def get_approximate_location_of_object(self, obj_name):
        """Get approximate location of object based on semantic map"""
        # In a real implementation, this would query a semantic map
        # For this example, return predefined locations
        locations = {
            'red cup': {'x': 2.0, 'y': 3.0},
            'living room': {'x': 5.0, 'y': 1.0},
            'kitchen counter': {'x': -1.0, 'y': 4.0}
        }
        return locations.get(obj_name, {'x': 0.0, 'y': 0.0})
    
    def get_destination_location(self, dest_name):
        """Get location of destination"""
        # In a real implementation, this would query a semantic map
        # For this example, return predefined locations
        locations = {
            'kitchen counter': {'x': -1.0, 'y': 4.0, 'z': 0.8}
        }
        return locations.get(dest_name, {'x': 0.0, 'y': 0.0, 'z': 0.0})
    
    def calculate_grasp_pose(self, detection):
        """Calculate appropriate grasp pose for detected object"""
        # Calculate position based on object center
        pose = detection.bbox.center.position  # Simplified for example
        
        # Adjust for proper grasp approach
        pose.x -= 0.1  # Approach from front
        pose.y -= 0.1  # Slight offset
        
        return pose
    
    def wait_for_navigation_completion(self):
        """Wait for navigation to complete"""
        # This would typically involve waiting for navigation feedback
        # For this example, use a timer
        self.get_logger().info("Waiting for navigation to complete...")
        self.create_timer(5.0, self.navigation_completed_callback)
    
    def wait_for_grasp_completion(self):
        """Wait for grasp action to complete"""
        # This would typically involve waiting for manipulation feedback
        # For this example, use a timer
        self.get_logger().info("Waiting for grasp to complete...")
        self.create_timer(3.0, self.grasp_completed_callback)
    
    def navigation_completed_callback(self):
        """Called when navigation is completed"""
        self.get_logger().info("Navigation completed")
        # Continue with next step in task
    
    def grasp_completed_callback(self):
        """Called when grasp is completed"""
        self.get_logger().info("Grasp completed")
        # Continue with next step in task

def main(args=None):
    rclpy.init(args=args)
    node = CompleteCapstoneNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example 2: Multi-Modal Interaction in Simulation

An example showing how to integrate all the systems in simulation before deploying on real hardware.

### Simulation Launch File

```xml
<!-- complete_capstone_simulation.launch.xml -->
<launch>
  <!-- Start Gazebo simulation -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gz_sim.launch.py">
    <arg name="gz_args" value="-r unitree_h1_world.sdf"/>
  </include>
  
  <!-- Start ROS 2 control for simulated robot -->
  <node pkg="controller_manager" exec="ros2_control_node" name="ros2_control_node">
    <param name="robot_description" value="$(var robot_description)"/>
  </node>
  
  <!-- Start navigation stack -->
  <include file="$(find-pkg-share nav2_bringup)/launch/navigation_launch.py"/>
  
  <!-- Start perception pipeline -->
  <node pkg="object_detection" exec="yolo_node" name="yolo_detector">
    <param name="model_path" value="models/yolo_sim.trt"/>
  </node>
  
  <!-- Start voice processing -->
  <node pkg="voice_processing" exec="stt_node" name="speech_to_text"/>
  <node pkg="voice_processing" exec="tts_node" name="text_to_speech"/>
  
  <!-- Start manipulation stack -->
  <include file="$(find-pkg-share moveit_bringup)/launch/moveit.launch.py"/>
  
  <!-- Start the complete capstone node -->
  <node pkg="physical_ai_examples" exec="complete_capstone_example.py" name="complete_capstone" output="screen"/>
  
</launch>
```

## Example 3: Deployment Integration

An example showing how to integrate the complete system for deployment on real hardware.

### Deployment Configuration

```yaml
# deployment_config.yaml
capstone_system:
  voice_processing:
    stt_model: "whisper-tiny"
    tts_model: "tacotron2+jasper"
    activation_phrase: "Hey Robot"
  navigation:
    planner: "nav2"
    global_frame: "map"
    robot_base_frame: "base_link"
    use_sim_time: false  # Important: disable sim time for real robot
  perception:
    detection_model: "yolo11-nano.trt"
    confidence_threshold: 0.7
    input_width: 640
    input_height: 480
  manipulation:
    planning_group: "right_arm"
    end_effector: "right_hand"
    grasp_approach_distance: 0.1
    grasp_grasp_distance: 0.05
  unitree_integration:
    use_unitree_sdk: true
    sdk_ip: "192.168.123.10"  # Default Unitree robot IP
    communication_port: 8080
  performance:
    update_rate: 10.0  # Hz
    timeout_duration: 5.0  # seconds
    max_retries: 3