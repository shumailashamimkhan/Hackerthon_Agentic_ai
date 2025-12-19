---
title: VLA Examples and Code Samples
sidebar_position: 5
---

# VLA Examples and Code Samples

## Example 1: Voice Command Processing Pipeline

This example demonstrates the complete pipeline from voice input to robot action execution:

```python
#!/usr/bin/env python3
"""
Example implementation of a complete Voice-Language-Action pipeline
that integrates speech recognition, natural language understanding,
and robot action execution.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from shu_msgs.action import HumanoidNavigation, HumanoidManipulation
import openai
import json
import time
import asyncio
from transformers import pipeline


class VLAPipelineNode(Node):
    """
    Complete VLA pipeline integrating voice input, language processing, 
    and robot action execution.
    """
    def __init__(self):
        super().__init__('vla_pipeline_node')
        
        # Initialize components
        self.speech_recognizer = self._initialize_speech_recognition()
        self.language_processor = self._initialize_language_processing()
        self.robot_controller = self._initialize_robot_controller()
        
        # Publishers/subscribers for the pipeline
        self.voice_input_sub = self.create_subscription(
            String, 
            'voice_input', 
            self.voice_callback, 
            10
        )
        
        self.interpretation_pub = self.create_publisher(
            String, 
            'interpreted_command', 
            10
        )
        
        self.action_feedback_pub = self.create_publisher(
            String, 
            'action_feedback', 
            10
        )
        
        # Action clients for robot commands
        self.nav_client = ActionClient(self, HumanoidNavigation, 'humanoid_navigate_to_pose')
        self.manip_client = ActionClient(self, HumanoidManipulation, 'humanoid_perform_manipulation')
        
        self.get_logger().info('VLA Pipeline Node initialized')

    def _initialize_speech_recognition(self):
        """Initialize speech recognition component (simplified placeholder)"""
        # In a real implementation, this would connect to Whisper or similar
        self.get_logger().info('Initializing speech recognition...')
        return {"status": "initialized"}

    def _initialize_language_processing(self):
        """Initialize language processing component"""
        # In a real implementation, this might use a local LLM or connect to an API
        self.get_logger().info('Initializing language processing...')
        return {
            "model": "local_transformer",  # Or "openai_gpt", etc.
            "tokenizer": "initialized"
        }

    def _initialize_robot_controller(self):
        """Initialize robot control interface"""
        self.get_logger().info('Initializing robot controller...')
        return {
            "navigation": True,
            "manipulation": True,
            "safety": True
        }

    def voice_callback(self, msg):
        """Process incoming voice input through the VLA pipeline"""
        try:
            # Step 1: Process the voice command
            command_text = msg.data
            self.get_logger().info(f'Received voice command: {command_text}')
            
            # Step 2: Interpret the language command
            interpretation = self.interpret_language_command(command_text)
            
            if interpretation:
                # Publish the interpretation for monitoring
                interpretation_msg = String()
                interpretation_msg.data = json.dumps(interpretation)
                self.interpretation_pub.publish(interpretation_msg)
                
                # Step 3: Execute the appropriate robot action
                success = self.execute_robot_action(interpretation)
                
                # Step 4: Provide feedback
                feedback_msg = String()
                feedback_msg.data = f"Command '{command_text}' executed successfully: {success}"
                self.action_feedback_pub.publish(feedback_msg)
                
                self.get_logger().info(f'Action execution result: {success}')
            else:
                self.get_logger().error(f'Could not interpret command: {command_text}')
                
        except Exception as e:
            self.get_logger().error(f'Error in VLA pipeline: {str(e)}')

    def interpret_language_command(self, command_text):
        """
        Interpret natural language command and extract actionable components
        """
        try:
            # This is a simplified interpretation - in practice, this would use 
            # more sophisticated NLP and context reasoning
            command_lower = command_text.lower()
            
            interpretation = {
                "raw_command": command_text,
                "action_type": self._classify_action_type(command_lower),
                "target_object": self._extract_target_object(command_lower),
                "target_location": self._extract_target_location(command_lower),
                "motion_parameters": self._extract_motion_params(command_lower),
                "confidence": 0.85  # Placeholder confidence
            }
            
            return interpretation
            
        except Exception as e:
            self.get_logger().error(f'Error interpreting command: {str(e)}')
            return None

    def _classify_action_type(self, command_text):
        """Classify the high-level action type from the command"""
        if any(word in command_text for word in ["go to", "navigate to", "move to", "walk to", "head to"]):
            return "navigation"
        elif any(word in command_text for word in ["pick up", "grasp", "get", "take", "lift"]):
            return "manipulation_grasp"
        elif any(word in command_text for word in ["put", "place", "set", "drop"]):
            return "manipulation_place"
        elif any(word in command_text for word in ["follow", "accompany", "escort"]):
            return "following"
        elif any(word in command_text for word in ["greet", "hello", "wave", "acknowledge"]):
            return "social_interaction"
        else:
            return "unknown"

    def _extract_target_object(self, command_text):
        """Extract the target object from the command"""
        # This would use more sophisticated NLP in practice
        objects = ["ball", "cup", "book", "box", "bottle", "apple", "pen", "tablet"]
        for obj in objects:
            if obj in command_text:
                # Check for adjectives before the object
                words = command_text.split()
                obj_idx = -1
                for i, word in enumerate(words):
                    if word == obj:
                        obj_idx = i
                        break
                
                if obj_idx > 0:
                    # Include possible adjective before object
                    adjective = words[obj_idx-1] if obj_idx > 0 else ""
                    return f"{adjective} {obj}".strip()
                else:
                    return obj
        return None

    def _extract_target_location(self, command_text):
        """Extract the target location from the command"""
        locations = ["kitchen", "living room", "bedroom", "office", "bathroom", 
                    "dining room", "hallway", "garage", "garden", "table", 
                    "counter", "couch", "chair", "bed", "door"]
        
        for loc in locations:
            if loc in command_text:
                return loc
        return None

    def _extract_motion_params(self, command_text):
        """Extract motion-related parameters from command"""
        params = {"speed": "normal", "caution": "medium"}
        
        if any(word in command_text for word in ["slowly", "carefully", "gently"]):
            params["speed"] = "slow"
            params["caution"] = "high"
        elif any(word in command_text for word in ["quickly", "fast", "hurry"]):
            params["speed"] = "fast"
        
        return params

    def execute_robot_action(self, interpretation):
        """Execute the appropriate robot action based on interpretation"""
        action_type = interpretation["action_type"]
        
        if action_type == "navigation":
            return self._execute_navigation_action(interpretation)
        elif action_type in ["manipulation_grasp", "manipulation_place"]:
            return self._execute_manipulation_action(interpretation)
        elif action_type == "following":
            return self._execute_following_action(interpretation)
        elif action_type == "social_interaction":
            return self._execute_social_action(interpretation)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def _execute_navigation_action(self, interpretation):
        """Execute navigation action"""
        try:
            # Wait for action server
            self.nav_client.wait_for_server()
            
            # Create goal
            goal_msg = HumanoidNavigation.Goal()
            
            # Determine target pose based on location
            location = interpretation["target_location"]
            if location:
                goal_msg.target_pose = self._get_pose_for_location(location)
            else:
                self.get_logger().error("No target location specified for navigation")
                return False
            
            # Set motion parameters
            goal_msg.motion_params.speed = self._get_speed_for_param(interpretation["motion_parameters"]["speed"])
            goal_msg.motion_params.caution_level = self._get_caution_for_param(interpretation["motion_parameters"]["caution"])
            
            # Send goal
            goal_future = self.nav_client.send_goal_async(goal_msg)
            
            # Wait for result
            rclpy.spin_until_future_complete(self, goal_future)
            
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Navigation goal rejected')
                return False
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            result = result_future.result().result
            return result.success
            
        except Exception as e:
            self.get_logger().error(f'Navigation execution error: {str(e)}')
            return False

    def _execute_manipulation_action(self, interpretation):
        """Execute manipulation action"""
        try:
            # Wait for action server
            self.manip_client.wait_for_server()
            
            # Create goal based on action type
            goal_msg = HumanoidManipulation.Goal()
            goal_msg.manipulation_type = interpretation["action_type"].split("_")[1]  # grasp or place
            
            # Set target object if specified
            target_obj = interpretation["target_object"]
            if target_obj:
                goal_msg.target_object_id = target_obj
                # In practice, we would need to know the object's location
                # For this example, we'll use a default pose
                goal_msg.target_pose = self._get_default_object_pose()
            else:
                self.get_logger().warn("No target object specified for manipulation")
            
            # Send goal
            goal_future = self.manip_client.send_async_goal(goal_msg)
            
            # Wait for result
            rclpy.spin_until_future_complete(self, goal_future)
            
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Manipulation goal rejected')
                return False
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            result = result_future.result().result
            return result.success
            
        except Exception as e:
            self.get_logger().error(f'Manipulation execution error: {str(e)}')
            return False

    def _execute_following_action(self, interpretation):
        """Execute following action (simplified)"""
        # This would typically activate a person-following behavior
        # For this example, we'll publish a command to a follower node
        try:
            follower_cmd_pub = self.create_publisher(String, 'follower_command', 10)
            cmd_msg = String()
            cmd_msg.data = "start_following"
            follower_cmd_pub.publish(cmd_msg)
            self.get_logger().info("Following command published")
            return True
        except Exception as e:
            self.get_logger().error(f'Following execution error: {str(e)}')
            return False

    def _execute_social_action(self, interpretation):
        """Execute social interaction action (e.g., waving)"""
        # This would activate a social interaction behavior
        try:
            social_cmd_pub = self.create_publisher(String, 'social_command', 10)
            cmd_msg = String()
            cmd_msg.data = "wave_hello"
            social_cmd_pub.publish(cmd_msg)
            self.get_logger().info("Social interaction command published")
            return True
        except Exception as e:
            self.get_logger().error(f'Social action execution error: {str(e)}')
            return False

    def _get_pose_for_location(self, location_name):
        """Get a predefined pose for a named location"""
        # In practice, this would come from a map or spatial memory
        location_poses = {
            "kitchen": PoseStamped(pose=[1.0, 2.0, 0.0, 0, 0, 0, 1]),
            "living room": PoseStamped(pose=[0.0, 0.0, 0.0, 0, 0, 0, 1]),
            "bedroom": PoseStamped(pose=[-2.0, 1.0, 0.0, 0, 0, 0, 1]),
            "office": PoseStamped(pose=[-1.0, -2.0, 0.0, 0, 0, 0, 1]),
        }
        
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        
        if location_name in location_poses:
            pose_stamped.pose = location_poses[location_name].pose
        else:
            # Default pose if location not found
            pose_stamped.pose.position.x = 0.0
            pose_stamped.pose.position.y = 0.0
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
        return pose_stamped

    def _get_speed_for_param(self, speed_param):
        """Convert speed parameter to numerical value"""
        speed_map = {
            "slow": 0.3,
            "normal": 0.6,
            "fast": 1.0
        }
        return speed_map.get(speed_param, 0.6)  # Default to normal

    def _get_caution_for_param(self, caution_param):
        """Convert caution parameter to numerical value"""
        caution_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 1.0
        }
        return caution_map.get(caution_param, 0.6)  # Default to medium

    def _get_default_object_pose(self):
        """Get a default pose for object manipulation"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        
        # Default: object in front of robot
        pose_stamped.pose.position.x = 0.5
        pose_stamped.pose.position.y = 0.0
        pose_stamped.pose.position.z = 0.8
        pose_stamped.pose.orientation.w = 1.0
        
        return pose_stamped


def main(args=None):
    rclpy.init(args=args)
    
    vla_node = VLAPipelineNode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        vla_node.get_logger().info('VLA Pipeline interrupted by user')
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 2: Integration with Unity Visualization

This example demonstrates how to visualize the VLA system in Unity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;
using System.Collections.Generic;

public class VLAUnityVisualizer : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("VLA Topics")]
    public string interpretedCommandTopic = "/interpreted_command";
    public string actionFeedbackTopic = "/action_feedback";
    public string voiceInputTopic = "/voice_input";
    
    [Header("Visualization")]
    public GameObject commandIndicatorPrefab;
    public Transform visualizationParent;
    public float visualizationDuration = 10.0f;
    
    private ROSTCPConnector ros;
    private Dictionary<string, GameObject> activeIndicators = new Dictionary<string, GameObject>();
    
    void Start()
    {
        ros = ROSTCPConnector.instance;
        if (ros == null)
        {
            Debug.LogError("ROSTCPConnector not found in scene! Please add it.");
            return;
        }
        
        // Subscribe to VLA topics
        ros.Subscribe<StringMsg>(interpretedCommandTopic, OnInterpretedCommandReceived);
        ros.Subscribe<StringMsg>(actionFeedbackTopic, OnActionFeedbackReceived);
        
        Debug.Log("VLA Unity Visualizer initialized");
    }
    
    void OnInterpretedCommandReceived(StringMsg msg)
    {
        try
        {
            // Parse the interpretation JSON
            var interpretation = JsonUtility.FromJson<CommandInterpretation>(msg.data);
            
            Debug.Log($"Received interpretation: {interpretation.action_type} for {interpretation.target_object} at {interpretation.target_location}");
            
            // Create visualization based on interpretation
            CreateCommandVisualization(interpretation);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error parsing interpretation: {e.Message}");
        }
    }
    
    void OnActionFeedbackReceived(StringMsg msg)
    {
        Debug.Log($"Action feedback: {msg.data}");
        
        // Update or remove visualization based on feedback
        UpdateActionFeedback(msg.data);
    }
    
    void CreateCommandVisualization(CommandInterpretation interpretation)
    {
        if (commandIndicatorPrefab == null) return;
        
        // Create a unique ID for this command visualization
        string cmdId = System.Guid.NewGuid().ToString();
        
        GameObject indicator = Instantiate(commandIndicatorPrefab, visualizationParent);
        indicator.name = $"CommandIndicator_{interpretation.action_type}_{cmdId}";
        
        // Configure the indicator based on interpretation
        CommandIndicator indicatorComponent = indicator.GetComponent<CommandIndicator>();
        if (indicatorComponent != null)
        {
            indicatorComponent.Configure(interpretation);
        }
        
        // Schedule destruction after duration
        StartCoroutine(DestroyAfterDelay(indicator, visualizationDuration));
        
        // Add to active indicators
        activeIndicators[cmdId] = indicator;
    }
    
    void UpdateActionFeedback(string feedback)
    {
        // In this example, we'll just log the feedback
        // In a more complex implementation, we could update visual indicators
        Debug.Log($"Action feedback received: {feedback}");
    }
    
    System.Collections.IEnumerator DestroyAfterDelay(GameObject obj, float delay)
    {
        yield return new WaitForSeconds(delay);
        if (obj != null && activeIndicators.ContainsValue(obj))
        {
            string keyToRemove = "";
            foreach(var kvp in activeIndicators)
            {
                if (kvp.Value == obj)
                {
                    keyToRemove = kvp.Key;
                    break;
                }
            }
            
            if (!string.IsNullOrEmpty(keyToRemove))
            {
                activeIndicators.Remove(keyToRemove);
            }
            
            Destroy(obj);
        }
    }
    
    // Send a test voice command
    void Update()
    {
        // For testing: send a voice command when pressing spacebar
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SendTestVoiceCommand("Go to the kitchen and bring me the red cup");
        }
    }
    
    public void SendTestVoiceCommand(string command)
    {
        if (ros != null)
        {
            var msg = new StringMsg(command);
            ros.Publish(voiceInputTopic, msg);
            Debug.Log($"Sent test voice command: {command}");
        }
    }
}

[System.Serializable]
public class CommandInterpretation
{
    public string raw_command;
    public string action_type;
    public string target_object;
    public string target_location;
    public MotionParams motion_parameters;
    public float confidence;
}

[System.Serializable]
public class MotionParams
{
    public string speed;
    public string caution;
}

// Example indicator component
public class CommandIndicator : MonoBehaviour
{
    public TextMesh actionText;
    public Renderer indicatorRenderer;
    public Color navigationColor = Color.blue;
    public Color manipulationColor = Color.green;
    public Color socialColor = Color.yellow;
    
    public void Configure(CommandInterpretation interpretation)
    {
        // Set the action text
        if (actionText != null)
        {
            actionText.text = $"{interpretation.action_type}\n{interpretation.target_object ?? "N/A"}\nto {interpretation.target_location ?? "N/A"}";
        }
        
        // Set the color based on action type
        if (indicatorRenderer != null)
        {
            switch (interpretation.action_type)
            {
                case "navigation":
                    indicatorRenderer.material.color = navigationColor;
                    break;
                case "manipulation_grasp":
                case "manipulation_place":
                    indicatorRenderer.material.color = manipulationColor;
                    break;
                case "social_interaction":
                    indicatorRenderer.material.color = socialColor;
                    break;
                default:
                    indicatorRenderer.material.color = Color.grey;
                    break;
            }
        }
        
        // Position the indicator in a visible location
        // This could be based on the target location in a more advanced implementation
        transform.position = Camera.main.transform.position + Camera.main.transform.forward * 3.0f;
    }
}
```

## Example 3: Advanced VLA with Context Awareness

This example demonstrates a more sophisticated VLA system with context awareness:

```python
#!/usr/bin/env python3
"""
Advanced VLA example with context awareness, memory, and adaptive behavior
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PointStamped
from shu_msgs.msg import ObjectDetectionArray, RobotState
from shu_msgs.action import HumanoidNavigation, HumanoidManipulation
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SpatialMemoryItem:
    """
    Represents a remembered object or location with spatial context
    """
    id: str
    type: str
    last_seen_pose: PoseStamped
    last_seen_time: float
    confidence: float = 1.0
    attributes: Dict[str, str] = None


class ContextAwareVLANode(Node):
    """
    Context-aware VLA system that remembers objects, locations, 
    and adapts to changing situations
    """
    def __init__(self):
        super().__init__('context_aware_vla_node')
        
        # Initialize context and memory
        self.spatial_memory = {}  # Dictionary of SpatialMemoryItem
        self.robot_state = RobotState()
        self.last_command_time = time.time()
        self.conversation_context = {}  # Remember recent conversation
        
        # QoS for sensor data
        sensor_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        
        # Publishers/subscribers
        self.voice_input_sub = self.create_subscription(
            String, 'voice_input', self.voice_callback, 10
        )
        
        self.object_detection_sub = self.create_subscription(
            ObjectDetectionArray, 'object_detections', self.detection_callback, sensor_qos
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, sensor_qos
        )
        
        self.robot_state_sub = self.create_subscription(
            RobotState, 'robot_state', self.robot_state_callback, sensor_qos
        )
        
        self.interpretation_pub = self.create_publisher(String, 'interpreted_command', 10)
        self.action_feedback_pub = self.create_publisher(String, 'action_feedback', 10)
        
        # Action clients
        self.nav_client = ActionClient(self, HumanoidNavigation, 'humanoid_navigate_to_pose')
        self.manip_client = ActionClient(self, HumanoidManipulation, 'humanoid_perform_manipulation')
        
        # Timer for periodic memory cleanup
        self.memory_cleanup_timer = self.create_timer(30.0, self.cleanup_old_memory)
        
        self.get_logger().info('Context-aware VLA Node initialized')

    def detection_callback(self, msg):
        """Update spatial memory with new object detections"""
        for detection in msg.detections:
            # Create/update spatial memory item
            memory_item = SpatialMemoryItem(
                id=detection.id,
                type=detection.type,
                last_seen_pose=detection.pose,
                last_seen_time=time.time(),
                confidence=detection.confidence,
                attributes=detection.attributes
            )
            
            self.spatial_memory[detection.id] = memory_item
            self.get_logger().debug(f'Updated spatial memory with {detection.type}: {detection.id}')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        # Update internal representation of joint states
        # This could be used for self-monitoring or to check if holding an object
        pass

    def robot_state_callback(self, msg):
        """Update robot state information"""
        self.robot_state = msg

    def voice_callback(self, msg):
        """Process voice command with context awareness"""
        try:
            command_text = msg.data
            self.get_logger().info(f'Received voice command with context: {command_text}')
            
            # Get current context
            context = self._build_context()
            
            # Interpret command with context
            interpretation = self.interpret_language_command_with_context(command_text, context)
            
            if interpretation:
                # Publish interpretation
                interpretation_msg = String()
                interpretation_msg.data = json.dumps(interpretation)
                self.interpretation_pub.publish(interpretation_msg)
                
                # Execute action
                success = self.execute_robot_action(interpretation)
                
                # Provide feedback
                feedback_msg = String()
                feedback_msg.data = f"Command '{command_text}' executed: {success}"
                self.action_feedback_pub.publish(feedback_msg)
                
                # Update conversation context
                self.conversation_context['last_command'] = command_text
                self.conversation_context['last_interpretation'] = interpretation
                self.conversation_context['last_success'] = success
                
            else:
                self.get_logger().error(f'Could not interpret command with context: {command_text}')
                
        except Exception as e:
            self.get_logger().error(f'Error in context-aware VLA pipeline: {str(e)}')

    def _build_context(self):
        """Build current context from internal state"""
        context = {
            'robot_pose': self.robot_state.current_pose if hasattr(self, 'robot_state') else None,
            'detected_objects': self.spatial_memory,
            'robot_capabilities': {
                'navigation': True,
                'manipulation': self.robot_state.manipulation_available if hasattr(self, 'robot_state') else True,
                'current_object_held': self.robot_state.object_held if hasattr(self, 'robot_state') else None
            },
            'time_of_day': self._get_time_of_day(),
            'recent_commands': self.conversation_context.get('last_command', ''),
            'conversation_history': self.conversation_context
        }
        
        return context

    def _get_time_of_day(self):
        """Simple time of day classification"""
        current_hour = time.localtime().tm_hour
        if 6 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 17:
            return "afternoon"
        elif 17 <= current_hour < 22:
            return "evening"
        else:
            return "night"

    def interpret_language_command_with_context(self, command_text, context):
        """Interpret command using context information"""
        try:
            interpretation = {
                "raw_command": command_text,
                "action_type": self._classify_action_type(command_text),
                "target_object": self._resolve_target_object(command_text, context),
                "target_location": self._resolve_target_location(command_text, context),
                "motion_parameters": self._extract_motion_params(command_text),
                "context_utilized": True,
                "relevant_objects_nearby": self._find_relevant_objects(command_text, context),
                "confidence": 0.9  # Higher confidence due to context use
            }
            
            return interpretation
            
        except Exception as e:
            self.get_logger().error(f'Error in contextual interpretation: {str(e)}')
            return None

    def _resolve_target_object(self, command_text, context):
        """Resolve target object using context and spatial memory"""
        # First, try to extract object from text
        potential_objects = self._extract_potential_objects(command_text)
        
        if not potential_objects:
            # If no explicit object, check for pronominal references
            if any(pronoun in command_text.lower() for pronoun in ["it", "that", "this", "the one"]):
                # Look for recently referenced object in conversation context
                last_obj = self.conversation_context.get('last_interpretation', {}).get('target_object')
                if last_obj:
                    return last_obj
        
        # Find the best matching object in spatial memory
        for obj_desc in potential_objects:
            for mem_id, memory_item in context['detected_objects'].items():
                if obj_desc.lower() in memory_item.type.lower() or obj_desc.lower() in (memory_item.attributes or {}).get('color', '').lower():
                    # Check if object is reachable
                    if self._is_object_reachable(memory_item, context):
                        return memory_item.id
        
        # If no close match, return the first potential object
        return potential_objects[0] if potential_objects else None

    def _resolve_target_location(self, command_text, context):
        """Resolve target location using context and map information"""
        # Extract potential locations from text
        potential_locations = self._extract_potential_locations(command_text)
        
        if not potential_locations:
            # Check for pronominal references to locations
            if any(word in command_text.lower() for word in ["there", "here", "that place"]):
                last_loc = self.conversation_context.get('last_interpretation', {}).get('target_location')
                if last_loc:
                    return last_loc
        
        # For this example, just return the first potential location
        # In a real implementation, this would use map information
        return potential_locations[0] if potential_locations else None

    def _find_relevant_objects(self, command_text, context):
        """Find objects that might be relevant to the command"""
        relevant = []
        
        for obj_id, memory_item in context['detected_objects'].items():
            # Check if object type or attributes match command
            if any(obj_term in command_text.lower() for obj_term in [memory_item.type.lower()] + 
                  [attr.lower() for attr in (memory_item.attributes or {}).values()]):
                # Include object if it's nearby (less than 3 meters)
                if self._is_object_nearby(memory_item, context):
                    relevant.append({
                        'id': obj_id,
                        'type': memory_item.type,
                        'distance': self._calculate_distance_to_object(memory_item, context)
                    })
        
        return relevant

    def _is_object_reachable(self, memory_item, context):
        """Check if an object is physically reachable by the robot"""
        if not context.get('robot_pose'):
            return True  # If no robot pose, assume reachable
        
        robot_pos = context['robot_pose'].pose.position
        obj_pos = memory_item.last_seen_pose.pose.position
        
        # Calculate distance
        dx = robot_pos.x - obj_pos.x
        dy = robot_pos.y - obj_pos.y
        dz = robot_pos.z - obj_pos.z
        distance = (dx*dx + dy*dy + dz*dz)**0.5
        
        # For humanoid, assume reachability within 2 meters for manipulation
        return distance < 2.0

    def _is_object_nearby(self, memory_item, context):
        """Check if an object is nearby (for relevance)"""
        if not context.get('robot_pose'):
            return True
        
        robot_pos = context['robot_pose'].pose.position
        obj_pos = memory_item.last_seen_pose.pose.position
        
        # Calculate distance
        dx = robot_pos.x - obj_pos.x
        dy = robot_pos.y - obj_pos.y
        dz = robot_pos.z - obj_pos.z
        distance = (dx*dx + dy*dy + dz*dz)**0.5
        
        # Consider "nearby" as within 5 meters
        return distance < 5.0

    def _calculate_distance_to_object(self, memory_item, context):
        """Calculate distance to an object"""
        if not context.get('robot_pose'):
            return float('inf')
        
        robot_pos = context['robot_pose'].pose.position
        obj_pos = memory_item.last_seen_pose.pose.position
        
        dx = robot_pos.x - obj_pos.x
        dy = robot_pos.y - obj_pos.y
        dz = robot_pos.z - obj_pos.z
        
        return (dx*dx + dy*dy + dz*dz)**0.5

    def cleanup_old_memory(self):
        """Remove old spatial memory entries"""
        current_time = time.time()
        expired_items = []
        
        for obj_id, memory_item in self.spatial_memory.items():
            # Remove entries older than 5 minutes
            if current_time - memory_item.last_seen_time > 300:
                expired_items.append(obj_id)
        
        for item_id in expired_items:
            del self.spatial_memory[item_id]
            self.get_logger().debug(f'Removed expired memory entry: {item_id}')

    def _extract_potential_objects(self, command_text):
        """Extract potential object identifiers from command text"""
        # In a real implementation, this would use more sophisticated NLP
        words = command_text.lower().split()
        objects = []
        
        # Common object types
        object_types = ["cup", "bottle", "book", "box", "ball", "apple", "pen", "tablet", 
                       "phone", "keys", "glasses", "laptop"]
        
        for word in words:
            if word in object_types:
                # Check if preceded by an adjective
                idx = words.index(word)
                if idx > 0:
                    adjective = words[idx-1]
                    objects.append(f"{adjective} {word}")
                else:
                    objects.append(word)
        
        return objects

    def _extract_potential_locations(self, command_text):
        """Extract potential location identifiers from command text"""
        words = command_text.lower().split()
        locations = []
        
        # Common locations
        location_names = ["kitchen", "living room", "bedroom", "office", "bathroom", "dining room", 
                         "hallway", "garage", "garden", "table", "counter", "couch", "chair", "bed"]
        
        for word in words:
            if word in location_names:
                locations.append(word)
        
        return locations

    # Other methods (_classify_action_type, _extract_motion_params, execute_robot_action, etc.)
    # would be similar to the previous example but with enhanced context awareness


def main(args=None):
    rclpy.init(args=args)
    
    vla_node = ContextAwareVLANode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        vla_node.get_logger().info('Context-aware VLA interrupted by user')
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 4: Unity-ROS Integration for VLA Simulation

This example demonstrates how to create a Unity visualization that works with the VLA system:

```csharp title="VLAIntegrationExample.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using System.Collections.Generic;
using System.Linq;

public class VLAIntegrationExample : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIp = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("VLA Topics")]
    public string[] visualizationTopics = {
        "/interpreted_command",
        "/action_feedback", 
        "/object_detections",
        "/robot_pose"
    };
    
    [Header("Prefabs")]
    public GameObject commandIndicatorPrefab;
    public GameObject objectMarkerPrefab;
    public GameObject robotModel;
    
    [Header("Visualization Settings")]
    public float visualizationLifetime = 10.0f;
    public float objectDetectionRadius = 0.5f;
    
    private ROSTCPConnector ros;
    private Dictionary<string, GameObject> activeMarkers = new Dictionary<string, GameObject>();
    private Dictionary<string, float> markerCreationTimes = new Dictionary<string, float>();
    
    void Start()
    {
        ros = ROSTCPConnector.instance;
        if (ros == null)
        {
            Debug.LogError("ROSTCPConnector not found! Add it to your scene.");
            return;
        }
        
        // Subscribe to all VLA topics
        ros.Subscribe<StringMsg>("/interpreted_command", OnInterpretedCommand);
        ros.Subscribe<StringMsg>("/action_feedback", OnActionFeedback);
        ros.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.geometry_msgs.PoseStamped>("/robot_pose", OnRobotPoseUpdate);
        
        Debug.Log("VLA Integration Example started");
    }
    
    void OnInterpretedCommand(StringMsg msg)
    {
        try 
        {
            // Parse the interpretation JSON
            VLAInterpretation interpretation = JsonUtility.FromJson<VLAInterpretation>(msg.data);
            
            // Create a visualization of the interpretation
            CreateInterpretationVisualization(interpretation);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error parsing VLA interpretation: {e.Message}");
        }
    }
    
    void OnActionFeedback(StringMsg msg)
    {
        Debug.Log($"Action Feedback: {msg.data}");
        
        // This could update existing visualizations
        // For simplicity, we'll just log it
    }
    
    void OnRobotPoseUpdate(Unity.Robotics.ROSTCPConnector.MessageTypes.geometry_msgs.PoseStamped msg)
    {
        // Update robot model position in Unity
        if (robotModel != null)
        {
            robotModel.transform.position = new Vector3((float)msg.pose.position.x, 
                                                       (float)msg.pose.position.y, 
                                                       (float)msg.pose.position.z);
                                                       
            robotModel.transform.rotation = new Quaternion((float)msg.pose.orientation.x,
                                                          (float)msg.pose.orientation.y,
                                                          (float)msg.pose.orientation.z,
                                                          (float)msg.pose.orientation.w);
        }
    }
    
    void CreateInterpretationVisualization(VLAInterpretation interpretation)
    {
        if (commandIndicatorPrefab == null) return;
        
        // Create an indicator for the interpretation
        GameObject indicator = Instantiate(commandIndicatorPrefab);
        
        // Name the indicator descriptively
        indicator.name = $"VLA_Cmd_{interpretation.action_type}_{System.DateTime.Now.Ticks}";
        
        // Position based on target if available
        // In a real implementation, this would use spatial context
        indicator.transform.position = Camera.main.transform.position + Camera.main.transform.forward * 3f;
        
        // Configure the indicator with interpretation data
        VLACommandIndicator indicatorComp = indicator.GetComponent<VLACommandIndicator>();
        if (indicatorComp != null)
        {
            indicatorComp.SetInterpretation(interpretation);
        }
        
        // Store reference and creation time for cleanup
        string indicatorId = indicator.GetInstanceID().ToString();
        activeMarkers[indicatorId] = indicator;
        markerCreationTimes[indicatorId] = Time.time;
        
        Debug.Log($"Created visualization for command: {interpretation.action_type}");
    }
    
    void Update()
    {
        // Clean up old visualizations
        CleanupOldVisualizations();
        
        // For testing, you can send sample commands
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            SendSampleCommand("Go to the kitchen and bring me the red cup");
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            SendSampleCommand("Navigate to the living room");
        }
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            SendSampleCommand("Pick up the blue ball");
        }
    }
    
    void CleanupOldVisualizations()
    {
        List<string> expiredIds = new List<string>();
        
        foreach (var kvp in markerCreationTimes)
        {
            if (Time.time - kvp.Value > visualizationLifetime)
            {
                expiredIds.Add(kvp.Key);
                
                // Destroy the marker if it exists
                if (activeMarkers.ContainsKey(kvp.Key) && activeMarkers[kvp.Key] != null)
                {
                    Destroy(activeMarkers[kvp.Key]);
                }
            }
        }
        
        // Remove expired entries
        foreach (string id in expiredIds)
        {
            markerCreationTimes.Remove(id);
            activeMarkers.Remove(id);
        }
    }
    
    public void SendSampleCommand(string command)
    {
        if (ros != null)
        {
            ros.Publish("/voice_input", new StringMsg(command));
            Debug.Log($"Sent sample command: {command}");
        }
    }
}

// Helper class to hold VLA interpretation data
[System.Serializable]
public class VLAInterpretation
{
    public string raw_command;
    public string action_type;
    public string target_object;
    public string target_location;
    public MotionParams motion_parameters;
    public bool context_utilized;
    public RelevantObject[] relevant_objects_nearby;
    public float confidence;
}

[System.Serializable]
public class MotionParams
{
    public string speed;
    public string caution;
}

[System.Serializable]
public class RelevantObject
{
    public string id;
    public string type;
    public float distance;
}

// Component for visualization indicators
public class VLACommandIndicator : MonoBehaviour
{
    public TextMesh commandText;
    public Renderer indicatorRenderer;
    public GameObject highlightSphere;
    
    public void SetInterpretation(VLAInterpretation interpretation)
    {
        if (commandText != null)
        {
            commandText.text = $"{interpretation.action_type}\n{interpretation.target_object}\nto {interpretation.target_location}";
        }
        
        // Color code based on action type
        if (indicatorRenderer != null)
        {
            switch (interpretation.action_type)
            {
                case "navigation":
                    indicatorRenderer.material.color = Color.blue;
                    break;
                case "manipulation_grasp":
                case "manipulation_place":
                    indicatorRenderer.material.color = Color.green;
                    break;
                case "social_interaction":
                    indicatorRenderer.material.color = Color.yellow;
                    break;
                default:
                    indicatorRenderer.material.color = Color.gray;
                    break;
            }
        }
        
        // Show highlight if confidence is high
        if (highlightSphere != null)
        {
            highlightSphere.SetActive(interpretation.confidence > 0.8);
        }
    }
}
```

## Exercise Set 3: Integration and Validation

### Exercise 3.1: End-to-End VLA Pipeline Test

**Objective**: Create a complete test that validates the entire VLA pipeline from voice input to action execution.

**Requirements**:
1. Create a ROS 2 launch file that starts all necessary components
2. Execute a sequence of complex voice commands
3. Monitor and validate the robot's responses
4. Measure performance metrics (accuracy, latency, etc.)

**Implementation**:
- Use the examples above to create a complete VLA pipeline
- Create test scenarios that validate each component
- Implement logging and metrics collection
- Design recovery procedures for failed commands

This completes the examples and code samples for the VLA module. These examples provide a comprehensive foundation for implementing Vision-Language-Action systems in Physical AI applications with humanoid robots, including both the ROS-based backend execution and Unity-based visualization components.