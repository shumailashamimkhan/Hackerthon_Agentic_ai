---
title: ROS Action Translation Examples
sidebar_position: 5
---

# ROS Action Translation Examples

## Introduction to ROS Action Translation in Physical AI

Translating high-level commands from cognitive systems (often based on LLM outputs) into executable ROS actions is a critical component of Physical AI systems. This translation process bridges the gap between abstract intentions and concrete robot behaviors, enabling humanoid robots to execute complex tasks expressed in natural language.

### The Translation Challenge

The translation process involves several key challenges:

1. **Semantic Gap**: Converting high-level goals ("bring me the red cup") to low-level commands
2. **Embodiment Constraints**: Accounting for robot-specific capabilities and limitations
3. **Environmental Context**: Adapting plans to current environmental conditions
4. **Safety Requirements**: Ensuring all actions are safe in the physical environment
5. **Temporal Coordination**: Orchestrating multiple simultaneous activities

## ROS Action Architecture for Physical AI

### Action Definition Structure

In ROS 2, actions are defined as a combination of three message types:

1. **Goal**: Requested action parameters
2. **Result**: Outcome of completed action
3. **Feedback**: Intermediate status updates during execution

For Physical AI applications, we often define custom action types:

```yaml
# HumanoidNavigation.action
# Goal - specifies the target pose
geometry_msgs/PoseStamped target_pose

# Result - specifies actual achieved pose and status
geometry_msgs/PoseStamped achieved_pose
builtin_interfaces/Duration execution_time
string outcome  # "success", "partial_success", "failure", "aborted"

# Feedback - provides updates during navigation
string status
float32 distance_remaining
geometry_msgs/PoseStamped current_pose
```

```yaml
# HumanoidManipulation.action
# Goal - specifies the manipulation task
string manipulation_type  # "grasp", "place", "handover", "press_button", etc.
geometry_msgs/PoseStamped target_pose
string target_object_id
string gripper_name

# Result - specifies the outcome
bool success
string object_status  # "held", "placed", "released"
sensor_msgs/JointState final_joint_state

# Feedback - provides updates during manipulation
string phase  # "approaching", "grasping", "lifting", "executing", "retracting"
geometry_msgs/PoseStamped current_ee_pose
float32 progress  # 0.0 to 1.0
```

### Action Servers for Humanoid Robotics

```python title="humanoid_navigation_action_server.py"
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
import time
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import the custom action
from shu_msgs.action import HumanoidNavigation
from nav2_msgs.action import NavigateToPose
from std_srvs.srv import Trigger


class HumanoidNavigationActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_action_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            HumanoidNavigation,
            'humanoid_navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Create client to underlying navigation system
        self.nav_client = self.create_client(NavigateToPose, 'navigate_to_pose')
        
        # Robot state and capabilities
        self.robot_pose = None
        self.is_moving = False
        self.navigation_speed = 0.5  # m/s
        
        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, 'navigation_status', 10)
        
        self.get_logger().info('Humanoid Navigation Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject incoming navigation goals"""
        self.get_logger().info(f'Received navigation goal: {goal_request.target_pose}')
        
        # Check if target is within robot's capabilities
        if self._is_valid_target(goal_request.target_pose):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject goal cancellation requests"""
        self.get_logger().info('Received request to cancel navigation goal')
        return CancelResponse.ACCEPT

    def _is_valid_target(self, target_pose):
        """Check if target pose is valid for navigation"""
        # Check if target is within robot workspace
        # For simplicity, check if z-coordinate is reasonable for bipedal locomotion
        if target_pose.pose.position.z > 0.5:  # Too high for humanoid to navigate
            return False
        
        # Check if target isn't too far
        if self.robot_pose:
            dist = self._calculate_distance(self.robot_pose, target_pose.pose)
            if dist > 50.0:  # Limit navigation distance
                return False
        
        return True

    def _calculate_distance(self, pose1, pose2):
        """Calculate 3D distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        dz = pose1.position.z - pose2.position.z
        return (dx*dx + dy*dy + dz*dz)**0.5

    def execute_callback(self, goal_handle):
        """Execute the navigation goal"""
        self.get_logger().info('Executing navigation goal...')
        
        # Get goal information
        target_pose = goal_handle.request.target_pose
        feedback_msg = HumanoidNavigation.Feedback()
        
        # Check if navigation is possible
        if not self.nav_client.service_is_ready():
            self.get_logger().error('Navigation service not available')
            result = HumanoidNavigation.Result()
            result.outcome = 'failure'
            result.execution_time = rclpy.duration.Duration(seconds=0).to_msg()
            goal_handle.abort()
            return result
        
        # Send goal to underlying navigation system
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = target_pose
        
        # Create a future to track the result
        self.nav_future = self.nav_client.send_async_goal(nav_goal)
        
        # Track execution
        start_time = time.time()
        current_distance = float('inf')
        
        while rclpy.ok():
            # Check if goal was cancelled
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Navigation goal cancelled')
                goal_handle.canceled()
                result = HumanoidNavigation.Result()
                result.outcome = 'cancelled'
                result.execution_time = rclpy.duration.Duration(seconds=time.time() - start_time).to_msg()
                return result
            
            # Publish feedback
            feedback_msg.status = f'Navigating to {target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f}'
            feedback_msg.distance_remaining = current_distance
            feedback_msg.current_pose = self._get_current_pose()
            
            goal_handle.publish_feedback(feedback_msg)
            
            # Check if navigation succeeded
            if self.nav_future.done():
                nav_result = self.nav_future.result()
                if nav_result.result.status == 1:  # SUCCESS
                    self.get_logger().info('Navigation completed successfully')
                    goal_handle.succeed()
                    result = HumanoidNavigation.Result()
                    result.achieved_pose = self._get_current_pose()
                    result.outcome = 'success'
                    result.execution_time = rclpy.duration.Duration(seconds=time.time() - start_time).to_msg()
                    return result
                else:
                    self.get_logger().error('Navigation failed')
                    goal_handle.abort()
                    result = HumanoidNavigation.Result()
                    result.achieved_pose = self._get_current_pose()
                    result.outcome = 'failure'
                    result.execution_time = rclpy.duration.Duration(seconds=time.time() - start_time).to_msg()
                    return result
            
            # Sleep briefly to allow other callbacks to execute
            time.sleep(0.1)

    def _get_current_pose(self):
        """Get robot's current pose (placeholder - would interface with localization system)"""
        # This would typically come from a localization system like AMCL or SLAM
        current_pose = PoseStamped()
        current_pose.header.frame_id = 'map'
        current_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Placeholder values - in reality would come from localization
        current_pose.pose.position.x = 0.0
        current_pose.pose.position.y = 0.0
        current_pose.pose.position.z = 0.0
        current_pose.pose.orientation.x = 0.0
        current_pose.pose.orientation.y = 0.0
        current_pose.pose.orientation.z = 0.0
        current_pose.pose.orientation.w = 1.0
        
        return current_pose


def main(args=None):
    rclpy.init(args=args)
    
    navigation_server = HumanoidNavigationActionServer()
    
    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(navigation_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        navigation_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

```python title="humanoid_manipulation_action_server.py"
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
import time
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

# Import the custom action
from shu_msgs.action import HumanoidManipulation


class HumanoidManipulationActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_manipulation_action_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            HumanoidManipulation,
            'humanoid_perform_manipulation',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Clients for underlying control systems
        self.arm_client = self.create_client(FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.hand_client = self.create_client(FollowJointTrajectory, '/hand_controller/follow_joint_trajectory')
        
        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, 'manipulation_status', 10)
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        # Internal state
        self.current_joint_states = JointState()
        self.is_manipulating = False
        
        self.get_logger().info('Humanoid Manipulation Action Server initialized')

    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.current_joint_states = msg

    def goal_callback(self, goal_request):
        """Accept or reject incoming manipulation goals"""
        self.get_logger().info(f'Received manipulation goal: {goal_request.manipulation_type} for {goal_request.target_object_id}')
        
        # Check if manipulation type is supported
        supported_types = ['grasp', 'place', 'handover', 'press_button', 'rotate_object']
        if goal_request.manipulation_type in supported_types:
            # Check if target is reachable
            if self._is_manipulation_feasible(goal_request):
                return GoalResponse.ACCEPT
            else:
                self.get_logger().warn('Manipulation not feasible')
                return GoalResponse.REJECT
        else:
            self.get_logger().error(f'Unsupported manipulation type: {goal_request.manipulation_type}')
            return GoalResponse.REJECT

    def _is_manipulation_feasible(self, goal_request):
        """Check if manipulation is physically feasible"""
        # Check if target pose is within reach
        # This would typically interface with inverse kinematics
        if goal_request.target_pose:
            # Simplified check - in reality, would use IK to verify reachability
            target_pos = goal_request.target_pose.pose.position
            if abs(target_pos.x) > 1.0 or abs(target_pos.y) > 0.8 or target_pos.z > 1.8:
                # Target is outside typical humanoid reach envelope
                return False
        
        return True

    def cancel_callback(self, goal_handle):
        """Accept or reject goal cancellation requests"""
        self.get_logger().info('Received request to cancel manipulation goal')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the manipulation goal"""
        self.get_logger().info(f'Executing {goal_handle.request.manipulation_type} manipulation...')
        
        # Get goal information
        manipulation_type = goal_handle.request.manipulation_type
        target_pose = goal_handle.request.target_pose
        target_object_id = goal_handle.request.target_object_id
        gripper_name = goal_handle.request.gripper_name
        
        feedback_msg = HumanoidManipulation.Feedback()
        
        # Check if controllers are available
        if not (self.arm_client.service_is_ready() and self.hand_client.service_is_ready()):
            self.get_logger().error('Arm or hand controllers not available')
            result = HumanoidManipulation.Result()
            result.success = False
            goal_handle.abort()
            return result
        
        # Depending on manipulation type, execute appropriate sequence
        success = False
        
        if manipulation_type == 'grasp':
            success = self._execute_grasp_sequence(target_pose, target_object_id, gripper_name, feedback_msg, goal_handle)
        elif manipulation_type == 'place':
            success = self._execute_place_sequence(target_pose, target_object_id, gripper_name, feedback_msg, goal_handle)
        elif manipulation_type == 'handover':
            success = self._execute_handover_sequence(target_pose, target_object_id, gripper_name, feedback_msg, goal_handle)
        else:
            self.get_logger().error(f'Unknown manipulation type: {manipulation_type}')
            result = HumanoidManipulation.Result()
            result.success = False
            goal_handle.abort()
            return result
        
        # Return result
        if success:
            goal_handle.succeed()
            result = HumanoidManipulation.Result()
            result.success = True
            result.object_status = 'held' if manipulation_type == 'grasp' else 'placed'
            result.final_joint_state = self.current_joint_states
            return result
        else:
            goal_handle.abort()
            result = HumanoidManipulation.Result()
            result.success = False
            result.object_status = 'none'
            result.final_joint_state = self.current_joint_states
            return result

    def _execute_grasp_sequence(self, target_pose, target_object_id, gripper_name, feedback_msg, goal_handle):
        """Execute grasping sequence"""
        try:
            # Phase 1: Approach
            feedback_msg.phase = 'approaching'
            feedback_msg.progress = 0.2
            goal_handle.publish_feedback(feedback_msg)
            
            # Move arm to approach position (slightly above target)
            approach_pose = PoseStamped()
            approach_pose.pose = target_pose.pose
            approach_pose.pose.position.z += 0.1  # 10cm above object
            self._move_arm_to_pose(approach_pose)
            
            # Phase 2: Descend to grasp
            feedback_msg.phase = 'descending_to_grasp'
            feedback_msg.progress = 0.5
            goal_handle.publish_feedback(feedback_msg)
            
            self._move_arm_to_pose(target_pose)
            
            # Phase 3: Grasp
            feedback_msg.phase = 'grasping'
            feedback_msg.progress = 0.8
            goal_handle.publish_feedback(feedback_msg)
            
            # Close gripper
            self._close_gripper(gripper_name)
            
            # Phase 4: Lift
            feedback_msg.phase = 'lifting_object'
            feedback_msg.progress = 0.9
            goal_handle.publish_feedback(feedback_msg)
            
            # Lift object slightly
            lift_pose = PoseStamped()
            lift_pose.pose = target_pose.pose
            lift_pose.pose.position.z += 0.1  # Lift 10cm
            self._move_arm_to_pose(lift_pose)
            
            # Success
            feedback_msg.phase = 'completed'
            feedback_msg.progress = 1.0
            goal_handle.publish_feedback(feedback_msg)
            
            return True
        except Exception as e:
            self.get_logger().error(f'Grasp sequence failed: {str(e)}')
            return False

    def _move_arm_to_pose(self, target_pose):
        """Move arm to specified pose using IK (simplified implementation)"""
        # This would normally involve calling an IK solver
        # For this example, we'll just send a placeholder trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = ['shoulder_pitch', 'shoulder_yaw', 'elbow_pitch', 'wrist_pitch']  # Placeholder names
        
        # Create a trajectory point (simplified - real implementation would calculate proper joint angles)
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0]  # Placeholder positions
        point.velocities = [0.0, 0.0, 0.0, 0.0]
        point.accelerations = [0.0, 0.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=2, nanosec=0)
        
        trajectory.points = [point]
        
        # Send trajectory to arm controller
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        
        future = self.arm_client.send_async_goal(goal)
        rclpy.spin_until_future_complete(self, future)

    def _close_gripper(self, gripper_name):
        """Close the gripper"""
        # Send command to close gripper
        trajectory = JointTrajectory()
        trajectory.joint_names = [f'{gripper_name}_joint_1', f'{gripper_name}_joint_2']  # Placeholder names
        
        point = JointTrajectoryPoint()
        point.positions = [0.5, 0.5]  # Closed position
        point.velocities = [0.0, 0.0]
        point.time_from_start = Duration(sec=1, nanosec=0)
        
        trajectory.points = [point]
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        
        future = self.hand_client.send_async_goal(goal)
        rclpy.spin_until_future_complete(self, future)


def main(args=None):
    rclpy.init(args=args)
    
    manipulation_server = HumanoidManipulationActionServer()
    
    try:
        rclpy.spin(manipulation_server)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Translation Patterns

### 1. Natural Language to Action Mapping

A key component in Physical AI systems is mapping natural language commands to ROS actions:

```python title="language_to_action_mapper.py"
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import spacy

class ActionType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    FOLLOW = "follow"
    GREET = "greet"
    RETRIEVE = "retrieve"
    PERFORM_TASK = "perform_task"

@dataclass
class ActionCommand:
    action_type: ActionType
    parameters: Dict[str, str]
    confidence: float
    context: Dict[str, str]  # Additional context from multimodal inputs

class LanguageToActionMapper:
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define action patterns
        self.navigation_patterns = [
            r'go to (the )?(?P<location>\w+)',
            r'move to (the )?(?P<location>\w+)',
            r'go (to the )?(?P<location>\w+)',
            r'travel to (the )?(?P<location>\w+)',
            r'walk to (the )?(?P<location>\w+)',
            r'navigate to (the )?(?P<location>\w+)'
        ]
        
        self.manipulation_patterns = [
            r'pick up (the )?(?P<object>\w+)',
            r'grasp (the )?(?P<object>\w+)',
            r'take (the )?(?P<object>\w+)',
            r'get (the )?(?P<object>\w+)',
            r'grab (the )?(?P<object>\w+)',
            r'retrieve (the )?(?P<object>\w+)',
            r'bring me (the )?(?P<object>\w+)',
            r'put (the )?(?P<object>\w+) (on|at|in) (?P<location>\w+)',
            r'place (the )?(?P<object>\w+) (on|at|in) (?P<location>\w+)',
            r'drop (the )?(?P<object>\w+) (on|at|in) (?P<location>\w+)'
        ]
        
        self.social_patterns = [
            r'(wave to|greet|say hello to) (the )?(?P<person>\w+)',
            r'follow (the )?(?P<person>\w+)',
            r'come with (the )?(?P<person>\w+)',
            r'escort (the )?(?P<person>\w+) to (the )?(?P<location>\w+)'
        ]
    
    def parse_command(self, natural_language: str) -> Optional[ActionCommand]:
        """
        Parse natural language command and return corresponding action
        """
        if not self.nlp:
            return None
            
        doc = self.nlp(natural_language.lower())
        
        # Try to match patterns
        for pattern in self.navigation_patterns:
            match = re.search(pattern, natural_language.lower())
            if match:
                location = match.group('location')
                return ActionCommand(
                    action_type=ActionType.NAVIGATE,
                    parameters={'target_location': location},
                    confidence=0.8,
                    context={}
                )
        
        for pattern in self.manipulation_patterns:
            match = re.search(pattern, natural_language.lower())
            if match:
                action = match.group(0).split()[0]  # First word indicates action type
                params = match.groupdict()
                
                if action in ['pick', 'grasp', 'take', 'get', 'grab', 'retrieve', 'bring']:
                    action_type = ActionType.GRASP
                    return ActionCommand(
                        action_type=action_type,
                        parameters=params,
                        confidence=0.7,
                        context={}
                    )
                elif action in ['put', 'place', 'drop']:
                    action_type = ActionType.PLACE
                    return ActionCommand(
                        action_type=action_type,
                        parameters=params,
                        confidence=0.7,
                        context={}
                    )
        
        for pattern in self.social_patterns:
            match = re.search(pattern, natural_language.lower())
            if match:
                params = match.groupdict()
                
                if 'wave' in natural_language or 'greet' in natural_language or 'hello' in natural_language:
                    action_type = ActionType.GREET
                    return ActionCommand(
                        action_type=action_type,
                        parameters=params,
                        confidence=0.8,
                        context={}
                    )
                elif 'follow' in natural_language or 'come with' in natural_language:
                    action_type = ActionType.FOLLOW
                    return ActionCommand(
                        action_type=action_type,
                        parameters=params,
                        confidence=0.8,
                        context={}
                    )
        
        # Use more sophisticated NLP for complex commands
        return self._advanced_nlp_parse(natural_language)
    
    def _advanced_nlp_parse(self, command: str) -> Optional[ActionCommand]:
        """
        Use advanced NLP techniques for complex command parsing
        """
        if not self.nlp:
            return None
            
        doc = self.nlp(command)
        
        # Extract entities and dependencies
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Find root verb
        root = [token for token in doc if token.head == token][0]
        
        # Determine action based on root verb and entities
        if root.lemma_ in ['go', 'move', 'navigate', 'walk', 'travel']:
            location_entities = [ent for ent in entities if ent[1] in ['LOC', 'FAC']]
            if location_entities:
                return ActionCommand(
                    action_type=ActionType.NAVIGATE,
                    parameters={'target_location': location_entities[0][0]},
                    confidence=0.6,
                    context={}
                )
        
        elif root.lemma_ in ['grasp', 'take', 'pick', 'carry', 'hold']:
            object_entities = [ent for ent in entities if ent[1] in ['OBJ', 'PRODUCT']]
            location_entities = [ent for ent in entities if ent[1] in ['LOC', 'FAC']]
            
            params = {}
            if object_entities:
                params['target_object'] = object_entities[0][0]
            if location_entities:
                params['target_location'] = location_entities[0][0]
            
            action_type = ActionType.GRASP if not location_entities else ActionType.RETRIEVE
            return ActionCommand(
                action_type=action_type,
                parameters=params,
                confidence=0.6,
                context={}
            )
        
        return None

# Example usage
def example_command_translation():
    mapper = LanguageToActionMapper()
    
    # Example commands
    commands = [
        "Go to the kitchen",
        "Pick up the red cup",
        "Place the book on the table",
        "Follow the person to the office",
        "Greet the visitor",
        "Bring me the blue pen from the desk"
    ]
    
    for cmd in commands:
        action_cmd = mapper.parse_command(cmd)
        if action_cmd:
            print(f"Command: '{cmd}' -> Action: {action_cmd.action_type}, Params: {action_cmd.parameters}, Confidence: {action_cmd.confidence:.2f}")
        else:
            print(f"Could not parse command: '{cmd}'")

if __name__ == "__main__":
    example_command_translation()
```

### 2. Cognitive Plan to ROS Action Translation

Translating high-level cognitive plans into sequences of ROS actions:

```python title="cognitive_plan_translator.py"
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
from shu_msgs.action import HumanoidNavigation, HumanoidManipulation
from geometry_msgs.msg import PoseStamped
import json

class PrimitiveAction(Enum):
    NAVIGATE_TO_POSE = "navigate_to_pose"
    GRASP_OBJECT = "grasp_object"
    PLACE_OBJECT = "place_object"
    PERFORM_ACTION_SEQUENCE = "perform_action_sequence"
    WAIT = "wait"
    DETECT_OBJECT = "detect_object"

@dataclass 
class PrimitiveStep:
    action_type: PrimitiveAction
    parameters: Dict[str, any]
    expected_outcome: str
    timeout: float = 30.0  # seconds

@dataclass
class CognitivePlanStep:
    step_number: int
    cognitive_action: str  # High-level action like "fetch_object"
    target_object: str
    target_location: str
    constraints: List[str]  # Movement constraints, safety constraints, etc.

class CognitivePlanTranslator:
    def __init__(self, robot_capabilities, environment_model):
        self.robot_capabilities = robot_capabilities
        self.environment_model = environment_model
        
    def translate_cognitive_plan(self, cognitive_plan: List[CognitivePlanStep]) -> List[PrimitiveStep]:
        """
        Translate a cognitive plan into a sequence of primitive ROS actions
        
        Args:
            cognitive_plan: List of high-level cognitive steps
            
        Returns:
            List of executable primitive steps
        """
        primitive_steps = []
        
        for cognitive_step in cognitive_plan:
            steps = self._translate_step(cognitive_step)
            primitive_steps.extend(steps)
        
        return primitive_steps
    
    def _translate_step(self, cognitive_step: CognitivePlanStep) -> List[PrimitiveStep]:
        """Translate a single cognitive step to primitive actions"""
        if cognitive_step.cognitive_action == "fetch_object":
            return self._translate_fetch_object(cognitive_step)
        elif cognitive_step.cognitive_action == "navigate_to_location":
            return self._translate_navigation_to_location(cognitive_step)
        elif cognitive_step.cognitive_action == "perform_task":
            return self._translate_task_performance(cognitive_step)
        else:
            # Unsupported cognitive action
            return [PrimitiveStep(
                action_type=PrimitiveAction.WAIT,
                parameters={'duration': 1.0},
                expected_outcome='No action taken for unsupported cognitive action',
                timeout=5.0
            )]
    
    def _translate_fetch_object(self, step: CognitivePlanStep) -> List[PrimitiveStep]:
        """Translate a fetch object action into primitive steps"""
        primitive_steps = []
        
        # First, navigate to location where object is expected
        if step.target_location:
            target_pose = self.environment_model.get_location_pose(step.target_location)
            primitive_steps.append(PrimitiveStep(
                action_type=PrimitiveAction.NAVIGATE_TO_POSE,
                parameters={'target_pose': target_pose},
                expected_outcome=f'Robot navigated to {step.target_location}',
                timeout=60.0
            ))
        
        # Then detect and identify the target object
        primitive_steps.append(PrimitiveStep(
            action_type=PrimitiveAction.DETECT_OBJECT,
            parameters={'target_object_class': step.target_object},
            expected_outcome=f'{step.target_object} detected in robot\'s field of view',
            timeout=10.0
        ))
        
        # Then grasp the object
        if hasattr(self.robot_capabilities, 'manipulation') and self.robot_capabilities.manipulation:
            primitive_steps.append(PrimitiveStep(
                action_type=PrimitiveAction.GRASP_OBJECT,
                parameters={'object_id': step.target_object},
                expected_outcome=f'{step.target_object} grasped successfully',
                timeout=30.0
            ))
        
        return primitive_steps
    
    def _translate_navigation_to_location(self, step: CognitivePlanStep) -> List[PrimitiveStep]:
        """Translate a navigation action into primitive steps"""
        primitive_steps = []
        
        # Get the target location pose
        target_pose = self.environment_model.get_location_pose(step.target_location)
        
        # Check if the robot needs to navigate through multiple waypoints
        path = self.environment_model.get_navigation_path(
            current_location=self.environment_model.get_robot_location(),
            target_location=step.target_location
        )
        
        # Add a step for each waypoint in the path
        for waypoint in path:
            primitive_steps.append(PrimitiveStep(
                action_type=PrimitiveAction.NAVIGATE_TO_POSE,
                parameters={'target_pose': waypoint},
                expected_outcome=f'Robot navigated to waypoint near {step.target_location}',
                timeout=60.0
            ))
        
        return primitive_steps
    
    def _translate_task_performance(self, step: CognitivePlanStep) -> List[PrimitiveStep]:
        """Translate a complex task performance action"""
        # This would be highly dependent on the specific task
        # For now, we'll implement a few common tasks
        
        if step.target_object == "handover":
            # Perform a handover action (e.g., give object to person)
            return [
                PrimitiveStep(
                    action_type=PrimitiveAction.NAVIGATE_TO_POSE,
                    parameters={'target_pose': self._get_person_pose(step.target_location)},
                    expected_outcome='Robot navigated to person for handover',
                    timeout=60.0
                ),
                PrimitiveStep(
                    action_type=PrimitiveAction.PLACE_OBJECT,
                    parameters={'object_id': 'held_object'},  # Currently held object
                    expected_outcome='Object placed in person\'s hands',
                    timeout=30.0
                )
            ]
        
        elif step.target_object == "press_button":
            # Navigate to location and press a button
            return [
                PrimitiveStep(
                    action_type=PrimitiveAction.NAVIGATE_TO_POSE,
                    parameters={'target_pose': self._get_button_location(step.target_location)},
                    expected_outcome='Robot navigated to button location',
                    timeout=60.0
                ),
                PrimitiveStep(
                    action_type=PrimitiveAction.PERFORM_ACTION_SEQUENCE,
                    parameters={
                        'sub_actions': [
                            {'type': 'move_arm_to_button', 'position': 'button_location'},
                            {'type': 'apply_force', 'magnitude': 10.0},  # Newtons
                            {'type': 'wait', 'duration': 2.0}
                        ]
                    },
                    expected_outcome='Button pressed successfully',
                    timeout=30.0
                )
            ]
        
        # Default: unrecognized task
        return [PrimitiveStep(
            action_type=PrimitiveAction.WAIT,
            parameters={'duration': 1.0},
            expected_outcome='Unrecognized task, no action performed',
            timeout=5.0
        )]
    
    def _get_person_pose(self, location_description: str) -> PoseStamped:
        """Get the pose of a person at the specified location"""
        # This would query the environment model for person locations
        # For now, return a placeholder
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 1.0
        pose.pose.position.y = 1.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        
        return pose
    
    def _get_button_location(self, location_description: str) -> PoseStamped:
        """Get the pose of a button at the specified location"""
        # This would query the environment model for button locations
        # For now, return a placeholder
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 1.5
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.0  # At button height
        pose.pose.orientation.w = 1.0
        
        return pose

# Example of integration with a high-level planner
class IntegratedCognitiveSystem:
    def __init__(self, plan_translator, action_servers):
        self.plan_translator = plan_translator
        self.action_servers = action_servers
        self.current_plan = []
        self.current_step_index = 0
    
    def execute_plan(self, cognitive_plan):
        """Execute a translated cognitive plan"""
        # Translate cognitive plan to primitive steps
        primitive_steps = self.plan_translator.translate_cognitive_plan(cognitive_plan)
        
        # Execute each primitive step
        for i, step in enumerate(primitive_steps):
            self.get_logger().info(f'Executing primitive step {i+1}/{len(primitive_steps)}: {step.action_type}')
            
            success = self._execute_primitive_step(step)
            
            if not success:
                self.get_logger().error(f'Failed to execute primitive step: {step.action_type}')
                # Could implement recovery actions here
                return False
        
        self.get_logger().info('Successfully executed cognitive plan')
        return True
    
    def _execute_primitive_step(self, step: PrimitiveStep):
        """Execute a single primitive step"""
        if step.action_type == PrimitiveAction.NAVIGATE_TO_POSE:
            return self._execute_navigation_step(step.parameters)
        elif step.action_type == PrimitiveAction.GRASP_OBJECT:
            return self._execute_manipulation_step(step.parameters, 'grasp')
        elif step.action_type == PrimitiveAction.PLACE_OBJECT:
            return self._execute_manipulation_step(step.parameters, 'place')
        elif step.action_type == PrimitiveAction.WAIT:
            return self._execute_wait_step(step.parameters)
        elif step.action_type == PrimitiveAction.DETECT_OBJECT:
            return self._execute_detection_step(step.parameters)
        else:
            self.get_logger().error(f'Unknown primitive action: {step.action_type}')
            return False
    
    def _execute_navigation_step(self, params):
        """Execute navigation to pose"""
        # Send navigation action goal
        goal = HumanoidNavigation.Goal()
        goal.target_pose = params['target_pose']
        
        # This would interface with the navigation action server
        # For now, return True to indicate success
        return True
    
    def _execute_manipulation_step(self, params, manipulation_type):
        """Execute manipulation action"""
        # Send manipulation action goal
        goal = HumanoidManipulation.Goal()
        goal.manipulation_type = manipulation_type
        if 'target_pose' in params:
            goal.target_pose = params['target_pose']
        if 'target_object_id' in params:
            goal.target_object_id = params['target_object_id']
        
        # This would interface with the manipulation action server
        return True
    
    def _execute_wait_step(self, params):
        """Execute a wait/delay action"""
        import time
        duration = params.get('duration', 1.0)
        time.sleep(duration)
        return True
    
    def _execute_detection_step(self, params):
        """Execute object detection action"""
        # Interface with perception system to detect objects
        target_class = params.get('target_object_class')
        
        # This would call the perception system
        # For now, return True to indicate success
        return True
```

### 3. Translation Validation and Verification

Implementing a system to validate that translations are appropriate:

```python title="translation_validator.py"
from typing import List, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class TranslationIssue:
    severity: str  # 'critical', 'warning', 'info'
    issue_type: str  # 'missing_capability', 'invalid_parameter', etc.
    description: str
    action_step: Any  # The problematic action step

class TranslationValidator:
    def __init__(self, robot_model, environment_model):
        self.robot_model = robot_model
        self.environment_model = environment_model
    
    def validate_translation(self, primitive_steps: List[PrimitiveStep]) -> List[TranslationIssue]:
        """Validate a sequence of primitive steps for correctness and feasibility"""
        issues = []
        
        for step in primitive_steps:
            step_issues = self._validate_step(step)
            issues.extend(step_issues)
        
        return issues
    
    def _validate_step(self, step: PrimitiveStep) -> List[TranslationIssue]:
        """Validate a single primitive step"""
        issues = []
        
        # Check if robot has capability to perform action
        capability = self._get_capability_for_action(step.action_type)
        if not capability:
            issues.append(TranslationIssue(
                severity='critical',
                issue_type='missing_capability',
                description=f'Robot does not have capability to perform action: {step.action_type}',
                action_step=step
            ))
            return issues  # Cannot proceed if missing capability
        
        # Validate action parameters
        param_issues = self._validate_parameters(step.action_type, step.parameters)
        issues.extend(param_issues)
        
        # Check environmental constraints
        env_issues = self._check_environmental_constraints(step)
        issues.extend(env_issues)
        
        # Check robot state constraints
        state_issues = self._check_state_dependencies(step)
        issues.extend(state_issues)
        
        return issues
    
    def _get_capability_for_action(self, action_type: PrimitiveAction):
        """Check if robot has capability for specified action"""
        capability_mapping = {
            PrimitiveAction.NAVIGATE_TO_POSE: self.robot_model.locmobility,
            PrimitiveAction.GRASP_OBJECT: self.robot_model.manipulation_capability,
            PrimitiveAction.PLACE_OBJECT: self.robot_model.manipulation_capability,
            PrimitiveAction.DETECT_OBJECT: self.robot_model.perception_capability,
            PrimitiveAction.WAIT: True  # All robots can wait
        }
        
        return capability_mapping.get(action_type, False)
    
    def _validate_parameters(self, action_type: PrimitiveAction, parameters: Dict[str, Any]) -> List[TranslationIssue]:
        """Validate action parameters"""
        issues = []
        
        if action_type == PrimitiveAction.NAVIGATE_TO_POSE:
            if 'target_pose' not in parameters:
                issues.append(TranslationIssue(
                    severity='critical',
                    issue_type='missing_parameter',
                    description='Missing target_pose parameter for navigation',
                    action_step=parameters
                ))
        
        elif action_type == PrimitiveAction.GRASP_OBJECT:
            if 'object_id' not in parameters:
                issues.append(TranslationIssue(
                    severity='critical', 
                    issue_type='missing_parameter',
                    description='Missing object_id parameter for grasp action',
                    action_step=parameters
                ))
            
            # Check if object is graspable
            obj_id = parameters['object_id']
            if not self._is_object_graspable(obj_id):
                issues.append(TranslationIssue(
                    severity='warning',
                    issue_type='ungraspable_object',
                    description=f'Object {obj_id} may not be graspable',
                    action_step=parameters
                ))
        
        elif action_type == PrimitiveAction.PLACE_OBJECT:
            if 'target_pose' not in parameters:
                issues.append(TranslationIssue(
                    severity='critical',
                    issue_type='missing_parameter',
                    description='Missing target_pose parameter for place action',
                    action_step=parameters
                ))
        
        return issues
    
    def _is_object_graspable(self, object_id: str) -> bool:
        """Check if object is graspable"""
        # This would query the environment model
        # For now, assume any object with known position is graspable
        return self.environment_model.has_object_info(object_id)
    
    def _check_environmental_constraints(self, step: PrimitiveStep) -> List[TranslationIssue]:
        """Check if environmental conditions allow step execution"""
        issues = []
        
        if step.action_type == PrimitiveAction.NAVIGATE_TO_POSE:
            target_pose = step.parameters.get('target_pose')
            if target_pose:
                if not self.environment_model.is_navigable(target_pose):
                    issues.append(TranslationIssue(
                        severity='critical',
                        issue_type='non_navigable_target',
                        description=f'Target pose {target_pose} is not navigable',
                        action_step=step
                    ))
        
        elif step.action_type == PrimitiveAction.GRASP_OBJECT:
            obj_id = step.parameters.get('object_id')
            if obj_id:
                obj_pose = self.environment_model.get_object_pose(obj_id)
                if obj_pose and not self._is_object_reachable(obj_pose):
                    issues.append(TranslationIssue(
                        severity='critical',
                        issue_type='object_not_reachable',
                        description=f'Object {obj_id} is not reachable by robot',
                        action_step=step
                    ))
        
        return issues
    
    def _is_object_reachable(self, object_pose) -> bool:
        """Check if object is physically reachable by robot"""
        # This would check if the pose is within robot's workspace
        # For now, a simplified check
        robot_pose = self.environment_model.get_robot_pose()
        
        # Calculate distance
        dx = object_pose.position.x - robot_pose.position.x
        dy = object_pose.position.y - robot_pose.position.y
        dz = object_pose.position.z - robot_pose.position.z
        distance = (dx**2 + dy**2 + dz**2)**0.5
        
        # Check against approximate reach
        return distance < self.robot_model.reach_distance
    
    def _check_state_dependencies(self, step: PrimitiveStep) -> List[TranslationIssue]:
        """Check if current robot state allows step execution"""
        issues = []
        
        if step.action_type == PrimitiveAction.PLACE_OBJECT:
            # Check if robot is holding an object
            if not self.robot_model.is_holding_object:
                issues.append(TranslationIssue(
                    severity='critical',
                    issue_type='invalid_state',
                    description='Cannot place object when robot is not holding anything',
                    action_step=step
                ))
        
        elif step.action_type == PrimitiveAction.GRASP_OBJECT:
            # Check if hand is free
            if self.robot_model.is_holding_object:
                issues.append(TranslationIssue(
                    severity='critical',
                    issue_type='invalid_state',
                    description='Cannot grasp object when robot is already holding another object',
                    action_step=step
                ))
        
        return issues

# Example of integration with validation
class ValidatedCognitiveSystem(IntegratedCognitiveSystem):
    def __init__(self, plan_translator, action_servers, validator):
        super().__init__(plan_translator, action_servers)
        self.validator = validator
    
    def execute_plan(self, cognitive_plan):
        """Execute a cognitive plan after validating the translation"""
        # Translate cognitive plan to primitive steps
        primitive_steps = self.plan_translator.translate_cognitive_plan(cognitive_plan)
        
        # Validate the translation
        issues = self.validator.validate_translation(primitive_steps)
        
        # Log any issues
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        if critical_issues:
            self.get_logger().error(f'Critical issues in plan translation: {len(critical_issues)}')
            for issue in critical_issues:
                self.get_logger().error(f'  - {issue.description}')
            return False
        
        warning_issues = [issue for issue in issues if issue.severity == 'warning']
        if warning_issues:
            self.get_logger().warn(f'Warning issues in plan translation: {len(warning_issues)}')
            for issue in warning_issues:
                self.get_logger().warn(f'  - {issue.description}')
        
        # Execute validated steps
        return super().execute_plan(cognitive_plan)
```

## Implementation in Unity

### ROS-Unity Integration for Action Monitoring

Creating a Unity component to monitor and visualize ROS actions:

```csharp title="ActionMonitor.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Actionlib;
using RosMessageTypes.Nav2;
using RosMessageTypes.Std;
using System.Collections.Generic;

public class ActionMonitor : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIpAddress = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("Action Monitoring")]
    public string navigationActionTopic = "navigate_to_pose/_action/status";
    public string manipulationActionTopic = "manipulation/_action/status";
    
    private ROSTCPConnector ros;
    private Dictionary<string, ActionStatusInfo> ongoingActions = new Dictionary<string, ActionStatusInfo>();
    
    [System.Serializable]
    public class ActionStatusInfo
    {
        public string actionName;
        public string status; // PENDING, ACTIVE, PREEMPTED, SUCCEEDED, ABORTED, REJECTED, PREEMPTING, RECALLING, RECALLED, LOST
        public float startTime;
        public float progress;
    }
    
    // Visualization elements
    public GameObject actionIndicatorPrefab;
    private Dictionary<string, GameObject> actionIndicators = new Dictionary<string, GameObject>();
    
    void Start()
    {
        ros = ROSTCPConnector.instance;
        
        if (ros == null)
        {
            Debug.LogError("Could not find ROSTCPConnector in scene! Please add it.");
            return;
        }
        
        // Subscribe to action status topics
        ros.Subscribe<GoalStatusArrayMsg>(navigationActionTopic, OnNavigationStatusReceived);
        ros.Subscribe<GoalStatusArrayMsg>(manipulationActionTopic, OnManipulationStatusReceived);
    }
    
    void OnNavigationStatusReceived(GoalStatusArrayMsg statusArray)
    {
        ProcessActionStatusUpdates(statusArray, "navigation");
    }
    
    void OnManipulationStatusReceived(GoalStatusArrayMsg statusArray)
    {
        ProcessActionStatusUpdates(statusArray, "manipulation");
    }
    
    void ProcessActionStatusUpdates(GoalStatusArrayMsg statusArray, string actionType)
    {
        foreach (var status in statusArray.status_list)
        {
            string actionId = status.goal_id.id;
            string statusText = GetStatusText(status.status);
            
            if (ongoingActions.ContainsKey(actionId))
            {
                // Update existing action status
                ActionStatusInfo existingAction = ongoingActions[actionId];
                existingAction.status = statusText;
                
                // Calculate progress (simplified calculation)
                float timeElapsed = Time.time - existingAction.startTime;
                existingAction.progress = Mathf.Min(1.0f, timeElapsed / 60.0f); // Assume max 60 seconds for any action
                
                UpdateActionIndicator(actionId, existingAction);
            }
            else if (statusText != "SUCCEEDED" && statusText != "ABORTED" && statusText != "REJECTED")
            {
                // New action started
                ActionStatusInfo newAction = new ActionStatusInfo();
                newAction.actionName = $"{actionType}_{actionId}";
                newAction.status = statusText;
                newAction.startTime = Time.time;
                newAction.progress = 0.0f;
                
                ongoingActions[actionId] = newAction;
                CreateActionIndicator(actionId, newAction);
            }
        }
    }
    
    string GetStatusText(byte statusCode)
    {
        switch (statusCode)
        {
            case 0: return "PENDING";
            case 1: return "ACTIVE";
            case 2: return "PREEMPTED";
            case 3: return "SUCCEEDED";
            case 4: return "ABORTED";
            case 5: return "REJECTED";
            case 6: return "PREEMPTING";
            case 7: return "RECALLING";
            case 8: return "RECALLED";
            case 9: return "LOST";
            default: return "UNKNOWN";
        }
    }
    
    void CreateActionIndicator(string actionId, ActionStatusInfo actionInfo)
    {
        if (actionIndicatorPrefab != null)
        {
            GameObject indicator = Instantiate(actionIndicatorPrefab, transform);
            indicator.name = $"ActionIndicator_{actionId}";
            
            // Add your custom visualization logic here
            ActionIndicatorUI ui = indicator.GetComponent<ActionIndicatorUI>();
            if (ui != null)
            {
                ui.SetActionInfo(actionInfo);
            }
            
            actionIndicators[actionId] = indicator;
        }
    }
    
    void UpdateActionIndicator(string actionId, ActionStatusInfo actionInfo)
    {
        if (actionIndicators.ContainsKey(actionId))
        {
            ActionIndicatorUI ui = actionIndicators[actionId].GetComponent<ActionIndicatorUI>();
            if (ui != null)
            {
                ui.SetActionInfo(actionInfo);
            }
        }
    }
    
    void Update()
    {
        // Clean up completed actions
        List<string> completedActions = new List<string>();
        foreach (var actionPair in ongoingActions)
        {
            if (actionPair.Value.status == "SUCCEEDED" || 
                actionPair.Value.status == "ABORTED" || 
                actionPair.Value.status == "REJECTED")
            {
                // Check if this action has been completed for more than 5 seconds
                if (Time.time - actionPair.Value.startTime > 5.0f)
                {
                    completedActions.Add(actionPair.Key);
                    
                    // Remove visualization
                    if (actionIndicators.ContainsKey(actionPair.Key))
                    {
                        Destroy(actionIndicators[actionPair.Key]);
                        actionIndicators.Remove(actionPair.Key);
                    }
                }
            }
        }
        
        // Remove completed actions from tracking
        foreach (string actionId in completedActions)
        {
            ongoingActions.Remove(actionId);
        }
    }
}
```

```csharp title="ActionIndicatorUI.cs" 
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ActionIndicatorUI : MonoBehaviour
{
    public Image progressBar;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI actionNameText;
    public Color activeColor = Color.green;
    public Color inactiveColor = Color.red;
    
    private ActionMonitor.ActionStatusInfo currentAction;
    
    public void SetActionInfo(ActionMonitor.ActionStatusInfo actionInfo)
    {
        currentAction = actionInfo;
        actionNameText.text = actionInfo.actionName;
        statusText.text = actionInfo.status;
        
        if (progressBar != null)
        {
            progressBar.fillAmount = actionInfo.progress;
        }
        
        // Change color based on status
        if (actionInfo.status == "ACTIVE" || actionInfo.status == "SUCCEEDED")
        {
            statusText.color = activeColor;
        }
        else if (actionInfo.status == "ABORTED" || actionInfo.status == "REJECTED")
        {
            statusText.color = inactiveColor;
        }
    }
    
    void Update()
    {
        if (currentAction != null)
        {
            // Update progress bar if it's an active action
            if (progressBar != null && currentAction.status == "ACTIVE")
            {
                // Recalculate progress based on elapsed time
                float elapsed = Time.time - currentAction.startTime;
                float estimatedDuration = 60.0f; // Default assumption
                float recalculatedProgress = Mathf.Min(1.0f, elapsed / estimatedDuration);
                
                // Smooth progress transition to avoid sudden jumps
                progressBar.fillAmount = Mathf.Lerp(progressBar.fillAmount, recalculatedProgress, Time.deltaTime * 2.0f);
            }
        }
    }
}
```

## Best Practices for ROS Action Translation

### 1. Error Handling and Recovery

```python title="robust_action_execution.py"
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from shu_msgs.action import HumanoidManipulation
import time


class RobustManipulationActionServer(Node):
    def __init__(self):
        super().__init__('robust_manipulation_action_server')
        
        # Action server with error handling
        self._action_server = ActionServer(
            self,
            HumanoidManipulation,
            'robust_perform_manipulation',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Retry mechanism parameters
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
        
        self.get_logger().info('Robust Manipulation Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject incoming manipulation goals"""
        self.get_logger().info(f'Received manipulation goal: {goal_request.manipulation_type}')
        
        # Validate goal parameters
        if self._validate_goal(goal_request):
            return GoalResponse.ACCEPT
        else:
            self.get_logger().warn('Goal validation failed')
            return GoalResponse.REJECT

    def _validate_goal(self, goal_request):
        """Validate goal parameters before execution"""
        # Implement validation logic
        if goal_request.manipulation_type not in ['grasp', 'place', 'handover', 'press_button']:
            return False
        
        # Verify target pose is reasonable
        if goal_request.target_pose:
            target_pos = goal_request.target_pose.pose.position
            if abs(target_pos.x) > 1.5 or abs(target_pos.y) > 1.0 or target_pos.z > 2.0:
                self.get_logger().warn('Target pose appears to be out of reach')
                return False
        
        return True

    def execute_callback(self, goal_handle):
        """Execute the manipulation goal with retry mechanism"""
        self.get_logger().info(f'Executing {goal_handle.request.manipulation_type} manipulation with robust execution...')
        
        success = False
        retry_count = 0
        
        # Prepare feedback message
        feedback_msg = HumanoidManipulation.Feedback()
        
        while not success and retry_count < self.max_retries:
            try:
                if retry_count > 0:
                    self.get_logger().info(f'Retrying action (attempt {retry_count + 1} of {self.max_retries})')
                    time.sleep(self.retry_delay)
                
                # Execute the manipulation sequence
                success = self._execute_manipulation_with_feedback(
                    goal_handle.request, 
                    feedback_msg, 
                    goal_handle
                )
                
                if success:
                    self.get_logger().info(f'Action completed successfully after {retry_count + 1} attempts')
                else:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        self.get_logger().warn(f'Action attempt failed, retrying...')
                    else:
                        self.get_logger().error(f'Action failed after {self.max_retries} attempts')
                        
            except Exception as e:
                self.get_logger().error(f'Exception during manipulation execution: {str(e)}')
                retry_count += 1
                if retry_count < self.max_retries:
                    self.get_logger().info(f'Retrying after exception (attempt {retry_count + 1} of {self.max_retries})')
        
        # Return result
        result = HumanoidManipulation.Result()
        if success:
            goal_handle.succeed()
            result.success = True
            result.object_status = self._get_object_status(goal_handle.request.manipulation_type)
        else:
            goal_handle.abort()
            result.success = False
            result.object_status = 'failed'
        
        return result

    def _execute_manipulation_with_feedback(self, request, feedback_msg, goal_handle):
        """Execute manipulation with detailed feedback reporting"""
        try:
            # Phase 1: Planning and preparation
            feedback_msg.phase = 'planning_approach'
            feedback_msg.progress = 0.05
            goal_handle.publish_feedback(feedback_msg)
            
            if not self._is_manipulation_feasible(request):
                self.get_logger().error('Manipulation not feasible')
                return False
            
            # Phase 2: Approach to target
            feedback_msg.phase = 'approaching_target'
            feedback_msg.progress = 0.2
            goal_handle.publish_feedback(feedback_msg)
            
            approach_success = self._execute_approach(request.target_pose)
            if not approach_success:
                self.get_logger().error('Approach to target failed')
                return False
            
            # Phase 3: Execute manipulation
            manipulation_type = request.manipulation_type
            if manipulation_type == 'grasp':
                feedback_msg.phase = 'grasping_object'
                feedback_msg.progress = 0.6
                goal_handle.publish_feedback(feedback_msg)
                
                grasp_success = self._execute_grasp(request)
                return grasp_success
                
            elif manipulation_type == 'place':
                feedback_msg.phase = 'placing_object'
                feedback_msg.progress = 0.6
                goal_handle.publish_feedback(feedback_msg)
                
                place_success = self._execute_place(request)
                return place_success
            
            # Add other manipulation types as needed...
            
            return False  # Unsupported manipulation type
            
        except Exception as e:
            self.get_logger().error(f'Error during manipulation execution: {str(e)}')
            return False

    def _is_manipulation_feasible(self, request):
        """Check if manipulation is physically feasible (implementation placeholder)"""
        # This would typically call IK solvers and other feasibility checks
        return True

    def _execute_approach(self, target_pose):
        """Execute approach to target (implementation placeholder)"""
        # This would move robot to appropriate position for manipulation
        return True

    def _execute_grasp(self, request):
        """Execute grasp operation (implementation placeholder)"""
        # This would handle the actual grasping sequence
        return True

    def _execute_place(self, request):
        """Execute place operation (implementation placeholder)"""
        # This would handle the placing sequence
        return True

    def _get_object_status(self, manipulation_type):
        """Get object status based on manipulation type"""
        if manipulation_type == 'grasp':
            return 'held'
        elif manipulation_type == 'place':
            return 'placed'
        elif manipulation_type == 'handover':
            return 'released'
        else:
            return 'unknown'
```

### 2. Performance Monitoring

```python title="action_performance_monitor.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from builtin_interfaces.msg import Duration
from shu_msgs.msg import ActionPerformance
import time
import statistics

class ActionPerformanceMonitor(Node):
    """
    Monitors ROS action execution performance and reports metrics
    """
    def __init__(self):
        super().__init__('action_performance_monitor')
        
        # Publishers for different metrics
        self.duration_pub = self.create_publisher(Float32, 'action_durations', 10)
        self.success_rate_pub = self.create_publisher(Float32, 'action_success_rates', 10)
        self.performance_pub = self.create_publisher(ActionPerformance, 'action_performance', 10)
        
        # Track action metrics
        self.action_metrics = {}  # Store metrics by action name
        self.running_average_window = 10
        
        # Timer for periodic reporting
        self.reporting_timer = self.create_timer(5.0, self.report_performance)
        
        self.get_logger().info('Action Performance Monitor initialized')

    def log_action_start(self, action_name, goal_id):
        """Log when an action starts executing"""
        if action_name not in self.action_metrics:
            self.action_metrics[action_name] = {
                'attempts': 0,
                'successes': 0,
                'durations': [],
                'recent_durations': []  # Rolling window for recent performance
            }
        
        # Record start time
        self.action_metrics[action_name]['start_times'] = self.action_metrics[action_name].get('start_times', {})
        self.action_metrics[action_name]['start_times'][goal_id] = self.get_clock().now()

    def log_action_completion(self, action_name, goal_id, success):
        """Log when an action completes"""
        if action_name in self.action_metrics:
            metrics = self.action_metrics[action_name]
            
            # Calculate duration
            start_time = metrics['start_times'].get(goal_id)
            if start_time:
                duration = (self.get_clock().now() - start_time).nanoseconds / 1e9  # Convert to seconds
                metrics['durations'].append(duration)
                
                # Keep track of recent durations for performance metrics
                metrics['recent_durations'].append(duration)
                if len(metrics['recent_durations']) > self.running_average_window:
                    metrics['recent_durations'] = metrics['recent_durations'][-self.running_average_window:]
                
                # Remove start time
                del metrics['start_times'][goal_id]
            
            # Update success/failure counts
            metrics['attempts'] += 1
            if success:
                metrics['successes'] += 1

    def report_performance(self):
        """Report action performance metrics periodically"""
        for action_name, metrics in self.action_metrics.items():
            if metrics['attempts'] > 0:
                # Calculate success rate
                success_rate = metrics['successes'] / metrics['attempts']
                
                # Calculate duration metrics
                if metrics['durations']:
                    avg_duration = sum(metrics['durations']) / len(metrics['durations'])
                    recent_avg = sum(metrics['recent_durations']) / len(metrics['recent_durations']) if metrics['recent_durations'] else avg_duration
                    
                    # Create performance message
                    perf_msg = ActionPerformance()
                    perf_msg.action_name = action_name
                    perf_msg.success_rate = success_rate
                    perf_msg.average_duration = avg_duration
                    perf_msg.recent_average_duration = recent_avg
                    perf_msg.total_attempts = metrics['attempts']
                    perf_msg.total_successes = metrics['successes']
                    perf_msg.timestamp = self.get_clock().now().to_msg()
                    
                    # Publish metrics
                    self.success_rate_pub.publish(Float32(data=success_rate))
                    self.duration_pub.publish(Float32(data=recent_avg))
                    self.performance_pub.publish(perf_msg)
                    
                    self.get_logger().info(
                        f'Action {action_name}: ' +
                        f'Success rate = {(success_rate * 100):.1f}%, ' +
                        f'Recent avg duration = {recent_avg:.2f}s, ' +
                        f'Total attempts = {metrics["attempts"]}'
                    )

    def get_action_statistics(self, action_name):
        """Get statistics for a specific action"""
        if action_name in self.action_metrics:
            metrics = self.action_metrics[action_name]
            if metrics['durations']:
                return {
                    'success_rate': metrics['successes'] / metrics['attempts'],
                    'average_duration': sum(metrics['durations']) / len(metrics['durations']),
                    'min_duration': min(metrics['durations']),
                    'max_duration': max(metrics['durations']),
                    'recent_average': sum(metrics['recent_durations']) / len(metrics['recent_durations']) if metrics['recent_durations'] else 0,
                    'total_attempts': metrics['attempts'],
                    'total_successes': metrics['successes']
                }
        return None
```

## Summary

ROS action translation is a complex but essential part of Physical AI systems that connect high-level cognitive capabilities with low-level robot control. Success in this domain requires:

1. **Careful Planning**: Understanding the relationship between cognitive goals and executable actions
2. **Validation**: Verifying that translation steps are feasible and safe
3. **Error Handling**: Robust error handling and recovery mechanisms
4. **Monitoring**: Performance monitoring to evaluate and improve translation systems
5. **Integration**: Seamless integration between simulation and real-world execution

The examples and best practices in this document provide a foundation for implementing effective ROS action translation systems in Physical AI applications. The combination of Gazebo simulation and Unity visualization enables comprehensive development, testing, and validation of these systems before deployment on real hardware.

These concepts apply to all levels of robot control, from simple navigation tasks to complex manipulation sequences and coordinated multi-robot activities. With proper implementation, these translation systems bridge the gap between high-level AI reasoning and physical robot execution, enabling the sophisticated Physical AI applications that are central to humanoid robotics.