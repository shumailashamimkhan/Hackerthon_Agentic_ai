---
title: VLA-Based Exercises
sidebar_position: 7
---

# VLA-Based Exercises

## Exercise Set 1: Voice Command Interpretation

### Exercise 1.1: Command Parsing and Intent Recognition

**Objective**: Implement a system that can accurately parse voice commands and identify the intent in a Physical AI context.

**Instructions**:
1. Create a Python class `VAInterpretationEngine` that takes natural language commands
2. Implement intent recognition for the following categories:
   - Navigation: "Go to the kitchen", "Move to the living room"
   - Manipulation: "Pick up the red cup", "Place the book on the table"
   - Interaction: "Wave to the person", "Follow John"
   - Complex: "Go to the kitchen and bring me the blue bottle"

3. The engine should output structured data including:
   - Intent type
   - Target objects
   - Target locations
   - Motion parameters (speed, caution level)

**Sample implementation**:

```python
import re
import json
from typing import Dict, List, Optional

class VAInterpretationEngine:
    def __init__(self):
        # Define patterns for different intents
        self.patterns = {
            'navigation': [
                r'go to (the )?(?P<location>\w+)',
                r'move to (the )?(?P<location>\w+)',
                r'navigate to (the )?(?P<location>\w+)',
                r'walk to (the )?(?P<location>\w+)'
            ],
            'manipulation': [
                r'pick up (the )?(?P<object>\w+)',
                r'grasp (the )?(?P<object>\w+)',
                r'get (the )?(?P<object>\w+)(?! from)',  # Negative lookahead
                r'bring me (the )?(?P<object>\w+)',
                r'place (the )?(?P<obj>\w+) on (the )?(?P<loc>\w+)'
            ],
            'interaction': [
                r'wave to (the )?(?P<entity>\w+)',
                r'follow (the )?(?P<entity>\w+)',
                r'greet (the )?(?P<entity>\w+)'
            ]
        }
    
    def parse_command(self, command: str) -> Dict:
        """Parse a natural language command and extract structured information"""
        command_lower = command.lower()
        result = {
            'raw_command': command,
            'intent': None,
            'target_object': None,
            'target_location': None,
            'motion_params': {
                'speed': 'normal',
                'caution': 'medium'
            },
            'confidence': 0.0
        }
        
        # Identify intent
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    result['intent'] = intent
                    groups = match.groupdict()
                    
                    # Extract object if present
                    if 'object' in groups:
                        result['target_object'] = groups['object']
                    elif 'obj' in groups:  # For composite patterns
                        result['target_object'] = groups['obj']
                    
                    # Extract location if present
                    if 'location' in groups:
                        result['target_location'] = groups['location']
                    elif 'loc' in groups:  # For composite patterns
                        result['target_location'] = groups['loc']
                    
                    # Extract entity for interaction
                    if 'entity' in groups:
                        result['target_entity'] = groups['entity']
                    
                    # Set high confidence for pattern matches
                    result['confidence'] = 0.9
                    break
            if result['intent']:
                break
        
        # If no intent matched, try more general classification
        if not result['intent']:
            if any(word in command_lower for word in ['go', 'move', 'navigate', 'walk', 'travel']):
                result['intent'] = 'navigation'
                result['confidence'] = 0.7
            elif any(word in command_lower for word in ['pick', 'grasp', 'get', 'bring', 'take', 'hold']):
                result['intent'] = 'manipulation'
                result['confidence'] = 0.7
            elif any(word in command_lower for word in ['wave', 'follow', 'greet', 'interact']):
                result['intent'] = 'interaction'
                result['confidence'] = 0.7
        
        # Infer motion parameters from command
        if 'slow' in command_lower or 'carefully' in command_lower:
            result['motion_params']['speed'] = 'slow'
            result['motion_params']['caution'] = 'high'
        elif 'fast' in command_lower or 'quickly' in command_lower:
            result['motion_params']['speed'] = 'fast'
            result['motion_params']['caution'] = 'low'
        
        return result

# Test the implementation
def test_interpreter():
    interpreter = VAInterpretationEngine()
    
    test_commands = [
        "Go to the kitchen",
        "Pick up the red cup",
        "Place the book on the table",
        "Wave to the person",
        "Navigate to the living room carefully",
        "Bring me the blue bottle quickly"
    ]
    
    for cmd in test_commands:
        result = interpreter.parse_command(cmd)
        print(f"Command: '{cmd}'")
        print(f"  Intent: {result['intent']}")
        print(f"  Target: {result['target_object'] or result['target_location'] or result.get('target_entity')}")
        print(f"  Parameters: {result['motion_params']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print()

if __name__ == "__main__":
    test_interpreter()
```

**Exercise Tasks**:
1. Implement the above class
2. Extend it to handle more complex commands with multiple clauses
3. Add confidence scoring based on multiple factors (pattern match, word similarity, context)
4. Test with various commands and evaluate accuracy

**Evaluation Criteria**:
- Correctly identifies intent in >90% of test commands
- Properly extracts target objects/locations in >85% of cases
- Appropriately sets motion parameters based on command modifiers
- Provides meaningful confidence scores

### Exercise 1.2: Voice-to-Action Mapping

**Objective**: Create a mapping system that translates interpreted voice commands to specific robot actions.

**Instructions**:
1. Build a mapping system that converts interpretation results to executable robot commands
2. Implement a command dispatcher that can handle different action types
3. Create validation functions to check if actions are feasible given robot state
4. Handle cases where commands cannot be executed (e.g. object not reachable)

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math

@dataclass
class RobotState:
    """Represents the current state of the robot"""
    position: Dict[str, float]  # x, y, z coordinates
    orientation: Dict[str, float]  # roll, pitch, yaw or quaternion
    joint_angles: Dict[str, float]  # current joint positions
    attached_object: Optional[str] = None  # object currently held, if any
    battery_level: float = 1.0  # 0.0 to 1.0

@dataclass
class ActionCommand:
    """Represents a command to send to the robot"""
    action_type: str  # 'navigation', 'manipulation', 'interaction'
    parameters: Dict[str, Any]  # specific to action type
    priority: int = 1  # lower is higher priority

class VoiceToActionMapper:
    def __init__(self, initial_robot_state: RobotState):
        self.robot_state = initial_robot_state
        self.action_history: List[ActionCommand] = []
    
    def map_interpretation_to_action(self, interpretation: Dict) -> Optional[ActionCommand]:
        """Map interpretation to an executable action command"""
        intent = interpretation['intent']
        
        if intent == 'navigation':
            return self._create_navigation_action(interpretation)
        elif intent == 'manipulation':
            return self._create_manipulation_action(interpretation)
        elif intent == 'interaction':
            return self._create_interaction_action(interpretation)
        else:
            print(f"Unknown intent: {intent}")
            return None
    
    def _create_navigation_action(self, interpretation: Dict) -> Optional[ActionCommand]:
        """Create navigation action from interpretation"""
        target_location = interpretation['target_location']
        if not target_location:
            print("No target location specified for navigation")
            return None
        
        # In a real system, this would get the actual coordinates
        # For this exercise, we'll use a lookup table
        location_coordinates = {
            'kitchen': {'x': 3.0, 'y': 1.0, 'z': 0.0},
            'living room': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'bedroom': {'x': -2.0, 'y': 1.5, 'z': 0.0},
            'office': {'x': -1.0, 'y': -2.0, 'z': 0.0}
        }
        
        if target_location not in location_coordinates:
            print(f"Unknown location: {target_location}")
            return None
        
        target_pose = location_coordinates[target_location]
        motion_params = interpretation['motion_params']
        
        # Create action command
        action = ActionCommand(
            action_type='navigation',
            parameters={
                'target_pose': target_pose,
                'speed': motion_params['speed'],
                'caution_level': motion_params['caution'],
                'path_planning_strategy': 'default'
            }
        )
        
        return action
    
    def _create_manipulation_action(self, interpretation: Dict) -> Optional[ActionCommand]:
        """Create manipulation action from interpretation"""
        target_object = interpretation['target_object']
        if not target_object:
            print("No target object specified for manipulation")
            return None
        
        # Check if robot is already holding something
        if self.robot_state.attached_object:
            print(f"Cannot manipulate '{target_object}' - already holding '{self.robot_state.attached_object}'")
            return None
        
        # For this exercise, assume object is at a fixed relative position
        # In reality, this would come from perception system
        relative_object_position = {
            'cup': {'x': 0.5, 'y': 0.2, 'z': 0.8},
            'book': {'x': 0.5, 'y': -0.1, 'z': 0.9},
            'bottle': {'x': 0.4, 'y': 0.3, 'z': 0.85}
        }
        
        if target_object not in relative_object_position:
            print(f"Unknown object: {target_object}")
            return None
        
        object_pose = relative_object_position[target_object]
        motion_params = interpretation['motion_params']
        
        action = ActionCommand(
            action_type='manipulation',
            parameters={
                'manipulation_type': 'grasp',
                'target_object': target_object,
                'target_pose': object_pose,
                'grip_type': 'pinch' if target_object in ['cup', 'bottle'] else 'power',
                'force_limit': 50.0,  # Newtons
                'speed': motion_params['speed']
            }
        )
        
        return action
    
    def _create_interaction_action(self, interpretation: Dict) -> Optional[ActionCommand]:
        """Create interaction action from interpretation"""
        target_entity = interpretation.get('target_entity')
        intent = interpretation['raw_command'].lower()
        
        action_type = 'gesture'
        interaction_params = {'gesture_type': 'wave', 'target_entity': target_entity}
        
        if 'follow' in intent:
            action_type = 'following'
            interaction_params = {
                'follow_type': 'person',
                'target_entity': target_entity,
                'following_distance': 1.0  # meter
            }
        elif 'greet' in intent:
            action_type = 'social'
            interaction_params = {
                'social_action': 'greeting',
                'target_entity': target_entity,
                'greeting_type': 'wave_and_speak'
            }
        
        action = ActionCommand(
            action_type=action_type,
            parameters=interaction_params
        )
        
        return action
    
    def validate_action_feasibility(self, action: ActionCommand) -> bool:
        """Check if the robot can execute the action given its current state"""
        if action.action_type == 'navigation':
            # Check if target is reachable based on current position
            target = action.parameters['target_pose']
            current = self.robot_state.position
            
            distance = math.sqrt(
                (target['x'] - current['x'])**2 + 
                (target['y'] - current['y'])**2 + 
                (target['z'] - current['z'])**2
            )
            
            # Assume robot has a maximum travel distance of 50m when battery is > 0.1
            max_distance = 50.0 if self.robot_state.battery_level > 0.1 else 0.0
            return distance <= max_distance
        
        elif action.action_type == 'manipulation':
            # Check if target is within reach
            target = action.parameters['target_pose']
            current = self.robot_state.position
            
            # Simplified reachability check (within 1.5m)
            distance = math.sqrt(
                (target['x'] - current['x'])**2 + 
                (target['y'] - current['y'])**2 + 
                (target['z'] - current['z'])**2
            )
            
            return distance <= 1.5  # Max reach distance
        
        # For other action types, assume feasible
        return True
    
    def execute_action(self, action: ActionCommand) -> bool:
        """Execute the action and update robot state accordingly"""
        if not self.validate_action_feasibility(action):
            print(f"Action not feasible: {action.action_type}")
            return False
        
        # Log the action
        self.action_history.append(action)
        
        # Update robot state based on the action
        if action.action_type == 'navigation':
            # Update robot's position to target
            self.robot_state.position = action.parameters['target_pose'].copy()
        
        elif action.action_type == 'manipulation' and action.parameters['manipulation_type'] == 'grasp':
            # Update robot state to show holding object
            self.robot_state.attached_object = action.parameters['target_object']
        
        elif action.action_type == 'manipulation' and action.parameters['manipulation_type'] == 'place':
            # Update robot state to show releasing object
            self.robot_state.attached_object = None
        
        return True

# Test the implementation
def test_vta_mapper():
    # Initialize robot state
    initial_state = RobotState(
        position={'x': 0.0, 'y': 0.0, 'z': 0.0},
        orientation={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        joint_angles={},
        battery_level=0.8
    )
    
    mapper = VoiceToActionMapper(initial_state)
    interpreter = VAInterpretationEngine()
    
    # Test commands
    test_commands = [
        "Go to the kitchen",
        "Go to the office",
        "Pick up the cup",
        "Place the cup on the table",
        "Wave to John"
    ]
    
    for cmd in test_commands:
        print(f"\nProcessing command: '{cmd}'")
        
        # Interpret the command
        interpretation = interpreter.parse_command(cmd)
        print(f"  Interpretation: {interpretation['intent']} -> {interpretation['target_object'] or interpretation['target_location']}")
        
        # Map to action
        action = mapper.map_interpretation_to_action(interpretation)
        if action:
            print(f"  Action created: {action.action_type} with params {action.parameters}")
            
            # Execute the action
            success = mapper.execute_action(action)
            print(f"  Execution result: {'Success' if success else 'Failed'}")
            print(f"  Updated robot state - Position: {mapper.robot_state.position}, Holding: {mapper.robot_state.attached_object}")
        else:
            print(f"  Could not create action for command: {cmd}")

if __name__ == "__main__":
    test_vta_mapper()
```

**Exercise Tasks**:
1. Implement the VoiceToActionMapper class as specified
2. Extend it to handle more complex manipulation actions (place, pour, etc.)
3. Add more sophisticated validation based on robot kinematics
4. Create a simulation environment to test action execution

**Evaluation Criteria**:
- Correctly maps voice interpretations to appropriate robot actions
- Properly validates action feasibility before execution
- Updates robot state correctly after action execution
- Handles edge cases (e.g., object already held, unreachable targets)

## Exercise Set 2: Language Understanding and Context Awareness  

### Exercise 2.1: Context-Aware Language Understanding

**Objective**: Implement a system that understands voice commands in the context of recent robot actions and environment state.

**Instructions**:
1. Create a ContextManager class that tracks recent robot actions, objects in view, and conversation history
2. Enhance the voice interpretation engine to consider context when resolving ambiguous references
3. Implement pronoun resolution (e.g., "it", "that", "him", "her")
4. Implement spatial reference resolution (e.g., "over there", "to the left")

```python
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ContextEntry:
    """Entry in the context memory"""
    timestamp: datetime
    entry_type: str  # 'action', 'object_seen', 'command', etc.
    content: Any
    
class ContextManager:
    def __init__(self, memory_duration_minutes=10):
        self.memory_duration = timedelta(minutes=memory_duration_minutes)
        self.context_entries: List[ContextEntry] = []
        self.current_environment = {
            'visible_objects': [],
            'known_locations': {},
            'people_present': []
        }
    
    def add_context_entry(self, entry_type: str, content: Any):
        """Add a new entry to the context memory"""
        entry = ContextEntry(
            timestamp=datetime.now(),
            entry_type=entry_type,
            content=content
        )
        self.context_entries.append(entry)
        
        # Clean up old entries
        self._cleanup_old_entries()
    
    def _cleanup_old_entries(self):
        """Remove context entries that are too old"""
        current_time = datetime.now()
        cutoff_time = current_time - self.memory_duration
        
        self.context_entries = [
            entry for entry in self.context_entries 
            if entry.timestamp > cutoff_time
        ]
    
    def get_recent_actions(self, minutes=5) -> List[ContextEntry]:
        """Get recent actions from context"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            entry for entry in self.context_entries
            if entry.entry_type == 'action' and entry.timestamp > cutoff
        ]
    
    def get_recent_objects(self, minutes=2) -> List[ContextEntry]:
        """Get recently seen objects from context"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            entry for entry in self.context_entries
            if entry.entry_type == 'object_seen' and entry.timestamp > cutoff
        ]
    
    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to a specific entity based on context"""
        if pronoun.lower() in ['it', 'that', 'this']:
            # Look for the most recently mentioned object
            recent_objects = self.get_recent_objects(2)
            if recent_objects:
                # Return the most recent object
                last_object = recent_objects[-1].content
                return last_object.get('id') or last_object.get('name')
        
        elif pronoun.lower() in ['him', 'her', 'them']:
            # Look for recently mentioned people
            recent_people = [
                entry for entry in self.context_entries
                if entry.entry_type == 'person_seen' and 
                   entry.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            if recent_people:
                last_person = recent_people[-1].content
                return last_person.get('name') or last_person.get('id')
        
        return None
    
    def resolve_spatial_reference(self, reference: str, robot_pose: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Resolve spatial references like 'over there', 'to the left', etc."""
        # This would connect to a spatial reasoning system
        # For this exercise, return a placeholder implementation
        if 'there' in reference:
            # 'Over there' - return a position in front of robot
            return {
                'x': robot_pose['x'] + 1.0,  # 1m in front
                'y': robot_pose['y'],
                'z': robot_pose['z']
            }
        elif 'left' in reference:
            # 'To the left' - return position to robot's left
            return {
                'x': robot_pose['x'] - 0.5,  # 0.5m to the left
                'y': robot_pose['y'],
                'z': robot_pose['z']
            }
        elif 'right' in reference:
            # 'To the right' - return position to robot's right
            return {
                'x': robot_pose['x'] + 0.5,  # 0.5m to the right
                'y': robot_pose['y'],
                'z': robot_pose['z']
            }
        
        return None

class ContextAwareInterpreter(VAInterpretationEngine):
    def __init__(self):
        super().__init__()
        self.context_manager = ContextManager()
    
    def parse_command_with_context(self, command: str, robot_state: RobotState) -> Dict:
        """Parse command using contextual information"""
        # First, get basic interpretation
        basic_interpretation = self.parse_command(command)
        
        # Now enhance with context
        enhanced_interpretation = self._enhance_with_context(basic_interpretation, robot_state)
        
        return enhanced_interpretation
    
    def _enhance_with_context(self, interpretation: Dict, robot_state: RobotState) -> Dict:
        """Enhance interpretation with contextual information"""
        # Check for pronouns that need resolution
        if not interpretation['target_object']:
            # Look for pronouns in the command
            command_lower = interpretation['raw_command'].lower()
            
            if any(pronoun in command_lower for pronoun in ['it', 'that', 'this']):
                resolved_target = self.context_manager.resolve_pronoun('it')
                if resolved_target:
                    interpretation['target_object'] = resolved_target
                    interpretation['confidence'] = max(interpretation['confidence'], 0.7)
        
        # Check for spatial references that need resolution
        if not interpretation['target_location']:
            command_lower = interpretation['raw_command'].lower()
            
            if any(ref in command_lower for ref in ['there', 'left', 'right', 'ahead', 'behind']):
                resolved_location = self.context_manager.resolve_spatial_reference(
                    interpretation['raw_command'], 
                    robot_state.position
                )
                if resolved_location:
                    interpretation['target_location'] = f"spatial_ref_{id(resolved_location)}"
                    # Store the actual coordinates somewhere else to access later
                    interpretation['resolved_coordinates'] = resolved_location
                    interpretation['confidence'] = max(interpretation['confidence'], 0.6)
        
        return interpretation
    
    def update_context(self, action_executed: ActionCommand, result: bool):
        """Update context based on executed action"""
        self.context_manager.add_context_entry('action', {
            'action': action_executed,
            'result': result,
            'timestamp': datetime.now()
        })

# Test the contextual interpreter
def test_contextual_interpreter():
    # Initialize robot state
    robot_state = RobotState(
        position={'x': 0.0, 'y': 0.0, 'z': 0.0},
        orientation={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        joint_angles={},
        battery_level=0.8
    )
    
    # Initialize systems
    context_interpreter = ContextAwareInterpreter()
    vta_mapper = VoiceToActionMapper(robot_state)
    
    # Simulate some context first
    context_interpreter.context_manager.add_context_entry('object_seen', {'id': 'red_cup', 'type': 'cup', 'color': 'red'})
    
    test_commands = [
        "Pick up the red cup",  # This establishes context
        "Put it on the table",  # 'it' should refer to red cup
        "Go over there",        # 'there' should be resolved spatially
        "Wave to him"           # 'him' would need to see a person first
    ]
    
    for cmd in test_commands:
        print(f"\nProcessing contextual command: '{cmd}'")
        
        # Interpret with context
        interpretation = context_interpreter.parse_command_with_context(cmd, robot_state)
        print(f"  Enhanced interpretation: {interpretation}")
        
        # Map to action
        action = vta_mapper.map_interpretation_to_action(interpretation)
        if action:
            print(f"  Action created: {action.action_type}")
            success = vta_mapper.execute_action(action)
            print(f"  Execution result: {'Success' if success else 'Failed'}")
            
            # Update context
            context_interpreter.update_context(action, success)
        else:
            print(f"  Could not create action for command: {cmd}")

if __name__ == "__main__":
    test_contextual_interpreter()
```

**Exercise Tasks**:
1. Implement the ContextManager class as specified
2. Extend the ContextAwareInterpreter to handle more complex contextual references
3. Add temporal reasoning (e.g., "the cup I saw earlier")
4. Create a simulation showing how context improves interpretation accuracy

**Evaluation Criteria**:
- Correctly resolves pronouns based on recent context
- Accurately interprets spatial references with robot position in mind
- Maintains coherent conversation context over multiple exchanges
- Improves overall command interpretation accuracy with context

## Exercise Set 3: Action Translation and Execution

### Exercise 3.1: ROS Action Client Implementation

**Objective**: Implement ROS 2 action clients to execute the actions generated by the VLA system.

**Instructions**:
1. Create a ROS 2 node that receives action commands from the VLA pipeline
2. Implement action clients for navigation and manipulation
3. Handle action feedback and result reporting
4. Implement error handling and recovery

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile

from shu_msgs.action import HumanoidNavigation, HumanoidManipulation
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
import json
import time

class VLAExecutionNode(Node):
    def __init__(self):
        super().__init__('vla_execution_node')
        
        # Action clients
        self.nav_client = ActionClient(self, HumanoidNavigation, 'humanoid_navigate_to_pose')
        self.manip_client = ActionClient(self, HumanoidManipulation, 'humanoid_perform_manipulation')
        
        # Subscriber for action commands from VLA pipeline
        self.command_sub = self.create_subscription(
            String,
            'vla_action_commands',
            self.command_callback,
            QoSProfile(depth=10)
        )
        
        # Publisher for execution results
        self.result_pub = self.create_publisher(
            String,
            'vla_execution_results',
            10
        )
        
        # Publisher for execution feedback
        self.feedback_pub = self.create_publisher(
            String,
            'vla_execution_feedback',
            10
        )
        
        self.get_logger().info('VLA Execution Node initialized')

    def command_callback(self, msg):
        """Process incoming action commands from VLA pipeline"""
        try:
            # Deserialize the command
            command_data = json.loads(msg.data)
            
            action_type = command_data.get('action_type')
            parameters = command_data.get('parameters', {})
            
            self.get_logger().info(f'Received {action_type} command with params: {parameters}')
            
            # Execute based on action type
            if action_type == 'navigation':
                success = self.execute_navigation_command(parameters)
            elif action_type == 'manipulation':
                success = self.execute_manipulation_command(parameters)
            elif action_type == 'interaction':
                success = self.execute_interaction_command(parameters)
            else:
                self.get_logger().error(f'Unknown action type: {action_type}')
                success = False
            
            # Publish result
            result_msg = String()
            result_msg.data = json.dumps({
                'command': command_data,
                'execution_success': success,
                'timestamp': time.time()
            })
            self.result_pub.publish(result_msg)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error decoding command JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Error executing command: {e}')

    def execute_navigation_command(self, params):
        """Execute a navigation command"""
        self.get_logger().info('Executing navigation command')
        
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False
        
        # Create goal
        goal_msg = HumanoidNavigation.Goal()
        
        # Set target pose from parameters
        target_pose_data = params.get('target_pose', {})
        goal_msg.target_pose.pose.position.x = target_pose_data.get('x', 0.0)
        goal_msg.target_pose.pose.position.y = target_pose_data.get('y', 0.0)
        goal_msg.target_pose.pose.position.z = target_pose_data.get('z', 0.0)
        
        # Default orientation (facing forward)
        goal_msg.target_pose.pose.orientation.w = 1.0
        
        # Set motion parameters
        speed_param = params.get('speed', 'normal')
        if speed_param == 'slow':
            goal_msg.motion_params.max_linear_speed = 0.3
            goal_msg.motion_params.max_angular_speed = 0.2
        elif speed_param == 'fast':
            goal_msg.motion_params.max_linear_speed = 1.0
            goal_msg.motion_params.max_angular_speed = 0.5
        else:  # normal
            goal_msg.motion_params.max_linear_speed = 0.6
            goal_msg.motion_params.max_angular_speed = 0.3
        
        # Set caution level
        caution_param = params.get('caution_level', 'medium')
        if caution_param == 'high':
            goal_msg.motion_params.footstep_planning_accuracy = 'precise'
            goal_msg.motion_params.obstacle_clearance = 0.8
        elif caution_param == 'low':
            goal_msg.motion_params.footstep_planning_accuracy = 'fast'
            goal_msg.motion_params.obstacle_clearance = 0.3
        else:  # medium
            goal_msg.motion_params.footstep_planning_accuracy = 'balanced'
            goal_msg.motion_params.obstacle_clearance = 0.5
        
        # Send goal
        goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )
        
        # Wait for result
        try:
            rclpy.spin_until_future_complete(self, goal_future)
            
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Navigation goal was rejected')
                return False
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            result = result_future.result().result
            self.get_logger().info(f'Navigation result: {result.success}')
            
            return result.success
            
        except Exception as e:
            self.get_logger().error(f'Error during navigation execution: {e}')
            return False
    
    def nav_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback_data = {
            'current_pose': {
                'x': feedback_msg.current_pose.pose.position.x,
                'y': feedback_msg.current_pose.pose.position.y,
                'z': feedback_msg.current_pose.pose.position.z
            },
            'distance_remaining': feedback_msg.distance_remaining,
            'status': feedback_msg.status,
            'timestamp': time.time()
        }
        
        feedback_msg_str = String()
        feedback_msg_str.data = json.dumps(feedback_data)
        self.feedback_pub.publish(feedback_msg_str)
        
        self.get_logger().debug(f'Navigation feedback: {feedback_data}')

    def execute_manipulation_command(self, params):
        """Execute a manipulation command"""
        self.get_logger().info('Executing manipulation command')
        
        # Wait for action server
        if not self.manip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Manipulation action server not available')
            return False
        
        # Create goal
        goal_msg = HumanoidManipulation.Goal()
        
        # Set manipulation type
        manip_type = params.get('manipulation_type', 'grasp')
        if manip_type == 'grasp':
            goal_msg.manipulation_type = 1  # Assuming 1 is grasp in the action definition
        elif manip_type == 'place':
            goal_msg.manipulation_type = 2  # Assuming 2 is place
        else:
            self.get_logger().error(f'Unknown manipulation type: {manip_type}')
            return False
        
        # Set target object if specified
        target_obj = params.get('target_object')
        if target_obj:
            goal_msg.target_object_id = target_obj
        
        # Set target pose if specified
        target_pose_data = params.get('target_pose')
        if target_pose_data:
            goal_msg.target_pose.pose.position.x = target_pose_data.get('x', 0.0)
            goal_msg.target_pose.pose.position.y = target_pose_data.get('y', 0.0)
            goal_msg.target_pose.pose.position.z = target_pose_data.get('z', 0.0)
            goal_msg.target_pose.pose.orientation.w = 1.0  # Default orientation
        
        # Set grip type
        grip_type = params.get('grip_type', 'pinch')
        goal_msg.grip_type = grip_type
        
        # Set force limit
        force_limit = params.get('force_limit', 50.0)
        goal_msg.force_limit = force_limit
        
        # Send goal
        goal_future = self.manip_client.send_goal_async(
            goal_msg,
            feedback_callback=self.manip_feedback_callback
        )
        
        # Wait for result
        try:
            rclpy.spin_until_future_complete(self, goal_future)
            
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Manipulation goal was rejected')
                return False
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            result = result_future.result().result
            self.get_logger().info(f'Manipulation result: {result.success}')
            
            return result.success
            
        except Exception as e:
            self.get_logger().error(f'Error during manipulation execution: {e}')
            return False

    def manip_feedback_callback(self, feedback_msg):
        """Handle manipulation feedback"""
        feedback_data = {
            'phase': feedback_msg.phase,
            'progress': feedback_msg.progress,
            'gripper_position': {
                'x': feedback_msg.gripper_pose.pose.position.x,
                'y': feedback_msg.gripper_pose.pose.position.y,
                'z': feedback_msg.gripper_pose.pose.position.z
            },
            'timestamp': time.time()
        }
        
        feedback_msg_str = String()
        feedback_msg_str.data = json.dumps(feedback_data)
        self.feedback_pub.publish(feedback_msg_str)
        
        self.get_logger().debug(f'Manipulation feedback: {feedback_data}')

    def execute_interaction_command(self, params):
        """Execute an interaction command (placeholder implementation)"""
        self.get_logger().info('Executing interaction command')
        
        # For interaction commands, we might trigger specific behaviors
        interaction_type = params.get('interaction_type', 'wave')
        
        # This is a simplified implementation - in reality would use different action servers
        # or service calls for different interaction types
        self.get_logger().info(f'Performing {interaction_type} interaction')
        
        # Simulate success after a short delay
        time.sleep(2.0)
        
        return True

def main(args=None):
    rclpy.init(args=args)
    
    node = VLAExecutionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('VLA Execution Node stopped cleanly')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Exercise Tasks**:
1. Implement the VLAExecutionNode as specified
2. Create custom action definitions if they don't exist
3. Add proper error handling and recovery mechanisms
4. Test the implementation with simulated action servers

**Evaluation Criteria**:
- Successfully executes navigation and manipulation actions
- Properly handles action feedback and results
- Implements robust error handling and recovery
- Integrates well with the rest of the VLA pipeline

### Exercise 3.2: Unity Visualization for Action Execution

**Objective**: Create a Unity visualization that shows the execution of actions from the VLA system.

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;
using System.Collections.Generic;
using System.Text.Json;

public class VLAActionVisualizer : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string executionResultsTopic = "/vla_execution_results";
    public string executionFeedbackTopic = "/vla_execution_feedback";
    
    [Header("Visualization")]
    public GameObject navigationIndicatorPrefab;
    public GameObject manipulationIndicatorPrefab;
    public GameObject interactionIndicatorPrefab;
    public Transform visualizationParent;
    
    [Header("Robot Model")]
    public GameObject robotModel;
    public float robotMoveSpeed = 2.0f;
    
    private Dictionary<string, GameObject> activeIndicators = new Dictionary<string, GameObject>();
    private Queue<ActionExecutionResult> executionResultQueue = new Queue<ActionExecutionResult>();
    private Queue<ActionExecutionFeedback> executionFeedbackQueue = new Queue<ActionExecutionFeedback>();
    
    void Start()
    {
        // Connect to ROS
        ROSTCPConnector ros = ROSTCPConnector.instance;
        
        // Subscribe to topics
        ros.Subscribe<StringMsg>(executionResultsTopic, OnExecutionResult);
        ros.Subscribe<StringMsg>(executionFeedbackTopic, OnExecutionFeedback);
        
        Debug.Log("VLA Action Visualizer initialized");
    }
    
    void OnExecutionResult(StringMsg msg)
    {
        try
        {
            // Parse the result JSON
            ActionExecutionResult result = JsonSerializer.Deserialize<ActionExecutionResult>(msg.data);
            
            // Process the result
            ProcessExecutionResult(result);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error parsing execution result: {e.Message}");
        }
    }
    
    void OnExecutionFeedback(StringMsg msg)
    {
        try
        {
            // Parse the feedback JSON
            ActionExecutionFeedback feedback = JsonSerializer.Deserialize<ActionExecutionFeedback>(msg.data);
            
            // Process the feedback
            ProcessExecutionFeedback(feedback);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error parsing execution feedback: {e.Message}");
        }
    }
    
    void ProcessExecutionResult(ActionExecutionResult result)
    {
        Debug.Log($"Execution result: {result.command.action_type} - Success: {result.execution_success}");
        
        // Visualize the result
        if (result.execution_success)
        {
            VisualizeSuccessfulExecution(result);
        }
        else
        {
            VisualizeFailedExecution(result);
        }
    }
    
    void ProcessExecutionFeedback(ActionExecutionFeedback feedback)
    {
        // Update any active visualizations based on feedback
        if (feedback.status == "navigating")
        {
            UpdateNavigationVisualization(feedback);
        }
        else if (feedback.phase == "grasping" || feedback.phase == "releasing")
        {
            UpdateManipulationVisualization(feedback);
        }
    }
    
    void VisualizeSuccessfulExecution(ActionExecutionResult result)
    {
        // Create appropriate visualization based on action type
        GameObject indicator = null;
        
        if (result.command.action_type == "navigation")
        {
            indicator = Instantiate(navigationIndicatorPrefab, visualizationParent);
            PlaceNavigationIndicator(indicator, result.command.parameters);
        }
        else if (result.command.action_type == "manipulation")
        {
            indicator = Instantiate(manipulationIndicatorPrefab, visualizationParent);
            PlaceManipulationIndicator(indicator, result.command.parameters);
        }
        else if (result.command.action_type == "interaction")
        {
            indicator = Instantiate(interactionIndicatorPrefab, visualizationParent);
            PlaceInteractionIndicator(indicator, result.command.parameters);
        }
        
        if (indicator != null)
        {
            // Store indicator for potential future updates
            string indicatorId = System.Guid.NewGuid().ToString();
            indicator.name = $"ResultIndicator_{result.command.action_type}_{indicatorId}";
            activeIndicators[indicatorId] = indicator;
            
            // Schedule removal after some time
            StartCoroutine(RemoveIndicatorAfterDelay(indicator, indicatorId, 5.0f));
            
            // If it's a navigation command and we have robot model, animate movement
            if (result.command.action_type == "navigation" && robotModel != null)
            {
                AnimateRobotToPose(robotModel, GetTargetPose(result.command.parameters));
            }
        }
    }
    
    void VisualizeFailedExecution(ActionExecutionResult result)
    {
        // Create visual indication of failure
        Debug.LogWarning($"Action failed: {result.command.action_type}");
        
        // Visualize failure (e.g., red X mark at attempted destination)
        GameObject failureIndicator = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        failureIndicator.transform.SetParent(visualizationParent);
        failureIndicator.name = "FailureIndicator";
        
        // Get target position and place failure indicator there
        if (result.command.action_type == "navigation")
        {
            var targetPos = GetTargetPosition(result.command.parameters);
            failureIndicator.transform.position = targetPos;
            failureIndicator.transform.localScale = new Vector3(0.2f, 0.1f, 0.2f);
        }
        
        // Change color to red to indicate failure
        var rend = failureIndicator.GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material.color = Color.red;
        }
        
        // Schedule removal
        StartCoroutine(RemoveIndicatorAfterDelay(failureIndicator, "", 5.0f));
    }
    
    void PlaceNavigationIndicator(GameObject indicator, Dictionary<string, object> parameters)
    {
        // Place navigation indicator at target location
        var targetPos = GetTargetPosition(parameters);
        indicator.transform.position = targetPos;
    }
    
    void PlaceManipulationIndicator(GameObject indicator, Dictionary<string, object> parameters)
    {
        // Place manipulation indicator at target object location
        // This would need to know the position of the target object
        indicator.transform.position = robotModel != null ? robotModel.transform.position + Vector3.forward * 1.5f : Vector3.zero;
    }
    
    void PlaceInteractionIndicator(GameObject indicator, Dictionary<string, object> parameters)
    {
        // Place interaction indicator near the robot or target entity
        indicator.transform.position = robotModel != null ? robotModel.transform.position + Vector3.up * 2f : Vector3.zero;
    }
    
    Vector3 GetTargetPosition(Dictionary<string, object> parameters)
    {
        // Extract position from parameters (assuming they contain target_pose with x, y, z)
        if (parameters.ContainsKey("target_pose"))
        {
            var poseObj = (Dictionary<string, object>)parameters["target_pose"];
            if (poseObj.ContainsKey("x") && poseObj.ContainsKey("y") && poseObj.ContainsKey("z"))
            {
                return new Vector3(
                    float.Parse(poseObj["x"].ToString()),
                    float.Parse(poseObj["y"].ToString()),
                    float.Parse(poseObj["z"].ToString())
                );
            }
        }
        return Vector3.zero; // default position if not found
    }
    
    Pose GetTargetPose(Dictionary<string, object> parameters)
    {
        // Extract pose from parameters
        var position = GetTargetPosition(parameters);
        var orientation = Quaternion.identity; // default orientation
        
        // If orientation data exists, extract it
        if (parameters.ContainsKey("target_pose"))
        {
            var poseObj = (Dictionary<string, object>)parameters["target_pose"];
            if (poseObj.ContainsKey("qx") || poseObj.ContainsKey("qy") || poseObj.ContainsKey("qz") || poseObj.ContainsKey("qw"))
            {
                float qx = poseObj.ContainsKey("qx") ? float.Parse(poseObj["qx"].ToString()) : 0f;
                float qy = poseObj.ContainsKey("qy") ? float.Parse(poseObj["qy"].ToString()) : 0f;
                float qz = poseObj.ContainsKey("qz") ? float.Parse(poseObj["qz"].ToString()) : 0f;
                float qw = poseObj.ContainsKey("qw") ? float.Parse(poseObj["qw"].ToString()) : 1f;
                
                orientation = new Quaternion(qx, qy, qz, qw);
            }
        }
        
        return new Pose(position, orientation);
    }
    
    IEnumerator RemoveIndicatorAfterDelay(GameObject indicator, string indicatorId, float delay)
    {
        yield return new WaitForSeconds(delay);
        
        if (indicator != null)
        {
            if (!string.IsNullOrEmpty(indicatorId) && activeIndicators.ContainsKey(indicatorId))
            {
                activeIndicators.Remove(indicatorId);
            }
            Destroy(indicator);
        }
    }
    
    void AnimateRobotToPose(GameObject robot, Pose targetPose)
    {
        // Simple animation to move robot to target pose
        // In a real implementation, this would use more sophisticated animation
        StartCoroutine(MoveRobotToPose(robot, targetPose, 2.0f)); // 2 seconds for move
    }
    
    IEnumerator MoveRobotToPose(GameObject robot, Pose targetPose, float duration)
    {
        Vector3 startPos = robot.transform.position;
        Quaternion startRot = robot.transform.rotation;
        
        float elapsedTime = 0;
        while (elapsedTime < duration)
        {
            robot.transform.position = Vector3.Lerp(startPos, targetPose.position, elapsedTime / duration);
            robot.transform.rotation = Quaternion.Slerp(startRot, targetPose.rotation, elapsedTime / duration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Final position to ensure accuracy
        robot.transform.position = targetPose.position;
        robot.transform.rotation = targetPose.rotation;
    }
    
    void UpdateNavigationVisualization(ActionExecutionFeedback feedback)
    {
        // Update any active navigation indicators based on current robot position
        if (feedback.current_pose != null)
        {
            Vector3 currentPosition = new Vector3(
                (float)feedback.current_pose.x,
                (float)feedback.current_pose.y,
                (float)feedback.current_pose.z
            );
            
            // Update robot model position if available
            if (robotModel != null)
            {
                robotModel.transform.position = currentPosition;
            }
        }
    }
    
    void UpdateManipulationVisualization(ActionExecutionFeedback feedback)
    {
        // Update manipulation indicators based on gripper position and phase
        Debug.Log($"Manipulation phase: {feedback.phase}, progress: {feedback.progress}");
    }
}

// Data classes for parsing JSON
[System.Serializable]
public class ActionExecutionResult
{
    public ActionCommand command;
    public bool execution_success;
    public double timestamp;
}

[System.Serializable]
public class ActionCommand
{
    public string action_type;
    public Dictionary<string, object> parameters;
    public int priority;
}

[System.Serializable]
public class ActionExecutionFeedback
{
    public string status; // For navigation
    public string phase;  // For manipulation
    public float progress;
    public PoseInfo current_pose;
    public double timestamp;
}

[System.Serializable]
public class PoseInfo
{
    public double x;
    public double y;
    public double z;
    public double qx;
    public double qy;
    public double qz;
    public double qw;
}
```

**Exercise Tasks**:
1. Implement the VLAActionVisualizer as specified
2. Create appropriate prefab indicators for different action types
3. Add animation for robot movement during navigation
4. Test integration with the ROS action execution pipeline

**Evaluation Criteria**:
- Visually represents different action types appropriately
- Updates in real-time based on action feedback
- Shows success or failure of action execution
- Integrates smoothly with Unity and ROS systems

## Exercise Set 4: Integration Challenge

### Exercise 4.1: Complete VLA Pipeline Implementation

**Objective**: Integrate all components (voice interpretation, context awareness, action mapping, execution, and visualization) into a complete working system.

**System Architecture**:
```
[Voice Input] → [VA Interpreter] → [Context Manager] → [Action Mapper] → [ROS Executor] → [Unity Visualizer]
```

**Requirements**:
1. All modules from the previous exercises must be integrated
2. Implement inter-module communication using ROS 2 topics/services
3. Test with realistic voice commands in simulation
4. Validate the complete pipeline with metrics (accuracy, response time, etc.)

**Implementation Steps**:

1. Create a launch file that starts all components:
```python
# launch/vla_complete_pipeline_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Start all VLA pipeline nodes
    ld = LaunchDescription()
    
    # VA interpretation node
    va_interp_node = Node(
        package='vla_pipeline',
        executable='va_interpreter_node',
        name='va_interpreter',
        parameters=[{'use_sim_time': True}]
    )
    
    # Context manager node
    context_node = Node(
        package='vla_pipeline',
        executable='context_manager_node',
        name='context_manager',
        parameters=[{'use_sim_time': True}]
    )
    
    # Action mapper node
    action_map_node = Node(
        package='vla_pipeline',
        executable='action_mapper_node',
        name='action_mapper',
        parameters=[{'use_sim_time': True}]
    )
    
    # Action execution node
    exec_node = Node(
        package='vla_pipeline',
        executable='vla_execution_node',
        name='vla_executor',
        parameters=[{'use_sim_time': True}]
    )
    
    ld.add_action(va_interp_node)
    ld.add_action(context_node)
    ld.add_action(action_map_node)
    ld.add_action(exec_node)
    
    return ld
```

2. Create a test script that runs through various scenarios:
```python
# test_vla_pipeline.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import time
import json

class VLAPipelineTester(Node):
    def __init__(self):
        super().__init__('vla_pipeline_tester')
        
        # Publisher for voice commands
        self.voice_cmd_pub = self.create_publisher(String, 'voice_input', 10)
        
        # Subscriber for execution results
        self.result_sub = self.create_subscription(
            String, 
            'vla_execution_results', 
            self.result_callback, 
            10
        )
        
        self.results_log = []
        self.test_scenario_idx = 0
        self.test_scenarios = [
            {"command": "Go to the kitchen", "expected_action": "navigation"},
            {"command": "Pick up the red cup", "expected_action": "manipulation"},
            {"command": "Wave to the person", "expected_action": "interaction"}
        ]
        
        # Timer to run tests
        self.test_timer = self.create_timer(5.0, self.run_next_test)
        
        self.get_logger().info('VLA Pipeline Tester initialized')
    
    def run_next_test(self):
        """Run the next test scenario"""
        if self.test_scenario_idx >= len(self.test_scenarios):
            self.get_logger().info('All tests completed!')
            self.evaluate_results()
            return
        
        scenario = self.test_scenarios[self.test_scenario_idx]
        self.get_logger().info(f'Running test {self.test_scenario_idx + 1}: {scenario["command"]}')
        
        # Publish the voice command
        cmd_msg = String()
        cmd_msg.data = scenario["command"]
        self.voice_cmd_pub.publish(cmd_msg)
        
        self.test_scenario_idx += 1
    
    def result_callback(self, msg):
        """Handle execution results"""
        try:
            result_data = json.loads(msg.data)
            self.results_log.append(result_data)
            self.get_logger().info(f'Execution result: {result_data["execution_success"]}')
        except Exception as e:
            self.get_logger().error(f'Error parsing result: {e}')
    
    def evaluate_results(self):
        """Evaluate and print test results"""
        self.get_logger().info('=== Test Results ===')
        for i, result in enumerate(self.results_log):
            scenario = self.test_scenarios[i]
            expected = scenario["expected_action"]
            actual = result["command"]["action_type"]
            success = result["execution_success"]
            
            status = "✓ PASS" if actual == expected and success else "✗ FAIL"
            self.get_logger().info(f'Test {i+1}: {status} - Expected: {expected}, Got: {actual}, Success: {success}')

def main(args=None):
    rclpy.init(args=args)
    
    tester = VLAPipelineTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Testing interrupted by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Exercise Tasks**:
1. Implement the complete integrated system
2. Test with various command scenarios
3. Measure system performance (latency, accuracy, etc.)
4. Document findings and potential improvements

**Evaluation Criteria**:
- Complete end-to-end functionality from voice input to robot action
- Proper integration between all system components
- Acceptable performance metrics (response time < 2 seconds, accuracy > 85%)
- Comprehensive testing across multiple scenarios

## Summary

These exercises provide hands-on experience with implementing Vision-Language-Action systems for Physical AI applications. By working through these exercises, you'll develop a deep understanding of:

1. How to process voice commands and map them to robot actions
2. How to maintain context for more natural interactions
3. How to execute actions safely and efficiently
4. How to visualize and monitor the system's behavior
5. How to integrate all components into a cohesive pipeline

Completing these exercises will give you practical experience with the core components of VLA systems that are essential for advanced Physical AI applications with humanoid robots.