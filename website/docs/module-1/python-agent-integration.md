---
title: Python Agent Integration with ROS Controllers
sidebar_position: 3
---

# Python Agent Integration with ROS Controllers

## Introduction to Python Agents in Physical AI

In the context of Physical AI, a "Python agent" refers to an intelligent software component written in Python that interfaces with ROS controllers to perceive the environment, reason about actions, and execute behaviors on a physical system. These agents embody the intelligence layer of the robotic nervous system, bridging the gap between perception and action.

### Why Python for Physical AI Agents?

Python is particularly well-suited for Physical AI agents due to:

- **Rich AI ecosystem**: Libraries like TensorFlow, PyTorch, scikit-learn, and OpenCV
- **Ease of prototyping**: Rapid development and testing of AI algorithms
- **Scientific computing**: NumPy, SciPy, and pandas for data processing
- **ROS integration**: Excellent support through the rclpy client library
- **Community support**: Large robotics and AI communities

### Agent Archetypes in Physical AI

Python agents in humanoid robotics typically fall into several categories:

- **Perception agents**: Analyze sensor data to understand the environment
- **Planning agents**: Generate sequences of actions to achieve goals
- **Learning agents**: Adapt behavior based on experience
- **Interaction agents**: Handle communication with humans and other systems
- **Control agents**: Coordinate low-level controllers for complex behaviors

## Setting Up Python-Agent ROS Integration

### ROS Client Library for Python (rclpy)

The `rclpy` package provides Python bindings to the ROS 2 client library:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
```

### Node Creation for Python Agents

Python agents typically extend the Node class to interface with the ROS system:

```python
class PhysicalAIAgentNode(Node):
    def __init__(self):
        super().__init__('physical_ai_agent')
        
        # Create subscribers for sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            qos_profile_sensor_data
        )
        
        # Create publishers for control commands
        self.command_publisher = self.create_publisher(
            JointTrajectory,
            'joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Create service clients for configuration
        self.get_params_client = self.create_client(
            GetParameters,
            'get_parameters'
        )
```

### Quality of Service for Physical AI

For real-time Physical AI applications, QoS profiles need careful consideration:

```python
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

# For sensor data (high frequency, can tolerate some drops)
qos_sensor = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    deadline=rclpy.duration.Duration(seconds=0.1)  # 10Hz deadline
)

# For control commands (reliable, low latency)
qos_control = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    lifespan=rclpy.duration.Duration(seconds=0.05)  # 50ms lifespan
)
```

## Integrating Python AI Libraries with ROS Controllers

### Message Conversion Patterns

Python AI libraries often expect data in NumPy arrays or tensor formats, requiring conversion from ROS messages:

```python
import numpy as np
from builtin_interfaces.msg import Time

def ros_to_numpy_joint_state(ros_msg):
    """Convert JointState ROS message to NumPy array for AI processing"""
    positions = np.array(ros_msg.position)
    velocities = np.array(ros_msg.velocity) if ros_msg.velocity else np.zeros_like(positions)
    efforts = np.array(ros_msg.effort) if ros_msg.effort else np.zeros_like(positions)
    
    # Combine into single state vector for ML models
    state_vector = np.concatenate([positions, velocities, efforts])
    
    return state_vector, ros_msg.header.stamp

def numpy_to_ros_trajectory(numpy_commands, joint_names):
    """Convert NumPy commands back to ROS JointTrajectory"""
    msg = JointTrajectory()
    msg.joint_names = joint_names
    
    # Create single trajectory point
    point = JointTrajectoryPoint()
    point.positions = numpy_commands.tolist()
    point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms
    msg.points = [point]
    
    return msg
```

### Asynchronous Processing

Physical AI agents should process sensor data asynchronously while maintaining real-time performance:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

class AsyncPhysicalAIAgent(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__()
        
        # Thread pool for CPU-intensive AI computations
        self.ai_executor = ThreadPoolExecutor(max_workers=2)
        
        # Lock for thread-safe state access
        self.state_lock = threading.Lock()
        self.current_state = np.zeros(0)
        
        # Timer for AI processing loop
        self.ai_timer = self.create_timer(0.05, self.process_ai_logic)  # 20Hz
        
    def joint_state_callback(self, msg):
        """Non-blocking sensor data reception"""
        state_vector, timestamp = ros_to_numpy_joint_state(msg)
        
        with self.state_lock:
            self.current_state = state_vector
            
    def process_ai_logic(self):
        """Run AI computations in background thread"""
        with self.state_lock:
            current_state_copy = self.current_state.copy()
        
        if len(current_state_copy) > 0:
            # Submit AI computation to thread pool
            future = self.ai_executor.submit(self.run_ai_model, current_state_copy)
            future.add_done_callback(self.on_ai_completion)
            
    def run_ai_model(self, state):
        """CPU-intensive AI processing (runs in background)"""
        # Example: Run a neural network for action selection
        # This runs outside the main ROS thread
        action = self.neural_network.predict(state)
        return action
        
    def on_ai_completion(self, future):
        """Called when AI processing is complete"""
        try:
            action = future.result()
            # Convert to ROS message and publish
            trajectory_cmd = numpy_to_ros_trajectory(action, self.joint_names)
            self.command_publisher.publish(trajectory_cmd)
        except Exception as e:
            self.get_logger().error(f'AI processing error: {e}')
```

## Example: Learning-Based Controller Integration

Let's explore a concrete example of integrating a reinforcement learning agent with ROS controllers:

```python
import torch
import torch.nn as nn
from stable_baselines3 import PPO

class RLHumanoidAgent(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__('rl_humanoid_agent')
        
        # Initialize RL model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rl_model = self.load_rl_model()  # Load trained model
        
        # Subscribe to observations
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_sensor
        )
        self.joint_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, qos_sensor
        )
        
        # Publish actions
        self.action_publisher = self.create_publisher(
            JointTrajectory, 'joint_trajectory_controller/joint_trajectory', qos_control
        )
        
        # Internal state
        self.current_observation = None
        self.observation_ready = False
        
        # Processing timer
        self.control_timer = self.create_timer(0.02, self.compute_action)  # 50Hz
        
    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        self.imu_data = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        ])
        self.update_observation()
        
    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity) if msg.velocity else np.zeros(len(msg.position))
        self.update_observation()
        
    def update_observation(self):
        """Update combined observation for RL model"""
        if hasattr(self, 'imu_data') and hasattr(self, 'joint_positions'):
            # Combine IMU and joint data into observation vector
            self.current_observation = np.concatenate([
                self.imu_data,           # 10 elements
                self.joint_positions,    # n joints
                self.joint_velocities    # n joints
            ])
            self.observation_ready = True
            
    def compute_action(self):
        """Compute action using RL model"""
        if self.observation_ready:
            # Prepare observation for model (normalize if needed)
            obs_tensor = torch.tensor(self.current_observation, dtype=torch.float32).unsqueeze(0)
            
            # Get action from model
            with torch.no_grad():
                action, _states = self.rl_model.predict(obs_tensor.cpu().numpy(), deterministic=True)
            
            # Convert action to joint commands
            joint_commands = self.scale_action_to_joints(action[0])
            
            # Create and publish trajectory command
            trajectory_msg = self.create_trajectory_message(joint_commands)
            self.action_publisher.publish(trajectory_msg)
            
    def load_rl_model(self):
        """Load trained reinforcement learning model"""
        # Load your trained model (example with Stable-Baselines3)
        model_path = self.get_parameter_or_declare(
            'model_path', 
            '/path/to/trained/model.zip'
        ).value
        
        return PPO.load(model_path)
        
    def scale_action_to_joints(self, action):
        """Map neural network output to joint commands"""
        # Scale action values to appropriate joint ranges
        # This depends on your specific humanoid robot
        scaled_commands = np.clip(action, -1.0, 1.0)  # Clamp to [-1, 1]
        return scaled_commands
        
    def create_trajectory_message(self, joint_commands):
        """Create JointTrajectory message from commands"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names  # Define based on your robot
        
        point = JointTrajectoryPoint()
        point.positions = joint_commands.tolist()
        point.time_from_start = Duration(sec=0, nanosec=20000000)  # 20ms to reach position
        
        msg.points = [point]
        return msg
```

## Advanced Integration Patterns

### Multi-Agent Coordination

For complex Physical AI applications, multiple Python agents may coordinate:

```python
class MultiAgentCoordinator(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__('multi_agent_coordinator')
        
        # Publisher for agent coordination
        self.coordination_pub = self.create_publisher(String, 'agent_coordination', 10)
        
        # Subscribers from other agents
        self.agent_status_subs = []
        for agent_id in range(5):  # Example: 5 agents
            sub = self.create_subscription(
                String, f'agent_{agent_id}_status', 
                lambda msg, aid=agent_id: self.agent_status_callback(msg, aid), 
                10
            )
            self.agent_status_subs.append(sub)
            
    def agent_status_callback(self, msg, agent_id):
        """Process status from individual agents"""
        # Implement coordination logic
        status = json.loads(msg.data)
        self.agents_status[agent_id] = status
        self.update_coordination()
```

### Safety Integration

Physical AI agents must incorporate safety considerations:

```python
class SafeAIAgent(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__('safe_ai_agent')
        
        # Initialize safety monitoring
        self.emergency_stop_active = False
        self.safety_monitor = SafetyMonitor()
        
        # Emergency stop publisher
        self.emergency_pub = self.create_publisher(Bool, 'emergency_stop', 1)
        
    def compute_action(self):
        """Safe action computation with validation"""
        if self.emergency_stop_active:
            return self.send_emergency_stop()
        
        # Normal action computation
        raw_action = self.run_ai_computation()
        
        # Safety validation
        if self.safety_monitor.is_safe(raw_action):
            self.execute_action(raw_action)
        else:
            self.get_logger().warn('Unsafe action detected, sending safe stop')
            self.send_safe_fallback()
```

## Performance Optimization

### Threading and Concurrency

For real-time Physical AI applications:

```python
class OptimizedPhysicalAgent(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__('optimized_physical_agent')
        
        # Dedicated threads for different tasks
        self.sensor_thread = threading.Thread(target=self.sensor_processing_loop)
        self.ai_thread = threading.Thread(target=self.ai_processing_loop)
        self.control_thread = threading.Thread(target=self.control_output_loop)
        
        # Thread-safe queues for inter-thread communication
        self.sensor_queue = queue.Queue(maxsize=10)
        self.ai_result_queue = queue.Queue(maxsize=5)
        
        # Start threads
        self.sensor_thread.start()
        self.ai_thread.start()
        self.control_thread.start()
```

### Memory Management

For resource-constrained humanoid platforms:

```python
import gc
import psutil

class MemoryEfficientAgent(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__('memory_efficient_agent')
        
        # Memory monitoring
        self.memory_limit = 0.8  # 80% of available memory
        self.monitor_timer = self.create_timer(1.0, self.check_memory_usage)
        
    def check_memory_usage(self):
        """Monitor and manage memory usage"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > self.memory_limit:
            # Force garbage collection
            gc.collect()
            
            # Consider deactivating non-critical components
            self.throttle_non_critical_ai()
```

## Simulation to Reality Transfer

Python agents need to work seamlessly in both simulation and reality:

```python
class SimulationAwareAgent(PhysicalAIAgentNode):
    def __init__(self):
        super().__init__('simulation_aware_agent')
        
        # Detect if running in simulation
        self.is_simulation = self.get_parameter_or_declare(
            'use_simulation', True
        ).value
        
        if self.is_simulation:
            # Adjust timing for simulation speed
            self.control_period = 0.01  # Faster in simulation
        else:
            # Real-time constraints
            self.control_period = 0.02  # 50Hz for real hardware
```

## Summary

Integrating Python agents with ROS controllers is fundamental to creating intelligent Physical AI systems. Key considerations include:

- **Architecture**: Proper separation of concerns between agents
- **Performance**: Threading and optimization for real-time requirements
- **Safety**: Built-in safeguards for physical systems
- **Reliability**: Error handling and fallback mechanisms
- **Scalability**: Support for multiple agents and complex behaviors

The Python ecosystem offers powerful tools for implementing sophisticated Physical AI agents, which can be seamlessly integrated with ROS controllers to create intelligent, embodied systems. The patterns and techniques shown here form the foundation for more advanced Physical AI applications in humanoid robotics.