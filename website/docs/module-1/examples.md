---
title: ROS 2 Examples and Code Samples
sidebar_position: 5
---

# ROS 2 Examples and Code Samples

This page contains practical examples and code samples for working with ROS 2 in the context of Physical AI and humanoid robotics.

## Simple Publisher/Subscriber Example

This example demonstrates the basic publish/subscribe pattern in ROS 2, which is fundamental to robot communication.

### Publisher Node

```python title="minimal_publisher.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node

```python title="minimal_subscriber.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

To run these examples:
```bash
# Terminal 1: Run the publisher
ros2 run my_package minimal_publisher

# Terminal 2: Run the subscriber
ros2 run my_package minimal_subscriber
```

## Service Example

This example demonstrates how to create and use services in ROS 2 for request/response communication.

### Service Definition (create in srv/AddTwoInts.srv)

```
int64 a
int64 b
---
int64 sum
```

### Service Server

```python title="add_two_ints_server.py"
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

```python title="add_two_ints_client.py"
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        f'Result of add_two_ints: for {sys.argv[1]} + {sys.argv[2]} = {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Action Example

This example shows how to implement actions for long-running tasks with feedback.

### Action Definition (create in action/Fibonacci.action)

```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

### Action Server

```python title="fibonacci_action_server.py"
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._goal_handle = None
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        if self._goal_handle is not None and self._goal_handle.is_active:
            self.get_logger().info('Rejecting new goal, previous goal still executing')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info(f'Received cancel request for goal')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        self._goal_handle = goal_handle

        # Calculate Fibonacci sequence
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if not goal_handle.is_active:
                self.get_logger().info('Goal aborted')
                return Fibonacci.Result()

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result


def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    executor = MultiThreadedExecutor()
    rclpy.spin(fibonacci_action_server, executor=executor)
    fibonacci_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Physical AI-Specific Examples

### Humanoid Joint State Publisher

This example demonstrates how to publish joint states for a humanoid robot.

```python title="humanoid_joint_publisher.py"
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class HumanoidJointStatePublisher(Node):
    def __init__(self):
        super().__init__('humanoid_joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Define humanoid joint names
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow'
        ]
        
        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)

    def timer_callback(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    humanoid_publisher = HumanoidJointStatePublisher()
    rclpy.spin(humanoid_publisher)
    humanoid_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Simple PID Controller Example

This example shows a basic PID controller implementation for joint control.

```python title="simple_pid_controller.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class SimplePIDController(Node):
    def __init__(self):
        super().__init__('simple_pid_controller')
        
        # PID parameters
        self.kp = 10.0
        self.ki = 0.1
        self.kd = 1.0
        
        self.previous_error = 0.0
        self.integral = 0.0
        
        # Desired position (for demonstration)
        self.target_position = 1.57  # 90 degrees
        
        # Subscription and publication
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)
            
        self.publisher = self.create_publisher(Float64, 'joint_command', 10)
        
        # Control timer (100 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.current_position = 0.0
        self.feedback_valid = False

    def joint_state_callback(self, msg):
        if len(msg.position) > 0:
            # Assuming first joint for simplicity
            self.current_position = msg.position[0]
            self.feedback_valid = True

    def control_loop(self):
        if not self.feedback_valid:
            return
            
        # Calculate error
        error = self.target_position - self.current_position
        
        # Calculate PID terms
        self.integral += error * 0.01  # dt = 0.01s
        derivative = (error - self.previous_error) / 0.01
        
        # Calculate output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update previous error
        self.previous_error = error
        
        # Publish control command
        cmd_msg = Float64()
        cmd_msg.data = float(output)
        self.publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    pid_controller = SimplePIDController()
    rclpy.spin(pid_controller)
    pid_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example Package Structure

For organizing your ROS 2 code, follow this standard package structure:

```
my_robot_pkg/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
├── test/
│   └── test_copyright.py
├── my_robot_pkg/
│   ├── __init__.py
│   ├── minimal_publisher.py
│   ├── minimal_subscriber.py
│   ├── add_two_ints_server.py
│   └── add_two_ints_client.py
└── example_interfaces/
    ├── action/
    │   └── Fibonacci.action
    ├── srv/
    │   └── AddTwoInts.srv
    └── msg/
        └── CustomMessage.msg
```

## Running Examples

To run these examples:

1. Create a new ROS 2 package:
   ```bash
   ros2 pkg create --build-type ament_python my_examples
   ```

2. Place the example files in the appropriate locations within your package

3. Build the package:
   ```bash
   colcon build --packages-select my_examples
   source install/setup.bash
   ```

4. Run the nodes:
   ```bash
   ros2 run my_examples minimal_publisher
   ros2 run my_examples minimal_subscriber
   ```

## Summary

This page provides practical examples of ROS 2 concepts relevant to Physical AI and humanoid robotics. These examples demonstrate fundamental ROS 2 communication patterns that form the foundation of complex robotic systems. As you work with more complex Physical AI examples, you'll build upon these basic patterns to create sophisticated robot applications.