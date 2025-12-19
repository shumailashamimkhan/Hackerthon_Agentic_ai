---
title: Nodes, Topics, Services, and Actions
sidebar_position: 2
---

# Nodes, Topics, Services, and Actions

## Understanding the ROS 2 Communication Paradigm

ROS 2 implements a distributed computing architecture where different processes (nodes) communicate with each other through a publish-subscribe model and request-response patterns. The four fundamental communication mechanisms in ROS 2 are:

- **Nodes**: Individual processes that perform computation
- **Topics**: Publish-subscribe channels for streaming data
- **Services**: Request-response communication for synchronous operations
- **Actions**: Communication for long-running tasks with feedback

These mechanisms form the backbone of any ROS 2 application, especially in Physical AI and humanoid robotics.

## Nodes: The Computational Units

### What is a Node?

A node is an executable that uses ROS 2 client libraries to communicate with other nodes. In the context of Physical AI:

- Each sensor (camera, LIDAR, IMU) runs in its own node
- Each controller (motor, servo, actuator) runs in its own node
- Each algorithmic component (perception, planning, learning) runs in its own node
- Each interface (GUI, web interface, mobile app) runs in its own node

### Node Characteristics

- **Lightweight**: Nodes are designed to be simple and focused on a single task
- **Modular**: Nodes can be developed, compiled, and run independently
- **Interchangeable**: Nodes can be swapped out with alternative implementations
- **Scalable**: Multiple nodes can run on single or multiple machines

### Node Lifecycle

In humanoid robotics applications, nodes typically follow this lifecycle:

1. **Initialization**: Connect to hardware, load configurations
2. **Activation**: Begin processing and communication
3. **Active**: Process callbacks and send/receive messages
4. **Deactivation**: Gracefully pause operations
5. **Cleanup**: Release resources and shut down

### Best Practices for Physical AI Nodes

- Keep nodes single-purpose and focused
- Handle failures gracefully with recovery mechanisms
- Implement proper logging and diagnostics
- Design with real-time constraints in mind
- Consider computational requirements for humanoid platforms

## Topics: Publish-Subscribe Communication

### Topic Fundamentals

Topics implement a publish-subscribe communication pattern where:

- One or more nodes publish messages to a topic
- Zero or more nodes subscribe to a topic
- Messages are sent asynchronously and can be lost
- No direct connection exists between publishers and subscribers

### Use Cases in Physical AI

Topics are ideal for:

- **Sensor data streams**: Camera images, LIDAR scans, IMU readings
- **Control commands**: Joint positions, velocities, torques
- **State information**: Robot pose, joint angles, battery levels
- **Event notifications**: Collision warnings, emergency stops

### Quality of Service (QoS) Settings

For real-time Physical AI applications, QoS settings are crucial:

- **Reliability**: Choose between reliable (ensure all messages arrive) or best-effort (faster but may lose messages)
- **Durability**: Whether late-joining subscribers get previous messages
- **History**: How many previous messages to store
- **Depth**: Buffer size for incoming/outgoing messages

```python
# Example: Image topic with appropriate QoS for real-time vision
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

image_qos = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)
```

### Topic Design for Physical AI

When designing topics for humanoid robotics:

1. **Minimize bandwidth**: Send only necessary data
2. **Optimize frequency**: Match sensor rates to algorithm requirements
3. **Consider time synchronization**: Use timestamps for sensor fusion
4. **Design for fault tolerance**: Handle dropped messages gracefully

## Services: Request-Response Communication

### Service Fundamentals

Services implement a synchronous request-response pattern where:

- One node provides a service
- Multiple clients can request the service
- Communication is synchronous - client waits for response
- Requests are processed sequentially by the service

### Use Cases in Physical AI

Services are appropriate for:

- **Configuration changes**: Update parameters, calibration
- **One-time queries**: Current state, available modes
- **State machine transitions**: Change operational mode
- **Resource allocation**: Request exclusive access to hardware

### Service Design for Physical AI

Consider time limits for services in Physical AI:

- **Fast services** (under 100ms): Configuration updates, simple queries
- **Medium services** (100ms-2s): Calibration, complex queries
- **Slow services** (over 2s): Should be converted to Actions instead

## Actions: Long-Running Tasks with Feedback

### Action Fundamentals

Actions are designed for long-running tasks with three components:

- **Goal**: Request for a long-running task
- **Feedback**: Periodic updates during execution
- **Result**: Final outcome when task is completed (success/failure)

### Use Cases in Physical AI

Actions are ideal for:

- **Navigation**: Move to a position with real-time feedback
- **Manipulation**: Grasp objects with progress updates
- **Learning**: Train models with progress reporting
- **Calibration**: Lengthy calibration procedures with status updates

### Action Design for Physical AI

When designing actions for humanoid robots:

- Define clear feedback intervals (every 100ms for navigation, 10ms for control)
- Implement proper cancellation mechanisms
- Design for graceful interruption and recovery
- Consider safety implications of long-running tasks

## Physical AI Communication Patterns

### The Nervous System Pattern

In humanoid robotics, ROS 2 communication patterns mirror biological nervous systems:

- **Sensory neurons**: Sensors publish data on topics
- **Motor neurons**: Controllers send commands on topics
- **Interneurons**: Processing nodes subscribe to sensors and publish to controllers
- **Central Pattern Generators**: Specialized nodes for rhythmic behaviors (walking, breathing)

### Hierarchical Control

Physical AI systems often implement hierarchical control:

- **Low-level**: Joint controllers (1 kHz) - Real-time nodes
- **Mid-level**: Motion planners (10-100 Hz) - Feedback control nodes
- **High-level**: Behavior planners (1-10 Hz) - Cognitive decision nodes

### Sensor Fusion Pattern

Multiple sensors contribute to perception:

- **Temporal fusion**: Combine readings over time
- **Spatial fusion**: Combine readings from different sensors
- **Modality fusion**: Combine different types of sensors

## Implementation Example: Humanoid Joint Controller

Here's how these concepts come together in a simple humanoid joint controller:

```
[Joint State Publisher] → [joint_states topic] → [Robot State Publisher]
                                    ↓
[Controller Manager] ← [controller_names service] → [Position Controller]
     ↑
[joint_trajectory_action] ← [MoveIt!] → [Planning Scene Monitor]
```

This architecture uses:
- **Topics** for continuous state updates
- **Services** for configuration queries
- **Actions** for trajectory execution
- **Nodes** for each component

## Designing for Performance in Physical AI

### Latency Requirements

Different Physical AI components have different latency requirements:

- **Control loops**: < 1ms (critical for stability)
- **Perception**: < 50ms (for real-time reaction)
- **Planning**: < 200ms (for smooth interaction)
- **Reasoning**: < 1000ms (for natural interaction)

### Bandwidth Considerations

Humanoid robots have limited computational resources:

- **Wireless bandwidth**: Often limited in robotics applications
- **Processing power**: Embedded platforms have constraints
- **Memory usage**: Efficient message passing is crucial
- **Energy efficiency**: Critical for mobile humanoid robots

## Summary

Understanding ROS 2 communication paradigms is essential for building robust Physical AI systems. The choice between topics, services, and actions depends on:

- **Frequency**: Continuous data → Topics, Occasional → Services/Actions
- **Latency**: Real-time → Topics, Blocking acceptable → Services/Actions
- **Duration**: Instantaneous → Services, Long-running → Actions
- **Reliability**: Critical → Services/Actions, Streaming → Topics

In the next section, we'll explore how to implement these concepts to integrate Python agents with ROS controllers, building on this foundational understanding.