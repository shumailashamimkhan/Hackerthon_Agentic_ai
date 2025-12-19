---
title: Introduction to ROS 2
sidebar_position: 1
---

# Introduction to ROS 2

## What is ROS 2?

Robot Operating System 2 (ROS 2) is the latest generation of the Robot Operating System, a flexible framework for writing robot software. Unlike its predecessor, ROS 2 is designed from the ground up to support modern robotics applications, including commercial products, safety-critical systems, and distributed multi-robot systems.

### ROS 2 vs. Traditional Robotics Frameworks

ROS 2 distinguishes itself from traditional robotics frameworks in several key ways:

- **Open Source**: Developed collaboratively by the Open Source Robotics Foundation and community partners
- **Middleware Agnostic**: Built on DDS (Data Distribution Service) for robust communication
- **Real-Time Support**: Designed with real-time capabilities in mind
- **Security Features**: Includes authentication, authorization, and encryption
- **Multi-Language Support**: Supports C++, Python, Rust, and more
- **Industry Adoption**: Used by organizations worldwide for commercial products

### The Nervous System Analogy

In the context of Physical AI, we think of ROS 2 as the **nervous system** of the robotic organism. Just as biological nervous systems transmit signals between different parts of the body and coordinate responses, ROS 2 provides the communication infrastructure that allows different components of a robot to work together:

- **Sensory Input**: Sensors like cameras, lidars, and IMUs send data through ROS topics
- **Processing Centers**: AI algorithms and controllers process sensory information
- **Motor Response**: Actuators and effectors carry out actions based on processed commands
- **Coordination**: Coordination nodes manage complex behaviors and decision-making

## Key Concepts in ROS 2

### Nodes
Nodes are the fundamental units of computation in ROS 2. Each node handles a specific task, such as sensor processing, control algorithm execution, or UI display. Nodes can be implemented in different programming languages and can run on the same or different machines.

### Topics & Messages
Topics provide a publish/subscribe communication mechanism between nodes. Publishers send messages on topics, and subscribers receive messages from topics. This decouples nodes from each other - publishers don't need to know who is subscribed, and subscribers don't need to know who is publishing.

### Services & Actions
For request/response communication, ROS 2 provides Services and Actions. Services enable synchronous request/response interactions, while Actions provide a more sophisticated interface for long-running tasks with feedback and goal management.

### Packages
ROS 2 organizes code into packages, which contain nodes, libraries, data files, and configuration. Packages provide modularity and enable code reuse across different robots and projects.

## Why ROS 2 for Physical AI?

ROS 2 is particularly well-suited for Physical AI applications due to several characteristics:

- **Distributed Architecture**: Enables complex robotic systems with many interacting components
- **Simulation Integration**: Tight integration with simulation tools like Gazebo
- **Hardware Abstraction**: Allows same code to run on simulation and real robots
- **Rich Ecosystem**: Extensive libraries for perception, navigation, manipulation, and control
- **Community Support**: Large community developing and maintaining packages
- **Industry Standard**: Widely adopted across academic and commercial robotics

## Core Architecture

ROS 2 architecture enables robots to scale from simple single-robot systems to complex multi-robot and multi-device deployments:

### Client Library
ROS 2 provides client libraries for multiple languages (rclcpp for C++, rclpy for Python, etc.) that implement the ROS 2 wire protocol and DDS communication.

### DDS Implementation
Different DDS implementations (Fast DDS, Cyclone DDS, RTI Connext) provide the underlying communication layer, offering different trade-offs in performance, portability, and features.

### RMW Layer
The ROS Middleware (RMW) layer abstracts the specific DDS implementation, allowing packages to work across different DDS vendors without code changes.

## Physical AI Applications of ROS 2

In the context of Physical AI and humanoid robotics, ROS 2 enables:

- **Embodied Cognition**: Integration of perception, reasoning, and action in the same framework
- **Multi-Modal Sensing**: Handling data from diverse sensor types (vision, touch, proprioception, etc.)
- **Human-Robot Interaction**: Integration of interfaces for natural human-robot communication
- **Embodied Learning**: Framework for acquiring behaviors through physical interaction
- **Simulation-Reality Transfer**: Tools to bridge simulation and real-world deployment

## Getting Started with ROS 2

Before diving deeper into ROS 2 concepts, make sure you have ROS 2 installed. For this book, we recommend using **Humble Hawksbill**, the latest LTS (Long Term Support) release, which provides:

- 5 years of support (until May 2027)
- Ubuntu 22.04 and Debian 11 support
- Python 3.10 compatibility
- Latest features and improvements

In the following sections, we'll dive into the specifics of ROS 2 architecture components and how they apply to humanoid robotics and Physical AI. We'll explore how to implement the nervous system of your humanoid robots using ROS 2 concepts.