---
title: ROS 2 Development Resources
sidebar_position: 4
---

# ROS 2 Development Resources

This page provides essential links and resources for working with ROS 2 in Physical AI and humanoid robotics applications. ROS 2 serves as the backbone for communication, device control, and system integration in most robotics applications.

## Official ROS 2 Resources

### Core Documentation
- [ROS 2 Documentation](https://docs.ros.org/en/humble/) - Official ROS 2 documentation for Humble Hawksbill (LTS)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html) - Step-by-step tutorials for ROS 2 concepts
- [ROS 2 Concepts](https://docs.ros.org/en/humble/Concepts.html) - Core ROS 2 architecture and concepts
- [ROS 2 Design](https://design.ros2.org/) - Design rationale and architecture decisions

### Installation and Setup
- [ROS 2 Installation Guide](https://docs.ros.org/en/humble/Installation.html) - Complete installation instructions
- [ROS 2 Setup Tutorial](https://docs.ros.org/en/humble/Tutorials/Configuring-ROS2-Environment.html) - Environment setup
- [ROS 2 Network Configuration](https://docs.ros.org/en/humble/How-To-Guides/Setting-up-ROS-2-bridge.html) - Network setup for distributed systems
- [ROS 2 Build Tools](https://docs.ros.org/en/humble/Installation/Ubuntu-Development-Setup.html) - Development environment setup

## ROS 2 Packages For Physical AI

### Navigation and Path Planning
- [Navigation2](https://navigation.ros.org/) - ROS 2 navigation stack with 2D and 3D navigation
- [Nav2 Examples](https://github.com/ros-planning/navigation2_tutorials) - Example navigation implementations
- [MoveIt 2](https://moveit.ros.org/) - Motion planning framework for manipulation
- [Behavior Trees](https://github.com/BehaviorTree/BehaviorTree.CPP) - Task execution framework for robotics

### Perception and Computer Vision
- [Vision](https://index.ros.org/search/?term=vision) - Various vision packages in ROS Index
- [OpenCV Integration](https://github.com/ros-perception/vision_opencv) - ROS 2 OpenCV interface
- [Point Cloud Library (PCL) Integration](https://github.com/ros-perception/pcl_ros) - 3D perception tools
- [Image Transport](https://github.com/ros-perception/image_transport) - Compressed image transport

### Robot Modeling and Simulation
- [URDF (Unified Robot Description Format)](https://wiki.ros.org/urdf) - Robot modeling framework
- [Xacro](https://wiki.ros.org/xacro) - XML macro language for URDF
- [Robot State Publisher](https://github.com/ros/robot_state_publisher) - Joint state to transform publisher
- [Joint State Publisher](https://github.com/ros/joint_state_publisher) - Joint state message publisher

### Control and Hardware Interfaces
- [ros2_control](https://control.ros.org/) - ROS 2 control framework
- [Hardware Interface](https://github.com/ros-controls/ros2_control/tree/master/hardware_interface) - Abstract hardware interface
- [Controller Manager](https://github.com/ros-controls/ros2_control/tree/master/controller_manager) - Controller management
- [Realtime Tools](https://github.com/ros-controls/realtime_tools) - Real-time safe utilities

## Humanoid-Specific ROS 2 Packages

### Humanoid Frameworks
- [Humanoid Navigation](https://github.com/roboticsgroup/humanoid_navigation) - Navigation packages for humanoid robots
- [HRP2 ROS Packages](https://github.com/tORK/hrp2_ros_stack) - ROS packages for HRP2 humanoid
- [Atlas ROS Packages](https://gitlab.com/aries-group/atlas_description) - ROS interfaces for Atlas humanoid
- [WALKGEN](https://github.com/ahornillos/walkgen) - Walk pattern generation for humanoid robots

### Bipedal Control
- [OpenHRP](https://github.com/fkanehiro/openhrp3) - Humanoid robot simulator with ROS interface
- [Biped Controller](https://github.com/ros-controls/ros2_controllers/tree/master/biped_controller) - Specialized controllers
- [Whole Body Control](https://github.com/stack-of-tasks/sot-core) - Whole body control framework
- [Cartesian Impedance Control](https://github.com/ros-controls/ros2_controllers/tree/master/cartesian_trajectory_controller)

### Simulation Integration
- [Gazebo ROS PKGs](https://gazebosim.org/docs/harmonic/ros_integration/) - ROS 2 integration for Gazebo
- [Ignition Gazebo](https://github.com/gazebosim/gz-sim) - Next-generation simulation platform
- [Unity Robotics](https://github.com/Unity-Technologies/ROS-TCP-Connector) - Unity-ROS connection
- [Isaac ROS](https://github.com/NVIDIA-ISAAC-ROS) - NVIDIA's ROS packages for accelerated robotics

## Development Tools and IDEs

### Editors and IDEs
- [VS Code with ROS Extension](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros) - Excellent ROS development support
- [Robot Operating System (ROS) Development Plugin for CLion](https://plugins.jetbrains.com/plugin/10947-robot-operating-system--ros--development) - JetBrains CLion plugin
- [Catkin Tools](https://catkin-tools.readthedocs.io/) - Command-line tools for building ROS workspaces
- [ROS Development Machine](https://github.com/adamheins/docker-ros-dev) - Docker environment for ROS development

### Debugging and Visualization
- [RViz2](https://github.com/ros2/rviz) - 3D visualization tool for ROS 2
- [rqt](https://wiki.ros.org/rqt) - Qt-based framework for GUI plugins
- [rosbag2](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html) - Data recording and playback
- [ros2topic, ros2service, ros2action](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools.html) - Command-line tools for introspection

### Testing and Quality Assurance
- [ROS 2 Testing](https://docs.ros.org/en/humble/How-To-Guides/Ament-CMake-Documentation.html) - Testing frameworks
- [Launch Testing](https://launch.ros.org/) - Test framework for launch files
- [ROS 2 Lint](https://github.com/ros-tooling/lint) - Code style and quality tools
- [Coverage Analysis](https://github.com/ros-tooling/action-ros-ci) - CI tools with coverage

## Community Resources

### Forums and Support
- [ROS Answers](https://answers.ros.org/questions/) - Official question and answer forum
- [ROS Discourse](https://discourse.ros.org/) - General discussions and announcements
- [ROS Melodic/Maintained Distributions](https://discourse.ros.org/c/general/) - Community support
- [GitHub Issues](https://github.com/ros2/ros2/issues) - Bug reports and feature requests

### Tutorials and Courses
- [ROS 2 Beginner Tutorials](https://docs.ros.org/en/humble/Tutorials.html#beginner-level) - Getting started with ROS 2
- [ROS 2 Intermediate Tutorials](https://docs.ros.org/en/humble/Tutorials.html#intermediate-level) - Advanced ROS 2 concepts
- [ROS Industrial Training](https://industrial-training-master.readthedocs.io/en/melodic/) - Industrial robotics applications
- [ROS Robot Programming Book](https://community.robotsource.org/t/ros-robot-programming-book-available-for-free-download/51) - Free book on ROS programming

### YouTube Channels and Videos
- [ROBOTIS OpenMANIPULATOR](https://www.youtube.com/playlist?list=PLRG6WP3c6MoR423Yo9fCdluaT7Z89zY9r) - ROS 2 tutorials
- [The Construct](https://www.youtube.com/c/TheConstruct) - Extensive ROS video tutorials
- [ROS](https://www.youtube.com/channel/UCXu4AXlYJf_BguzC9ftltNQ) - Official ROS YouTube channel
- [NVIDIA Robotics](https://www.youtube.com/c/NVIDIADeveloper) - Isaac ROS tutorials

## ROS 2 for Physical AI Specific Applications

### Machine Learning Integration
- [ROS 2 ML Examples](https://github.com/ros2/examples/tree/master/rclpy/services/minimal_client) - Basic ML service examples
- [TensorFlow in ROS 2](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/ros2) - RealSense with ROS 2
- [PyTorch Integration](https://github.com/ros-controls/ros2_control_demos) - Example neural network implementations
- [OpenVINO ROS](https://github.com/intel/ros_openvino_toolkit) - Intel OpenVINO integration

### Simulation to Real Transfer
- [Gazebo ROS Control](https://github.com/ros-simulation/gazebo_ros_pkgs/wiki) - Control integration for simulation
- [Transfer Learning Tutorials](https://navigation.ros.org/tutorials/docs/navigation2_with_gazebo_simulation.html) - Sim-to-real navigation
- [Domain Randomization](https://github.com/oleg-dsmk/isaac_ros_examples) - Isaac examples with randomization
- [Robot Calibration](https://github.com/ros-industrial-consortium/descartes) - Advanced calibration techniques

### Multi-Robot Systems
- [Multi-Robot Packages](https://github.com/ros-planning/navigation2/tree/main/demos/multirobot_simulation) - Navigation2 multi-robot examples
- [Robot Communication](https://github.com/ethz-asl/ros_communication_patterns) - Reliable robot-to-robot communication
- [Swarm Robotics](https://github.com/dennisss/madrob) - Multi-agent robotics
- [Distributed Control](https://github.com/dennisss/swarm_ros) - Distributed control algorithms

## Best Practices for Physical AI Development

### Code Organization
- [ROS 2 Package Conventions](https://docs.ros.org/en/humble/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html) - Style guidelines
- [Repository Structure](https://github.com/ros2/rclpy/tree/master/examples) - Example repository organization
- [Component-Based Architecture](https://github.com/ros2/rclcpp/tree/master/rclcpp_components) - Reusable components
- [Lifecycle Nodes](https://github.com/ros2/demos/tree/master/lifecycle) - Managing node lifecycles

### Performance Optimization
- [Real-time Programming](https://github.com/ros2/realtime_support) - Real-time considerations in ROS 2
- [Memory Management](https://docs.ros.org/en/humble/How-To-Guides/Real-time-linux.html) - Efficient memory usage
- [Threading Models](https://github.com/ros2/examples/tree/master/rclpy/executors) - Concurrency in ROS 2
- [QoS Configuration](https://github.com/ros2/examples/tree/master/rclpy/topics) - Quality of Service for performance

### Safety and Reliability
- [Safety Controllers](https://github.com/ros-controls/ros2_control/tree/master/controller_interface) - Safety-focused controllers
- [Diagnostic Aggregator](https://github.com/ros/diagnostics/tree/master/diagnostic_aggregator) - System health monitoring
- [Fault Tolerance](https://github.com/ros2/demos/tree/master/dummy_robot) - Handling component failures
- [Security Guidelines](https://docs.ros.org/en/humble/How-To-Guides/Setting-up-Secure-Communication.html) - Secure ROS 2 communication

## Troubleshooting Resources

### Common Issues
- [Troubleshooting Guide](https://docs.ros.org/en/humble/Troubleshooting.html) - Common ROS 2 issues
- [Performance Issues](https://github.com/ros2/demos/tree/master/performance_test) - Performance troubleshooting
- [Networking FAQ](https://github.com/eProsima/Fast-DDS/blob/master/docs/fastdds/ros2/getting_started/net_faq.md) - DDS networking issues
- [Build System Issues](https://index.ros.org/doc/ros2/Troubleshooting/DDS-tuning/) - FastDDS configuration

### Debugging Tools
- [ROS 2 Doctor](https://github.com/ros2/ros2cli/tree/master/ros2doctor) - System diagnostic tool
- [Performance Testing](https://github.com/ros2/demos/tree/master/performance_test) - Performance measurement tools
- [Network Analysis](https://github.com/ros-tools/rosbags) - Network traffic analysis
- [Memory Analysis](https://github.com/ros-tooling/system_metrics_collector) - System resource monitoring

This resource page provides a comprehensive collection of links and information for working with ROS 2 in Physical AI applications. Bookmark this page for quick access to the most relevant ROS 2 resources for your humanoid robotics development.