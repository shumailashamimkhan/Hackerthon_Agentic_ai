---
title: Edge AI Kit Setup Guide
sidebar_position: 2
---

# Edge AI Kit Setup Guide

## Introduction to Edge AI for Physical AI Systems

Edge AI processing is crucial for Physical AI systems, especially humanoid robots that need to make real-time decisions without relying on cloud connectivity. Edge AI kits enable deployment of AI models directly on the robot platform, reducing latency and increasing autonomy.

## Hardware Selection for Edge AI

### Recommended Edge AI Platforms

#### NVIDIA Jetson Series
- **Jetson Orin NX (16GB)**: Most powerful option for complex humanoid behaviors
- **Jetson AGX Orin**: Excellent for advanced perception and navigation
- **Jetson Xavier NX**: Good balance of performance and power efficiency
- **Jetson Nano**: For lightweight inference tasks

#### Specifications Comparison
| Platform | GPU Cores | CPU Cores | RAM | Power | AI Perf (INT8) | Use Case |
|----------|-----------|-----------|------|--------|-----------------|----------|
| Jetson Orin NX | 2048 | 8 | 16GB | 15-25W | 77 TOPS | Complex humanoid behaviors |
| Jetson AGX Orin | 2048 | 12 | 32/64GB | 15-60W | 275 TOPS | Advanced AI workloads |
| Jetson Xavier NX | 384 | 6 | 8GB | 10-25W | 21 TOPS | Moderate ML tasks |
| Jetson Nano | 128 | 4 | 4GB | 5-15W | 0.5 TOPS | Lightweight inference |

### Alternative Platforms

#### Intel-Based Options
- **Intel Neural Compute Stick 2**: USB-based AI acceleration
- **Intel UP Squared AI Vision Dev Kit**: x86-based with AI extensions
- **Google Coral Dev Board**: Edge TPU-based acceleration

#### Performance Comparison
- **NVIDIA Jetson**: Superior for computer vision and complex neural networks
- **Intel Options**: Better for traditional ML and some computer vision tasks
- **Google Coral**: Excellent for MobileNet-based models and specific AI tasks

## Setup Procedure

### Initial Hardware Setup

#### 1. Unboxing and Physical Assembly
- Mount Jetson carrier board securely in robot chassis
- Connect appropriate power supply (with proper voltage regulation)
- Verify all mounting screws are tightened to specifications
- Connect cooling solution (heat sink/fan) appropriately

#### 2. Power and Cooling Connection
- Use regulated 12V-19V power supply depending on Jetson model
- Ensure cooling fan is connected and functional
- Check power connector orientation before plugging in
- Allow jetson to boot and verify power LED illuminates

### Software Installation

#### Option 1: JetPack Installation
For NVIDIA Jetson platforms, use JetPack:

```bash
# 1. Download JetPack from NVIDIA developer portal
# 2. Run the installer:
sudo ./JetPack-<version>-linux-x64_b<build>.run

# 3. Select components:
#    - Linux for Tegra (L4T)
#    - CUDA Toolkit
#    - cuDNN
#    - TensorRT
#    - OpenCV
#    - VPI
#    - Isaac ROS packages

# 4. Complete installation and reboot
```

#### Option 2: Custom Ubuntu Installation
Alternatively, use standard Ubuntu with NVIDIA drivers:

```bash
# 1. Flash Ubuntu 22.04 to SD card
# 2. Boot jetson and update system:
sudo apt update && sudo apt upgrade -y

# 3. Install NVIDIA drivers:
sudo apt install nvidia-jetpack -y

# 4. Install CUDA and required libraries:
sudo apt install cuda-libraries-dev-l4t cuda-cudart-dev-l4t
```

### Environment Configuration

#### 1. Python Environment Setup
```bash
# Install Python packages essential for robotics
sudo apt install python3-pip python3-dev python3-venv

# Create virtual environment for robotics projects
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate
pip install --upgrade pip setuptools wheel

# Install core robotics libraries
pip install numpy scipy matplotlib jupyter
pip install opencv-contrib-python-headless
```

#### 2. ROS 2 Installation
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros-iron/setup.bash | sudo bash

# Install ROS 2 Iron packages
sudo apt install ros-iron-desktop
sudo apt install python3-colcon-common-extensions
sudo apt install ros-iron-ros-gz  # For simulation bridge

# Source ROS environment
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### 3. AI Framework Installation
```bash
# Install PyTorch optimized for Jetson
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorRT Python bindings
sudo apt install python3-libnvinfer-dev

# Install Triton Inference Server (if needed)
pip install tritonclient[all]

# Install additional ML libraries
pip install scikit-learn pandas pillow
pip install stable-baselines3[extra]  # For reinforcement learning
```

### Performance Optimization

#### 1. Jetson Clocks Configuration
```bash
# Check current mode
sudo nvpmodel -q

# Set to MAX performance mode (for development)
sudo nvpmodel -m 0

# Apply maximum clocks
sudo jetson_clocks
```

#### 2. Swap Configuration
```bash
# Create swap file for additional virtual memory
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 3. Thermal Management
```bash
# Install jetson stats for monitoring
pip install jetson-stats
sudo systemctl restart jetson_stats.service

# Configure thermal management
cat << EOF | sudo tee /etc/systemd/system/thermal-manager.service
[Unit]
Description=Thermal Management for Jetson
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/thermal_monitor.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable thermal-manager.service
```

## Integration with Robotics Stack

### Isaac ROS Integration

#### Installation
```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src

# Clone Isaac ROS packages
cd ~/isaac_ros_ws/src
git clone -b iron https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b iron https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone -b iron https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
git clone -b iron https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git

# Install dependencies
cd ~/isaac_ros_ws
rosdep install --from-paths src --ignore-src -r -y

# Build packages
colcon build --merge-install --packages-select \
    isaac_ros_common \
    isaac_ros_visual_slam \
    isaac_ros_apriltag \
    isaac_ros_pose_estimation
```

#### Configuration for Humanoid Robots
```yaml
# config/humanoid_isaac_perception.yaml
camera_info_topic: "/camera/color/camera_info"
image_topic: "/camera/color/image_rect_color"
enable_fisheye_distortion: false
rectified_images: true
map_frame: "map"
tracking_frame: "base_link"
robot_base_frame: "base_link"
use_sim_time: false

# SLAM parameters for humanoid navigation
slam_config:
  num_keyframes_per_graph_node: 2
  min_num_stereo_matches: 10
  max_num_tracked_landmarks: 150
  min_z: 0.1
  max_z: 5.0
  num_frames_per_stereo_pair: 1
```

### Perception Pipeline Setup

#### Camera Integration
```python
# Example perception node for Jetson
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

class JetsonPerceptionNode(Node):
    def __init__(self):
        super().__init__('jetson_perception_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load AI model optimized for Jetson
        self.model = self.load_optimized_model()
        
        # Create subscription and publisher
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        self.perception_publisher = self.create_publisher(
            PerceptionResult,  # Custom message type
            '/humanoid/perception_result',
            10
        )
        
        self.get_logger().info('Jetson Perception Node Initialized')
    
    def load_optimized_model(self):
        """Load a TensorRT or INT8 optimized model for Jetson"""
        # Use TensorRT optimized model if available
        try:
            import tensorrt as trt
            # Load optimized model
            return self.load_trt_model()
        except ImportError:
            # Fall back to PyTorch model
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.eval()
            return model
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run inference with the AI model
            results = self.model(cv_image)
            
            # Process results
            if isinstance(results, torch.Tensor):
                # PyTorch model output
                predictions = results.cpu().numpy()
            else:
                # Other model output format
                predictions = results
            
            # Publish perception results
            perception_msg = PerceptionResult()
            # Process predictions into perception message
            # ...
            
            self.perception_publisher.publish(perception_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in perception pipeline: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    perception_node = JetsonPerceptionNode()
    
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Real-time Considerations

#### Latency Optimization
- Use TensorRT for inference acceleration
- Optimize model for INT8 precision where possible
- Use appropriate batch sizes (often 1 for real-time robotics)
- Minimize data copying between CPU and GPU

#### Memory Management
```python
class OptimizedPerceptionNode(Node):
    def __init__(self):
        # Pre-allocate tensors to minimize memory allocation
        self.input_tensor = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device='cuda')
        self.output_tensor = torch.zeros((1, 100, 5), dtype=torch.float32, device='cuda')  # Example shape
        
        # Use tensor pools for frequent allocations
        self.tensor_pool = []
    
    def reuse_tensor(self, shape, dtype=torch.float32, device='cuda'):
        """Reuse tensors from pool to reduce allocation overhead"""
        for i, tensor in enumerate(self.tensor_pool):
            if tensor.shape == shape and tensor.dtype == dtype and tensor.device == torch.device(device):
                return self.tensor_pool.pop(i)
        
        # Create new tensor if none available
        return torch.zeros(shape, dtype=dtype, device=device)
```

## Deployment Considerations

### Power Management

#### Energy Efficiency Strategies
- Use appropriate performance modes for different activities
- Implement power scaling based on computational demand
- Optimize model precision vs. accuracy trade-offs
- Consider duty cycles for non-critical operations

#### Performance Profiles
```bash
# Create performance profile script
cat << 'EOF' > /home/user/performance_manager.sh
#!/bin/bash

case "$1" in
  "low_power")
    # For standby or minimal processing
    sudo nvpmodel -m 3
    ;;
  "balanced")
    # For normal operations
    sudo nvpmodel -m 1
    ;;
  "high_performance")
    # For intensive AI processing
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ;;
  *)
    echo "Usage: $0 {low_power|balanced|high_performance}"
    exit 1
    ;;
esac
EOF

chmod +x /home/user/performance_manager.sh
```

### Thermal Management

#### Active Cooling Control
```python
import subprocess

class ThermalManager:
    def __init__(self, node):
        self.node = node
        self.temperature_threshold = 75.0  # Celsius
        
    def check_temperature(self):
        """Check Jetson's thermal zones"""
        try:
            with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0  # Convert from millidegrees
                
            if temp > self.temperature_threshold:
                self.node.get_logger().warn(f'High temperature detected: {temp:.2f}°C')
                self.reduce_workload()
            else:
                # Gradually increase performance if temperatures allow
                self.restore_performance()
                
            return temp
        except Exception as e:
            self.node.get_logger().error(f'Error reading temperature: {e}')
            return None
    
    def reduce_workload(self):
        """Reduce computational load to lower temperature"""
        subprocess.run(["sudo", "nvpmodel", "-m", "3"])  # Switch to low power mode
        # Reduce sensor update rates
        # Reduce AI model complexity
        # Pause non-critical operations
        pass
    
    def restore_performance(self):
        """Restore normal performance if thermal conditions allow"""
        subprocess.run(["sudo", "nvpmodel", "-m", "1"])  # Switch to balanced mode
        pass
```

### Safety Considerations

#### Watchdog Implementation
```python
import threading
import time

class EdgeAISafetyManager:
    def __init__(self, node):
        self.node = node
        self.watchdog_timer = None
        self.timeout = 5.0  # seconds
        self.safety_callback = None
        
    def start_watchdog(self, safety_cb):
        """Start safety watchdog to reset system if AI hangs"""
        self.safety_callback = safety_cb
        self.reset_watchdog()
        
    def reset_watchdog(self):
        """Reset the safety timer"""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
        
        self.watchdog_timer = threading.Timer(self.timeout, self.safety_callback)
        self.watchdog_timer.start()
    
    def stop_watchdog(self):
        """Stop the safety watchdog"""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
            self.watchdog_timer = None
```

## Best Practices for Edge AI in Physical AI

### Model Optimization
- Use TensorRT for optimized inference on NVIDIA platforms
- Apply quantization techniques (INT8) where precision allows
- Implement pruning to reduce model size
- Consider knowledge distillation for smaller student models

### Deployment Strategies
- Use containerization (Docker) for consistent deployments
- Implement A/B testing for model updates
- Monitor inference performance and accuracy
- Plan for over-the-air updates for deployed robots

### Performance Monitoring
- Track GPU utilization and memory usage
- Monitor inference latency and throughput
- Log model accuracy metrics
- Monitor power consumption patterns

## Troubleshooting

### Common Issues

#### Power Issues
- **Insufficient Power Supply**: Verify power supply can deliver required current
- **Thermal Throttling**: Check cooling solution and thermal paste
- **Voltage Drops**: Use thick gauge wires for high-current applications

#### Performance Issues
- **Slow Inference**: Check model optimization and batch sizes
- **Memory Exhaustion**: Monitor memory usage and implement pooling
- **Thermal Limitations**: Enhance cooling or reduce computational load

#### Compatibility Issues
- **Library Version Mismatches**: Ensure CUDA, cuDNN, and framework versions are compatible
- **Driver Issues**: Update to JetPack-recommended driver versions
- **Hardware Limitations**: Verify model complexity matches platform capabilities

### Diagnostic Tools
```bash
# Jetson diagnostics
sudo jetson_clocks --show  # Show current clock status
jtop  # Real-time monitoring of Jetson performance
sudo tegrastats  # Low-level hardware statistics

# Memory and performance
free -h  # Check system memory
nvidia-smi  # Check GPU usage
iotop  # Check disk I/O performance
```

## Maintenance Guidelines

### Regular Maintenance Tasks
- Update JetPack when new stable releases are available
- Clean cooling fins and check fan operation
- Monitor storage space and clear logs regularly
- Verify backup of important configurations and models

### Long-term Considerations
- Plan for component end-of-life and obsolescence
- Maintain compatibility across software updates
- Backup hardware configurations and performance profiles
- Document all custom optimizations and settings

## Summary

Setting up an effective Edge AI kit for Physical AI and humanoid robotics requires careful consideration of computing power, thermal management, power requirements, and software optimization. The Jetson platform provides an excellent balance of AI performance and power efficiency for robotic applications.

Proper configuration of the Edge AI environment enables humanoid robots to run sophisticated AI algorithms with low latency and high autonomy. This is critical for real-time Physical AI applications where cloud connectivity may not be available or reliable.

The combination of optimized AI frameworks, appropriate hardware selection, and careful thermal and power management ensures that humanoid robots can demonstrate advanced physical AI capabilities in real-world scenarios.