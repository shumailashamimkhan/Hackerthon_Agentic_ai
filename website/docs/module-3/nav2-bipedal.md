---
title: Nav2 Path Planning for Bipedal Humanoid Movement
sidebar_position: 4
---

# Nav2 Path Planning for Bipedal Humanoid Movement

## Introduction to Humanoid Navigation Challenges

Navigation for bipedal humanoid robots presents unique challenges that differ significantly from wheeled or tracked robots. Unlike conventional mobile robots, humanoid robots must maintain balance while navigating, deal with discrete foot placements, and handle complex multi-contact dynamics. This requires specialized path planning approaches that account for bipedal locomotion constraints.

### Key Differences from Wheeled Navigation

1. **Discrete Contact Points**: Humanoid robots have intermittent contact with the ground through feet
2. **Balance Constraints**: Must maintain center of mass within support polygon
3. **Step Sequence Planning**: Need to plan stable step sequences rather than continuous paths
4. **Dynamic Stability**: Gait must be dynamically stable at all times
5. **Terrain Negotiation**: Handle stairs, curbs, and varied terrain patterns

### Physical AI Context

In the Physical AI paradigm, navigation for humanoid robots must integrate perception, planning, and control in a cohesive way. The robot must:
- Perceive the environment to understand traversable areas
- Plan paths that consider both geometric and dynamic constraints
- Execute locomotion with adaptive gait patterns
- Maintain balance and stability throughout movement

## Nav2 Architecture for Humanoid Robots

### Traditional Nav2 vs. Humanoid Nav2

Traditional Nav2 is designed primarily for ground vehicles with continuous contact and differential or Ackermann steering. For humanoid robots, we need to adapt the architecture to handle:

- **Feet-based movement**: Instead of continuous motion, movement is achieved through step-by-step progression
- **Balance-aware planning**: The path planner must account for balance constraints
- **Step sequence generation**: The controller must generate safe stepping sequences
- **Dynamic gait adaptation**: Adjust gait patterns based on terrain and environment

### Modified Nav2 Stack Components

```
┌─────────────────────────────────────────────────────────┐
│                     Nav2 Humanoid Stack                 │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Localization│  │ Humanoid     │  │ Footstep      │  │
│  │ (AMCL/SLAM) │  │ Costmap      │  │ Planner       │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│         │                 │                  │          │
│         ▼                 ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Path Planner (Global)                  ││
│  │  • A* with balance constraints                      ││
│  │  • Step-aware heuristics                          ││
│  │  • Dynamic stability considerations               ││
│  └─────────────────────────────────────────────────────┘│
│                           │                             │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │            Path Post-Processor                      ││
│  │  • Step sequence generation                         ││
│  │  • Balance preservation                           ││
│  │  • Swing-foot trajectory planning                 ││
│  └─────────────────────────────────────────────────────┘│
│                           │                             │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │            Local Planner (Humanoid)                 ││
│  │  • Footstep tracking                                ││
│  │  • Balance feedback control                       ││
│  │  • Gait adaptation                                ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Humanoid-Specific Components

#### 1. Humanoid Costmap Layer

The humanoid-specific costmap layer differs from traditional navigation by incorporating:

- **Stability Constraints**: Regions marked as high-cost if they compromise robot stability
- **Foot Placement Validity**: Valid footholds based on terrain geometry and friction
- **Step Reachability**: Areas reachable given the robot's leg length and step constraints
- **Balance Margin**: Extra cost for areas that reduce balance stability

```cpp title="humanoid_costmap_plugin.cpp"
#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav2_costmap_2d/layered_costmap.hpp>

namespace nav2_humanoid_layers
{

class HumanoidCostmapLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  HumanoidCostmapLayer();

  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw,
                    double* min_x, double* min_y, double* max_x, double* max_y);
  virtual void updateCosts(nav2_costmap_2d::Costmap2D& master_grid,
                   int min_i, int min_j, int max_i, int max_j);

private:
  // Humanoid-specific parameters
  double step_length_max_;
  double step_width_max_;
  double stability_margin_; 
  double min_support_area_;
  
  // Balance model parameters
  double com_height_;
  double max_lean_angle_;
  
  void addStabilityCosts(unsigned char* master_array, int min_i, int min_j, 
                         int max_i, int max_j, unsigned int size_xi);
  void addFootholdValidityCosts(unsigned char* master_array, int min_i, int min_j,
                               int max_i, int max_j, unsigned int size_xi);
  void applyStepReachability(unsigned char* master_array, int min_i, int min_j,
                            int max_i, int max_j, unsigned int size_xi);
};

HumanoidCostmapLayer::HumanoidCostmapLayer() {}

void HumanoidCostmapLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  
  // Get humanoid-specific parameters
  nh.param("step_length_max", step_length_max_, 0.3);      // meters
  nh.param("step_width_max", step_width_max_, 0.2);       // meters  
  nh.param("stability_margin", stability_margin_, 0.1);   // meters
  nh.param("min_support_area", min_support_area_, 0.01);  // sq meters
  nh.param("com_height", com_height_, 0.8);               // meters
  nh.param("max_lean_angle", max_lean_angle_, M_PI/6);    // radians
}

void HumanoidCostmapLayer::updateBounds(double robot_x, double robot_y, double robot_yaw,
                               double* min_x, double* min_y, double* max_x, double* max_y)
{
  *min_x = std::min(*min_x, robot_x - step_length_max_ - 1.0);
  *min_y = std::min(*min_y, robot_y - step_width_max_ - 1.0);
  *max_x = std::max(*max_x, robot_x + step_length_max_ + 1.0);
  *max_y = std::max(*max_y, robot_y + step_width_max_ + 1.0);
}

void HumanoidCostmapLayer::updateCosts(nav2_costmap_2d::Costmap2D& master_grid,
                              int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) return;

  addStabilityCosts(master_array, min_i, min_j, max_i, max_j, size_x_);
  addFootholdValidityCosts(master_array, min_i, min_j, max_i, max_j, size_x_);
  applyStepReachability(master_array, min_i, min_j, max_i, max_j, size_x_);
}

void HumanoidCostmapLayer::addStabilityCosts(unsigned char* master_array, int min_i, int min_j, 
                                     int max_i, int max_j, unsigned int size_xi)
{
  for (int j = min_j; j < max_j; j++) {
    for (int i = min_i; i < max_i; i++) {
      int index = nav2_costmap_2d::getIndex(size_xi, i, j);
      
      double world_x, world_y;
      layered_costmap_->getCostmap()->mapToWorld(i, j, world_x, world_y);
      
      // Calculate if this position would compromise stability
      double stability_cost = calculateStabilityCost(world_x, world_y);
      unsigned char current_cost = master_array[index];
      
      if (stability_cost > current_cost) {
        master_array[index] = (unsigned char)std::min(stability_cost, 254.0);
      }
    }
  }
}

} // namespace nav2_humanoid_layers
```

#### 2. Footstep Planner

The footstep planner is responsible for converting global plans into safe stepping sequences:

```python title="footstep_planner.py"
#!/usr/bin/env python3
"""
Footstep planner for bipedal humanoid navigation
Implements RRT-based planning for safe foot placements
"""

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import heapq
import math

class FootstepPlanner:
    def __init__(self):
        rospy.init_node('footstep_planner')
        
        # Parameters
        self.max_step_length = rospy.get_param('~max_step_length', 0.3)  # meters
        self.max_step_width = rospy.get_param('~max_step_width', 0.2)    # meters
        self.min_step_length = rospy.get_param('~min_step_length', 0.05)  # meters
        self.max_step_turn = rospy.get_param('~max_step_turn', math.pi/3)  # radians
        self.com_height = rospy.get_param('~com_height', 0.8)  # meters
        self.support_polygon_margin = rospy.get_param('~support_polygon_margin', 0.05)  # meters
        
        # Publishers
        self.footstep_path_pub = rospy.Publisher('/footstep_path', Path, queue_size=1)
        self.footstep_viz_pub = rospy.Publisher('/footstep_viz', MarkerArray, queue_size=1)
        
        # Subscribers
        self.global_path_sub = rospy.Subscriber('/global_plan', Path, self.global_path_callback)
        
        # Robot state
        self.current_left_foot = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_right_foot = np.array([0.0, 0.0, 0.0]) # x, y, theta
        self.support_foot = 'left'  # Which foot to move next
        
        # Visualization
        self.marker_id = 0
        
    def global_path_callback(self, msg):
        """Process global path and generate footstep sequence"""
        # Convert global path waypoints to footstep sequence
        footstep_sequence = self.generate_footsteps_from_path(msg.poses)
        
        # Create and publish path of footsteps
        footstep_path = Path()
        footstep_path.header = msg.header
        footstep_path.header.frame_id = "map"
        
        for i, step in enumerate(footstep_sequence):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = step[0]  # x position
            pose.pose.position.y = step[1]  # y position
            pose.pose.position.z = 0.0      # foot height
            # Set orientation based on step direction
            q = self.euler_to_quaternion(0, 0, step[2])  # theta
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            footstep_path.poses.append(pose)
        
        self.footstep_path_pub.publish(footstep_path)
        
        # Publish visualization
        self.visualize_footsteps(footstep_path)
    
    def generate_footsteps_from_path(self, path_poses):
        """
        Generate safe footstep sequence from global path
        Uses a sampling-based approach with balance constraints
        """
        if len(path_poses) < 2:
            return []
        
        footstep_sequence = []
        
        # Start with current robot position
        current_pos = np.array([0.0, 0.0])
        current_theta = 0.0  # Robot orientation
        
        for i in range(1, len(path_poses)):
            # Target position from global path
            target_pos = np.array([path_poses[i].pose.position.x,
                                  path_poses[i].pose.position.y])
            
            # Calculate direction to target
            direction_vec = target_pos - current_pos
            distance = np.linalg.norm(direction_vec)
            direction_unit = direction_vec / distance if distance > 0 else np.array([1, 0])
            
            # For humanoid, we need to step toward target while maintaining balance
            # Alternate between left and right foot movements
            steps_needed = int(np.ceil(distance / self.max_step_length))
            
            for step in range(steps_needed):
                # Calculate next step position based on balance constraints
                next_step = self.calculate_next_balanced_step(
                    current_pos, current_theta, direction_unit, target_pos, 
                    len(footstep_sequence) % 2 == 0  # Alternate feet
                )
                
                if next_step is not None:
                    footstep_sequence.append(next_step)
                    current_pos = np.array([next_step[0], next_step[1]])
                    current_theta = next_step[2]
        
        return footstep_sequence
    
    def calculate_next_balanced_step(self, current_pos, current_theta, direction, target, is_left_step):
        """
        Calculate next foot placement that maintains balance
        Implements a stability-based sampling approach
        """
        # Potential step positions in direction of movement
        step_candidates = []
        
        # Sample potential step positions
        for step_dist in np.linspace(self.min_step_length, self.max_step_length, 5):
            for step_angle_offset in np.linspace(-self.max_step_turn, self.max_step_turn, 7):
                # Calculate step position
                step_angle = current_theta + step_angle_offset
                step_x = current_pos[0] + step_dist * math.cos(step_angle)
                step_y = current_pos[1] + step_dist * math.sin(step_angle)
                
                # Calculate resulting support polygon
                if is_left_step:
                    # Moving left foot, right foot stays in place
                    support_points = [self.get_right_foot_pos(), (step_x, step_y)]
                else:
                    # Moving right foot, left foot stays in place  
                    support_points = [(step_x, step_y), self.get_left_foot_pos()]
                
                # Check if COM projection is within support polygon with margin
                com_proj = self.predict_com_position_after_step(step_x, step_y, is_left_step)
                if self.is_com_stable(com_proj, support_points):
                    # Additional scoring based on direction toward target
                    score = self.score_step_toward_target((step_x, step_y), target)
                    step_candidates.append(((step_x, step_y, step_angle), score))
        
        # Return best candidate
        if step_candidates:
            best_step = max(step_candidates, key=lambda x: x[1])
            return (*best_step[0][:2], best_step[0][2])
        
        return None  # No stable step found
    
    def is_com_stable(self, com_proj, support_points):
        """Check if center of mass projection is within support polygon"""
        # For simplicity, treat as rectangular support with margin
        # In practice, this would be a more complex polygon check
        
        if len(support_points) < 2:
            return False
        
        # Calculate support polygon bounds with stability margin
        min_x = min(p[0] for p in support_points) - self.support_polygon_margin
        max_x = max(p[0] for p in support_points) + self.support_polygon_margin
        min_y = min(p[1] for p in support_points) - self.support_polygon_margin
        max_y = max(p[1] for p in support_points) + self.support_polygon_margin
        
        # Check if COM projection is within bounds
        return min_x <= com_proj[0] <= max_x and min_y <= com_proj[1] <= max_y
    
    def predict_com_position_after_step(self, step_x, step_y, is_left_step):
        """Predict CoM position after taking the step"""
        # Simplified model: assume CoM stays at fixed position relative to stance foot
        if is_left_step:
            # When moving left foot, CoM shifts toward right foot
            stance_foot = self.get_right_foot_pos()
        else:
            stance_foot = self.get_left_foot_pos()
        
        # Predict CoM position based on single support phase dynamics
        return np.array([stance_foot[0], stance_foot[1]])  # Simplified model
    
    def score_step_toward_target(self, step_pos, target):
        """Score step based on progress toward target"""
        current_to_target = np.array(target) - np.array(self.current_robot_pos)
        step_to_target = np.array(target) - np.array(step_pos)
        
        # Prefer steps that make progress toward target
        progress = np.linalg.norm(current_to_target) - np.linalg.norm(step_to_target)
        direction_alignment = np.dot(current_to_target / np.linalg.norm(current_to_target), 
                                   step_to_target / np.linalg.norm(step_to_target))
        
        return progress + 0.5 * direction_alignment
    
    def get_left_foot_pos(self):
        """Get current left foot position"""
        # In practice, this would come from TF or joint states
        return (self.current_left_foot[0], self.current_left_foot[1])
    
    def get_right_foot_pos(self):
        """Get current right foot position""" 
        # In practice, this would come from TF or joint states
        return (self.current_right_foot[0], self.current_right_foot[1])
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [x, y, z, w]
    
    def visualize_footsteps(self, footstep_path):
        """Visualize footsteps in RViz"""
        marker_array = MarkerArray()
        
        for i, pose in enumerate(footstep_path.poses):
            # Footstep marker
            marker = Marker()
            marker.header = footstep_path.header
            marker.ns = "footsteps"
            marker.id = self.marker_id
            self.marker_id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            marker.pose = pose.pose
            marker.scale.x = 0.1  # Length of arrow
            marker.scale.y = 0.05  # Width of arrow
            marker.scale.z = 0.05  # Height of arrow
            
            if i % 2 == 0:  # Left foot
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # Blue
            else:  # Right foot
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0  # Red
                
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        self.footstep_viz_pub.publish(marker_array)

def main():
    try:
        planner = FootstepPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
```

#### 3. Gait Pattern Generator

For smooth and stable bipedal locomotion, we need to implement gait pattern generation:

```python title="gait_pattern_generator.py"
#!/usr/bin/env python3
"""
Gait pattern generator for humanoid robots
Implements stable walking patterns with balance control
"""

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import math

class GaitPatternGenerator:
    def __init__(self):
        rospy.init_node('gait_pattern_generator')
        
        # Gait parameters
        self.step_height = rospy.get_param('~step_height', 0.1)  # meters
        self.step_duration = rospy.get_param('~step_duration', 1.0)  # seconds
        self.stride_length = rospy.get_param('~stride_length', 0.3)  # meters
        self.stride_width = rospy.get_param('~stride_width', 0.2)   # meters
        self.com_height = rospy.get_param('~com_height', 0.8)       # meters
        
        # Walking pattern parameters
        self.walking_frequency = rospy.get_param('~walking_frequency', 0.5)  # Hz
        self.zmp_margin = rospy.get_param('~zmp_margin', 0.05)  # Zero Moment Point safety margin
        
        # Publishers and subscribers
        self.joint_cmd_pub = rospy.Publisher('/joint_commands', Float64MultiArray, queue_size=1)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # Gait state
        self.current_phase = 0.0  # 0.0 to 1.0 for gait cycle
        self.is_stepping = False
        self.next_foot = 'left'  # Which foot to step with next
        
        # Timing
        self.last_update_time = rospy.Time.now()
        self.timer = rospy.Timer(rospy.Duration(0.01), self.gait_update_callback)
        
        # Joint names for humanoid
        self.joint_names = [
            # Left leg
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            # Right leg
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]
        
        # Current joint positions (initialized to neutral stance)
        self.current_positions = np.zeros(len(self.joint_names))
        
    def cmd_vel_callback(self, msg):
        """Receive velocity commands and adjust gait parameters"""
        # Map linear/angular velocities to gait parameters
        self.desired_speed = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.turn_rate = msg.angular.z
        
        # Adjust step frequency based on desired speed
        if self.desired_speed > 0.01:  # If moving forward
            self.walking_frequency = 0.5 + 0.5 * (self.desired_speed / 1.0)  # Scale frequency with speed
        else:
            self.walking_frequency = 0.5  # Default walking frequency
    
    def gait_update_callback(self, event):
        """Main gait update loop"""
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        self.last_update_time = current_time
        
        if dt > 0:
            self.current_phase += self.walking_frequency * dt
            if self.current_phase >= 1.0:
                self.current_phase -= 1.0  # Wrap around
            
            # Generate gait pattern
            joint_commands = self.generate_gait_pattern(self.current_phase)
            
            # Publish joint commands
            cmd_msg = Float64MultiArray()
            cmd_msg.data = joint_commands
            self.joint_cmd_pub.publish(cmd_msg)
    
    def generate_gait_pattern(self, phase):
        """
        Generate joint positions for current gait phase
        Implements a simplified inverted pendulum model for balance
        """
        # Simplified gait - in practice, this would use more sophisticated inverse kinematics
        commands = np.zeros(len(self.joint_names))
        
        # Determine which foot is swing vs stance
        left_swing_phase = self.modulate_phase(phase, 0.0, 0.5)  # Left swings in first half
        right_swing_phase = self.modulate_phase(phase, 0.5, 1.0)  # Right swings in second half
        
        # Left leg pattern
        left_foot_pos = self.calculate_foot_trajectory(left_swing_phase, 'left')
        
        # Right leg pattern  
        right_foot_pos = self.calculate_foot_trajectory(right_swing_phase, 'right')
        
        # Convert foot positions to joint angles (simplified)
        left_joint_angles = self.inverse_kinematics_leg(left_foot_pos, 'left')
        right_joint_angles = self.inverse_kinematics_leg(right_foot_pos, 'right')
        
        # Apply joint angles to command arrays
        commands[0:6] = left_joint_angles  # Left leg joints
        commands[6:12] = right_joint_angles  # Right leg joints
        
        return commands
    
    def modulate_phase(self, phase, start, end):
        """Modulate phase to a specific range [start, end]"""
        if start < end:
            # Normal range
            adjusted = (phase - start) / (end - start)
        else:
            # Wraps around
            adjusted = ((phase - start + 1.0) % 1.0) / (end - start + 1.0)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, adjusted)) if (end - start) != 0 else 0.0
    
    def calculate_foot_trajectory(self, swing_phase, foot_side):
        """
        Calculate foot trajectory during swing phase
        Uses a simplified 3D trajectory with lift and landing
        """
        # Calculate horizontal movement (forward and lateral)
        forward_progress = self.stride_length * swing_phase * (2 - swing_phase)  # Parabolic profile
        lateral_offset = self.stride_width/2 if foot_side == 'left' else -self.stride_width/2
        
        # Calculate vertical movement for step clearance
        vertical_lift = self.step_height * math.sin(math.pi * swing_phase)
        
        # Calculate if in swing phase (simplified)
        if swing_phase <= 1.0:
            # This foot is swinging
            return np.array([forward_progress, lateral_offset, vertical_lift])
        else:
            # This foot is in stance, keep at ground level
            return np.array([self.stride_length * (swing_phase - 1.0), lateral_offset, 0.0])
    
    def inverse_kinematics_leg(self, foot_position, leg_side):
        """
        Simplified inverse kinematics for a 6-DOF leg
        Returns joint angles for desired foot position
        """
        # Simplified 3D position
        x, y, z = foot_position
        
        # Calculate leg angles (simplified 3D model)
        # This is a simplified version - full implementation would use more complex analytical IK
        
        # Hip joints
        hip_yaw = math.atan2(y, x) if x != 0 else 0
        hip_roll = math.atan2(z, abs(y)) if y != 0 else 0
        hip_pitch = math.atan2(x, z) if z != 0 else 0
        
        # Knee angle (based on leg extension)
        leg_length = math.sqrt(x*x + y*y + z*z)
        # Simplified knee angle calculation (in practice, this would solve the full IK)
        knee_angle = max(-0.5, min(1.5, math.pi - (leg_length / 0.5)))  # Knee constraints
        
        # Ankle joints
        ankle_pitch = -hip_pitch  # Compensate for hip
        ankle_roll = -hip_roll    # Compensate for hip
        
        return np.array([hip_yaw, hip_roll, hip_pitch, knee_angle, ankle_pitch, ankle_roll])
    
    def generate_balance_adjustments(self, com_error, zmp_error):
        """
        Generate balance adjustments based on COM and ZMP errors
        """
        # Calculate corrective torques based on errors
        com_correction = 5.0 * com_error  # PD controller gain
        zmp_correction = 3.0 * zmp_error
        
        # Apply corrections to ankle joints for balance
        # This would modify the joint commands generated above
        pass

def main():
    try:
        gpg = GaitPatternGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
```

## Advanced Navigation Techniques for Humanoids

### Model Predictive Control for Navigation

For more advanced humanoid navigation, we can implement Model Predictive Control (MPC) that considers the robot's full dynamics:

```python title="humanoid_mpc_controller.py"
#!/usr/bin/env python3
"""
Model Predictive Control for humanoid navigation
Optimizes both footstep planning and balance control
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
from scipy.optimize import minimize
import math

class HumanoidMPCController:
    def __init__(self):
        rospy.init_node('humanoid_mpc_controller')
        
        # MPC parameters
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 10)
        self.dt = rospy.get_param('~control_dt', 0.1)  # Time step
        self.step_interval = rospy.get_param('~step_interval', 0.5)  # Sec between steps
        
        # Robot parameters
        self.com_height = rospy.get_param('~com_height', 0.8)
        self.max_foot_step = rospy.get_param('~max_foot_step', 0.3)
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/joint_commands', Float64MultiArray, queue_size=1)
        self.path_sub = rospy.Subscriber('/footstep_path', Path, self.path_callback)
        
        # Internal state
        self.current_state = np.zeros(6)  # [x, y, theta, vx, vy, omega]
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.last_step_time = rospy.Time.now()
        
    def path_callback(self, msg):
        """Update the global path"""
        self.waypoints = [(wp.pose.position.x, wp.pose.position.y) 
                         for wp in msg.poses]
        self.current_waypoint_idx = 0
    
    def predict_com_motion(self, initial_state, control_inputs, horizon):
        """
        Predict Center of Mass motion over the horizon
        Uses simplified inverted pendulum model for efficiency
        """
        # Simplified inverted pendulum model: x_ddot = g/h * x
        # Where g is gravity, h is COM height, x is COM position deviation
        
        g = 9.81  # gravity constant
        h = self.com_height  # COM height
        omega = np.sqrt(g / h)  # Natural frequency of pendulum
        
        predicted_states = [initial_state.copy()]
        
        current_state = initial_state.copy()
        
        for k in range(horizon):
            # Inverted pendulum dynamics
            # dx/dt = v
            # dv/dt = omega^2 * x (for x-direction)
            # Similar for y-direction
            
            # Simplified prediction using control inputs
            next_state = current_state.copy()
            
            # Apply control input (simplified model)
            if len(control_inputs) > k:
                u = control_inputs[k]  # Control input
                next_state[0] += next_state[3] * self.dt  # x += vx*dt
                next_state[1] += next_state[4] * self.dt  # y += vy*dt
                next_state[2] += next_state[5] * self.dt  # theta += omega*dt
                
                # Acceleration based on control input
                next_state[3] += u[0] * self.dt  # vx += ax*dt
                next_state[4] += u[1] * self.dt  # vy += ay*dt
                next_state[5] += u[2] * self.dt  # omega += alpha*dt
                
            predicted_states.append(next_state)
            current_state = next_state
            
        return predicted_states
    
    def mpc_objective(self, control_sequence, current_state, target_states):
        """
        Objective function for MPC optimization
        Minimizes tracking error while satisfying constraints
        """
        n_controls = self.prediction_horizon
        n_states = 6
        control_sequence = control_sequence.reshape((n_controls, 3))  # 3D control (x, y, theta)
        
        # Predict states over horizon
        predicted_states = self.predict_com_motion(current_state, control_sequence, self.prediction_horizon)
        
        # Calculate cost: tracking error + control effort
        total_cost = 0.0
        
        for k in range(self.prediction_horizon):
            # Tracking error cost
            if k < len(target_states):
                tracking_error = predicted_states[k+1] - target_states[k]
                total_cost += 0.5 * tracking_error @ tracking_error  # Quadratic cost
            
            # Control effort cost
            control_effort = control_sequence[k] @ control_sequence[k]
            total_cost += 0.1 * control_effort  # Small weight on control effort
        
        return total_cost
    
    def compute_optimal_control(self, current_state, current_waypoint):
        """
        Compute optimal control using MPC formulation
        """
        # Define target states over the prediction horizon
        target_states = []
        for k in range(self.prediction_horizon):
            # Calculate desired position at time step k
            dt_k = (k + 1) * self.dt
            target_pos = current_waypoint
            # Add simple temporal interpolation toward the waypoint
            # In practice, this would be more sophisticated
            target_state = np.array([
                target_pos[0], target_pos[1], 0.0,  # pos x, y, theta
                0.0, 0.0, 0.0  # vel x, y, omega
            ])
            target_states.append(target_state)
        
        # Initial guess for control sequence
        initial_controls = np.zeros(self.prediction_horizon * 3)
        
        # Define constraints (simplified - in reality would be more complex)
        constraints = []
        
        # Optimization
        result = minimize(
            fun=self.mpc_objective,
            x0=initial_controls,
            args=(current_state, target_states),
            method='SLSQP',
            constraints=constraints,
            options={'disp': False, 'maxiter': 100}
        )
        
        if result.success:
            optimal_controls = result.x.reshape((self.prediction_horizon, 3))
            # Return the first control in sequence
            return optimal_controls[0]
        else:
            # Return zero control if optimization fails
            return np.zeros(3)
    
    def step_timing_controller(self):
        """Handle step timing based on MPC predictions"""
        current_time = rospy.Time.now()
        time_since_last_step = (current_time - self.last_step_time).to_sec()
        
        if time_since_last_step >= self.step_interval and self.waypoints:
            # Time for next step
            if self.current_waypoint_idx < len(self.waypoints):
                target_pos = self.waypoints[self.current_waypoint_idx]
                
                # Compute optimal control using MPC
                optimal_control = self.compute_optimal_control(self.current_state, target_pos)
                
                # Convert control to joint commands
                joint_commands = self.convert_control_to_joints(optimal_control)
                
                # Publish commands
                cmd_msg = Float64MultiArray()
                cmd_msg.data = joint_commands
                self.cmd_pub.publish(cmd_msg)
                
                # Update last step time
                self.last_step_time = current_time
                
                # Update waypoint if close enough
                distance_to_waypoint = math.sqrt(
                    (self.current_state[0] - target_pos[0])**2 +
                    (self.current_state[1] - target_pos[1])**2
                )
                
                if distance_to_waypoint < 0.1:  # Threshold for reaching waypoint
                    self.current_waypoint_idx += 1
    
    def convert_control_to_joints(self, com_control):
        """
        Convert COM control commands to joint space commands
        Uses simplified mapping - in practice would use full IK/whole-body control
        """
        # This is a placeholder - real implementation would use inverse kinematics
        # and whole-body control to convert COM forces/torques to joint efforts
        joint_commands = np.zeros(12)  # Assuming 6 joints per leg
        
        # Simplified mapping (in real implementation, use advanced control)
        # Map x,y control to hip joints
        joint_commands[2] = com_control[0] * 0.1  # Left hip pitch
        joint_commands[8] = com_control[0] * 0.1  # Right hip pitch
        
        # Map y control to hip roll
        joint_commands[1] = com_control[1] * 0.1  # Left hip roll
        joint_commands[7] = com_control[1] * 0.1  # Right hip roll
        
        # Map theta control to ankle
        joint_commands[4] = com_control[2] * 0.1  # Left ankle pitch
        joint_commands[10] = com_control[2] * 0.1  # Right ankle pitch
        
        return joint_commands

def main():
    try:
        controller = HumanoidMPCController()
        
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            controller.step_timing_controller()
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
```

## Integration Example: Complete Navigation System

Here's how to integrate all components in a launch file:

```xml title="launch/humanoid_navigation.launch.py">
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file', 
        default=os.path.join(get_package_share_directory('my_humanoid_nav'), 
                           'config', 'humanoid_nav_params.yaml'))
    
    ld = LaunchDescription()
    
    # Include Nav2 lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server', 
                                   'planner_server', 
                                   'controller_server',
                                   'recoveries_server',
                                   'bt_navigator',
                                   'waypoint_follower']}]
    )
    
    # Global planner with humanoid constraints
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('global_costmap/costmap_raw', '/humanoid_costmap/costmap_raw'),
                   ('global_costmap/costmap', '/humanoid_costmap/costmap')]
    )
    
    # Local controller for humanoid stepping
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server', 
        name='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', '/humanoid_cmd_vel')]
    )
    
    # Footstep planner node
    footstep_planner = Node(
        package='my_humanoid_nav',
        executable='footstep_planner',
        name='footstep_planner',
        parameters=[params_file],
        remappings=[('/global_plan', '/plan')]
    )
    
    # Gait pattern generator
    gait_generator = Node(
        package='my_humanoid_nav',
        executable='gait_pattern_generator', 
        name='gait_pattern_generator',
        parameters=[params_file]
    )
    
    # MPC controller (if using advanced control)
    mpc_controller = Node(
        package='my_humanoid_nav',
        executable='humanoid_mpc_controller',
        name='mpc_controller',
        parameters=[params_file]
    )
    
    # Add all actions to the launch description
    ld.add_action(lifecycle_manager)
    ld.add_action(planner_server)
    ld.add_action(controller_server) 
    ld.add_action(footstep_planner)
    ld.add_action(gait_generator)
    # Uncomment the next line if using MPC controller
    # ld.add_action(mpc_controller)
    
    return ld
```

And a configuration file for the complete system:

```yaml title="config/humanoid_nav_params.yaml"
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    set_initial_pose: true
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_sigma_hit: 0.2
    max_beams: 60
    max_particles: 2000
    min_particles: 100
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 40
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "package://nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "package://nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_consistent_localizer_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transformer_bt_node
    - nav2_get_costmap_service_bt_node
    - nav2_get_costmap_action_bt_node
    - nav2_get_local_plan_action_bt_node
    - nav2_get_global_plan_action_bt_node
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["general_goal_checker"] 
    controller_plugins: ["HumanoidMpcController"]
    
    # HumanoidMpcController
    HumanoidMpcController:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 20
      control_horizon: 2
      timestep: 0.1
      cost_function_weight: [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      control_variance: [0.8, 0.8, 0.8]
      nominal_control: [0.0, 0.0, 0.0]
      control_bounds: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
      trajectory_generator_plugin: "dwb_plugins::DWBLocalPlanner"
      penalize_negative_x: true

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3
      plugins: ["humanoid_layer", "inflation_layer"]
      humanoid_layer:
        plugin: "nav2_humanoid_layers::HumanoidCostmapLayer"
        enabled: True
        footprint_padding: 0.01
        step_length_max: 0.3
        step_width_max: 0.2
        stability_margin: 0.1
        min_support_area: 0.01
        com_height: 0.8
        max_lean_angle: 0.524
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        enabled: True
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: "map"
      robot_base_frame: "base_link" 
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"