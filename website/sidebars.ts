import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for our Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'overview',
        'prerequisites',
        'how-to-use',
      ],
      link: {
        type: 'doc',
        id: 'intro',
      },
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/intro-ros2',
        'module-1/nodes-topics-services-actions',
        'module-1/python-agent-integration',
        'module-1/urdf-humanoid-modeling',
        'module-1/examples',
        'module-1/exercises',
      ],
      link: {
        type: 'generated-index',
        slug: 'module-1',
      },
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/physics-simulation',
        'module-2/gazebo-setup',
        'module-2/unity-visualization',
        'module-2/sensor-simulation',
        'module-2/examples',
        'module-2/exercises',
      ],
      link: {
        type: 'generated-index',
        slug: 'module-2',
      },
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3/isaac-sim-intro',
        'module-3/synthetic-data',
        'module-3/isaac-ros',
        'module-3/nav2-bipedal',
        'module-3/examples/isaac-examples',
        'module-3/exercises',
      ],
      link: {
        type: 'generated-index',
        slug: 'module-3',
      },
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4/vla-intro',
        'module-4/whisper-integration',
        'module-4/cognitive-planning',
        'module-4/multimodal-interaction',
        'module-4/ros-translation',
        'module-4/examples/vla-examples',
        'module-4/exercises',
      ],
      link: {
        type: 'generated-index',
        slug: 'module-4',
      },
    },
    {
      type: 'category',
      label: 'Module 5: Capstone Project',
      items: [
        'module-5/capstone-intro',
        'module-5/simulated-humanoid',
        'module-5/natural-language',
        'module-5/navigation-object-recognition',
        'module-5/manipulation',
        'module-5/jetson-deployment',
        'module-5/unitree-deployment'
      ],
      link: {
        type: 'generated-index',
        slug: 'module-5',
      },
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/digital-twin-workstation',
        'appendices/edge-ai-kit',
        'appendices/proxy-robots',
        'appendices/ros2-links',
        'appendices/isaac-sim-links',
        'appendices/unity-links',
        'appendices/llm-links'
      ],
      link: {
        type: 'generated-index',
        slug: 'appendices',
      },
    }
  ],
};

export default sidebars;
