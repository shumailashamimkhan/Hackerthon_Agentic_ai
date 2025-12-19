# Data Model: Book Layout - Physical AI & Humanoid Robotics

## Overview
This document defines the data models for the Physical AI & Humanoid Robotics educational book project. Since this is primarily a content-focused documentation site, the data model focuses on the content organization and metadata.

## Content Entities

### Module
**Description**: A major section of the book covering a specific aspect of Physical AI and robotics
**Fields**:
- id: String (unique identifier, e.g. "module-1")
- title: String (e.g. "The Robotic Nervous System (ROS 2)")
- description: String (brief description of the module)
- order: Integer (sequence number 1-5)
- prerequisites: [String] (list of prerequisite concepts)
- learning_outcomes: [String] (list of outcomes after completing the module)
- duration_hours: Integer (estimated time to complete)
- content_path: String (path to content files)

### Chapter
**Description**: A subdivision within a module, representing a focused topic
**Fields**:
- id: String (unique identifier, e.g. "module-1-chapter-1")
- module_id: String (reference to parent module)
- title: String (e.g. "Introduction to ROS 2 Architecture")
- description: String (brief description of the chapter)
- order: Integer (sequence within the module)
- content_path: String (path to the chapter content file)
- estimated_reading_time: Integer (in minutes)

### ContentPage
**Description**: An individual page of content, which could be a chapter, appendix, or standalone article
**Fields**:
- id: String (unique identifier)
- title: String (page title)
- path: String (relative path from docs root)
- type: Enum ["chapter", "appendix", "blog", "resource", "exercise"]
- metadata: Object (frontmatter data like authors, date, tags)
- content: String (Markdown content or path to content file)
- related_pages: [String] (IDs of related pages)

### LearningOutcome
**Description**: A measurable outcome that students should achieve after completing content
**Fields**:
- id: String (unique identifier)
- description: String (what the student should be able to do)
- module_id: String (which module this outcome belongs to)
- verification_method: String (how to verify the outcome - project, quiz, simulation)
- difficulty: Enum ["beginner", "intermediate", "advanced"]

### Resource
**Description**: Supplementary materials like code samples, simulation environments, hardware guides
**Fields**:
- id: String (unique identifier)
- title: String (resource name)
- type: Enum ["code_sample", "simulation", "hardware_guide", "tutorial", "video", "paper"]
- url: String (path or external URL)
- description: String (what the resource contains)
- related_modules: [String] (list of relevant module IDs)
- tags: [String] (technology tags like "ROS2", "Gazebo", "Isaac", etc.)

### CodeSample
**Description**: Embedded or referenced code examples in the documentation
**Fields**:
- id: String (unique identifier)
- title: String (brief description of the sample)
- language: String (programming language)
- code: String (the actual code)
- description: String (explanation of the code)
- usage_context: String (where in the content this appears)
- related_chapters: [String] (chapters that reference this sample)

### SimulationEnvironment
**Description**: Digital twin or simulation environments referenced in the content
**Fields**:
- id: String (unique identifier)
- name: String (e.g. "Gazebo Simulation for Humanoid Locomotion")
- type: Enum ["gazebo", "isaac_sim", "unity", "custom"]
- description: String (what this simulation demonstrates)
- setup_guide_path: String (path to setup instructions)
- assets: [String] (list of required models/URDFs/scenes)
- integration_notes: String (how it connects to the content)
- complexity: Enum ["basic", "intermediate", "advanced"]

## Relationships

- Module contains many Chapter
- Chapter belongs to one Module  
- ContentPage may reference many CodeSample
- Module has many LearningOutcome
- Module may reference many Resource
- SimulationEnvironment relates to many Chapter (many-to-many)

## Validation Rules

1. Module.order must be between 1 and 5 (inclusive)
2. Chapter.order must be unique within its parent Module
3. ContentPage.path must be unique across the site
4. LearningOutcome must be connected to a valid Module
5. CodeSample.language must be a supported language (Python, C++, etc.)
6. SimulationEnvironment.type must be one of the defined enum values

## State Transitions

Content pages have the following lifecycle states:
- draft: Initial state, content is being created
- review: Content ready for review by subject matter experts
- approved: Content verified and approved for publication
- published: Content available in production
- archived: Content no longer current but kept for historical reference