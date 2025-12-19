// Content models for the Physical AI & Humanoid Robotics Book
// Based on the data model defined in the project specification

// Module: A major section of the book covering a specific aspect of Physical AI and robotics
export interface Module {
  id: string; // unique identifier, e.g. "module-1"
  title: string; // e.g. "The Robotic Nervous System (ROS 2)"
  description: string; // brief description of the module
  order: number; // sequence number 1-5
  prerequisites: string[]; // list of prerequisite concepts
  learningOutcomes: string[]; // list of outcomes after completing the module
  durationHours: number; // estimated time to complete
  contentPath: string; // path to content files
  chapters: Chapter[]; // list of chapters in this module
}

// Chapter: A subdivision within a module, representing a focused topic
export interface Chapter {
  id: string; // unique identifier, e.g. "module-1-chapter-1"
  moduleId: string; // reference to parent module
  title: string; // e.g. "Introduction to ROS 2 Architecture"
  description: string; // brief description of the chapter
  order: number; // sequence within the module
  contentPath: string; // path to the chapter content file
  estimatedReadingTime: number; // in minutes
  learningOutcomes: LearningOutcome[]; // outcomes for this chapter
  resources: Resource[]; // resources related to this chapter
  codeSamples: CodeSample[]; // code samples in this chapter
}

// ContentPage: An individual page of content, which could be a chapter, appendix, or standalone article
export interface ContentPage {
  id: string; // unique identifier
  title: string; // page title
  path: string; // relative path from docs root
  type: 'chapter' | 'appendix' | 'blog' | 'resource' | 'exercise'; // content type
  metadata: Record<string, any>; // frontmatter data like authors, date, tags
  content: string; // the actual content
  relatedPages: string[]; // IDs of related pages
}

// LearningOutcome: A measurable outcome that students should achieve after completing content
export interface LearningOutcome {
  id: string; // unique identifier
  description: string; // what the student should be able to do
  moduleId: string; // which module this outcome belongs to
  verificationMethod: string; // how to verify the outcome - project, quiz, simulation
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

// Resource: Supplementary materials like code samples, simulation environments, hardware guides
export interface Resource {
  id: string; // unique identifier
  title: string; // resource name
  type: 'code_sample' | 'simulation' | 'hardware_guide' | 'tutorial' | 'video' | 'paper';
  url: string; // path or external URL
  description: string; // what the resource contains
  relatedModules: string[]; // list of relevant module IDs
  tags: string[]; // technology tags like "ROS2", "Gazebo", "Isaac", etc.
}

// CodeSample: Embedded or referenced code examples in the documentation
export interface CodeSample {
  id: string; // unique identifier
  title: string; // brief description of the sample
  language: string; // programming language
  code: string; // the actual code content
  description: string; // explanation of the code
  usageContext: string; // where in the content this appears
  relatedChapters: string[]; // chapters that reference this sample
}

// SimulationEnvironment: Digital twin or simulation environments referenced in the content
export interface SimulationEnvironment {
  id: string; // unique identifier
  name: string; // e.g. "Gazebo Simulation for Humanoid Locomotion"
  type: 'gazebo' | 'isaac_sim' | 'unity' | 'custom';
  description: string; // what this simulation demonstrates
  setupGuidePath: string; // path to setup instructions
  assets: string[]; // list of required models/URDFs/scenes
  integrationNotes: string; // how it connects to the content
  complexity: 'basic' | 'intermediate' | 'advanced';
}

// Content state enum for tracking lifecycle
export enum ContentState {
  draft = "draft",
  review = "review",
  approved = "approved",
  published = "published",
  archived = "archived"
}