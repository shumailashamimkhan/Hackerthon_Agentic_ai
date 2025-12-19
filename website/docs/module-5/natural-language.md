---
title: Natural Language Command Processing
sidebar_position: 3
---

# Natural Language Command Processing

## Introduction to Language Processing for Humanoid Robots

Natural Language Processing (NLP) for humanoid robots enables intuitive human-robot interaction through everyday language commands. In Physical AI applications, this involves not just understanding language, but connecting linguistic concepts to physical actions and environmental contexts. This creates unique challenges compared to traditional virtual assistants that don't need to affect the physical world.

### The Language-to-Action Pipeline

In Physical AI systems, the natural language processing pipeline extends beyond understanding to execution:

1. **Speech Recognition**: Converting spoken language to text
2. **Natural Language Understanding (NLU)**: Extracting meaning and intent
3. **Context Resolution**: Grounding language in physical context
4. **Action Planning**: Translating language to executable robot behaviors
5. **Execution Monitoring**: Ensuring actions align with command intent
6. **Feedback Generation**: Providing status and confirmation to user

### Challenges in Physical AI NLP

Unlike general-purpose language models, humanoid robotics NLP must address:

- **Embodied Grounding**: Connecting words to physical objects and locations
- **Actionability**: Ensuring commands result in safe, achievable robot behaviors
- **Multimodal Integration**: Combining language with visual, auditory, and proprioceptive information
- **Real-time Processing**: Operating within real-time constraints of physical systems
- **Robustness**: Handling ambiguous, incomplete, or noisy language inputs
- **Safety**: Preventing dangerous interpretations of natural language commands

## Architecture for NLP in Humanoid Systems

### Component-Based Architecture

A typical natural language processing pipeline for humanoid robots consists of several interconnected components:

```
[Speech Input] → [ASR] → [Text] → [NLU] → [Structured Intent] 
                                            ↓
                                    [Context Integrator] 
                                            ↓
                                    [Action Planner] 
                                            ↓
                                    [Execution System]
```

Each component must be designed for the specific requirements of physical AI systems.

### ASR (Automatic Speech Recognition) Systems

#### Whisper Integration for Humanoid Robotics
Whisper is particularly valuable for robotics due to its multilingual capabilities and robustness:

```python
import whisper
import torch
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SpokenCommand:
    text: str
    confidence: float
    timestamp: float
    speaker_id: Optional[str] = None
    audio_data: Optional[np.ndarray] = None

class WhisperASRInterface:
    def __init__(self, model_size="small", language="en", sample_rate=16000):
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        
        # Load Whisper model
        self.model = whisper.load_model(model_size)
        
        # Audio processing
        self.audio_buffer = np.array([])
        self.buffer_size = sample_rate * 2  # 2 seconds of audio
        self.min_speech_duration = sample_rate * 0.5  # 0.5 seconds minimum
        
        # Thread safety
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.is_listening = False
        
        self.get_logger().info(f'Whisper ASR interface initialized with {model_size} model')
    
    def start_listening(self):
        """Start the audio processing thread"""
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self.listening_thread.start()
    
    def stop_listening(self):
        """Stop the audio processing thread"""
        self.is_listening = False
        if hasattr(self, 'listening_thread'):
            self.listening_thread.join()
    
    def process_audio_chunk(self, audio_data):
        """Process an incoming audio chunk"""
        # Add to processing queue
        self.audio_queue.put(audio_data)
    
    def _audio_processing_loop(self):
        """Continuously process audio and detect speech commands"""
        while self.is_listening:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                
                # Process when buffer has enough data
                if len(self.audio_buffer) >= self.buffer_size:
                    # Check for speech activity before processing
                    if self._has_speech_activity(self.audio_buffer):
                        # Process with Whisper
                        result = self.model.transcribe(
                            self.audio_buffer,
                            language=self.language,
                            temperature=0.0,  # Deterministic output
                            compression_ratio_threshold=2.0,
                            logprob_threshold=-1.0,
                            no_speech_threshold=0.6
                        )
                        
                        if result and result["text"].strip():
                            confidence = self._estimate_confidence(result)
                            
                            if confidence > 0.7:  # Confidence threshold
                                command = SpokenCommand(
                                    text=result["text"].strip(),
                                    confidence=confidence,
                                    timestamp=time.time()
                                )
                                self.command_queue.put(command)
                    
                    # Reset buffer (keeping some overlap for continuity)
                    self.audio_buffer = self.audio_buffer[-int(self.sample_rate * 0.5):]  # 0.5s overlap
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio processing: {e}')
                continue
    
    def _has_speech_activity(self, audio_buffer):
        """Simple voice activity detection"""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_buffer**2))
        
        # Simple threshold (in practice, use more sophisticated VAD)
        threshold = 0.01  # Adjust based on environment
        return rms > threshold
    
    def _estimate_confidence(self, whisper_result):
        """Estimate confidence from Whisper results"""
        if "avg_logprob" in whisper_result:
            avg_logprob = whisper_result["avg_logprob"]
            # Convert to confidence (0 to 1 scale)
            # Log prob is typically negative, more negative is less confident
            confidence = max(0.0, min(1.0, (avg_logprob + 1.5) / 2.5))  # Normalize based on expected range
        else:
            confidence = 0.5  # Default if no confidence available
        
        return confidence
    
    def get_processed_command(self):
        """Get the next processed command from the queue"""
        try:
            command = self.command_queue.get_nowait()
            return command
        except queue.Empty:
            return None

# Example of usage in a humanoid system
class HumanoidCommandProcessor:
    def __init__(self):
        self.asr_interface = WhisperASRInterface(model_size="base")
        self.nlu_system = NaturalLanguageUnderstandingSystem()
        self.context_resolver = ContextResolver()
        self.action_planner = ActionPlanner()
        
        # Start listening
        self.asr_interface.start_listening()
        
    def process_commands(self):
        """Process any available commands"""
        command = self.asr_interface.get_processed_command()
        if command:
            # Process through NLU pipeline
            intent = self.nlu_system.parse_intent(command.text)
            
            # Resolve with context
            resolved_intent = self.context_resolver.resolve(intent, self.get_current_context())
            
            # Plan actions
            action_plan = self.action_planner.create_plan(resolved_intent)
            
            # Execute actions (would typically be asynchronous)
            self.execute_action_plan(action_plan)
            
            # Confirm execution
            self.provide_feedback_to_user(action_plan, command.confidence)
    
    def get_current_context(self):
        """Get current robot and environment context"""
        # This would pull from robot state and perception systems
        return {
            'robot_pose': self.get_robot_position(),
            'detected_objects': self.get_perceived_objects(),
            'current_task': self.get_active_task(),
            'battery_level': self.get_battery_level(),
            'last_interaction': self.get_last_interaction_time()
        }
```

### Natural Language Understanding (NLU)

#### Intent Classification for Physical AI

Unlike general NLP systems, Physical AI NLU must be aware of robot capabilities and environmental constraints:

```python
from enum import Enum
from typing import NamedTuple, List
import re
import spacy
from dataclasses import dataclass

class IntentType(Enum):
    NAVIGATE = "navigate"
    MANIPULATE = "manipulate"
    INTERACT = "interact"
    QUERY = "query"
    FOLLOW = "follow"
    STOP = "stop"
    WAIT = "wait"

@dataclass
class ParsedIntent:
    intent_type: IntentType
    entities: Dict[str, List[str]]
    confidence: float
    raw_text: str

class NaturalLanguageUnderstandingSystem:
    def __init__(self):
        # Load spaCy model for linguistic processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define intent patterns specific to robotics
        self.intent_patterns = {
            IntentType.NAVIGATE: [
                r'\bgo to (the )?(?P<location>\w+)\b',
                r'\bmove to (the )?(?P<location>\w+)\b',
                r'\bmarch to (the )?(?P<location>\w+)\b',
                r'\bwalk to (the )?(?P<location>\w+)\b',
                r'\brun to (the )?(?P<location>\w+)\b',
                r'\bproceed to (the )?(?P<location>\w+)\b',
                r'\bhead to (the )?(?P<location>\w+)\b',
                r'\btravel to (the )?(?P<location>\w+)\b',
                r'\bnavigate to (the )?(?P<location>\w+)\b'
            ],
            IntentType.MANIPULATE: [
                r'\bpick up (the )?(?P<object>\w+)\b',
                r'\bgrasp (the )?(?P<object>\w+)\b',
                r'\breach for (the )?(?P<object>\w+)\b',
                r'\bgrab (the )?(?P<object>\w+)\b',
                r'\btake (the )?(?P<object>\w+)\b',
                r'\blift (the )?(?P<object>\w+)\b',
                r'\bstretch for (the )?(?P<object>\w+)\b',
                r'\breach over to grab (the )?(?P<object>\w+)\b',
                r'\bcapture (the )?(?P<object>\w+)\b',
                r'\bcollect (the )?(?P<object>\w+)\b',
                r'\bgather (the )?(?P<object>\w+)\b',
                r'\bget (the )?(?P<object>\w+)\b',
                r'\bretrieve (the )?(?P<object>\w+)\b',
                r'\bfetch (the )?(?P<object>\w+)\b',
                r'\bplace (the )?(?P<obj>\w+) (on|at|in) (the )?(?P<loc>\w+)\b',
                r'\bput (the )?(?P<obj>\w+) (on|at|in) (the )?(?P<loc>\w+)\b',
                r'\bset (the )?(?P<obj>\w+) (on|at|in) (the )?(?P<loc>\w+)\b',
                r'\bdrop (the )?(?P<obj>\w+) (on|at|in) (the )?(?P<loc>\w+)\b',
                r'\brelease (the )?(?P<obj>\w+)\b'
            ],
            IntentType.INTERACT: [
                r'\bgreet (the )?(?P<person>\w+)\b',
                r'\bwave to (the )?(?P<person>\w+)\b',
                r'\bsay hello to (the )?(?P<person>\w+)\b',
                r'\bintroduce yourself to (the )?(?P<person>\w+)\b',
                r'\bfollow (the )?(?P<person>\w+)\b',
                r'\bescort (the )?(?P<person>\w+) to (the )?(?P<location>\w+)\b',
                r'\bcollaborate with (the )?(?P<person>\w+)\b',
                r'\bassist (the )?(?P<person>\w+)\b',
                r'\bhelp (the )?(?P<person>\w+)\b',
                r'\bshake hands with (the )?(?P<person>\w+)\b'
            ],
            IntentType.QUERY: [
                r'\bwhere is (the )?(?P<entity>\w+)\b',
                r'\bwhat is (the )?(?P<entity>\w+)\b',
                r'\bhow many (are there|do you see|can you spot)\b',
                r'\bdescribe (the )?(?P<scene>\w+|area|environment|room)\b',
                r'\bwhat do you see\b',
                r'\breport on\b',
                r'\bstatus of\b',
                r'\btell me about\b'
            ],
            IntentType.STOP: [
                r'\bstop\b',
                r'\bhalt\b',
                r'\bfreeze\b',
                r'\bpause\b',
                r'\bemergency stop\b',
                r'\bsafety stop\b'
            ],
            IntentType.WAIT: [
                r'\bwait\b',
                r'\bwait for me\b',
                r'\bstand by\b',
                r'\bstay here\b',
                r'\bstay put\b',
                r'\bremain\b'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'location': [
                'kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'dining room',
                'hallway', 'garage', 'garden', 'patio', 'entrance', 'exit',
                'table', 'chair', 'couch', 'counter', 'desk', 'shelf', 'cabinet'
            ],
            'object': [
                'cup', 'bottle', 'book', 'ball', 'box', 'phone', 'tablet', 'keys',
                'glasses', 'apple', 'banana', 'toy', 'tool', 'pen', 'pencil'
            ],
            'person': [
                'person', 'man', 'woman', 'child', 'adult', 'human', 'operator', 'user'
            ]
        }
    
    def parse_intent(self, text: str) -> ParsedIntent:
        """Parse the intent from text using pattern matching"""
        text_lower = text.lower()
        doc = self.nlp(text) if self.nlp else None
        
        # Pattern-based matching
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract entities from the match
                    entities = match.groupdict()
                    
                    # If no named entities were extracted by regex, try NER
                    if not entities and doc:
                        entities = self._extract_entities_ner(doc)
                    
                    # Calculate confidence based on match strength
                    confidence = self._calculate_pattern_confidence(match, text)
                    
                    return ParsedIntent(
                        intent_type=intent_type,
                        entities=entities,
                        confidence=confidence,
                        raw_text=text
                    )
        
        # If no pattern matches, try NER approach
        if doc:
            return self._parse_intent_ner(doc, text)
        
        # Default: Unknown intent with low confidence
        return ParsedIntent(
            intent_type=None,
            entities={},
            confidence=0.1,
            raw_text=text
        )
    
    def _extract_entities_ner(self, doc) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER"""
        entities = {'object': [], 'location': 'person': []}
        
        for ent in doc.ents:
            if ent.label_ in ['OBJECT', 'FACILITY', 'LOCATION']:
                entities['location'].append(ent.text)
            elif ent.label_ in ['PERSON']:
                entities['person'].append(ent.text)
        
        # Also extract noun phrases that might represent objects
        for token in doc:
            if token.pos_ == 'NOUN':
                # Add noun and adjacent adjectives
                phrase = token.text
                if token.i > 0 and doc[token.i - 1].pos_ == 'ADJ':
                    phrase = doc[token.i - 1].text + " " + phrase
                entities['object'].append(phrase)
        
        return entities
    
    def _parse_intent_ner(self, doc, text: str) -> ParsedIntent:
        """Parse intent using spaCy NER and dependency parsing"""
        # Analyze dependencies to identify intent
        for token in doc:
            if token.lemma_ in ['go', 'move', 'navigate', 'walk', 'head', 'proceed', 'travel']:
                return ParsedIntent(IntentType.NAVIGATE, self._extract_entities_ner(doc), 0.6, text)
            elif token.lemma_ in ['pick', 'grasp', 'grab', 'take', 'lift', 'get', 'fetch', 'collect']:
                return ParsedIntent(IntentType.MANIPULATE, self._extract_entities_ner(doc), 0.6, text)
            elif token.lemma_ in ['greet', 'wave', 'follow', 'escort', 'assist', 'help']:
                return ParsedIntent(IntentType.INTERACT, self._extract_entities_ner(doc), 0.6, text)
            elif token.lemma_ in ['stop', 'halt', 'freeze', 'pause']:
                return ParsedIntent(IntentType.STOP, {}, 0.9, text)
            elif token.lemma_ in ['wait', 'stay', 'remain']:
                return ParsedIntent(IntentType.WAIT, {}, 0.8, text)
        
        return ParsedIntent(None, {}, 0.1, text)
    
    def _calculate_pattern_confidence(self, match, text: str) -> float:
        """Calculate confidence based on pattern match strength"""
        # Longer, more specific matches are more confident
        matched_text = match.group(0)
        confidence = 0.7 + (len(matched_text) / len(text)) * 0.2
        return min(confidence, 1.0)

class ContextResolver:
    """Resolves ambiguous language references using environmental context"""
    
    def __init__(self):
        self.object_memory = {}  # Stores recently seen objects
        self.location_memory = {}  # Stores location coordinates
        self.conversation_context = {}  # Stores conversational references
    
    def resolve(self, intent: ParsedIntent, current_context: Dict) -> ParsedIntent:
        """Resolve entities in the intent using current context"""
        if not intent.entities:
            return intent
        
        resolved_entities = {}
        
        # Resolve location references
        for entity_type, entity_values in intent.entities.items():
            if entity_type == 'location':
                resolved_locations = []
                for loc in entity_values:
                    resolved_location = self._resolve_location(loc, current_context)
                    resolved_locations.append(resolved_location)
                resolved_entities[entity_type] = resolved_locations
            
            elif entity_type == 'object':
                resolved_objects = []
                for obj in entity_values:
                    resolved_object = self._resolve_object(obj, current_context)
                    resolved_objects.append(resolved_object)
                resolved_entities[entity_type] = resolved_objects
                
            elif entity_type == 'person':
                resolved_persons = []
                for person in entity_values:
                    resolved_person = self._resolve_person(person, current_context)
                    resolved_persons.append(resolved_person)
                resolved_entities[entity_type] = resolved_persons
            
            else:
                resolved_entities[entity_type] = entity_values
        
        intent.entities = resolved_entities
        return intent
    
    def _resolve_location(self, location_desc: str, context: Dict) -> Dict:
        """Resolve a location description using context"""
        # Look up in known locations
        if location_desc in context.get('known_locations', {}):
            return context['known_locations'][location_desc]
        
        # Use conversational context (e.g., "over there" relative to robot)
        if location_desc in ['there', 'over there', 'that place']:
            # Use robot's current orientation and a relative offset
            robot_pose = context.get('robot_pose', {'x': 0, 'y': 0, 'theta': 0})
            # "There" might mean 1 meter in front of robot
            import math
            resolved_x = robot_pose['x'] + math.cos(robot_pose['theta'])
            resolved_y = robot_pose['y'] + math.sin(robot_pose['theta'])
            
            return {'x': resolved_x, 'y': resolved_y, 'description': location_desc, 'reference_type': 'relative'}
        
        # Default: return description only if location is unknown
        return {'description': location_desc, 'resolved': False}
    
    def _resolve_object(self, object_desc: str, context: Dict) -> Dict:
        """Resolve an object description using perception context"""
        # Look for objects in current perception
        detected_objects = context.get('detected_objects', [])
        
        for obj in detected_objects:
            # Check if description matches
            if object_desc.lower() in obj.get('name', '').lower() or \
               object_desc.lower() in obj.get('class', '').lower() or \
               object_desc.lower() in obj.get('attributes', {}).get('color', '').lower():
                return {
                    'id': obj.get('id'),
                    'name': obj.get('name'),
                    'class': obj.get('class'),
                    'position': obj.get('position'),
                    'description': object_desc,
                    'confidence': obj.get('confidence', 0.8)
                }
        
        # Check object memory for previously seen objects
        if object_desc in self.object_memory:
            return self.object_memory[object_desc]
        
        # If can't resolve, return description only
        return {'description': object_desc, 'resolved': False}
    
    def _resolve_person(self, person_desc: str, context: Dict) -> Dict:
        """Resolve a person description using context"""
        detected_people = context.get('detected_people', [])
        
        for person in detected_people:
            if person_desc.lower() in person.get('name', '').lower() or \
               person_desc.lower() in person.get('role', '').lower():
                return {
                    'id': person.get('id'),
                    'name': person.get('name'),
                    'position': person.get('position'),
                    'description': person_desc,
                    'confidence': person.get('confidence', 0.9)
                }
        
        # If person is mentioned as "the person" or "someone", find closest
        if person_desc in ['the person', 'someone', 'that person'] and detected_people:
            closest_person = min(detected_people, 
                                key=lambda p: p.get('distance_from_robot', float('inf')))
            if closest_person.get('distance_from_robot', float('inf')) < 5.0:  # Within 5m
                return closest_person
        
        return {'description': person_desc, 'resolved': False}
```

## Implementation with Large Language Models

### LLM Integration for Physical AI

Large Language Models bring general knowledge and reasoning capabilities to Physical AI systems:

```python
import openai
import json
from typing import Dict, Any
import re

class LLMPhysicalAIProcessor:
    """
    Uses Large Language Models to enhance natural language processing
    for physical AI applications, adding reasoning and planning capabilities.
    """
    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        if api_key:
            openai.api_key = api_key
        self.model_name = model_name
        
        self.system_prompt = """
You are an expert assistant that interprets natural language commands for humanoid robots operating in Physical AI contexts. Your role is to:

1. Understand natural language commands for humanoid robots
2. Break down complex commands into executable steps
3. Identify physical objects, locations, and people in the environment
4. Consider robot capabilities and limitations
5. Account for safety and environmental constraints
6. Plan appropriate robot responses

Think step by step about the command and respond with a structured JSON object containing the action to take.
"""
    
    def process_command_with_llm(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command using a Large Language Model with contextual information
        """
        prompt = f"""
Given the following command and context, determine the appropriate robot response:

COMMAND: {command}

CONTEXT: 
- Robot capabilities: {context.get('robot_capabilities', {})}
- Current position: {context.get('robot_position', {})}
- Detected objects: {context.get('detected_objects', [])}
- Known locations: {context.get('known_locations', {})}
- Battery level: {context.get('battery_level', 1.0)}

Provide a structured response in JSON format:
{{
  "intent": "navigation|manipulation|interaction|query|stop|wait",
  "action_sequence": [
    {{
      "step": 1,
      "action": "navigate|grasp|place|speak|turn|wait",
      "parameters": {{"target_location": "...", "target_object": "...", "message": "..."}},
      "preconditions": ["robot_must_be_stationary", "safety_check_passes"],
      "expected_outcomes": ["robot_moves_to_location", "object_grasped"],
      "success_criteria": "description of success"
    }}
  ],
  "confidence": 0.0-1.0,
  "reasoning": "explain the reasoning behind your interpretation",
  "potential_issues": ["list any concerns or assumptions"]
}}

Example of good response for "Go to the kitchen and bring me the red cup":
{{
  "intent": "manipulation",
  "action_sequence": [
    {{
      "step": 1,
      "action": "navigate", 
      "parameters": {{"target_location": "kitchen"}},
      "preconditions": ["path_is_clear", "battery_level_sufficient"],
      "expected_outcomes": ["robot_arrives_at_kitchen"],
      "success_criteria": "robot's position is within 0.5m of kitchen center"
    }},
    {{
      "step": 2,
      "action": "perceive",
      "parameters": {{"focus_area": "countertops", "object_types": ["cup"]}},
      "preconditions": ["robot_has_arrived_at_kitchen"],
      "expected_outcomes": ["red_cup_detected"],
      "success_criteria": "at least one red cup identified in perception"
    }},
    {{
      "step": 3,
      "action": "grasp",
      "parameters": {{"target_object": "red_cup"}},
      "preconditions": ["red_cup_is_reachable", "gripper_is_empty"],
      "expected_outcomes": ["red_cup_grasped"],
      "success_criteria": "force sensor confirms grasp, object no longer visible in previous location"
    }},
    {{
      "step": 4,
      "action": "navigate",
      "parameters": {{"target_location": "operator_position"}},
      "preconditions": ["red_cup_grasped"],
      "expected_outcomes": ["robot_returns_to_operator"],
      "success_criteria": "robot is within 1m of operator"
    }},
    {{
      "step": 5,
      "action": "place",
      "parameters": {{"placement_type": "handoff_to_human"}},
      "preconditions": ["robot_is_near_operator"],
      "expected_outcomes": ["cup_placed_for_handoff"],
      "success_criteria": "operator_can_easily_grasp_the_cup"
    }}
  ],
  "confidence": 0.85,
  "reasoning": "The command involves fetching an object, requiring navigation to location, object identification, grasping, returning to operator, and handing off",
  "potential_issues": ["red cup may not be where expected in kitchen", "operator position not specified in command"]
}}
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1500
            )
            
            # Extract JSON from response
            response_text = response.choices[0].message.content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                structured_response = json.loads(json_str)
                return structured_response
            else:
                # If no JSON found, return basic interpretation
                return {
                    "intent": "unknown",
                    "action_sequence": [],
                    "confidence": 0.1,
                    "reasoning": f"Could not extract structured response from: {response_text}",
                    "potential_issues": ["LLM response format not recognized"]
                }
                
        except Exception as e:
            return {
                "intent": "unknown",
                "action_sequence": [],
                "confidence": 0.0,
                "reasoning": f"Error calling LLM: {str(e)}",
                "potential_issues": ["LLM service unavailable"]
            }
    
    def validate_action_plan(self, plan: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> List[str]:
        """
        Validate that the action plan is executable with robot capabilities
        """
        issues = []
        
        for step in plan.get('action_sequence', []):
            action = step.get('action', '')
            
            # Check navigation capability
            if action == 'navigate':
                if not robot_capabilities.get('navigation_enabled', False):
                    issues.append(f"Step {step.get('step')} requires navigation but robot doesn't have navigation capability")
                    
            # Check manipulation capability
            elif action in ['grasp', 'place']:
                if not robot_capabilities.get('manipulation_enabled', False):
                    issues.append(f"Step {step.get('step')} requires manipulation but robot doesn't have manipulation capability")
                    
            # Check speaking capability
            elif action == 'speak':
                if not robot_capabilities.get('speaking_enabled', False):
                    issues.append(f"Step {step.get('step')} requires speaking but robot doesn't have speech capability")
        
        return issues
```

## Safety and Validation in Language Processing

### Command Validation Framework

```python
class CommandValidator:
    """Validates commands for safety and feasibility before execution"""
    
    def __init__(self, robot_model, environment_model):
        self.robot_model = robot_model
        self.environment_model = environment_model
    
    def validate_intent(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that an intent is safe and feasible"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        if intent.intent_type == IntentType.NAVIGATE:
            validation_result = self._validate_navigation(intent, context)
        elif intent.intent_type == IntentType.MANIPULATE:
            validation_result = self._validate_manipulation(intent, context)
        elif intent.intent_type == IntentType.INTERACT:
            validation_result = self._validate_interaction(intent, context)
        elif intent.intent_type == IntentType.STOP:
            # Stop commands are generally valid
            validation_result['valid'] = True
        
        return validation_result
    
    def _validate_navigation(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate navigation commands"""
        results = {'valid': True, 'warnings': [], 'errors': [], 'suggestions': []}
        
        if 'location' not in intent.entities:
            results['errors'].append("Navigation intent without target location")
            results['valid'] = False
            return results
        
        target_location = intent.entities['location']
        if isinstance(target_location, list) and target_location:
            target_location = target_location[0]  # Use first location if multiple
        
        # Check if location exists in environment
        if isinstance(target_location, dict) and not target_location.get('resolved', True):
            results['errors'].append(f"Unknown location: {target_location.get('description', 'unspecified')}")
            results['valid'] = False
            return results
        
        # Check navigation feasibility
        if not self._is_navigable(target_location):
            results['errors'].append(f"Location {target_location} is not navigable")
            results['valid'] = False
        
        # Check battery level for long trips
        if self._is_long_distance_travel(target_location, context.get('robot_pose')):
            battery_level = context.get('battery_level', 1.0)
            if battery_level < 0.3:
                results['warnings'].append("Battery level low for long navigation")
                results['suggestions'].append("Consider charging before executing this command")
        
        # Check for safety constraints
        if self._is_hazardous_area(target_location):
            results['errors'].append(f"Navigation to {target_location} may be hazardous")
            results['valid'] = False
        
        return results
    
    def _validate_manipulation(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate manipulation commands"""
        results = {'valid': True, 'warnings': [], 'errors': [], 'suggestions': []}
        
        if 'object' not in intent.entities:
            results['errors'].append("Manipulation intent without target object")
            results['valid'] = False
            return results
        
        target_object = intent.entities['object']
        if isinstance(target_object, list) and target_object:
            target_object = target_object[0]
        
        # Check if object exists and is manipulable
        if isinstance(target_object, dict):
            if not target_object.get('resolved', True):
                results['errors'].append(f"Unknown object: {target_object.get('description', 'unspecified')}")
                results['valid'] = False
                return results
            
            # Check if object is graspable
            obj_class = target_object.get('class', target_object.get('description', '')).lower()
            if self._is_graspable_object(obj_class):
                # Check if object is reachable
                obj_position = target_object.get('position')
                if obj_position and not self._is_reachable(obj_position):
                    results['errors'].append(f"Object {obj_class} is not currently reachable")
                    results['valid'] = False
            else:
                results['errors'].append(f"Object {obj_class} is not manipulable")
                results['valid'] = False
        else:
            results['errors'].append(f"Could not resolve target object: {target_object}")
            results['valid'] = False
        
        return results
    
    def _validate_interaction(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate interaction commands"""
        results = {'valid': True, 'warnings': [], 'errors': [], 'suggestions': []}
        
        if 'person' not in intent.entities:
            # If no person specified, interaction might be with operator
            # Check if any person is detected
            if not context.get('detected_people', []):
                results['warnings'].append("No person detected for interaction command")
                results['suggestions'].append("Specify which person to interact with or ensure detection system is active")
        
        # Check for safe interaction parameters
        if intent.raw_text.lower().contains('attack') or intent.raw_text.lower().contains('hurt'):
            results['errors'].append("Interaction command contains unsafe terminology")
            results['valid'] = False
        
        return results
    
    def _is_navigable(self, location) -> bool:
        """Check if a location is navigable by the robot"""
        # This would check the environment model for obstacle-free paths
        # For this example, assume all known locations are navigable
        return True
    
    def _is_long_distance_travel(self, target_location, current_pose) -> bool:
        """Check if navigation would be long distance"""
        # Calculate distance to target and return True if it exceeds threshold
        # This is a simplified implementation
        return False
    
    def _is_hazardous_area(self, location) -> bool:
        """Check if location is potentially hazardous"""
        # This would check the environment model for hazards
        return False
    
    def _is_graspable_object(self, obj_class) -> bool:
        """Check if object class is manipulable"""
        # Define which objects are graspable
        graspable_classes = [
            'cup', 'bottle', 'book', 'ball', 'box', 'phone', 'tablet', 'keys', 
            'glasses', 'apple', 'banana', 'toy', 'tool', 'pen', 'pencil'
        ]
        return any(grasp_class in obj_class.lower() for grasp_class in graspable_classes)
    
    def _is_reachable(self, obj_position) -> bool:
        """Check if an object is within robot reach"""
        # This would check robot kinematics and joint limits
        # For this example, assume objects within 1 meter of current position are reachable
        return True  # Placeholder implementation
```

## Unity Integration for Language Processing

### Visualization of Language Understanding

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Nlp;  // fictional message type for NLP results
using System.Collections.Generic;
using System.Linq;

public class LanguageUnderstandingVisualizer : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string nlpResultTopic = "/vla/nlp_results";
    
    [Header("Visualization")]
    public GameObject intentMarkerPrefab;
    public GameObject objectHighlightPrefab;
    public GameObject pathVisualizationPrefab;
    public GameObject speechBubblePrefab;
    
    [Header("Colors & Appearance")]
    public Color navigateColor = Color.blue;
    public Color manipulationColor = Color.green;
    public Color interactColor = Color.yellow;
    public Color queryColor = Color.cyan;
    public Color errorColor = Color.red;
    
    private Dictionary<string, GameObject> activeVisualizations = new Dictionary<string, GameObject>();
    private ROSTCPConnector ros;
    
    void Start()
    {
        ros = ROSTCPConnector.instance;
        ros.Subscribe<StringMsg>(nlpResultTopic, OnNLPResultReceived);
    }
    
    void OnNLPResultReceived(StringMsg msg)
    {
        try
        {
            // Parse the NLP result JSON
            NLPResult nlpResult = JsonUtility.FromJson<NLPResult>(msg.data);
            
            // Clear old visualizations
            ClearPreviousVisualizations();
            
            // Visualize based on intent
            VisualizeIntent(nlpResult);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error visualizing NLP result: {e.Message}");
        }
    }
    
    void VisualizeIntent(NLPResult result)
    {
        if (result.intent == "navigate")
        {
            VisualizeNavigationIntent(result);
        }
        else if (result.intent == "manipulation")
        {
            VisualizeManipulationIntent(result);
        }
        else if (result.intent == "interaction")
        {
            VisualizeInteractionIntent(result);
        }
        else if (result.intent == "query")
        {
            VisualizeQueryIntent(result);
        }
        else
        {
            // Unknown intent - visualize with error color
            VisualizeGenericIntent(result);
        }
    }
    
    void VisualizeNavigationIntent(NLPResult result)
    {
        // Create path visualization
        if (result.action_sequence != null)
        {
            foreach (var step in result.action_sequence)
            {
                if (step.action == "navigate" && step.parameters.ContainsKey("target_location"))
                {
                    // Get target location coordinates from parameters
                    var locationParam = step.parameters["target_location"];
                    // In practice, this would get actual coordinates from environment model
                    
                    // For now, create a marker at a representative position
                    Vector3 targetPos = new Vector3(2f, 0f, 0.5f); // Placeholder position
                    CreateNavigationMarker(targetPos, result.confidence);
                }
            }
        }
        
        // Create intent marker
        CreateIntentMarker(navigateColor, result.reasoning, "NAVIGATE");
    }
    
    void VisualizeManipulationIntent(NLPResult result)
    {
        // Highlight target object if specified
        if (result.action_sequence != null)
        {
            foreach (var step in result.action_sequence)
            {
                if (step.action == "grasp" && step.parameters.ContainsKey("target_object"))
                {
                    string targetObject = step.parameters["target_object"];
                    HighlightObject(targetObject);
                }
            }
        }
        
        // Create intent marker
        CreateIntentMarker(manipulationColor, result.reasoning, "MANIPULATE");
    }
    
    void VisualizeInteractionIntent(NLPResult result)
    {
        // Visualize interaction with person
        if (result.action_sequence != null)
        {
            foreach (var step in result.action_sequence)
            {
                if (step.action == "greet" || step.action == "wave" || step.action == "follow")
                {
                    // Highlight the person to interact with
                    if (step.parameters.ContainsKey("target_person"))
                    {
                        string targetPerson = step.parameters["target_person"];
                        HighlightPerson(targetPerson);
                    }
                }
            }
        }
        
        // Create intent marker
        CreateIntentMarker(interactColor, result.reasoning, "INTERACT");
    }
    
    void VisualizeQueryIntent(NLPResult result)
    {
        // Create visualization for query intent
        CreateIntentMarker(queryColor, result.reasoning, "QUERY");
    }
    
    void VisualizeGenericIntent(NLPResult result)
    {
        // For unknown intents, visualize with error color
        CreateIntentMarker(errorColor, result.reasoning, "UNKNOWN");
    }
    
    GameObject CreateNavigationMarker(Vector3 position, float confidence)
    {
        GameObject marker = Instantiate(intentMarkerPrefab);
        marker.transform.position = position;
        marker.name = $"NavigationIntent_{System.DateTime.Now.Ticks}";
        
        // Color based on confidence
        Renderer rend = marker.GetComponent<Renderer>();
        if (rend != null)
        {
            Color markerColor = Color.Lerp(Color.gray, navigateColor, confidence);
            rend.material.color = markerColor;
        }
        
        // Add confidence indicator
        TextMesh textMesh = marker.GetComponentInChildren<TextMesh>();
        if (textMesh != null)
        {
            textMesh.text = $"Go There\nConf: {(int)(confidence * 100)}%";
        }
        
        activeVisualizations[$"navigation_{position}"] = marker;
        return marker;
    }
    
    void HighlightObject(string objectName)
    {
        // Find object in scene by name and highlight it
        GameObject[] objects = GameObject.FindGameObjectsWithTag("InteractiveObject");
        
        foreach (GameObject obj in objects)
        {
            if (obj.name.ToLower().Contains(objectName.ToLower()))
            {
                // Add highlight visualization
                GameObject highlight = Instantiate(objectHighlightPrefab, obj.transform);
                highlight.name = $"ObjectHighlight_{objectName}";
                
                // Store for cleanup later
                activeVisualizations[$"highlight_{obj.name}"] = highlight;
            }
        }
    }
    
    void HighlightPerson(string personName)
    {
        // Find person in scene and highlight
        GameObject[] people = GameObject.FindGameObjectsWithTag("Person");
        
        foreach (GameObject person in people)
        {
            if (person.name.ToLower().Contains(personName.ToLower()))
            {
                // Add highlight visualization
                GameObject highlight = Instantiate(objectHighlightPrefab, person.transform);
                highlight.name = $"PersonHighlight_{personName}";
                
                // Modify color for interaction
                Renderer rend = highlight.GetComponent<Renderer>();
                if (rend != null)
                {
                    rend.material.color = interactColor;
                }
                
                activeVisualizations[$"person_highlight_{person.name}"] = highlight;
            }
        }
    }
    
    GameObject CreateIntentMarker(Color color, string reasoning, string intentName)
    {
        GameObject marker = Instantiate(intentMarkerPrefab);
        marker.name = $"IntentMarker_{intentName}";
        
        // Position marker above robot or in center of view
        marker.transform.position = Camera.main.transform.position + Camera.main.transform.forward * 5f;
        
        Renderer rend = marker.GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material.color = color;
        }
        
        TextMesh textMesh = marker.GetComponentInChildren<TextMesh>();
        if (textMesh != null)
        {
            textMesh.text = $"{intentName}\n{reasoning.Substring(0, Mathf.Min(20, reasoning.Length))}...";
        }
        
        activeVisualizations[$"intent_{intentName}"] = marker;
        return marker;
    }
    
    void ClearPreviousVisualizations()
    {
        List<string> keysToRemove = new List<string>();
        
        foreach (var kvp in activeVisualizations)
        {
            if (kvp.Value != null)  // Check if object still exists
            {
                Destroy(kvp.Value);
            }
            keysToRemove.Add(kvp.Key);
        }
        
        foreach (string key in keysToRemove)
        {
            activeVisualizations.Remove(key);
        }
    }
    
    void OnDestroy()
    {
        ClearPreviousVisualizations();
    }
}

// Data class to hold NLP results
[System.Serializable]
public class NLPResult
{
    public string intent;
    public ActionStep[] action_sequence;
    public float confidence;
    public string reasoning;
    public string[] potential_issues;
    public System.Collections.Generic.Dictionary<string, string> entities;
}

[System.Serializable]
public class ActionStep
{
    public int step;
    public string action;
    public System.Collections.Generic.Dictionary<string, string> parameters;
    public string[] preconditions;
    public string[] expected_outcomes;
    public string success_criteria;
}
```

## Testing and Validation

### Test Suite for NLP Components

```python
import unittest
from unittest.mock import Mock, MagicMock

class TestNLPComponents(unittest.TestCase):
    def setUp(self):
        self.nlp_system = NaturalLanguageUnderstandingSystem()
        self.context_resolver = ContextResolver()
        self.validator = CommandValidator(Mock(), Mock())
    
    def test_intent_classification_navigate(self):
        """Test navigation intent classification"""
        result = self.nlp_system.parse_intent("Go to the kitchen")
        self.assertEqual(result.intent_type, IntentType.NAVIGATE)
        self.assertIn('kitchen', result.entities.get('location', []))
    
    def test_intent_classification_manipulate(self):
        """Test manipulation intent classification"""
        result = self.nlp_system.parse_intent("Pick up the red cup")
        self.assertEqual(result.intent_type, IntentType.MANIPULATE)
        self.assertIn('cup', result.entities.get('object', []))
    
    def test_context_resolution(self):
        """Test context-based entity resolution"""
        intent = ParsedIntent(
            intent_type=IntentType.NAVIGATE,
            entities={'location': ['there']},
            confidence=0.7,
            raw_text="Go over there"
        )
        
        context = {
            'robot_pose': {'x': 0, 'y': 0, 'theta': 0},
            'known_locations': {'kitchen': {'x': 3.0, 'y': 1.0, 'z': 0.0}}
        }
        
        resolved = self.context_resolver.resolve(intent, context)
        # Check that 'there' was resolved to a proper location
        self.assertTrue(len(resolved.entities['location']) > 0)
    
    def test_command_validation_safe_navigation(self):
        """Test validation of safe navigation command"""
        intent = ParsedIntent(
            intent_type=IntentType.NAVIGATE,
            entities={'location': [{'description': 'kitchen', 'resolved': True}]},
            confidence=0.9,
            raw_text="Go to the kitchen"
        )
        
        context = {
            'robot_pose': {'x': 0, 'y': 0},
            'battery_level': 0.8
        }
        
        validation = self.validator.validate_intent(intent, context)
        self.assertTrue(validation['valid'])
    
    def test_command_validation_unsafe_manipulation(self):
        """Test validation of unsafe manipulation command"""
        intent = ParsedIntent(
            intent_type=IntentType.MANIPULATE,
            entities={'object': [{'description': 'elephant', 'resolved': False}]},
            confidence=0.5,
            raw_text="Pick up the elephant"
        )
        
        context = {}
        
        validation = self.validator.validate_intent(intent, context)
        self.assertFalse(validation['valid'])
        self.assertGreater(len(validation['errors']), 0)

if __name__ == '__main__':
    unittest.main()
```

## Summary

Natural Language Command Processing is a critical component of Physical AI systems that enables intuitive human-robot interaction. The implementation combines several technologies:

1. **Automatic Speech Recognition**: Converting speech to text using systems like Whisper
2. **Natural Language Understanding**: Parsing intent and extracting entities
3. **Context Integration**: Grounding language in physical and situational context
4. **Large Language Models**: Adding reasoning and planning capabilities
5. **Action Planning**: Translating linguistic commands to executable robot behaviors
6. **Validation and Safety**: Ensuring commands are safe and feasible before execution

The integration with Unity provides visualization of the language understanding process, making it easier for developers and users to understand how the robot interprets and plans to execute commands. Together, these systems enable more natural and intuitive interaction with humanoid robots in Physical AI applications.

These exercises have demonstrated the core components needed for effective Natural Language Command Processing in humanoid robotics, including practical implementations for both simulation and real-world deployment. The combination of accurate NLP, contextual awareness, and safety validation enables robust and reliable Physical AI systems.