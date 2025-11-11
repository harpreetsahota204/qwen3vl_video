"""
FiftyOne integration for Qwen3-VL video understanding model.

This module provides a config-based implementation of the Qwen3-VL model
for video understanding tasks in FiftyOne, supporting:
- Video description and summarization
- Temporal event localization
- Object tracking with bounding boxes
- Video OCR and text extraction
- Spatial understanding
- Comprehensive video analysis

The model can output multiple label types in a single forward pass:
- Sample-level: Classifications, TemporalDetections
- Frame-level: Detections (objects, OCR text)

"""

import logging
import json
import re
from typing import Dict, List, Optional, Union, Any

import torch
import numpy as np

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.utils.import_utils import is_flash_attn_2_available

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device for inference.
    
    Checks for available devices in priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Task-specific prompts and configurations
# Each operation defines a default prompt template
# Parsing is fully dynamic - any JSON keys in output will be parsed automatically
OPERATIONS = {
    "comprehensive": {
        "prompt": """Analyze this video comprehensively in JSON format:

{
  "summary": "Brief 2-3 sentence description",
  "objects": [{"name": "object name", "first_appears": "mm:ss.ff", "last_appears": "mm:ss.ff"}],
  "events": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}],
  "text_content": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "text": "..."}],
  "scene_info": {"setting": "...", "time_of_day": "...", "location_type": "..."},
  "object_count": {"count": number, "activities": [...]}
}"""
    },
    "description": {
        "prompt": "Provide a detailed description of what happens in this video."
    },
    "temporal_localization": {
        "prompt": """Localize activity events in the video. Output start and end timestamp for each event.
Provide in JSON format with 'mm:ss.ff' format:
[{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}]"""
    },
    "tracking": {
        "prompt": """Track all objects in this video.
For each frame where objects appear, provide:
- time: timestamp (mm:ss.ff)
- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale
- label: object label
Output in JSON: [{"time": "mm:ss.ff", "bbox_2d": [...], "label": "..."}, ...]"""
    },
    "ocr": {
        "prompt": """Extract all text appearing in this video.
For each text instance, provide:
- time: timestamp (mm:ss.ff)
- text: the actual text content
- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale
Output in JSON: [{"time": "mm:ss.ff", "text": "...", "bbox_2d": [...]}, ...]"""
    },
}


class Qwen3VLVideoModelConfig(fout.TorchImageModelConfig):
    """Configuration for Qwen3-VL video model.
    
    This config class defines all parameters for video processing, model inference,
    and output parsing. It inherits from TorchImageModelConfig to leverage FiftyOne's
    built-in configuration parsing utilities.
    
    Args:
        model_path (str): HuggingFace model identifier or local path to model
            Examples: "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"
            Default: "Qwen/Qwen3-VL-8B-Instruct"
        
        # Video processing parameters
        total_pixels (int): Maximum total pixels for quality/memory tradeoff
            Controls video resolution and memory usage. Higher = better quality but more memory.
            
        min_pixels (int): Minimum pixel threshold for video processing
            Default: 64*32*32
            
        max_frames (int): Maximum number of frames to sample from video
            Controls temporal coverage. More frames = better temporal understanding.

        sample_fps (int/float): Frame sampling rate
            Controls how densely frames are sampled
            
        image_patch_size (int): Patch size for vision encoder
            Default: 16 (typically don't need to change)
        
        # Generation parameters
        max_new_tokens (int): Maximum tokens to generate in response
            Default: 8192. Increase for longer/more detailed outputs.
            
        do_sample (bool): Whether to use sampling (True) vs greedy decoding (False)
            Default: False (greedy decoding for deterministic outputs)
            
        temperature (float): Sampling temperature (only used if do_sample=True)
            Higher = more creative/random. Default: 0.7
            
        top_p (float): Nucleus sampling parameter (only used if do_sample=True)
            Default: 0.8
            
        top_k (int): Top-k sampling parameter (only used if do_sample=True)
            Default: 20
            
        repetition_penalty (float): Penalty for repeating tokens
            Default: 1.0 (no penalty)
        
        # Operation configuration
        operation (str): Operation type - selects default prompt from OPERATIONS dict
            Options:
            - "comprehensive": All analysis types (summary, events, tracking, OCR)
            - "description": Video description only
            - "temporal_localization": Event detection with timestamps
            - "tracking": Object tracking with bounding boxes
            - "ocr": Text extraction with locations
            Default: "comprehensive"
            
        custom_prompt (str): Custom prompt to override default operation prompt
            If provided, this prompt is used instead of OPERATIONS[operation]["prompt"]
            Allows full flexibility for custom analysis tasks.
            Default: None (uses operation default)
        
        # Output configuration
        output_keys (list): List of JSON keys to parse from model output
            If None, parses all keys found in JSON output.
            If specified, only parses the listed keys.
            Example: ["summary", "events", "objects"]
            Default: None (parse all)
    
    Example:
        # Comprehensive analysis with custom settings
        config = Qwen3VLVideoModelConfig({
            "model_path": "Qwen/Qwen3-VL-8B-Instruct",
            "operation": "comprehensive",
            "max_frames": 120,
            "sample_fps": 10,
            "total_pixels": 2048*32*32,
            "output_keys": ["summary", "events", "objects"]  # Only parse these
        })
        
        # Custom prompt example
        config = Qwen3VLVideoModelConfig({
            "custom_prompt": "Find all vehicles and people in this video...",
            "output_keys": ["vehicles", "people"]
        })
    """
    
    def __init__(self, d):
        super().__init__(d)
        
        # Model parameters
        # HuggingFace model identifier or local path
        self.model_path = self.parse_string(d, "model_path", default="Qwen/Qwen3-VL-8B-Instruct")
        
        # Video processing parameters
        # Controls quality/memory tradeoff - higher = better quality but more VRAM
        self.total_pixels = self.parse_number(d, "total_pixels", default=2048*32*32)
        # Minimum pixel threshold (typically don't change)
        self.min_pixels = self.parse_number(d, "min_pixels", default=64*32*32)
        # Number of frames to sample - more frames = better temporal understanding
        self.max_frames = self.parse_number(d, "max_frames", default=120)
        # Frame sampling rate - higher = denser sampling
        self.sample_fps = self.parse_number(d, "sample_fps", default=10)
        # Vision encoder patch size (typically don't change)
        self.image_patch_size = self.parse_number(d, "image_patch_size", default=16)
        
        # Text generation parameters
        # Maximum length of generated response
        self.max_new_tokens = self.parse_number(d, "max_new_tokens", default=8192)
        # Use sampling (True) vs greedy decoding (False)
        self.do_sample = self.parse_bool(d, "do_sample", default=False)
        # Sampling temperature - higher = more creative (only if do_sample=True)
        self.temperature = self.parse_number(d, "temperature", default=0.7)
        # Nucleus sampling threshold (only if do_sample=True)
        self.top_p = self.parse_number(d, "top_p", default=0.8)
        # Top-k sampling parameter (only if do_sample=True)
        self.top_k = self.parse_number(d, "top_k", default=20)
        # Penalty for token repetition
        self.repetition_penalty = self.parse_number(d, "repetition_penalty", default=1.0)
        
        # Operation configuration
        # Selects default prompt from OPERATIONS dict
        self.operation = self.parse_string(d, "operation", default="comprehensive")
        # Optional custom prompt to override operation default
        self.custom_prompt = self.parse_string(d, "custom_prompt", default=None)
        
        # Output parsing configuration
        # If None, parses all JSON keys; if list, only parses specified keys
        self.output_keys = self.parse_array(d, "output_keys", default=None)


class Qwen3VLVideoModel(fom.SamplesMixin, fom.Model):
    """FiftyOne wrapper for Qwen3-VL video understanding model.
    
    This model processes videos directly from file paths (no manual frame extraction)
    and can output multiple label types in a single forward pass. It inherits from
    both SamplesMixin (for per-sample field access) and Model (base FiftyOne interface).
    
    The model supports flexible operations through the OPERATIONS dict and can parse
    custom JSON outputs with automatic type detection.
    
    Sample-level labels (stored in sample fields):
        - summary: Video description (plain text string)
        - events: Temporal detections for events spanning time ranges (fo.TemporalDetections)
        - objects: Object appearances over time (fo.TemporalDetections)
        - scene_info_*: Scene classifications (fo.Classification) - one per scene_info key
        - Any custom string keys from JSON output (stored as plain text or parsed based on structure)
    
    Frame-level labels (stored in sample.frames[N] fields):
        - objects: Object bounding boxes per frame (fo.Detections)
        - text_content: Text bounding boxes per frame (fo.Detections)
        - Any custom keys with time-based detections
    """
    
    def __init__(self, config):
        # Initialize SamplesMixin to enable per-sample field access
        fom.SamplesMixin.__init__(self)
        self.config = config
        
        # Detect and set best available device (cuda > mps > cpu)
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Lazy loading - model and processor loaded on first predict() call
        # This avoids loading heavy models during initialization
        self._processor = None  # HuggingFace AutoProcessor
        self._model = None      # HuggingFace AutoModelForVision2Seq

        self._fields = {} 
    
    @property
    def media_type(self):
        """Media type this model operates on.
        
        Returns "video" to indicate this model processes video files.
        FiftyOne uses this to validate dataset compatibility.
        """
        return "video"

    @property
    def needs_fields(self):
        """Dict mapping model-specific keys to sample field names.
        
        Allows the model to access specific fields from samples during prediction.
        Inherited from SamplesMixin.
        
        Example:
            model.needs_fields = {"prompt_field": "custom_prompts"}
            # Now predict() will read prompts from sample["custom_prompts"]
        """
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        """Set the fields this model needs from samples."""
        self._fields = fields

    def _get_field(self):
        """Get the field name to use for prompt lookup from sample.
        
        Checks needs_fields for "prompt_field" key first, otherwise returns
        the first available field name.
        
        Returns:
            str or None: Field name to read from sample, or None if not configured
        """
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        
        return prompt_field
    
    @property
    def operation(self):
        """Current operation type.
        
        Returns the operation name which determines the default prompt used.
        Can be changed at runtime to switch between different analysis modes.
        """
        return self.config.operation
    
    @operation.setter
    def operation(self, value):
        """Set operation type with validation.
        
        Args:
            value (str): Operation name, must be a key in OPERATIONS dict
        
        Raises:
            ValueError: If operation is not in OPERATIONS
        """
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self.config.operation = value
    
    @property
    def prompt(self):
        """Get current prompt - custom or default for operation.
        
        Returns custom_prompt if set, otherwise returns the default prompt
        for the current operation from OPERATIONS dict.
        
        Returns:
            str: The prompt text to use for inference
        """
        if self.config.custom_prompt:
            return self.config.custom_prompt
        return OPERATIONS[self.config.operation]["prompt"]
    
    @prompt.setter
    def prompt(self, value):
        """Set custom prompt, overriding operation default.
        
        Args:
            value (str): Custom prompt text
        """
        self.config.custom_prompt = value
    
    def _load_model(self):
        """Load Qwen3-VL model and processor from HuggingFace.
        
        Loads both the processor (for tokenization and vision processing) and
        the model (for inference). Uses "auto" for dtype and device_map to let
        HuggingFace automatically select optimal settings.
        
        Optimizations:
        - Enables Flash Attention 2 if available for faster inference
        - Uses automatic dtype selection
        - Uses automatic device mapping
        
        The model is set to eval mode for inference.
        """
        logger.info(f"Loading Qwen3-VL model from {self.config.model_path}")
        
        # Load processor for tokenization and vision processing
        self._processor = AutoProcessor.from_pretrained(self.config.model_path)
        
        # Prepare model loading kwargs with optimizations
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        
        # Enable Flash Attention 2 if using CUDA and it's available
        # Check device first (cheap), then flash attention availability
        if self.device == "cuda" and torch.cuda.is_available():
            if is_flash_attn_2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2 for optimized inference")
            else:
                logger.info("Flash Attention 2 not available, using default attention")
        else:
            logger.info(f"Using {self.device} device with default attention")
        
        # Load model with optimizations
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        
        # Set to evaluation mode (disables dropout, etc.)
        self._model.eval()
        
        logger.info("Model loaded successfully")
    
    def predict(self, arg, sample=None):
        """Run inference on a video file.
        
        This is the main prediction method called by FiftyOne's apply_model() function.
        It processes a video and returns a dict with both sample-level and frame-level
        labels that FiftyOne automatically routes to the correct storage locations.
        
        Since this model inherits from SamplesMixin, it requires the sample parameter
        to access the video filepath and metadata.
        
        Prompt Priority (in order):
        1. Sample field value (if needs_fields is configured)
        2. Custom prompt (if config.custom_prompt is set)
        3. Default prompt for operation (from OPERATIONS dict)
        
        Args:
            arg: Unused (for compatibility with Model interface)
            sample (fo.Sample): FiftyOne sample (required)
                Used to:
                - Get video filepath (sample.filepath)
                - Get video FPS from metadata
                - Read per-sample prompts if needs_fields is configured
                - Convert timestamps to frame numbers for temporal detections
            
        Returns:
            dict: Mixed dictionary with sample-level and frame-level labels
            {
                # Sample-level labels (string keys)
                "summary": "Plain text description",
                "events": fo.TemporalDetections(...),
                "objects": fo.TemporalDetections(...),
                "scene_info_setting": fo.Classification(...),
                "scene_info_time_of_day": fo.Classification(...),
                
                # Frame-level labels (integer keys = frame numbers)
                1: {"objects": fo.Detections(...), "text_content": fo.Detections(...)},
                5: {"objects": fo.Detections(...), "text_content": fo.Detections(...)},
                ...
            }
            
            FiftyOne's apply_model() automatically handles this structure:
            - String keys with string values → sample["{label_field}_{key}"] = string
            - String keys with Label values → sample["{label_field}_{key}"] = Label
            - Integer keys → sample.frames[frame_num]["{label_field}_{field}"]
        """
        # Lazy load model on first use
        if self._model is None:
            logger.info("Model not loaded, loading now...")
            self._load_model()
        
        # Get video file path from sample (required for SamplesMixin models)
        if sample is None:
            raise ValueError("Sample is required for video processing")
        
        video_path = sample.filepath
        logger.info(f"Processing video: {video_path}")
        
        # Validate metadata upfront for operations that need it
        needs_metadata = self.config.operation in [
            "comprehensive", "temporal_localization", "tracking", "ocr"
        ]
        
        if needs_metadata and not hasattr(sample, 'metadata'):
            raise ValueError(
                f"Operation '{self.config.operation}' requires sample metadata for timestamp conversion. "
                f"Call dataset.compute_metadata() before applying model."
            )
        
        # Determine prompt to use (priority: sample field > custom > operation default)
        prompt = self.prompt  # Start with property (custom or operation default)
        
        # Override with sample field value if configured via needs_fields
        if self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)
        
        # Build messages in Qwen3-VL format
        # Video is passed as file path, model handles frame extraction internally
        messages = [{
            "role": "user",
            "content": [
                {
                    "video": video_path,
                    "total_pixels": self.config.total_pixels,  # Quality/memory control
                    "min_pixels": self.config.min_pixels,      # Minimum threshold
                    "max_frames": self.config.max_frames,      # Temporal coverage
                    "sample_fps": self.config.sample_fps       # Sampling density
                },
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Run model inference
        try:
            output_text = self._run_inference(messages)
            logger.debug(f"Model output: {output_text[:200]}...")
        except Exception as e:
            logger.error(f"Inference failed for video {video_path}: {e}")
            raise
        
        # Parse text output into FiftyOne label objects
        try:
            labels = self._parse_output(output_text, video_path, sample)
            logger.debug(f"Parsed {len(labels)} label fields")
        except Exception as e:
            logger.error(f"Parsing failed for video {video_path}: {e}")
            logger.error(f"Raw output was: {output_text[:500]}")
            raise
        
        return labels
    
    def _run_inference(self, messages):
        """Run model inference on processed messages.
        
        Handles the complete inference pipeline:
        1. Apply chat template to format messages
        2. Process video (extract and encode frames)
        3. Prepare model inputs and move to device
        4. Generate response
        5. Decode output text
        
        Args:
            messages (list): List of message dicts in Qwen3-VL format
        
        Returns:
            str: Generated text output from model
        """
        
        
        # Step 1: Apply chat template to format conversation
        # Converts messages to the format expected by the model
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Step 2: Process vision info - extracts and encodes video frames
        # This handles frame sampling based on max_frames and sample_fps
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=self.config.image_patch_size,
            return_video_metadata=True
        )
        
        # Step 3: Unpack video inputs and metadata
        # video_inputs contains tuples of (frames, metadata)
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        
        # Step 4: Prepare final model inputs and move to device
        # Tokenizes text and prepares vision tensors
        # Use model's actual device (from device_map="auto")
        device = next(self._model.parameters()).device
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(device)
        
        # Step 5: Generate response
        with torch.no_grad():
            # Base generation kwargs
            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "repetition_penalty": self.config.repetition_penalty,
            }
            
            # Add sampling parameters if using sampling (not greedy)
            if self.config.do_sample:
                gen_kwargs.update({
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                })
            
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        
        # Step 6: Decode generated tokens to text
        # Extract only the newly generated tokens (exclude input prompt)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return output_text[0]
    
    def _parse_output(self, output_text, video_path, sample):
        """Parse model output into FiftyOne labels.
        
        Flexibly parses JSON output with automatic type detection based on structure.
        Supports both predefined operations and custom prompts with arbitrary JSON keys.
        
        Parsing Strategy:
        1. Extract JSON from output (handles markdown code blocks and raw lists)
        2. If JSON is a list, wrap in dict with operation-specific key
        3. Determine which keys to parse (all or config.output_keys subset)
        4. For each key, detect type based on value structure:
           - String → Plain string (for summary, description, etc.)
           - Dict → Multiple Classifications (for scene_info, etc.) or skip if not parseable
           - List with temporal data → fo.TemporalDetections (sample-level)
           - List with bbox+time → fo.Detections per frame (frame-level)
        
        Args:
            output_text (str): Raw text output from model
            video_path (str): Path to video file
            sample (fo.Sample): FiftyOne sample for metadata
        
        Returns:
            dict: Mixed sample-level (string keys) and frame-level (int keys) labels
        """
        labels = {}
        parsed_json = self._extract_json(output_text)
        
        # Fallback if no JSON found - treat entire output as plain text summary
        if not parsed_json:
            return {"summary": output_text}
        
        # Handle list-type JSON (wrap in dict with operation-appropriate key)
        if isinstance(parsed_json, list):
            # Detect what type of list based on operation
            if self.config.operation == "temporal_localization":
                parsed_json = {"events": parsed_json}
            elif self.config.operation == "tracking":
                parsed_json = {"objects": parsed_json}
            elif self.config.operation == "ocr":
                parsed_json = {"text_content": parsed_json}
            else:
                # Unknown list type - skip parsing
                logger.warning(f"Received list JSON but don't know how to parse for operation '{self.config.operation}'")
                return {}
        
        # Determine which keys to parse (all or selective)
        keys = self.config.output_keys or list(parsed_json.keys())
        
        # Parse each key based on its value structure
        for key in keys:
            if key not in parsed_json:
                continue
            
            value = parsed_json[key]
            
            # Dispatch based on value type
            if isinstance(value, str):
                # String values stored as plain text (not wrapped in Classification)
                # Used for: summary, description, etc.
                labels[key] = value
            elif isinstance(value, dict):
                # Dict values - parse as nested classifications only if all values are simple types
                # Skip dicts like object_count that have mixed/complex structures
                if self._is_simple_dict(value):
                    self._parse_dict_value(key, value, labels)
                else:
                    # Skip complex dicts that don't fit the scene_info pattern
                    logger.debug(f"Skipping complex dict for key '{key}'")
            elif isinstance(value, list) and value:
                # List values - detect structure and parse appropriately
                self._parse_list_value(key, value, labels, video_path, sample)
        
        return labels
    
    def _parse_list_value(self, key, value, labels, video_path, sample):
        """Dispatch list parsing based on item structure.
        
        Examines the first item in the list to determine type, then parses accordingly:
        - Temporal events: Items with "start", "end", "description" keys
        - Object appearances: Items with "name", "first_appears", "last_appears" keys
        - Text content: Items with "start", "end", "text" keys
        - Object detections: Items with "time", "bbox_2d", "label" keys
        - OCR detections: Items with "time", "text", "bbox_2d" keys
        
        Args:
            key (str): JSON key name (becomes field name)
            value (list): List of items to parse
            labels (dict): Labels dict to update (modified in place)
            video_path (str): Path to video file
            sample (fo.Sample): FiftyOne sample
        """
        first = value[0]
        
        # Check structure of first item to determine type
        if self._has_keys(first, ["start", "end", "description"]):
            # Temporal events - sample-level TemporalDetections
            events = self._parse_temporal_events(value, video_path, sample)
            if events:
                labels[key] = events
        elif self._has_keys(first, ["name", "first_appears", "last_appears"]):
            # Object appearances - convert to temporal detections
            events = self._parse_object_appearances(value, video_path, sample)
            if events:
                labels[key] = events
        elif self._has_keys(first, ["start", "end", "text"]):
            # Text content temporal - convert to temporal detections
            events = self._parse_text_temporal(value, video_path, sample)
            if events:
                labels[key] = events
        elif self._has_keys(first, ["time", "bbox_2d", "label"]):
            # Object detections - frame-level Detections
            self._merge_frame_labels(labels, self._parse_frame_detections(value, video_path, sample), key)
        elif self._has_keys(first, ["time", "text", "bbox_2d"]):
            # OCR detections - frame-level Detections with text attribute
            self._merge_frame_labels(labels, self._parse_frame_detections(value, video_path, sample, "text", "text"), key)
    
    def _parse_dict_value(self, key, value, labels):
        """Parse dict values as multiple Classifications.
        
        Used for nested structures like scene_info where each key-value pair
        becomes a separate Classification. Values are converted to sentence case.
        
        Example:
            Input: {"setting": "indoor", "time_of_day": "night"}
            Output: 
                labels["scene_info_setting"] = fo.Classification(label="Indoor")
                labels["scene_info_time_of_day"] = fo.Classification(label="Night")
        
        Args:
            key (str): Parent key name (e.g., "scene_info")
            value (dict): Dict with key-value pairs
            labels (dict): Labels dict to update (modified in place)
        """
        for subkey, subvalue in value.items():
            # Convert value to string and apply sentence case
            label_text = str(subvalue).capitalize()
            # Create field name: parent_key + "_" + subkey
            field_name = f"{key}_{subkey}"
            labels[field_name] = fol.Classification(label=label_text)
    
    def _merge_frame_labels(self, labels, frame_detections, key):
        """Merge frame-level detections into labels dict.
        
        Takes frame-level detections and merges them into the main labels dict
        using integer frame numbers as keys. Multiple fields can exist per frame.
        
        Args:
            labels (dict): Main labels dict (modified in place)
            frame_detections (dict): Dict of {frame_num: fo.Detections}
            key (str): Field name for these detections
        """
        for frame_num, dets in frame_detections.items():
            # Create frame entry if doesn't exist
            if frame_num not in labels:
                labels[frame_num] = {}
            # Add detections under field name
            labels[frame_num][key] = dets
    
    def _extract_json(self, text):
        """Extract JSON from model output.
        
        Handles two common formats:
        1. JSON wrapped in markdown code blocks: ```json {...} ```
        2. Raw JSON text
        
        Args:
            text (str): Model output text
        
        Returns:
            dict/list or None: Parsed JSON object, or None if parsing fails
        """
        # Try to find JSON in markdown code block first
        json_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text
        
        # Attempt to parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def _has_keys(self, item, required_keys):
        """Check if item is dict with all required keys.
        
        Used for automatic type detection based on JSON structure.
        
        Args:
            item: Item to check (typically dict)
            required_keys (list): List of required key names
        
        Returns:
            bool: True if item is dict and has all required keys
        """
        return isinstance(item, dict) and all(k in item for k in required_keys)
    
    def _is_simple_dict(self, value):
        """Check if dict has only simple string/number values (not lists or nested dicts).
        
        Used to determine if a dict like scene_info should be parsed as Classifications,
        or if it's a complex structure like object_count that should be skipped.
        
        Args:
            value (dict): Dictionary to check
        
        Returns:
            bool: True if all values are strings or numbers
        """
        if not isinstance(value, dict):
            return False
        
        for v in value.values():
            # Allow strings and numbers, reject lists and dicts
            if not isinstance(v, (str, int, float, bool)):
                return False
        
        return True
    
    def _parse_object_appearances(self, objects_list, video_path, sample):
        """Parse object appearances into TemporalDetections.
        
        Converts object appearance/disappearance times into temporal detections.
        Each object becomes a temporal detection spanning from first to last appearance.
        
        Expected JSON structure:
            [
                {"name": "car", "first_appears": "00:05.00", "last_appears": "00:15.50"},
                {"name": "person", "first_appears": "00:10.00", "last_appears": "00:20.00"},
                ...
            ]
        
        Args:
            objects_list (list): List of dicts with "name", "first_appears", "last_appears"
            video_path (str): Path to video file
            sample (fo.Sample): FiftyOne sample for metadata
        
        Returns:
            fo.TemporalDetections or None: Container with temporal detections for each object
        """
        if not objects_list:
            return None
        
        detections = []
        for obj in objects_list:
            # Convert timestamps to seconds
            first_sec = self._timestamp_to_seconds(obj.get("first_appears", "00:00.00"))
            last_sec = self._timestamp_to_seconds(obj.get("last_appears", "00:00.00"))
            
            # Create temporal detection for object's presence duration
            try:
                detection = fol.TemporalDetection.from_timestamps(
                    [first_sec, last_sec],
                    label=obj.get("name", "object"),
                    sample=sample
                )
                detections.append(detection)
            except Exception as e:
                logger.warning(f"Failed to create object appearance detection: {e}")
                continue
        
        # Alert if complete failure
        if not detections and objects_list:
            logger.error(
                f"Failed to parse ALL {len(objects_list)} object appearances. "
                f"Check video metadata and timestamp format."
            )
        
        if detections:
            return fol.TemporalDetections(detections=detections)
        return None
    
    def _parse_text_temporal(self, text_list, video_path, sample):
        """Parse text content temporal ranges into TemporalDetections.
        
        Converts text visibility periods into temporal detections.
        Each text instance becomes a temporal detection for when it's visible.
        
        Expected JSON structure:
            [
                {"start": "00:05.00", "end": "00:10.00", "text": "STOP"},
                {"start": "00:15.00", "end": "00:20.00", "text": "Main Street"},
                ...
            ]
        
        Args:
            text_list (list): List of dicts with "start", "end", "text"
            video_path (str): Path to video file
            sample (fo.Sample): FiftyOne sample for metadata
        
        Returns:
            fo.TemporalDetections or None: Container with temporal detections for each text
        """
        if not text_list:
            return None
        
        detections = []
        for text_item in text_list:
            # Convert timestamps to seconds
            start_sec = self._timestamp_to_seconds(text_item.get("start", "00:00.00"))
            end_sec = self._timestamp_to_seconds(text_item.get("end", "00:00.00"))
            
            # Create temporal detection for text visibility period
            try:
                detection = fol.TemporalDetection.from_timestamps(
                    [start_sec, end_sec],
                    label=text_item.get("text", "text"),
                    sample=sample
                )
                detections.append(detection)
            except Exception as e:
                logger.warning(f"Failed to create text temporal detection: {e}")
                continue
        
        # Alert if complete failure
        if not detections and text_list:
            logger.error(
                f"Failed to parse ALL {len(text_list)} text temporal detections. "
                f"Check video metadata and timestamp format."
            )
        
        if detections:
            return fol.TemporalDetections(detections=detections)
        return None
    
    def _parse_temporal_events(self, events_list, video_path, sample):
        """Parse temporal events into TemporalDetections.
        
        Converts model's timestamp-based events into FiftyOne's frame-based
        TemporalDetections. Uses TemporalDetection.from_timestamps() which
        automatically converts seconds to frame numbers using video metadata.
        
        Expected JSON structure:
            [
                {"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "event name"},
                {"start": "00:05.00", "end": "00:10.50", "description": "car enters"},
                ...
            ]
        
        Args:
            events_list (list): List of dicts with "start", "end", "description" keys
            video_path (str): Path to video file (for FPS fallback)
            sample (fo.Sample): FiftyOne sample (must have metadata for timestamp conversion)
        
        Returns:
            fo.TemporalDetections or None: Container with all temporal detections, or None if empty
        """
        if not events_list:
            return None
        
        detections = []
        for event in events_list:
            # Convert "mm:ss.ff" timestamps to seconds
            start_sec = self._timestamp_to_seconds(event.get("start", "00:00.00"))
            end_sec = self._timestamp_to_seconds(event.get("end", "00:00.00"))
            
            # Use FiftyOne's from_timestamps() to convert seconds to frame numbers
            # This uses the sample's video metadata (FPS) for accurate conversion
            try:
                detection = fol.TemporalDetection.from_timestamps(
                    [start_sec, end_sec],
                    label=event.get("description", "event"),
                    sample=sample
                )
                detections.append(detection)
            except Exception as e:
                logger.warning(f"Failed to create temporal detection: {e}")
                continue
        
        # Alert if complete failure
        if not detections and events_list:
            logger.error(
                f"Failed to parse ALL {len(events_list)} temporal events. "
                f"Check video metadata and timestamp format."
            )
        
        # Return container with all detections, or None if none were created
        if detections:
            return fol.TemporalDetections(detections=detections)
        return None
    
    def _parse_frame_detections(self, items_list, video_path, sample, label_key="label", text_key=None):
        """Parse frame-level detections (objects or OCR).
        
        Unified method for parsing both object detections and OCR detections.
        Converts model's timestamp+bbox format to FiftyOne's frame-level Detections.
        
        Coordinate Conversion:
        - Model outputs: [x1, y1, x2, y2] in 0-1000 scale (relative)
        - FiftyOne expects: [x, y, width, height] in 0-1 scale (relative)
        - Formula: x = x1/1000, y = y1/1000, w = (x2-x1)/1000, h = (y2-y1)/1000
        
        Expected JSON structure:
            [
                {"time": "mm:ss.ff", "bbox_2d": [100, 200, 300, 400], "label": "car"},
                {"time": "00:05.00", "bbox_2d": [150, 250, 350, 450], "label": "person"},
                ...
            ]
        
        Or for OCR:
            [
                {"time": "mm:ss.ff", "text": "STOP", "bbox_2d": [100, 200, 300, 400]},
                ...
            ]
        
        Args:
            items_list (list): List of dicts with "time", "bbox_2d", and label/text keys
            video_path (str): Path to video file
            sample (fo.Sample): FiftyOne sample (for FPS)
            label_key (str): Key to use for detection label (default: "label")
            text_key (str, optional): If provided, also store this key as custom attribute
                Used for OCR to store text content separately
            
        Returns:
            dict: Mapping of {frame_number: fo.Detections} for all frames with detections
        """
        # Get video FPS for timestamp → frame number conversion
        fps = self._get_video_fps(video_path, sample)
        frame_detections = {}
        
        for item in items_list:
            # Convert timestamp to 1-based frame number
            frame_num = int(self._timestamp_to_seconds(item.get("time", "00:00.00")) * fps) + 1
            
            # Get bounding box coordinates
            bbox = item.get("bbox_2d", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue  # Skip invalid bboxes
            
            # Validate and clip bbox coordinates to 0-1000 range
            x1, y1, x2, y2 = bbox[:4]
            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            x2 = max(0, min(1000, x2))
            y2 = max(0, min(1000, y2))
            
            # Ensure x2 > x1 and y2 > y1 (positive width/height)
            if x2 <= x1 or y2 <= y1:
                logger.debug(f"Skipping invalid bbox with negative dimensions: {bbox}")
                continue
            
            # Convert coordinates: 0-1000 scale → 0-1 scale, corner format → xywh format
            x, y, w, h = x1/1000, y1/1000, (x2-x1)/1000, (y2-y1)/1000
            
            # Create detection with converted coordinates
            detection = fol.Detection(label=item.get(label_key, ""), bounding_box=[x, y, w, h])
            
            # For OCR, also store text as custom attribute
            if text_key:
                detection[text_key] = item.get(text_key, "")
            
            # Add detection to frame (create Detections container if needed)
            frame_detections.setdefault(frame_num, fol.Detections(detections=[])).detections.append(detection)
        
        return frame_detections
    
    def _timestamp_to_seconds(self, timestamp_str):
        """Convert 'mm:ss.ff' timestamp to seconds.
        
        Parses Qwen3-VL's timestamp format and converts to total seconds.
        
        Format: "mm:ss.ff" where:
        - mm = minutes (2 digits)
        - ss = seconds (2 digits)
        - ff = centiseconds/frames (2 digits, treated as hundredths of second)
        
        Examples:
            "00:05.00" → 5.0 seconds
            "01:23.45" → 83.45 seconds
            "00:00.50" → 0.5 seconds
        
        Args:
            timestamp_str: Timestamp string in "mm:ss.ff" format
            
        Returns:
            float: Time in seconds, or 0.0 if parsing fails
        """
        # Parse mm:ss.ff format using regex
        match = re.match(r'(\d+):(\d+)\.(\d+)', str(timestamp_str))
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            centiseconds = int(match.group(3))
            
            # Convert to total seconds
            total_seconds = minutes * 60 + seconds + centiseconds / 100.0
            return total_seconds
        
        # Return 0 if format doesn't match (beginning of video)
        return 0.0
    
    def _get_video_fps(self, video_path, sample):
        """Get video FPS from sample metadata or video file.
        
        Tries to get FPS in priority order:
        1. From sample.metadata.frame_rate (if sample has metadata computed)
        2. From video file using decord VideoReader
        3. Default to 30.0 FPS if all else fails
        
        Args:
            video_path (str): Path to video file
            sample (fo.Sample): FiftyOne sample (may have metadata)
            
        Returns:
            float: Video frames per second
        """
        # Try to get from sample metadata (fastest, most reliable)
        if sample and hasattr(sample, 'metadata') and getattr(sample.metadata, 'frame_rate', None):
                return sample.metadata.frame_rate
        
        # Fallback: read directly from video file
        try:
            from decord import VideoReader, cpu
            return VideoReader(video_path, ctx=cpu(0)).get_avg_fps()
        except Exception as e:
            # Last resort: use common default
            logger.warning(f"Could not determine FPS: {e}. Using default 30.")
            return 30.0
    

