"""
FiftyOne integration for Qwen3-VL video understanding model.

This module provides an operation-driven implementation of the Qwen3-VL model
for video understanding tasks in FiftyOne. Each operation has a fixed prompt
and predictable parsing strategy:

Operations:
- description: Plain text video description
- temporal_localization: Temporal event detection
- tracking: Frame-level object tracking with bounding boxes
- ocr: Frame-level text extraction with bounding boxes
- comprehensive: Flexible multi-label analysis
- custom: User-defined prompts with plain text output

Output Types:
- Sample-level: Classifications, TemporalDetections, plain text
- Frame-level: Detections (objects, OCR text)

Key Design Principles:
- Operation determines both prompt and parsing behavior
- Prompts are immutable per operation (except custom)
- Predictable outputs for each operation type

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
  "summary": "Brief description of the video",
  "objects": [{"name": "object name", "first_appears": "mm:ss.ff", "last_appears": "mm:ss.ff"}],
  "events": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "event description"}],
  "text_content": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "text": "text content"}],
  "scene_info": {"setting": "<one-word-description-of-setting>", "time_of_day": "<one-word-description-of-time>", "location_type": "<one-word-description-of-location>"},
  "activities": {"primary_activity": "activity name", "secondary_activities": "comma-separated activities"}
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
        "prompt": """Track all objects in this video. For each frame where objects appear, provide:
- time: timestamp (mm:ss.ff)
- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale
- label: object label
Output in JSON: [{"time": "mm:ss.ff", "bbox_2d": [...], "label": "..."}, ...]"""
    },
    "ocr": {
        "prompt": """Extract all text appearing in this video. For each text instance, provide:
- time: timestamp (mm:ss.ff)
- text: the actual text content
- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale
Output in JSON: [{"time": "mm:ss.ff", "text": "...", "bbox_2d": [...]}, ...]"""
    },
    "custom": {
        "prompt": None  # User must provide via custom_prompt
    },
}


class Qwen3VLVideoModelConfig(fout.TorchImageModelConfig):
    """Configuration for Qwen3-VL video model.
    
    This config class defines all parameters for video processing, model inference,
    and output parsing. It inherits from TorchImageModelConfig to leverage FiftyOne's
    built-in configuration parsing utilities.
    
    Key Parameters:
        model_path: HuggingFace model identifier (default: "Qwen/Qwen3-VL-8B-Instruct")
        total_pixels: Max pixels for quality/memory tradeoff (default: 2048*32*32)
        max_frames: Max frames to sample from video (default: 120)
        sample_fps: Frame sampling rate (default: 10)
        max_new_tokens: Max tokens to generate (default: 8192)
        operation: Operation type - "comprehensive", "description", "temporal_localization", 
                  "tracking", "ocr", or "custom" (default: "comprehensive")
        custom_prompt: Required for operation="custom", forbidden otherwise
    
    Operation Constraints:
        - Each operation has a fixed prompt and parsing strategy
        - custom_prompt is ONLY allowed when operation="custom"
        - For operation="custom", custom_prompt is REQUIRED
    """
    
    def __init__(self, d):
        super().__init__(d)
        
        # Model parameters
        self.model_path = self.parse_string(d, "model_path", default="Qwen/Qwen3-VL-8B-Instruct")
        
        # Video processing parameters
        self.total_pixels = self.parse_number(d, "total_pixels", default=2048*32*32)
        self.min_pixels = self.parse_number(d, "min_pixels", default=64*32*32)
        self.max_frames = self.parse_number(d, "max_frames", default=120)
        self.sample_fps = self.parse_number(d, "sample_fps", default=10)
        self.image_patch_size = self.parse_number(d, "image_patch_size", default=16)
        
        # Text generation parameters
        self.max_new_tokens = self.parse_number(d, "max_new_tokens", default=8192)
        self.do_sample = self.parse_bool(d, "do_sample", default=True)
        self.temperature = self.parse_number(d, "temperature", default=0.7)
        self.top_p = self.parse_number(d, "top_p", default=0.8)
        self.top_k = self.parse_number(d, "top_k", default=20)
        self.repetition_penalty = self.parse_number(d, "repetition_penalty", default=1.0)
        
        # Operation configuration
        self.operation = self.parse_string(d, "operation", default="comprehensive")
        self.custom_prompt = self.parse_string(d, "custom_prompt", default=None)
        
        # Validate operation and custom_prompt relationship
        if self.operation != "custom" and self.custom_prompt is not None:
            raise ValueError("custom_prompt only allowed when operation='custom'")
        
        if self.operation == "custom" and self.custom_prompt is None:
            raise ValueError("custom_prompt required when operation='custom'")


class Qwen3VLVideoModel(fom.SamplesMixin, fom.Model):
    """FiftyOne wrapper for Qwen3-VL video understanding model.
    
    This model processes videos directly from file paths using operation-driven
    parsing. Each operation has a fixed prompt and predictable output structure.
    It inherits from both SamplesMixin (for per-sample field access) and Model 
    (base FiftyOne interface).
    
    Operations and Outputs:
        - description: Plain text summary in "summary" field
        - temporal_localization: fo.TemporalDetections in "events" field
        - tracking: fo.Detections per frame in frame.objects
        - ocr: fo.Detections per frame in frame.text_content
        - comprehensive: Mixed labels with flexible schema parsing
        - custom: Plain text summary with user-defined prompt
    
    For comprehensive operation, possible labels include:
        Sample-level: summary, events, objects, scene_info_*, activities_*
        Frame-level: objects, text_content (stored in sample.frames[N])
    """
    
    def __init__(self, config):
        fom.SamplesMixin.__init__(self)
        self.config = config
        
        # Detect and set best available device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Lazy loading - model and processor loaded on first predict() call
        self._processor = None
        self._model = None
        self._fields = {}
    
    @property
    def media_type(self):
        """Media type this model operates on (video)."""
        return "video"

    @property
    def needs_fields(self):
        """Dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        """Set the fields this model needs from samples."""
        self._fields = fields

    # Video processing parameters
    @property
    def total_pixels(self):
        """Max pixels for quality/memory tradeoff."""
        return self.config.total_pixels
    
    @total_pixels.setter
    def total_pixels(self, value):
        """Set max pixels."""
        self.config.total_pixels = value
    
    @property
    def min_pixels(self):
        """Min pixels threshold."""
        return self.config.min_pixels
    
    @min_pixels.setter
    def min_pixels(self, value):
        """Set min pixels."""
        self.config.min_pixels = value
    
    @property
    def max_frames(self):
        """Max frames to sample from video."""
        return self.config.max_frames
    
    @max_frames.setter
    def max_frames(self, value):
        """Set max frames."""
        self.config.max_frames = value
    
    @property
    def sample_fps(self):
        """Frame sampling rate."""
        return self.config.sample_fps
    
    @sample_fps.setter
    def sample_fps(self, value):
        """Set frame sampling rate."""
        self.config.sample_fps = value
    
    @property
    def image_patch_size(self):
        """Image patch size for vision processing."""
        return self.config.image_patch_size
    
    @image_patch_size.setter
    def image_patch_size(self, value):
        """Set image patch size."""
        self.config.image_patch_size = value
    
    # Text generation parameters
    @property
    def max_new_tokens(self):
        """Max tokens to generate."""
        return self.config.max_new_tokens
    
    @max_new_tokens.setter
    def max_new_tokens(self, value):
        """Set max new tokens."""
        self.config.max_new_tokens = value
    
    @property
    def do_sample(self):
        """Whether to use sampling (vs greedy decoding)."""
        return self.config.do_sample
    
    @do_sample.setter
    def do_sample(self, value):
        """Set sampling strategy."""
        self.config.do_sample = value
    
    @property
    def temperature(self):
        """Sampling temperature (higher = more random)."""
        return self.config.temperature
    
    @temperature.setter
    def temperature(self, value):
        """Set temperature."""
        self.config.temperature = value
    
    @property
    def top_p(self):
        """Nucleus sampling threshold."""
        return self.config.top_p
    
    @top_p.setter
    def top_p(self, value):
        """Set top_p."""
        self.config.top_p = value
    
    @property
    def top_k(self):
        """Top-k sampling parameter."""
        return self.config.top_k
    
    @top_k.setter
    def top_k(self, value):
        """Set top_k."""
        self.config.top_k = value
    
    @property
    def repetition_penalty(self):
        """Repetition penalty (higher = less repetition)."""
        return self.config.repetition_penalty
    
    @repetition_penalty.setter
    def repetition_penalty(self, value):
        """Set repetition penalty."""
        self.config.repetition_penalty = value
    
    # Operation configuration
    @property
    def operation(self):
        """Current operation type (determines prompt and parsing)."""
        return self.config.operation
    
    @operation.setter
    def operation(self, value):
        """Set operation type with validation."""
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self.config.operation = value
    
    @property
    def prompt(self):
        """Get current prompt for the operation.
        
        For custom operation, returns the user-provided custom_prompt.
        For all other operations, returns the predefined prompt.
        """
        if self.config.operation == "custom":
            return self.config.custom_prompt  # Guaranteed to exist by validation
        return OPERATIONS[self.config.operation]["prompt"]
    
    @prompt.setter
    def prompt(self, value):
        """Set custom prompt (only valid for custom operation)."""
        if self.config.operation != "custom":
            raise ValueError("Cannot set prompt for predefined operations. Use operation='custom'")
        self.config.custom_prompt = value
    
    def _load_model(self):
        """Load Qwen3-VL model and processor from HuggingFace.
        
        Loads both the processor (for tokenization and vision processing) and
        the model (for inference). Enables Flash Attention 2 if available for 
        faster inference on CUDA devices.
        """
        logger.info(f"Loading Qwen3-VL model from {self.config.model_path}")
        
        # Load processor for tokenization and vision processing
        self._processor = AutoProcessor.from_pretrained(self.config.model_path)
        
        # Prepare model loading kwargs with optimizations
        model_kwargs = {"dtype": "auto", "device_map": self.device}
        
        # Enable Flash Attention 2 if using CUDA and it's available
        if self.device == "cuda" and is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        # Load model with optimizations and set to eval mode
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_path, **model_kwargs
        ).eval()
        
        logger.info("Model loaded successfully")
    
    def predict(self, arg, sample=None):
        """Run inference on a video file.
        
        This is the main prediction method called by FiftyOne's apply_model().
        
        Prompt Priority:
        - For predefined operations: Uses fixed operation prompt (immutable)
        - For custom operation: Uses config.custom_prompt, or can be overridden
          per-sample via needs_fields
        
        Args:
            arg: Unused (for compatibility with Model interface)
            sample: FiftyOne sample (required for video filepath and metadata)
            
        Returns:
            dict: Mixed dictionary with sample-level (string keys) and 
                  frame-level (integer keys) labels
        """
        # Lazy load model on first use
        if self._model is None:
            self._load_model()
        
        if sample is None:
            raise ValueError("Sample is required for video processing")
        
        # Validate metadata for operations that need timestamp conversion
        needs_metadata = self.config.operation in [
            "comprehensive", "temporal_localization", "tracking", "ocr"
        ]
        if needs_metadata and not hasattr(sample, 'metadata'):
            raise ValueError(
                f"Operation '{self.config.operation}' requires sample metadata. "
                f"Call dataset.compute_metadata() first."
            )
        
        # Determine prompt to use
        prompt = self.prompt  # Get operation prompt
        
        # For custom operation only, allow per-sample prompt override via needs_fields
        if self.config.operation == "custom":
            prompt_field = self._fields.get("prompt_field") or next(iter(self._fields.values()), None)
            if prompt_field:
                field_value = sample.get_field(prompt_field)
                if field_value:
                    prompt = str(field_value)
        
        # Build messages in Qwen3-VL format
        # Video is passed as file path, model handles frame extraction internally
        messages = [{
            "role": "user",
            "content": [
                {
                    "video": sample.filepath,
                    "total_pixels": self.config.total_pixels,
                    "min_pixels": self.config.min_pixels,
                    "max_frames": self.config.max_frames,
                    "sample_fps": self.config.sample_fps
                },
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Run inference and parse results
        output_text = self._run_inference(messages)
        logger.debug(f"Model output: {output_text[:200]}...")
        
        labels = self._parse_output(output_text, sample)
        logger.debug(f"Parsed {len(labels)} label fields")
        
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
            messages: List of message dicts in Qwen3-VL format
        
        Returns:
            str: Generated text output from model
        """
        # Step 1: Apply chat template to format conversation
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Step 2: Process vision info - extracts and encodes video frames
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=self.config.image_patch_size,
            return_video_metadata=True
        )
        
        # Step 3: Unpack video inputs and metadata
        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        
        # Step 4: Prepare final model inputs and move to device
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
        
        # Step 5: Generate response with appropriate sampling strategy
        with torch.no_grad():
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
        
        # Step 6: Decode generated tokens to text (exclude input prompt)
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
    
    def _extract_json(self, text):
        """Extract JSON from model output.
        
        Handles two common formats:
        1. JSON wrapped in markdown code blocks: ```json {...} ```
        2. Raw JSON text
        
        Args:
            text: Model output text
        
        Returns:
            Parsed JSON object (dict or list), or None if parsing fails
        """
        # Try to find JSON in markdown code block first
        json_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def _parse_temporal_only(self, json_data, sample):
        """Parse output as temporal detections only.
        
        Used for temporal_localization operation. Expects a list of events
        with start/end timestamps and descriptions.
        
        Args:
            json_data: Parsed JSON (list or dict)
            sample: FiftyOne sample
            
        Returns:
            dict: {"events": fo.TemporalDetections} (empty if no detections)
        """
        # Handle both list and dict with "events" key
        if isinstance(json_data, list):
            items = json_data if json_data else []
        elif isinstance(json_data, dict) and "events" in json_data:
            items = json_data["events"] if json_data["events"] else []
        else:
            if json_data:
                logger.warning("Expected list or dict with 'events' key for temporal_localization")
            items = []
        
        if not items:
            # Return empty TemporalDetections container
            return {"events": fol.TemporalDetections(detections=[])}
        
        detections = self._parse_temporal_detections(items, sample, "events")
        return {"events": detections if detections else fol.TemporalDetections(detections=[])}
    
    def _parse_tracking_only(self, json_data, sample):
        """Parse output as object tracking detections only.
        
        Used for tracking operation. Expects a list of frame-level detections
        with time, bbox_2d, and label.
        
        Args:
            json_data: Parsed JSON (list or dict)
            sample: FiftyOne sample
            
        Returns:
            dict: Frame-level labels {frame_num: {"objects": fo.Detections}}
        """
        # Handle both list and dict with "objects" key
        if isinstance(json_data, list):
            items = json_data if json_data else []
        elif isinstance(json_data, dict) and "objects" in json_data:
            items = json_data["objects"] if json_data["objects"] else []
        else:
            if json_data:
                logger.warning("Expected list or dict with 'objects' key for tracking")
            items = []
        
        if not items:
            # No detections - return empty frame-level dict (FiftyOne handles this gracefully)
            return {}
        
        frame_detections = self._parse_frame_detections(items, sample, text_key=None)
        
        if not frame_detections:
            return {}
        
        # Convert to final format
        labels = {}
        for frame_num, dets in frame_detections.items():
            labels[frame_num] = {"objects": dets}
        
        return labels
    
    def _parse_ocr_only(self, json_data, sample):
        """Parse output as OCR detections only.
        
        Used for ocr operation. Expects a list of frame-level detections
        with time, text, and bbox_2d.
        
        Args:
            json_data: Parsed JSON (list or dict)
            sample: FiftyOne sample
            
        Returns:
            dict: Frame-level labels {frame_num: {"text_content": fo.Detections}}
        """
        # Handle both list and dict with "text_content" key
        if isinstance(json_data, list):
            items = json_data if json_data else []
        elif isinstance(json_data, dict) and "text_content" in json_data:
            items = json_data["text_content"] if json_data["text_content"] else []
        else:
            if json_data:
                logger.warning("Expected list or dict with 'text_content' key for ocr")
            items = []
        
        if not items:
            # No detections - return empty frame-level dict (FiftyOne handles this gracefully)
            return {}
        
        frame_detections = self._parse_frame_detections(items, sample, text_key="text")
        
        if not frame_detections:
            return {}
        
        # Convert to final format
        labels = {}
        for frame_num, dets in frame_detections.items():
            labels[frame_num] = {"text_content": dets}
        
        return labels
    
    def _parse_comprehensive(self, json_data, sample):
        """Parse output with flexible schema-based parsing.
        
        Used for comprehensive operation. Automatically detects structure
        and parses all JSON keys based on their value types.
        
        Args:
            json_data: Parsed JSON (dict)
            sample: FiftyOne sample
            
        Returns:
            dict: Mixed sample-level and frame-level labels (may be empty)
        """
        if not json_data:
            # Return empty dict - FiftyOne will handle gracefully
            return {}
        
        if isinstance(json_data, list):
            logger.warning("Expected dict for comprehensive operation, got list")
            return {}
        
        labels = {}
        
        # Parse each key based on its value structure
        for key, value in json_data.items():
            
            # Dispatch based on value type
            if isinstance(value, str):
                # String values stored as plain text
                labels[key] = value
            elif isinstance(value, dict) and self._is_simple_dict(value):
                # Simple dicts parsed as nested classifications
                self._parse_dict_value(key, value, labels)
            elif isinstance(value, list) and value:
                # Lists - detect structure and parse appropriately
                self._parse_list_value(key, value, labels, sample)
        
        return labels
    
    def _parse_output(self, output_text, sample):
        """Parse model output into FiftyOne labels using operation-driven dispatch.
        
        Each operation has a specific parsing strategy:
        - description/custom: Plain text output (no JSON parsing)
        - temporal_localization: Parse as temporal detections
        - tracking: Parse as frame-level object detections
        - ocr: Parse as frame-level text detections
        - comprehensive: Flexible schema-based parsing
        
        Args:
            output_text: Raw text output from model
            sample: FiftyOne sample for metadata
        
        Returns:
            dict: Mixed sample-level (string keys) and frame-level (int keys) labels
        """
        # Operations that return plain text (no JSON parsing)
        if self.config.operation in ["description", "custom"]:
            return {"summary": output_text}
        
        # Operations that require JSON parsing
        json_data = self._extract_json(output_text)
        
        # Dispatch to operation-specific parser
        if self.config.operation == "temporal_localization":
            return self._parse_temporal_only(json_data, sample)
        
        elif self.config.operation == "tracking":
            return self._parse_tracking_only(json_data, sample)
        
        elif self.config.operation == "ocr":
            return self._parse_ocr_only(json_data, sample)
        
        elif self.config.operation == "comprehensive":
            return self._parse_comprehensive(json_data, sample)
        
        else:
            logger.warning(f"Unknown operation: {self.config.operation}")
            return {"summary": output_text}
    
    def _is_simple_dict(self, value):
        """Check if dict has only simple string/number values.
        
        Used to determine if a dict should be parsed as Classifications,
        or if it's a complex structure that should be skipped.
        
        Args:
            value: Dictionary to check
        
        Returns:
            bool: True if all values are strings or numbers
        """
        return isinstance(value, dict) and all(
            isinstance(v, (str, int, float, bool)) for v in value.values()
        )
    
    def _parse_dict_value(self, key, value, labels):
        """Parse dict values as Classifications.
        
        Used for nested structures like scene_info and activities where each 
        key-value pair becomes a separate field.
        
        Special handling:
        - Fields ending with "activities" (plural) are parsed as fo.Classifications
        - All other fields are parsed as fo.Classification (single value)
        
        Args:
            key: Parent key name (e.g., "scene_info")
            value: Dict with key-value pairs
            labels: Labels dict to update (modified in place)
        """
        for subkey, subvalue in value.items():
            field_name = f"{key}_{subkey}"
            
            # Parse fields ending in "activities" as multiple Classifications
            if subkey.endswith("activities"):
                if isinstance(subvalue, str):
                    # Parse comma-separated values
                    items = [item.strip().capitalize() for item in subvalue.split(',') if item.strip()]
                    labels[field_name] = fol.Classifications(
                        classifications=[fol.Classification(label=item) for item in items]
                    )
                else:
                    # Single value - still wrap in Classifications for consistency
                    labels[field_name] = fol.Classifications(
                        classifications=[fol.Classification(label=str(subvalue).capitalize())]
                    )
            else:
                # Regular fields - single Classification
                labels[field_name] = fol.Classification(label=str(subvalue).capitalize())
    
    def _parse_list_value(self, key, value, labels, sample):
        """Dispatch list parsing based on item structure.
        
        Examines the first item in the list to determine type, then parses accordingly:
        - Temporal events: Items with "start", "end", "description"
        - Object appearances: Items with "name", "first_appears", "last_appears"
        - Text content: Items with "start", "end", "text"
        - Object detections: Items with "time", "bbox_2d", "label"
        - OCR detections: Items with "time", "text", "bbox_2d"
        
        Args:
            key: JSON key name (becomes field name)
            value: List of items to parse
            labels: Labels dict to update (modified in place)
            sample: FiftyOne sample
        """
        first = value[0]
        
        # Define structure patterns with (required_keys, parser, label_type)
        patterns = [
            (["start", "end", "description"], self._parse_temporal_detections, "events"),
            (["name", "first_appears", "last_appears"], self._parse_temporal_detections, "objects"),
            (["start", "end", "text"], self._parse_temporal_detections, "text"),
            (["time", "bbox_2d", "label"], self._parse_frame_detections, None),
            (["time", "text", "bbox_2d"], self._parse_frame_detections, "text"),
        ]
        
        # Match structure and dispatch to appropriate parser
        for required_keys, parser, label_type in patterns:
            if all(k in first for k in required_keys):
                if parser == self._parse_frame_detections:
                    # Frame-level detections
                    frame_labels = parser(value, sample, label_type)
                    self._merge_frame_labels(labels, frame_labels, key)
                else:
                    # Sample-level temporal detections
                    detections = parser(value, sample, label_type)
                    if detections:
                        labels[key] = detections
                return
    
    def _parse_temporal_detections(self, items, sample, label_type):
        """Parse temporal detections (unified method for events, objects, text).
        
        Converts model's timestamp-based temporal data into FiftyOne's frame-based
        TemporalDetections. Uses TemporalDetection.from_timestamps() which
        automatically converts seconds to frame numbers using video metadata.
        
        Args:
            items: List of dicts with temporal information
            sample: FiftyOne sample (must have metadata for timestamp conversion)
            label_type: Type of temporal data ("events", "objects", or "text")
            
        Returns:
            fo.TemporalDetections or None: Container with all temporal detections
        """
        detections = []
        
        for item in items:
            # Determine timestamps and label based on type
            if label_type == "events":
                start = item.get("start", "00:00.00")
                end = item.get("end", "00:00.00")
                label = str(item.get("description", "event")).capitalize()
            elif label_type == "objects":
                start = item.get("first_appears", "00:00.00")
                end = item.get("last_appears", "00:00.00")
                label = str(item.get("name", "object")).capitalize()
            else:  # text
                start = item.get("start", "00:00.00")
                end = item.get("end", "00:00.00")
                label = str(item.get("text", "text")).capitalize()
            
            # Convert timestamps to seconds
            start_sec = self._timestamp_to_seconds(start)
            end_sec = self._timestamp_to_seconds(end)
            
            # Use FiftyOne's from_timestamps() to convert to frame numbers
            detection = fol.TemporalDetection.from_timestamps(
                [start_sec, end_sec], label=label, sample=sample
            )
            detections.append(detection)
        
        return fol.TemporalDetections(detections=detections) if detections else None
    
    def _parse_frame_detections(self, items, sample, text_key=None):
        """Parse frame-level detections (objects or OCR).
        
        Unified method for parsing both object detections and OCR detections.
        Converts model's timestamp+bbox format to FiftyOne's frame-level Detections.
        
        Coordinate Conversion:
        - Model outputs: [x1, y1, x2, y2] in 0-1000 scale (relative)
        - FiftyOne expects: [x, y, width, height] in 0-1 scale (relative)
        
        Args:
            items: List of dicts with "time", "bbox_2d", and label/text keys
            sample: FiftyOne sample (for FPS)
            text_key: If provided, also store this key as custom attribute (for OCR)
            
        Returns:
            dict: Mapping of {frame_number: fo.Detections}
        """
        fps = self._get_video_fps(sample)
        frame_detections = {}
        
        for item in items:
            # Convert timestamp to 1-based frame number
            frame_num = int(self._timestamp_to_seconds(item.get("time", "00:00.00")) * fps) + 1
            
            # Get and validate bounding box
            bbox = item.get("bbox_2d", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            
            # Clip coordinates to 0-1000 range
            x1, y1, x2, y2 = [max(0, min(1000, c)) for c in bbox[:4]]
            
            # Skip invalid boxes with negative dimensions
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Convert: 0-1000 scale → 0-1 scale, corner format → xywh format
            x, y, w, h = x1/1000, y1/1000, (x2-x1)/1000, (y2-y1)/1000
            
            # Create detection with appropriate label
            label = item.get("text" if text_key else "label", "")
            detection = fol.Detection(label=label, bounding_box=[x, y, w, h])
            
            # For OCR, also store text as custom attribute
            if text_key:
                detection[text_key] = item.get(text_key, "")
            
            # Add detection to frame
            if frame_num not in frame_detections:
                frame_detections[frame_num] = fol.Detections(detections=[])
            frame_detections[frame_num].detections.append(detection)
        
        return frame_detections
    
    def _merge_frame_labels(self, labels, frame_detections, key):
        """Merge frame-level detections into labels dict.
        
        Takes frame-level detections and merges them into the main labels dict
        using integer frame numbers as keys. Multiple fields can exist per frame.
        
        Args:
            labels: Main labels dict (modified in place)
            frame_detections: Dict of {frame_num: fo.Detections}
            key: Field name for these detections
        """
        for frame_num, dets in frame_detections.items():
            if frame_num not in labels:
                labels[frame_num] = {}
            labels[frame_num][key] = dets
    
    def _timestamp_to_seconds(self, timestamp_str):
        """Convert 'mm:ss.ff' timestamp to seconds.
        
        Parses Qwen3-VL's timestamp format and converts to total seconds.
        Format: "mm:ss.ff" where mm=minutes, ss=seconds, ff=centiseconds
        
        Args:
            timestamp_str: Timestamp string in "mm:ss.ff" format
            
        Returns:
            float: Time in seconds, or 0.0 if parsing fails
        """
        match = re.match(r'(\d+):(\d+)\.(\d+)', str(timestamp_str))
        if not match:
            return 0.0
        
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        centiseconds = int(match.group(3))
        
        return minutes * 60 + seconds + centiseconds / 100.0
    
    def _get_video_fps(self, sample):
        """Get video FPS from sample metadata or video file.
        
        Tries to get FPS from sample.metadata.frame_rate first (fastest).
        Falls back to reading from video file using decord.
        
        Args:
            sample: FiftyOne sample
            
        Returns:
            float: Video frames per second
        """
        # Try to get from sample metadata (fastest, most reliable)
        if hasattr(sample, 'metadata') and hasattr(sample.metadata, 'frame_rate'):
            return sample.metadata.frame_rate
        
        # Fallback: read directly from video file
        from decord import VideoReader, cpu
        return VideoReader(sample.filepath, ctx=cpu(0)).get_avg_fps()