# Qwen3-VL Video Model for FiftyOne

![image](qwen3vl_video_fo.gif)

A FiftyOne zoo model integration for Qwen3-VL that enables comprehensive video understanding with multiple label types in a single forward pass.

## Quick Start

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/qwen3vl_video",
    overwrite=True
)

# Load a video dataset
dataset = foz.load_zoo_dataset("quickstart-video", max_samples=5)
dataset.compute_metadata()  # Required for temporal/spatial operations

# Load the model (defaults to comprehensive operation)
model = foz.load_zoo_model("Qwen/Qwen3-VL-8B-Instruct")

# Apply to dataset
dataset.apply_model(model, label_field="analysis")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

## Features

- 🎬 **Video Description** - Natural language video summaries
- ⏱️ **Temporal Event Localization** - Events with precise start/end timestamps
- 📦 **Object Tracking** - Track objects across frames with bounding boxes
- 📝 **Video OCR** - Extract text from frames with spatial locations
- 🎯 **Comprehensive Analysis** - All of the above in one pass
- 🔧 **Custom Prompts** - Full control over model behavior

## Supported Models

- `Qwen/Qwen3-VL-2B-Instruct` (4-6GB VRAM)
- `Qwen/Qwen3-VL-4B-Instruct` (8-12GB VRAM) 
- `Qwen/Qwen3-VL-8B-Instruct` (16-24GB VRAM) **[Recommended]**

## Operation Modes

The model supports 6 operation modes, each with a fixed prompt and predictable output format:

### 1. Comprehensive (Default)
Analyzes video for all aspects: description, events, objects, scene info, activities.

```python
model = foz.load_zoo_model("Qwen/Qwen3-VL-8B-Instruct")
model.operation = "comprehensive"
dataset.apply_model(model, label_field="analysis")
```

**Output fields:**
- `analysis_summary` - Video description (string)
- `analysis_events` - Temporal events (fo.TemporalDetections)
- `analysis_objects` - Object appearances (fo.TemporalDetections)
- `analysis_scene_info_*` - Scene classifications
- `analysis_activities_*` - Activity classifications
- `sample.frames[N].objects` - Frame-level object detections
- `sample.frames[N].text_content` - Frame-level OCR

### 2. Description
Simple video description without structured output.

```python
model.operation = "description"
dataset.apply_model(model, label_field="desc")
```

**Output fields:**
- `desc_summary` - Plain text description

**✅ Does NOT require metadata**

### 3. Temporal Localization
Detects and localizes events in time.

```python
model.operation = "temporal_localization"
dataset.apply_model(model, label_field="events")
```

**Output fields:**
- `events` - fo.TemporalDetections with start/end frames

### 4. Tracking
Tracks objects across frames with bounding boxes.

```python
model.operation = "tracking"
dataset.apply_model(model, label_field="tracking")
```

**Output fields:**
- `sample.frames[N].objects` - fo.Detections per frame

⚠️ **Note:** Tracking can be slow and results may vary. Test on a small subset first.

### 5. OCR
Extracts text from video frames with bounding boxes.

```python
model.operation = "ocr"
dataset.apply_model(model, label_field="ocr")
```

**Output fields:**
- `sample.frames[N].text_content` - fo.Detections with text labels

### 6. Custom
Full control over prompts for specialized use cases.

```python
model.operation = "custom"
model.custom_prompt = """Analyze this video and identify:
- Changes in lighting
- Unusual events
- Weather changes

Output in JSON format:
{
  "lighting_changes": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}],
  "unusual_events": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}],
  "weather": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}]
}

Return empty list [] if nothing detected.
"""

dataset.apply_model(model, label_field="custom_analysis")
```

**Output fields:**
- `custom_analysis_result` - Raw text output (you post-process as needed)

**Custom mode is perfect for:**
- Domain-specific analysis
- Specialized event detection
- Custom JSON schemas you'll parse yourself
- Experimenting with prompts

#### Post-Processing Custom Output

Custom operation returns raw text, which you can parse into FiftyOne labels:

```python
import fiftyone as fo
import json
import re

def parse_time_str(timestamp_str):
    """Convert 'mm:ss.ff' timestamp to seconds."""
    match = re.match(r'(\d+):(\d+)\.(\d+)', str(timestamp_str))
    if not match:
        return 0.0
    
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    centiseconds = int(match.group(3))
    
    return minutes * 60 + seconds + centiseconds / 100.0

def parse_events_to_temporal_detections(events, label, sample):
    """Convert event list to FiftyOne TemporalDetections."""
    detections = []
    for event in events:
        start_sec = parse_time_str(event["start"])
        end_sec = parse_time_str(event["end"])
        detection = fo.TemporalDetection.from_timestamps(
            [start_sec, end_sec],
            label=label,
            sample=sample
        )
        detection.set_attribute_value("description", event["description"])
        detections.append(detection)
    return fo.TemporalDetections(detections=detections)

def clean_and_parse_json(text):
    """Remove markdown code blocks and parse JSON."""
    text = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()
    return json.loads(text)

# Process all samples
for sample in dataset.iter_samples(autosave=True):
    content_str = sample["custom_analysis_result"]
    events_dict = clean_and_parse_json(content_str)
    
    for category, events in events_dict.items():
        if events:
            sample[category] = parse_events_to_temporal_detections(events, category, sample)
```

**Example: Parsing as Classifications**

For categorical analysis like content rating or sentiment:

```python
model.operation = "custom"
model.custom_prompt = """Analyze this video and provide:
{
  "content_type": "educational/entertainment/promotional/other",
  "safety_rating": "safe/moderate/unsafe",
  "primary_activity": "sports/cooking/gaming/vlog/other"
}
"""

dataset.apply_model(model, label_field="analysis")

# Parse into Classifications
for sample in dataset.iter_samples(autosave=True):
    content_str = sample["analysis_result"]
    result = clean_and_parse_json(content_str)
    
    # Add as Classifications
    sample["content_type"] = fo.Classification(label=result["content_type"])
    sample["safety_rating"] = fo.Classification(label=result["safety_rating"])
    sample["primary_activity"] = fo.Classification(label=result["primary_activity"])
```

## Dynamic Reconfiguration

The model supports dynamic property changes without reloading:

```python
# Load once
model = foz.load_zoo_model("Qwen/Qwen3-VL-8B-Instruct")

# Switch operations (no reload!)
model.operation = "description"
dataset.apply_model(model, label_field="desc")

model.operation = "ocr"
dataset.apply_model(model, label_field="ocr")

model.operation = "temporal_localization"
dataset.apply_model(model, label_field="events")

# Adjust video processing
model.max_frames = 60
model.sample_fps = 5

# Tune generation
model.temperature = 0.9
model.max_new_tokens = 4096
```

**All configurable properties:**
```python
# Video processing
model.total_pixels = 2048*32*32
model.min_pixels = 64*32*32
model.max_frames = 120
model.sample_fps = 10
model.image_patch_size = 16

# Text generation
model.max_new_tokens = 8192
model.do_sample = True
model.temperature = 0.7
model.top_p = 0.8
model.top_k = 20
model.repetition_penalty = 1.0

# Operation
model.operation = "comprehensive"  # or other modes
model.custom_prompt = "..."  # for custom operation
```

## Installation

### Prerequisites
```bash
pip install fiftyone
```

### Model Dependencies
When you first load the model, FiftyOne will automatically install:
- `transformers>=4.37.0`
- `torch`
- `torchvision` 
- `qwen-vl-utils`
- `decord`

Or install manually:
```bash
pip install transformers torch torchvision qwen-vl-utils decord
```

## Memory Configuration

### Low Memory (4-6GB VRAM)
```python
model = foz.load_zoo_model(
    "Qwen/Qwen3-VL-2B-Instruct",
    total_pixels=5*1024*32*32,
    max_frames=32,
    sample_fps=0.5
)
```

### Balanced (8-12GB VRAM)
```python
model = foz.load_zoo_model(
    "Qwen/Qwen3-VL-4B-Instruct",
    total_pixels=20*1024*32*32,
    max_frames=64,
    sample_fps=1
)
```

### High Quality (16-24GB VRAM)
```python
model = foz.load_zoo_model(
    "Qwen/Qwen3-VL-8B-Instruct",
    total_pixels=128*1024*32*32,
    max_frames=256,
    sample_fps=2
)
```

## Important: Metadata Requirement

Most operations require video metadata for timestamp and frame conversion:

```python
# ALWAYS do this first!
dataset.compute_metadata()

# Then apply model
model = foz.load_zoo_model("Qwen/Qwen3-VL-8B-Instruct")
dataset.apply_model(model, label_field="analysis")
```

**Operations requiring metadata:**
- ✅ `comprehensive`
- ✅ `temporal_localization`
- ✅ `tracking`
- ✅ `ocr`

**Operations NOT requiring metadata:**
- ❌ `description`
- ❌ `custom` (if not using temporal features)

## Complete Example

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import load_from_hub

# Register model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/qwen3vl_video",
    overwrite=True
)

# Load dataset
dataset = load_from_hub(
    "harpreetsahota/random_short_videos",
    dataset_name="random_short_videos"
)

# Compute metadata (required!)
dataset.compute_metadata()

# Load model
model = foz.load_zoo_model("Qwen/Qwen3-VL-8B-Instruct")

# Run comprehensive analysis
model.operation = "comprehensive"
dataset.apply_model(model, label_field="comprehensive", skip_failures=True)

# Add descriptions
model.operation = "description"
dataset.apply_model(model, label_field="desc", skip_failures=True)

# Extract text with OCR
model.operation = "ocr"
dataset.apply_model(model, label_field="ocr", skip_failures=True)

# Detect temporal events
model.operation = "temporal_localization"
dataset.apply_model(model, label_field="events", skip_failures=True)

# Custom analysis
model.operation = "custom"
model.custom_prompt = """Analyze for safety concerns:
{
  "potential_hazards": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}],
  "safety_rating": "low/medium/high"
}
"""
dataset.apply_model(model, label_field="safety", skip_failures=True)

# Launch app
session = fo.launch_app(dataset)
```

## Coordinate System

Bounding boxes follow FiftyOne's format:
```python
[x, y, width, height]  # All values in [0, 1] relative coordinates
```

Where:
- `x` = top-left x (fraction of image width)
- `y` = top-left y (fraction of image height)
- `width` = box width (fraction of image width)
- `height` = box height (fraction of image height)

## Timestamp Format

Model outputs use `mm:ss.ff` format (minutes:seconds.centiseconds).
The integration automatically converts these to frame numbers using video FPS.

## Troubleshooting

### "Sample metadata required"
```python
# Fix: Compute metadata first
dataset.compute_metadata()
```

### Out of memory errors
```python
# Reduce processing parameters
model.total_pixels = 1024*32*32  # Lower resolution
model.max_frames = 32            # Fewer frames
model.sample_fps = 0.5           # Lower sampling rate
```

### Empty results / No detections
This is normal! Not all videos contain text (OCR) or trackable objects. The model will create empty label containers instead of raising errors.

### Slow inference
```python
# Use smaller model
model = foz.load_zoo_model("Qwen/Qwen3-VL-2B-Instruct")

# Process fewer frames
model.max_frames = 32
model.sample_fps = 0.5

# Reduce video quality
model.total_pixels = 5*1024*32*32
```

## Advanced: Building Multi-Field Models

This implementation demonstrates how to create FiftyOne models that add multiple fields in one inference pass:

```python
def predict(self, arg, sample=None):
    """Return dict with mixed sample/frame-level labels"""
    return {
        # Sample-level (string keys)
        "summary": "Video description",
        "events": fo.TemporalDetections(detections=[...]),
        
        # Frame-level (integer keys)
        1: {"objects": fo.Detections(detections=[...])},
        5: {"objects": fo.Detections(detections=[...])}
    }
```

See the source code in `zoo.py` for a complete reference implementation.

## License

- FiftyOne: Apache 2.0
- Qwen3-VL: Apache 2.0 (see [model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct))
- This integration: Apache 2.0

## Citation

```bibtex
@article{qwen3vl2024,
  title={Qwen3-VL: Towards Versatile Vision-Language Understanding},
  author={Qwen Team},
  year={2024}
}

@misc{fiftyone2020,
  title={FiftyOne},
  author={Voxel51},
  year={2020},
  howpublished={\url{https://fiftyone.ai}}
}
```

## Support & Contributing

- 🐛 **Issues**: [GitHub Issues](https://github.com/harpreetsahota204/qwen3vl_video/issues)
- 📖 **FiftyOne Docs**: [docs.voxel51.com](https://docs.voxel51.com)
- 💬 **FiftyOne Community**: [Slack](https://slack.voxel51.com)
- 🤖 **Qwen3-VL**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## Acknowledgments

Built on top of:
- [FiftyOne](https://fiftyone.ai) by Voxel51
- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) by Qwen Team
- [Transformers](https://github.com/huggingface/transformers) by HuggingFace
