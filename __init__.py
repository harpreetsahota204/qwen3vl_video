import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import Qwen3VLVideoModel, Qwen3VLVideoModelConfig

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the Qwen3-VL model from HuggingFace.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load a Qwen3-VL video model for use with FiftyOne.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID or path to model files
            Default: "Qwen/Qwen3-VL-8B-Instruct"
        **kwargs: Additional config parameters:
            - operation: Operation type (comprehensive, description, tracking, ocr, temporal_localization)
            - custom_prompt: Custom prompt to override operation default
            - max_frames: Maximum frames to sample (default: 120)
            - sample_fps: Frame sampling rate (default: 10)
            - total_pixels: Quality/memory tradeoff (default: 2048*32*32)
            - max_new_tokens: Max tokens to generate (default: 8192)
            - output_keys: List of JSON keys to parse (default: None, parses all)
        
    Returns:
        Qwen3VLVideoModel: Initialized model ready for inference
    
    Example:
        model = load_model(
            model_path="Qwen/Qwen3-VL-8B-Instruct",
            operation="comprehensive",
            max_frames=120,
            sample_fps=10
        )
        dataset.apply_model(model, label_field="predictions")
    """
    if model_path is None:
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
    
    config_dict = {"model_path": model_path}
    config_dict.update(kwargs)
    
    config = Qwen3VLVideoModelConfig(config_dict)
    return Qwen3VLVideoModel(config)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    inputs = types.Object()
    
    # Operation selection
    inputs.enum(
        "operation",
        values=[
            "comprehensive",
            "description", 
            "temporal_localization",
            "tracking",
            "ocr"
        ],
        default="comprehensive",
        label="Operation",
        description="Type of video analysis to perform",
    )
    
    # Custom prompt
    inputs.str(
        "custom_prompt",
        default=None,
        required=False,
        label="Custom Prompt",
        description="Optional custom prompt to override operation default",
    )
    
    # Video processing parameters
    inputs.int(
        "max_frames",
        default=120,
        label="Max Frames",
        description="Maximum number of frames to sample from video",
    )
    
    inputs.int(
        "sample_fps",
        default=10,
        label="Sample FPS",
        description="Frame sampling rate (-1 for uniform sampling)",
    )
    
    inputs.int(
        "total_pixels",
        default=2048*32*32,
        label="Total Pixels",
        description="Maximum total pixels (quality/memory tradeoff)",
    )
    
    # Generation parameters
    inputs.int(
        "max_new_tokens",
        default=8192,
        label="Max New Tokens",
        description="Maximum tokens to generate in response",
    )
    
    inputs.bool(
        "do_sample",
        default=False,
        label="Use Sampling",
        description="Use sampling (True) vs greedy decoding (False)",
    )
    
    # Output filtering
    inputs.list(
        "output_keys",
        types.String(),
        default=None,
        required=False,
        label="Output Keys",
        description="Specific JSON keys to parse (leave empty to parse all)",
    )
    
    return types.Property(inputs)

