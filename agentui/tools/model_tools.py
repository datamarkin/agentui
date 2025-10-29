"""
ML Model nodes for object detection, segmentation, and other computer vision tasks.

This module integrates the Mozo library for model serving with PixelFlow for
unified detection output format.

See mozo.md and pixelflow.md for detailed documentation.
"""

from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2

from ..core.tool import Tool, ToolOutput, Port, PortType

# Try to import mozo library
try:
    from mozo.manager import ModelManager
    MOZO_AVAILABLE = True
except ImportError:
    MOZO_AVAILABLE = False
    print("Warning: mozo library not installed. Install with: pip install mozo")

# Try to import pixelflow library
try:
    import pixelflow as pf
    PIXELFLOW_AVAILABLE = True
except ImportError:
    PIXELFLOW_AVAILABLE = False
    print("Warning: pixelflow library not installed. Install with: pip install pixelflow")


class MozoModelToolBase(Tool):
    """
    Base class for nodes using the Mozo model serving library.

    Provides:
    - Shared ModelManager instance (singleton pattern)
    - Image format conversion helpers (PIL ↔ numpy)
    - Model lifecycle management
    """

    # Shared ModelManager instance across all model nodes
    _model_manager: Optional[ModelManager] = None

    def __init__(self, tool_id: Optional[str] = None, **kwargs):
        super().__init__(tool_id, **kwargs)

        if not MOZO_AVAILABLE:
            raise ImportError(
                "Mozo model nodes require the mozo library. "
                "Install with: pip install mozo"
            )

        if not PIXELFLOW_AVAILABLE:
            raise ImportError(
                "Mozo model nodes require the pixelflow library. "
                "Install with: pip install pixelflow"
            )

        # Initialize shared model manager if not already done
        if MozoModelToolBase._model_manager is None:
            MozoModelToolBase._model_manager = ModelManager()

    @property
    def model_manager(self) -> ModelManager:
        """Get the shared model manager instance"""
        return MozoModelToolBase._model_manager

    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV numpy array (BGR format)"""
        # Convert to RGB first (in case it's RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array (RGB)
        np_image = np.array(image)

        # Convert RGB to BGR for OpenCV
        cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        return cv2_image

    @staticmethod
    def cv2_to_pil(image: np.ndarray) -> Image.Image:
        """Convert OpenCV numpy array (BGR) to PIL Image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        return pil_image

    @classmethod
    def cleanup_models(cls, inactive_seconds: int = 600) -> int:
        """
        Clean up inactive models to free memory.

        Args:
            inactive_seconds: Unload models not used in this many seconds

        Returns:
            Number of models unloaded
        """
        if cls._model_manager is not None:
            return cls._model_manager.cleanup_inactive_models(inactive_seconds)
        return 0

    @classmethod
    def unload_all_models(cls) -> int:
        """Unload all loaded models"""
        if cls._model_manager is not None:
            return cls._model_manager.unload_all_models()
        return 0


class ObjectDetection(MozoModelToolBase):
    """
    Universal object detection supporting multiple frameworks via Mozo.

    Supports:
    - Detectron2: Faster R-CNN, RetinaNet, and other detection architectures
    - YOLOv8: Nano, Small, Medium, Large, XLarge variants

    Output format: PixelFlow Detections (unified format for all frameworks)

    Note: Model variants are defined in registry.py NODE_METADATA (single source of truth)
    """

    @property
    def tool_type(self) -> str:
        return "ObjectDetection"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Detected objects (PixelFlow Detections)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get parameters
            framework = self.parameters.get('framework', 'detectron2')
            model_variant = self.parameters.get('model_variant', 'faster_rcnn_R_50_FPN_3x')
            confidence_threshold = self.parameters.get('confidence_threshold', 0.5)
            device = self.parameters.get('device', 'cpu')

            # Convert PIL Image to OpenCV format (BGR numpy array)
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo (lazy loads if needed)
            print(f"{self.tool_type}: Loading model '{framework}/{model_variant}' on {device}...")
            model = self.model_manager.get_model(framework, model_variant)

            # Run prediction - returns PixelFlow Detections
            print(f"{self.tool_type}: Running inference...")
            detections = model.predict(cv2_image)

            # Filter by confidence threshold
            if confidence_threshold > 0.0:
                detections = detections.filter_by_confidence(confidence_threshold)

            print(f"{self.tool_type}: Found {len(detections)} detections")

            # Set outputs
            # Note: We output the PixelFlow Detections object directly
            # It can be converted to dict/json later if needed
            self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class InstanceSegmentation(MozoModelToolBase):
    """
    Universal instance segmentation supporting multiple frameworks via Mozo.

    Supports:
    - Detectron2: Mask R-CNN variants (R50, R101, X101)
    - YOLOv8: Segmentation variants (yolov8n-seg, yolov8s-seg, etc.)

    Output includes:
    - Bounding boxes
    - Segmentation masks (binary masks or polygons)
    - Class labels and confidence scores

    Output format: PixelFlow Detections (unified format for all frameworks)

    Note: Model variants are defined in registry.py NODE_METADATA (single source of truth)
    """

    @property
    def tool_type(self) -> str:
        return "InstanceSegmentation"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Detected objects with masks (PixelFlow Detections)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get parameters
            framework = self.parameters.get('framework', 'detectron2')
            model_variant = self.parameters.get('model_variant', 'mask_rcnn_R_50_FPN_3x')
            confidence_threshold = self.parameters.get('confidence_threshold', 0.5)
            device = self.parameters.get('device', 'cpu')

            # Convert PIL Image to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo
            print(f"{self.tool_type}: Loading model '{framework}/{model_variant}' on {device}...")
            model = self.model_manager.get_model(framework, model_variant)

            # Run prediction
            print(f"{self.tool_type}: Running inference...")
            detections = model.predict(cv2_image)

            # Filter by confidence
            if confidence_threshold > 0.0:
                detections = detections.filter_by_confidence(confidence_threshold)

            # Count detections with masks
            mask_count = sum(1 for det in detections if det.masks)
            print(f"{self.tool_type}: Found {len(detections)} detections ({mask_count} with masks)")

            # Set outputs
            self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class DepthEstimation(MozoModelToolBase):
    """
    Monocular depth estimation using Depth Anything models via Mozo.

    Outputs:
    - Depth map as grayscale image (PIL Image)
    - Normalized depth values (0=near, 255=far)

    Available models:
    - small: Fast, lower memory (~350MB)
    - base: Balanced speed/accuracy (~1.3GB)
    - large: Highest accuracy (~1.3GB)
    """

    @property
    def tool_type(self) -> str:
        return "DepthEstimation"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "depth_map": Port("depth_map", PortType.IMAGE, "Depth map (grayscale PIL Image)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get parameters
            model_variant = self.parameters.get('model_variant', 'small')
            device = self.parameters.get('device', 'cpu')

            # Convert PIL Image to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo
            print(f"{self.tool_type}: Loading model 'depth_anything/{model_variant}' on {device}...")
            model = self.model_manager.get_model('depth_anything', model_variant)

            # Run prediction - returns PIL Image (depth map)
            print(f"{self.tool_type}: Running inference...")
            depth_map = model.predict(cv2_image)

            print(f"{self.tool_type}: Depth map generated (size: {depth_map.size})")

            # Set output
            self.outputs["depth_map"] = ToolOutput(depth_map, PortType.IMAGE)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class DatamarkinDetection(MozoModelToolBase):
    """
    Datamarkin Cloud Detection - Use your custom trained models via API.

    Supports:
    - Custom object detection models
    - Keypoint detection models
    - Instance segmentation models

    No local model loading - all inference happens via Datamarkin cloud service.
    Requires a training_id and optional bearer_token for authentication.

    Output format: PixelFlow Detections (unified format for all frameworks)
    """

    @property
    def tool_type(self) -> str:
        return "DatamarkinDetection"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Detected objects (PixelFlow Detections)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get essential parameters only
            training_id = self.parameters.get('training_id', '')
            bearer_token = self.parameters.get('bearer_token', None)

            # Validate training_id
            if not training_id:
                print(f"{self.tool_type}: No training_id provided. Please enter your Datamarkin model ID.")
                return False

            # Convert PIL to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo (Mozo handles base_url and timeout)
            print(f"{self.tool_type}: Connecting to Datamarkin API...")
            print(f"  Training ID: {training_id}")

            model = self.model_manager.get_model(
                'datamarkin',
                training_id,
                bearer_token=bearer_token
            )

            # Run cloud inference - return all detections (no filtering)
            print(f"{self.tool_type}: Running cloud inference...")
            detections = model.predict(cv2_image)

            print(f"{self.tool_type}: Found {len(detections)} detections")

            # Set outputs (all detections, no filtering)
            self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class OCRDetection(MozoModelToolBase):
    """
    Optical Character Recognition - Extract text from images.

    Supports multiple OCR engines:
    - PaddleOCR: PP-OCRv5 with 80+ languages (mobile/server variants)
    - EasyOCR: User-friendly OCR with good general-purpose accuracy

    Output format: PixelFlow Detections with text bounding boxes
    Each detection contains:
    - bbox: Text region coordinates
    - text: Transcribed text content
    - confidence: Recognition confidence score
    """

    @property
    def tool_type(self) -> str:
        return "OCRDetection"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Text detections (PixelFlow Detections)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get parameters
            framework = self.parameters.get('framework', 'paddleocr')
            variant = self.parameters.get('variant', 'mobile')
            language = self.parameters.get('language', 'en')

            # Convert PIL to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo
            print(f"{self.tool_type}: Loading {framework} model (variant: {variant})...")
            model = self.model_manager.get_model(framework, variant)

            # Run OCR with language parameter
            print(f"{self.tool_type}: Running OCR (language: {language})...")
            detections = model.predict(cv2_image, language=language)

            print(f"{self.tool_type}: Found {len(detections)} text regions")

            # Set outputs
            self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class Florence2(MozoModelToolBase):
    """
    Florence-2 Multi-Task Vision Model - One model for detection, captioning, OCR, and segmentation.

    Available tasks:
    - detection: Object detection with bounding boxes
    - detection_with_caption: Detection with descriptive captions
    - captioning: Basic image captions
    - detailed_captioning: Detailed image descriptions
    - more_detailed_captioning: Comprehensive descriptions
    - ocr: Extract text from images
    - ocr_with_region: Extract text with bounding boxes
    - segmentation: Instance segmentation (requires prompt)

    Output depends on task:
    - Detection/OCR tasks → DETECTIONS (PixelFlow Detections)
    - Captioning tasks → JSON (OpenAI-compatible format with text)
    """

    @property
    def tool_type(self) -> str:
        return "Florence2"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        # NOTE: Both ports are always defined (static ports required for UI initialization)
        # Only one will be populated during execution based on task parameter:
        # - Captioning tasks → "result" populated
        # - Detection/OCR tasks → "detections" populated
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Detected objects/text (PixelFlow Detections)"),
            "result": Port("result", PortType.JSON, "Caption/text result (OpenAI format)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get parameters
            task = self.parameters.get('task', 'detection')
            prompt = self.parameters.get('prompt', '')  # For segmentation task

            # Convert PIL to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo
            print(f"{self.tool_type}: Loading Florence-2 for task: {task}...")
            model = self.model_manager.get_model('florence2', task)

            # Run inference
            print(f"{self.tool_type}: Running {task}...")

            # Segmentation task requires prompt
            if task == 'segmentation' and prompt:
                result = model.predict(cv2_image, prompt=prompt)
            else:
                result = model.predict(cv2_image)

            # Handle different output types
            if 'caption' in task:
                # Captioning tasks return OpenAI-compatible dict
                print(f"{self.tool_type}: Generated caption")
                self.outputs["result"] = ToolOutput(result, PortType.JSON)
            else:
                # Detection/OCR tasks return PixelFlow Detections
                print(f"{self.tool_type}: Found {len(result)} detections")
                self.outputs["detections"] = ToolOutput(result, PortType.DETECTIONS)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class VisualQuestionAnswering(MozoModelToolBase):
    """
    Visual Question Answering - Answer questions about images using vision-language models.

    Supports multiple VQA frameworks:
    - blip_vqa: Salesforce BLIP for visual question answering (base, capfilt-large)
    - qwen2.5_vl: Qwen2.5-VL for advanced vision-language understanding (7b-instruct)
    - qwen3_vl: Qwen3-VL with chain-of-thought reasoning (2b-thinking)

    Input: Image + Question text
    Output: Answer in OpenAI-compatible JSON format
    """

    @property
    def tool_type(self) -> str:
        return "VisualQuestionAnswering"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image (PIL Image)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "answer": Port("answer", PortType.JSON, "Answer (OpenAI-compatible format)")
        }

    def process(self) -> bool:
        try:
            # Get input image
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            pil_image = self.inputs["image"].data

            # Get parameters
            framework = self.parameters.get('framework', 'blip_vqa')
            variant = self.parameters.get('variant', 'base')
            question = self.parameters.get('question', 'What is in this image?')

            # Validate question
            if not question or not question.strip():
                print(f"{self.tool_type}: No question provided")
                return False

            # Convert PIL to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)

            # Get model from Mozo
            print(f"{self.tool_type}: Loading {framework} model (variant: {variant})...")
            model = self.model_manager.get_model(framework, variant)

            # Run VQA
            print(f"{self.tool_type}: Asking: '{question}'...")
            answer = model.predict(cv2_image, question=question)

            print(f"{self.tool_type}: Answer received")

            # Set outputs (OpenAI-compatible format)
            self.outputs["answer"] = ToolOutput(answer, PortType.JSON)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


class StabilityInpainting(MozoModelToolBase):
    """
    Stable Diffusion 2 Inpainting - Generate and modify image content using text prompts.

    Inpainting allows you to:
    - Remove objects from images
    - Replace regions with generated content
    - Fill in missing areas
    - Modify specific parts guided by text

    Requires:
    - Original image
    - Mask image (white = areas to inpaint, black = keep original)
    - Text prompt describing what to generate

    Output: Inpainted image (PIL Image)
    """

    @property
    def tool_type(self) -> str:
        return "StabilityInpainting"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Original image (PIL Image)"),
            "mask": Port("mask", PortType.IMAGE, "Mask image (white = inpaint, black = keep)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "inpainted_image": Port("inpainted_image", PortType.IMAGE, "Inpainted result (PIL Image)")
        }

    def process(self) -> bool:
        try:
            # Get input images
            if "image" not in self.inputs:
                print(f"{self.tool_type}: No input image")
                return False

            if "mask" not in self.inputs:
                print(f"{self.tool_type}: No mask image provided")
                return False

            pil_image = self.inputs["image"].data
            pil_mask = self.inputs["mask"].data

            # Get parameters
            prompt = self.parameters.get('prompt', 'high quality, detailed')
            negative_prompt = self.parameters.get('negative_prompt', 'lowres, bad quality')
            num_inference_steps = self.parameters.get('num_inference_steps', 50)
            guidance_scale = self.parameters.get('guidance_scale', 7.5)

            # Validate prompt
            if not prompt or not prompt.strip():
                print(f"{self.tool_type}: No prompt provided")
                return False

            # Convert PIL to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)
            cv2_mask = self.pil_to_cv2(pil_mask)

            # Get model from Mozo
            print(f"{self.tool_type}: Loading Stable Diffusion 2 Inpainting...")
            model = self.model_manager.get_model('stability_inpainting', 'default')

            # Run inpainting
            print(f"{self.tool_type}: Inpainting with prompt: '{prompt}'...")
            result_image = model.predict(
                cv2_image,
                mask=cv2_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

            print(f"{self.tool_type}: Inpainting complete")

            # Convert result back to PIL if needed
            if isinstance(result_image, np.ndarray):
                result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

            # Set outputs
            self.outputs["inpainted_image"] = ToolOutput(result_image, PortType.IMAGE)

            return True

        except Exception as e:
            print(f"{self.tool_type} error: {e}")
            import traceback
            traceback.print_exc()
            return False


# Export available model nodes
MODEL_TOOLS = [
    ObjectDetection,
    InstanceSegmentation,
    DepthEstimation,
    DatamarkinDetection,
    OCRDetection,
    Florence2,
    VisualQuestionAnswering,
    StabilityInpainting
] if MOZO_AVAILABLE and PIXELFLOW_AVAILABLE else []
