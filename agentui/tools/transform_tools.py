"""
PixelFlow Transform Tools for AgentUI

This module provides comprehensive image and detection transformation capabilities
using the pixelflow.transform module. Includes both image-only transforms and
detection-aware transforms that update coordinates automatically.

Tools are organized into:
- Image-only geometric transforms (rotate, flip, crop)
- Image-only enhancement transforms (CLAHE, auto-contrast, gamma, normalize)
- Detection-aware transforms (rotate/flip/crop with coordinate updates)
- Detection utility transforms (align, update bbox, add padding, crop around)
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.tool import Tool, ToolOutput, Port, PortType

# Try to import pixelflow library
try:
    import pixelflow as pf
    PIXELFLOW_AVAILABLE = True
except ImportError:
    PIXELFLOW_AVAILABLE = False
    print("Warning: pixelflow library not installed. Install with: pip install pixelflow")


class PixelFlowTransformToolBase(Tool):
    """Base class for transform tools using the pixelflow library"""

    def __init__(self, tool_id: Optional[str] = None, **kwargs):
        super().__init__(tool_id, **kwargs)
        if not PIXELFLOW_AVAILABLE:
            raise ImportError("PixelFlow transform tools require the pixelflow library. Install with: pip install pixelflow")

    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV numpy array (RGB -> BGR)"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        cv2_image = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV numpy array to PIL Image (BGR -> RGB)"""
        # Convert BGR to RGB
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
            rgb_image = cv2_image[:, :, ::-1]
        else:
            rgb_image = cv2_image
        return Image.fromarray(rgb_image)


# ============================================================================
# Image-Only Geometric Transforms
# ============================================================================

class RotateImage(PixelFlowTransformToolBase):
    """Rotate image by specified angle"""

    @property
    def tool_type(self) -> str:
        return "RotateImage"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def description(self) -> str:
        return "Rotate image by angle (counter-clockwise)"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "rotated_image": Port("rotated_image", PortType.IMAGE, "Rotated image")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get parameters
            angle = self.parameters.get("angle", 0)
            center_x = self.parameters.get("center_x", None)
            center_y = self.parameters.get("center_y", None)
            fillcolor_hex = self.parameters.get("fillcolor", None)

            # Parse center
            h, w = cv2_image.shape[:2]
            if center_x is not None and center_y is not None:
                center = (int(center_x * w), int(center_y * h))
            else:
                center = None

            # Parse fillcolor (hex to BGR tuple)
            fillcolor = None
            if fillcolor_hex:
                fillcolor_hex = fillcolor_hex.lstrip('#')
                r, g, b = tuple(int(fillcolor_hex[i:i+2], 16) for i in (0, 2, 4))
                fillcolor = (b, g, r)  # BGR for OpenCV

            # Rotate using pixelflow
            rotated = pf.transform.rotate(
                cv2_image,
                angle=angle,
                center=center,
                fillcolor=fillcolor
            )

            # Convert back to PIL
            result = self.cv2_to_pil(rotated)
            self.outputs["rotated_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"RotateImage error: {e}")
            import traceback
            traceback.print_exc()
            return False


class FlipImage(PixelFlowTransformToolBase):
    """Flip image horizontally or vertically"""

    @property
    def tool_type(self) -> str:
        return "FlipImage"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def description(self) -> str:
        return "Flip image horizontally or vertically"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "flipped_image": Port("flipped_image", PortType.IMAGE, "Flipped image")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get direction parameter
            direction = self.parameters.get("direction", "horizontal")

            # Flip using pixelflow
            if direction == "horizontal":
                flipped = pf.transform.flip_horizontal(cv2_image)
            else:  # vertical
                flipped = pf.transform.flip_vertical(cv2_image)

            # Convert back to PIL
            result = self.cv2_to_pil(flipped)
            self.outputs["flipped_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"FlipImage error: {e}")
            import traceback
            traceback.print_exc()
            return False


class CropImage(PixelFlowTransformToolBase):
    """Crop image to bounding box"""

    @property
    def tool_type(self) -> str:
        return "CropImage"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def description(self) -> str:
        return "Crop image to specified bounding box"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "cropped_image": Port("cropped_image", PortType.IMAGE, "Cropped image")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get crop parameters
            x1 = self.parameters.get("x1", 0)
            y1 = self.parameters.get("y1", 0)
            x2 = self.parameters.get("x2", cv2_image.shape[1])
            y2 = self.parameters.get("y2", cv2_image.shape[0])
            bbox = [x1, y1, x2, y2]

            # Crop using pixelflow
            cropped = pf.transform.crop(cv2_image, bbox)

            # Convert back to PIL
            result = self.cv2_to_pil(cropped)
            self.outputs["cropped_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"CropImage error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# Image-Only Enhancement Transforms
# ============================================================================

class EnhanceCLAHE(PixelFlowTransformToolBase):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement"""

    @property
    def tool_type(self) -> str:
        return "EnhanceCLAHE"

    @property
    def category(self) -> str:
        return "Enhance"

    @property
    def description(self) -> str:
        return "Apply CLAHE contrast enhancement"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "enhanced_image": Port("enhanced_image", PortType.IMAGE, "Enhanced image")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get parameters
            clip_limit = self.parameters.get("clip_limit", 2.0)
            tile_size = self.parameters.get("tile_size", 8)

            # Apply CLAHE using pixelflow
            enhanced = pf.transform.clahe(
                cv2_image,
                clip_limit=clip_limit,
                tile_size=(tile_size, tile_size)
            )

            # Convert back to PIL
            result = self.cv2_to_pil(enhanced)
            self.outputs["enhanced_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"EnhanceCLAHE error: {e}")
            import traceback
            traceback.print_exc()
            return False


class AutoContrast(PixelFlowTransformToolBase):
    """Apply automatic contrast adjustment"""

    @property
    def tool_type(self) -> str:
        return "AutoContrast"

    @property
    def category(self) -> str:
        return "Enhance"

    @property
    def description(self) -> str:
        return "Apply automatic contrast adjustment by stretching histogram"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "enhanced_image": Port("enhanced_image", PortType.IMAGE, "Enhanced image")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get cutoff parameter
            cutoff = self.parameters.get("cutoff", 1.0)

            # Apply auto contrast using pixelflow
            enhanced = pf.transform.auto_contrast(cv2_image, cutoff=cutoff)

            # Convert back to PIL
            result = self.cv2_to_pil(enhanced)
            self.outputs["enhanced_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"AutoContrast error: {e}")
            import traceback
            traceback.print_exc()
            return False


class GammaCorrection(PixelFlowTransformToolBase):
    """Apply gamma correction for brightness adjustment"""

    @property
    def tool_type(self) -> str:
        return "GammaCorrection"

    @property
    def category(self) -> str:
        return "Enhance"

    @property
    def description(self) -> str:
        return "Apply gamma correction (<1 brightens, >1 darkens)"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "adjusted_image": Port("adjusted_image", PortType.IMAGE, "Gamma-corrected image")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get gamma parameter
            gamma = self.parameters.get("gamma", 1.0)

            # Apply gamma correction using pixelflow
            adjusted = pf.transform.gamma_correction(cv2_image, gamma=gamma)

            # Convert back to PIL
            result = self.cv2_to_pil(adjusted)
            self.outputs["adjusted_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"GammaCorrection error: {e}")
            import traceback
            traceback.print_exc()
            return False


class NormalizeImage(PixelFlowTransformToolBase):
    """Normalize image for model input using mean and standard deviation"""

    @property
    def tool_type(self) -> str:
        return "NormalizeImage"

    @property
    def category(self) -> str:
        return "Enhance"

    @property
    def description(self) -> str:
        return "Normalize image for neural network input"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "normalized_image": Port("normalized_image", PortType.IMAGE, "Normalized image (float32)")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get preset or custom parameters
            preset = self.parameters.get("preset", "imagenet")

            if preset == "imagenet":
                # ImageNet normalization (BGR format for OpenCV)
                mean = (0.406, 0.456, 0.485)  # BGR order
                std = (0.225, 0.224, 0.229)
            else:
                # Custom normalization
                mean_r = self.parameters.get("mean_r", 0.5)
                mean_g = self.parameters.get("mean_g", 0.5)
                mean_b = self.parameters.get("mean_b", 0.5)
                std_r = self.parameters.get("std_r", 0.5)
                std_g = self.parameters.get("std_g", 0.5)
                std_b = self.parameters.get("std_b", 0.5)
                mean = (mean_b, mean_g, mean_r)  # BGR order
                std = (std_b, std_g, std_r)

            # Apply normalization using pixelflow
            normalized = pf.transform.normalize(cv2_image, mean=mean, std=std)

            # Convert float32 back to uint8 for display (scale back to 0-255)
            # Note: This is for visualization only, actual model input would use float32
            display_image = ((normalized * np.array(std).reshape(1, 1, 3) +
                             np.array(mean).reshape(1, 1, 3)) * 255).astype(np.uint8)

            # Convert back to PIL
            result = self.cv2_to_pil(display_image)
            self.outputs["normalized_image"] = ToolOutput(result, PortType.IMAGE)
            return True

        except Exception as e:
            print(f"NormalizeImage error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# Detection-Aware Geometric Transforms
# ============================================================================

class RotateWithDetections(PixelFlowTransformToolBase):
    """Rotate image and update detection coordinates automatically"""

    @property
    def tool_type(self) -> str:
        return "RotateWithDetections"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def description(self) -> str:
        return "Rotate image and detections together (modifies detections in-place)"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image"),
            "detections": Port("detections", PortType.DETECTIONS, "Detections to transform")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "rotated_image": Port("rotated_image", PortType.IMAGE, "Rotated image"),
            "rotated_detections": Port("rotated_detections", PortType.DETECTIONS, "Rotated detections")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs or "detections" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            detections = self.inputs["detections"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get parameters
            angle = self.parameters.get("angle", 0)
            center_x = self.parameters.get("center_x", None)
            center_y = self.parameters.get("center_y", None)
            fillcolor_hex = self.parameters.get("fillcolor", None)
            track_metadata = self.parameters.get("track_metadata", True)

            # Parse center
            h, w = cv2_image.shape[:2]
            if center_x is not None and center_y is not None:
                center = (int(center_x * w), int(center_y * h))
            else:
                center = None

            # Parse fillcolor
            fillcolor = None
            if fillcolor_hex:
                fillcolor_hex = fillcolor_hex.lstrip('#')
                r, g, b = tuple(int(fillcolor_hex[i:i+2], 16) for i in (0, 2, 4))
                fillcolor = (b, g, r)  # BGR

            # Rotate using pixelflow (modifies detections in-place)
            rotated_img, rotated_dets = pf.transform.rotate_detections(
                cv2_image,
                detections,
                angle=angle,
                center=center,
                fillcolor=fillcolor,
                track_metadata=track_metadata
            )

            # Convert back to PIL
            result_image = self.cv2_to_pil(rotated_img)
            self.outputs["rotated_image"] = ToolOutput(result_image, PortType.IMAGE)
            self.outputs["rotated_detections"] = ToolOutput(rotated_dets, PortType.DETECTIONS)
            return True

        except Exception as e:
            print(f"RotateWithDetections error: {e}")
            import traceback
            traceback.print_exc()
            return False


class FlipWithDetections(PixelFlowTransformToolBase):
    """Flip image and update detection coordinates automatically"""

    @property
    def tool_type(self) -> str:
        return "FlipWithDetections"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def description(self) -> str:
        return "Flip image and detections together (modifies detections in-place)"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image"),
            "detections": Port("detections", PortType.DETECTIONS, "Detections to transform")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "flipped_image": Port("flipped_image", PortType.IMAGE, "Flipped image"),
            "flipped_detections": Port("flipped_detections", PortType.DETECTIONS, "Flipped detections")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs or "detections" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            detections = self.inputs["detections"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get parameters
            direction = self.parameters.get("direction", "horizontal")
            track_metadata = self.parameters.get("track_metadata", True)

            # Flip using pixelflow
            if direction == "horizontal":
                flipped_img, flipped_dets = pf.transform.flip_horizontal_detections(
                    cv2_image, detections, track_metadata=track_metadata
                )
            else:  # vertical
                flipped_img, flipped_dets = pf.transform.flip_vertical_detections(
                    cv2_image, detections, track_metadata=track_metadata
                )

            # Convert back to PIL
            result_image = self.cv2_to_pil(flipped_img)
            self.outputs["flipped_image"] = ToolOutput(result_image, PortType.IMAGE)
            self.outputs["flipped_detections"] = ToolOutput(flipped_dets, PortType.DETECTIONS)
            return True

        except Exception as e:
            print(f"FlipWithDetections error: {e}")
            import traceback
            traceback.print_exc()
            return False


class CropWithDetections(PixelFlowTransformToolBase):
    """Crop image and filter/translate detection coordinates"""

    @property
    def tool_type(self) -> str:
        return "CropWithDetections"

    @property
    def category(self) -> str:
        return "Transform"

    @property
    def description(self) -> str:
        return "Crop image and filter detections to crop region"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image"),
            "detections": Port("detections", PortType.DETECTIONS, "Detections to filter/transform")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "cropped_image": Port("cropped_image", PortType.IMAGE, "Cropped image"),
            "cropped_detections": Port("cropped_detections", PortType.DETECTIONS, "Filtered detections")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs or "detections" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            detections = self.inputs["detections"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get crop parameters
            x1 = self.parameters.get("x1", 0)
            y1 = self.parameters.get("y1", 0)
            x2 = self.parameters.get("x2", cv2_image.shape[1])
            y2 = self.parameters.get("y2", cv2_image.shape[0])
            bbox = [x1, y1, x2, y2]
            track_metadata = self.parameters.get("track_metadata", True)

            # Crop using pixelflow
            cropped_img, cropped_dets = pf.transform.crop_detections(
                cv2_image,
                detections,
                bbox=bbox,
                track_metadata=track_metadata
            )

            # Convert back to PIL
            result_image = self.cv2_to_pil(cropped_img)
            self.outputs["cropped_image"] = ToolOutput(result_image, PortType.IMAGE)
            self.outputs["cropped_detections"] = ToolOutput(cropped_dets, PortType.DETECTIONS)
            return True

        except Exception as e:
            print(f"CropWithDetections error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# Detection Utility Transforms
# ============================================================================

class CropAroundDetections(PixelFlowTransformToolBase):
    """Extract individual crops for each detection"""

    @property
    def tool_type(self) -> str:
        return "CropAroundDetections"

    @property
    def category(self) -> str:
        return "Detection"

    @property
    def description(self) -> str:
        return "Extract cropped images around each detection bbox"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image"),
            "detections": Port("detections", PortType.DETECTIONS, "Detections to crop around")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "crops": Port("crops", PortType.ARRAY, "List of cropped images")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs or "detections" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            detections = self.inputs["detections"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get padding parameters
            padding_mode = self.parameters.get("padding_mode", "uniform")

            if padding_mode == "uniform":
                padding = self.parameters.get("padding", 0.0)
            else:  # custom
                padding = {
                    'left': self.parameters.get("padding_left", 0.0),
                    'right': self.parameters.get("padding_right", 0.0),
                    'top': self.parameters.get("padding_top", 0.0),
                    'bottom': self.parameters.get("padding_bottom", 0.0)
                }

            # Crop around each detection using pixelflow
            crops_cv2 = pf.transform.crop_around_detections(
                cv2_image,
                detections,
                padding=padding
            )

            # Convert crops to PIL images
            crops_pil = [self.cv2_to_pil(crop) for crop in crops_cv2]

            self.outputs["crops"] = ToolOutput(crops_pil, PortType.ARRAY)
            return True

        except Exception as e:
            print(f"CropAroundDetections error: {e}")
            import traceback
            traceback.print_exc()
            return False


class AlignDetections(PixelFlowTransformToolBase):
    """Rotate image to align two keypoints to target angle"""

    @property
    def tool_type(self) -> str:
        return "AlignDetections"

    @property
    def category(self) -> str:
        return "Detection"

    @property
    def description(self) -> str:
        return "Rotate image so two keypoints form specified angle"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image"),
            "detections": Port("detections", PortType.DETECTIONS, "Detections with keypoints")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "aligned_image": Port("aligned_image", PortType.IMAGE, "Aligned image"),
            "aligned_detections": Port("aligned_detections", PortType.DETECTIONS, "Aligned detections")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs or "detections" not in self.inputs:
                return False

            pil_image = self.inputs["image"].data
            detections = self.inputs["detections"].data
            cv2_image = self.pil_to_cv2(pil_image)

            # Get parameters
            point1_name = self.parameters.get("point1_name", "p0")
            point2_name = self.parameters.get("point2_name", "p9")
            target_angle = self.parameters.get("target_angle", 0.0)
            detection_index = self.parameters.get("detection_index", 0)

            # Align using pixelflow
            aligned_img, aligned_dets = pf.transform.rotate_to_align(
                cv2_image,
                detections,
                point1_name=point1_name,
                point2_name=point2_name,
                target_angle=target_angle,
                detection_index=detection_index
            )

            # Convert back to PIL
            result_image = self.cv2_to_pil(aligned_img)
            self.outputs["aligned_image"] = ToolOutput(result_image, PortType.IMAGE)
            self.outputs["aligned_detections"] = ToolOutput(aligned_dets, PortType.DETECTIONS)
            return True

        except Exception as e:
            print(f"AlignDetections error: {e}")
            import traceback
            traceback.print_exc()
            return False


class UpdateBBoxFromKeypoints(PixelFlowTransformToolBase):
    """Recalculate detection bounding boxes from keypoints"""

    @property
    def tool_type(self) -> str:
        return "UpdateBBoxFromKeypoints"

    @property
    def category(self) -> str:
        return "Detection"

    @property
    def description(self) -> str:
        return "Update detection bboxes based on keypoint positions"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Detections with keypoints")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "updated_detections": Port("updated_detections", PortType.DETECTIONS, "Detections with updated bboxes")
        }

    def process(self) -> bool:
        try:
            if "detections" not in self.inputs:
                return False

            detections = self.inputs["detections"].data

            # Get parameters
            keypoint_mode = self.parameters.get("keypoint_mode", "all")
            track_metadata = self.parameters.get("track_metadata", True)

            # Parse keypoint names
            keypoint_names = None
            if keypoint_mode == "custom":
                keypoint_names_str = self.parameters.get("keypoint_names", "")
                if keypoint_names_str:
                    keypoint_names = [name.strip() for name in keypoint_names_str.split(",")]

            # Update bbox using pixelflow (modifies in-place)
            updated_dets = pf.transform.update_bbox_from_keypoints(
                detections,
                keypoint_names=keypoint_names,
                track_metadata=track_metadata
            )

            self.outputs["updated_detections"] = ToolOutput(updated_dets, PortType.DETECTIONS)
            return True

        except Exception as e:
            print(f"UpdateBBoxFromKeypoints error: {e}")
            import traceback
            traceback.print_exc()
            return False


class AddPadding(PixelFlowTransformToolBase):
    """Add padding to detection bounding boxes"""

    @property
    def tool_type(self) -> str:
        return "AddPadding"

    @property
    def category(self) -> str:
        return "Detection"

    @property
    def description(self) -> str:
        return "Add padding to all detection bounding boxes"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "Detections to pad")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "padded_detections": Port("padded_detections", PortType.DETECTIONS, "Detections with padded bboxes")
        }

    def process(self) -> bool:
        try:
            if "detections" not in self.inputs:
                return False

            detections = self.inputs["detections"].data

            # Get parameters
            padding_mode = self.parameters.get("padding_mode", "uniform")
            reference = self.parameters.get("reference", "shorter")
            track_metadata = self.parameters.get("track_metadata", True)

            if padding_mode == "uniform":
                padding = self.parameters.get("padding", 0.1)
            else:  # custom
                padding = {
                    'left': self.parameters.get("padding_left", 0.1),
                    'right': self.parameters.get("padding_right", 0.1),
                    'top': self.parameters.get("padding_top", 0.1),
                    'bottom': self.parameters.get("padding_bottom", 0.1)
                }

            # Add padding using pixelflow (modifies in-place)
            padded_dets = pf.transform.add_padding(
                detections,
                padding=padding,
                reference=reference,
                track_metadata=track_metadata
            )

            self.outputs["padded_detections"] = ToolOutput(padded_dets, PortType.DETECTIONS)
            return True

        except Exception as e:
            print(f"AddPadding error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# Export List
# ============================================================================

TRANSFORM_TOOLS = [
    # Image-only geometric transforms
    RotateImage,
    FlipImage,
    CropImage,

    # Image-only enhancement transforms
    EnhanceCLAHE,
    AutoContrast,
    GammaCorrection,
    NormalizeImage,

    # Detection-aware geometric transforms
    RotateWithDetections,
    FlipWithDetections,
    CropWithDetections,

    # Detection utility transforms
    CropAroundDetections,
    AlignDetections,
    UpdateBBoxFromKeypoints,
    AddPadding
] if PIXELFLOW_AVAILABLE else []
