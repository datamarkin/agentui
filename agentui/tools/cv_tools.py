"""
Computer Vision nodes for advanced image analysis and processing
"""

import os
import json
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ..core.tool import (
    Tool, ToolOutput, Port, PortType
)


# Analysis nodes
class DominantColorTool(Tool):
    """Extract dominant color from image"""

    @property
    def tool_type(self) -> str:
        return "DominantColor"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "color": Port("color", PortType.STRING, "Dominant color as hex"),
            "rgb": Port("rgb", PortType.JSON, "RGB values as object")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            image = self.inputs["image"].data

            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get dominant color using color quantization
            # Resize for faster processing
            small_image = image.resize((150, 150))
            result = small_image.quantize(colors=1, method=2)
            palette = result.getpalette()

            # Get the dominant color (first color in 1-color palette)
            r, g, b = palette[0], palette[1], palette[2]

            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            rgb_obj = {"r": r, "g": g, "b": b}

            self.outputs["color"] = ToolOutput(hex_color, PortType.STRING)
            self.outputs["rgb"] = ToolOutput(rgb_obj, PortType.JSON)
            return True
        except Exception as e:
            print(f"DominantColor error: {e}")
            return False


class QualityAnalysisTool(Tool):
    """Analyze image quality metrics"""

    @property
    def tool_type(self) -> str:
        return "QualityAnalysis"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "quality_score": Port("quality_score", PortType.NUMBER, "Overall quality score 0-1"),
            "metrics": Port("metrics", PortType.JSON, "Detailed quality metrics")
        }

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            image = self.inputs["image"].data

            # Simple quality metrics (placeholder implementation)
            width, height = image.size
            pixel_count = width * height

            # Convert to grayscale for analysis
            gray = image.convert('L')
            pixels = np.array(gray)

            # Calculate metrics
            sharpness = np.var(pixels)  # Variance as sharpness measure
            brightness = np.mean(pixels) / 255.0
            contrast = np.std(pixels) / 255.0

            # Normalize sharpness (simple heuristic)
            sharpness_score = min(sharpness / 1000.0, 1.0)

            # Calculate overall quality score
            quality_score = (sharpness_score + contrast) / 2.0

            metrics = {
                "sharpness": float(sharpness_score),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "resolution": {"width": width, "height": height},
                "pixel_count": pixel_count
            }

            self.outputs["quality_score"] = ToolOutput(quality_score, PortType.NUMBER)
            self.outputs["metrics"] = ToolOutput(metrics, PortType.JSON)
            return True
        except Exception as e:
            print(f"QualityAnalysis error: {e}")
            return False


# Combiner nodes
class VisualizeDetectionsTool(Tool):
    """Draw detection boxes on image (PixelFlow Detections only)"""

    @property
    def tool_type(self) -> str:
        return "VisualizeDetections"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Input image"),
            "detections": Port("detections", PortType.DETECTIONS, "Detection results (PixelFlow Detections)")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Annotated image")}

    def process(self) -> bool:
        try:
            if "image" not in self.inputs or "detections" not in self.inputs:
                return False

            image = self.inputs["image"].data.copy()
            detections = self.inputs["detections"].data

            draw = ImageDraw.Draw(image)

            # Draw each detection
            for detection in detections:
                if not hasattr(detection, 'bbox'):
                    continue

                # PixelFlow format: detection.bbox is [x1, y1, x2, y2]
                x1, y1, x2, y2 = detection.bbox
                class_name = detection.class_name if hasattr(detection, 'class_name') else "unknown"
                confidence = detection.confidence if hasattr(detection, 'confidence') else 0.0

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                draw.text((x1, y1 - 20), label, fill="red")

            self.outputs["image"] = ToolOutput(image, PortType.IMAGE)
            return True
        except Exception as e:
            print(f"VisualizeDetections error: {e}")
            import traceback
            traceback.print_exc()
            return False


class BlendImagesTool(Tool):
    """Blend two images together"""

    @property
    def tool_type(self) -> str:
        return "BlendImages"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "background": Port("background", PortType.IMAGE, "Background image"),
            "foreground": Port("foreground", PortType.IMAGE, "Foreground image")
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Blended image")}

    def process(self) -> bool:
        try:
            if "background" not in self.inputs or "foreground" not in self.inputs:
                return False

            bg = self.inputs["background"].data
            fg = self.inputs["foreground"].data
            alpha = self.parameters.get('alpha', 0.5)

            # Resize foreground to match background
            fg_resized = fg.resize(bg.size)

            # Blend images
            blended = Image.blend(bg, fg_resized, alpha)

            self.outputs["image"] = ToolOutput(blended, PortType.IMAGE)
            return True
        except Exception as e:
            print(f"BlendImages error: {e}")
            return False