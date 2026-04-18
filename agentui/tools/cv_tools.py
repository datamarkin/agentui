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