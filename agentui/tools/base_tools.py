import os
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np

from ..core.tool import (
    Tool, ToolOutput, Port, PortType,
    InputTool
)


class MediaInputTool(InputTool):
    """Input tool for loading media (images, videos, etc.)"""

    @property
    def tool_type(self) -> str:
        return "MediaInput"

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Loaded image")}

    def process(self) -> bool:
        try:
            image_path = self.parameters.get('path')
            image_data = self.parameters.get('data')  # base64 encoded image

            # Better error messages for debugging
            if not image_path and not image_data:
                print(f"MediaInput error: No image path or data provided. Please upload an image or set a file path.")
                return False

            if image_path:
                if not os.path.exists(image_path):
                    print(f"MediaInput error: File not found: {image_path}")
                    return False
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"MediaInput error: Failed to open image file '{image_path}': {e}")
                    return False
            elif image_data:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                except Exception as e:
                    print(f"MediaInput error: Failed to decode base64 image data: {e}")
                    return False

            self.outputs["image"] = ToolOutput(image, PortType.IMAGE)
            return True
        except Exception as e:
            print(f"MediaInput error: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


class ResizeTool(Tool):
    """Resize image tool"""

    @property
    def tool_type(self) -> str:
        return "Resize"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Output image")}

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            image = self.inputs["image"].data
            width = self.parameters.get('width', 800)
            height = self.parameters.get('height', 600)

            resized_image = image.resize((width, height), Image.LANCZOS)
            self.outputs["image"] = ToolOutput(resized_image, PortType.IMAGE)
            return True
        except Exception as e:
            print(f"Resize error: {e}")
            return False


class BlurTool(Tool):
    """Apply blur filter to image"""

    @property
    def tool_type(self) -> str:
        return "Blur"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Output image")}

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            image = self.inputs["image"].data
            radius = self.parameters.get('radius', 2.0)

            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            self.outputs["image"] = ToolOutput(blurred_image, PortType.IMAGE)
            return True
        except Exception as e:
            print(f"Blur error: {e}")
            return False


class ConvertFormatTool(Tool):
    """Convert image format"""

    @property
    def tool_type(self) -> str:
        return "ConvertFormat"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Output image")}

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            image = self.inputs["image"].data
            format_type = self.parameters.get('format', 'RGB')

            if format_type == 'grayscale':
                converted_image = image.convert('L')
            elif format_type == 'RGB':
                converted_image = image.convert('RGB')
            elif format_type == 'RGBA':
                converted_image = image.convert('RGBA')
            else:
                converted_image = image

            self.outputs["image"] = ToolOutput(converted_image, PortType.IMAGE)
            return True
        except Exception as e:
            print(f"ConvertFormat error: {e}")
            return False


class SaveImageTool(Tool):
    """Save image to file"""

    @property
    def tool_type(self) -> str:
        return "SaveImage"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Image to save")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"path": Port("path", PortType.STRING, "Saved file path")}

    def process(self) -> bool:
        try:
            if "image" not in self.inputs:
                return False

            image = self.inputs["image"].data
            path = self.parameters.get('path', 'output.jpg')

            # Ensure directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

            image.save(path)
            self.outputs["path"] = ToolOutput(path, PortType.STRING)
            return True
        except Exception as e:
            print(f"SaveImage error: {e}")
            return False


