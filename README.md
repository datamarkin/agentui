# AgentUI

Visual workflow builder for computer vision and AI. Create image processing pipelines by connecting tools in a drag-and-drop interface, then export and run them programmatically.

**Part of the [Datamarkin](https://datamarkin.com) ecosystem** - Built on [PixelFlow](https://pixelflow.datamarkin.com) and [Mozo](https://mozo.datamarkin.com) for production-ready computer vision.

## What It Is

AgentUI is a web-first tool builder that lets you:
- **Build visually**: Drag and drop tools on a canvas to create workflows
- **Connect tools**: Wire outputs to inputs with type-safe connections
- **Execute**: Run workflows in the browser and see results instantly
- **Export**: Save workflows as JSON for version control and programmatic execution
- **Integrate**: Use as a Python library in your own applications

Think of it as a visual programming environment for computer vision tasks.

## Quick Start

```bash
# Install
pip install agentui

# Start the server
agentui start

# Open http://localhost:8000 in your browser
```

That's it. The UI is already bundled - no separate build step needed.

## What You Can Build

### ML-Powered Tools
- **Object Detection**: Detect objects using YOLOv8 or Detectron2 (80 COCO classes)
- **Instance Segmentation**: Get pixel-level masks for detected objects
- **Depth Estimation**: Generate depth maps from single images (Depth Anything)

### Image Processing
- **Transforms**: Rotate, flip, crop images with automatic detection coordinate updates
- **Enhancement**: CLAHE enhancement, auto-contrast, gamma correction, image normalization
- **Analysis**: Color analysis, quality metrics, dominant color extraction
- **Blending**: Combine multiple images with alpha blending

### Annotation & Privacy
- **Draw Detections**: Bounding boxes, labels, masks, polygons
- **Privacy Protection**: Blur or pixelate regions automatically
- **Object Tracking**: Track objects across video frames
- **Zone Analysis**: Monitor object presence in defined areas

### Input/Output
- **Load**: Images from files or base64 data
- **Save**: Export processed images to disk
- **Web Display**: Convert images to base64 for browser display

## Usage

### Web Interface

1. **Add tools**: Drag tools from the left palette onto the canvas
2. **Connect**: Click and drag from output ports to input ports
3. **Configure**: Select a tool to edit its parameters in the right panel
4. **Execute**: Click "Run Workflow" to process
5. **View Results**: See outputs in the results panel
6. **Export**: Save your workflow as JSON

### Programmatic Usage

```python
from agentui import Workflow

# Load a workflow created in the UI
workflow = Workflow.load('my_workflow.json')

# Run with an image
result = workflow.run(image='test.jpg')

# Access outputs
detections = result['detections']  # PixelFlow Detections object
print(f"Found {len(detections)} objects")

# Batch processing (automatic)
result = workflow.run(image=['img1.jpg', 'img2.jpg', 'img3.jpg'])
for i, dets in enumerate(result['detections']):
    print(f"Image {i}: {len(dets)} objects")
```

### Workflow Design Philosophy

**AgentUI is designed for visual workflow creation:**
- Create workflows using the drag-and-drop UI
- Export as JSON for version control
- Load and execute programmatically with the Python API

**Why not build workflows in code?** The visual interface is the fastest way to prototype CV pipelines. The Python API focuses on *execution* (loading and running workflows), not construction. This separation keeps the codebase simple and the workflow format UI-native.

## The Datamarkin Ecosystem

AgentUI integrates two powerful libraries:

- **[PixelFlow](https://pixelflow.datamarkin.com)**: Computer vision primitives (annotation, tracking, spatial analysis)
- **[Mozo](https://mozo.datamarkin.com)**: Universal model serving (object detection, segmentation, depth estimation)

These libraries are maintained by the same team and designed to work together seamlessly.

## Development

### UI Development

Only needed if you're modifying the UI:

```bash
cd ui
npm install
npm run dev  # Development server with hot reload at http://localhost:5173

# When done
npm run build  # Builds to ../agentui/static/
```

### Adding Custom Tools

Tools are Python classes that inherit from `Tool`:

```python
from agentui.core.tool import Tool, ToolOutput, Port, PortType

class MyCustomTool(Tool):
    @property
    def tool_type(self) -> str:
        return "MyCustomTool"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Output image")}

    def process(self) -> bool:
        image = self.inputs["image"].data
        # Do something with the image
        self.outputs["image"] = ToolOutput(processed_image, PortType.IMAGE)
        return True
```

Tools are automatically discovered by the registry. See `CLAUDE.md` for detailed development guidance.

## Installation for Development

```bash
git clone <repository-url>
cd agentui

# Python setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Start server
agentui start

# Optional: UI development (only if modifying Svelte code)
cd ui
npm install
npm run build
```

## Roadmap

Future additions will focus on:
- Additional ML models (OCR, classification, keypoint detection)
- Vision-language models (GPT-4V, Claude, Gemini, Qwen-VL)
- Cloud storage integrations (S3, GCS, Azure)
- Advanced tracking and analytics
- Real-time streaming workflows


## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Complete developer guide and architecture documentation

## Requirements

- Python 3.9+
- Optional: Node.js 18+ (only for UI development)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please check `CLAUDE.md` for development guidelines and architecture overview.

---

**Built by [Datamarkin](https://datamarkin.com)** - Making computer vision accessible through visual workflows.
