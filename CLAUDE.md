# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### CLI Commands
```bash
# Start the server (default port 8000)
agentui start

# Start on custom port
agentui start --port 3000

# Start with hot reload (development)
agentui start --reload

# Show version
agentui version

# Show system info and dependencies
agentui info

# Execute workflow from file
agentui run workflow.json

# Show help
agentui --help
```

### UI Development
```bash
# Build UI (required after Svelte changes)
cd ui && npm run build

# Development with hot reload
cd ui && npm run dev  # Runs on port 5173

# Install UI dependencies
cd ui && npm install
```

### Testing & Validation
```bash
# Kill processes on port 8000
lsof -ti:8000 | xargs kill -9

# Test Python installation
python -c "from agentui.core.workflow import Workflow; print('✅ Python OK')"
```

## Architecture Overview

AgentUI is a **web-first computer vision workflow builder** with these key architectural layers.

### Frontend (Svelte Flow)
- **Svelte Flow Canvas**: Drag-and-drop node-based workflow editor
- **Component Architecture**: Modular UI with shared utilities
- **Build System**: Vite builds to `../agentui/static/` for single-server deployment

### Backend (Python)
- **Tool System**: Plugin-based tools with automatic registry discovery
- **Workflow Engine**: DAG execution with topological sorting
- **FastAPI Server**: Serves UI and provides REST API endpoints
- **PixelFlow Integration**: External library for computer vision processing
  - **We are the maintainers** of PixelFlow - can extend capabilities as needed for agentui
- **Mozo Integration**: Universal model serving library for ML models
  - **We are the maintainers** of Mozo - can add new model families and frameworks as needed

### Key Design Patterns
1. **Terminal Tool Results**: Workflows return results only from tools with no outgoing connections
2. **Auto-batching**: Tools can process single items or lists automatically via `process_with_auto_batching()`
3. **Port Type System**: Strongly-typed data flow with validation (`PortType.IMAGE`, `PortType.STRING`, `PortType.DETECTIONS`, etc.)
4. **Placeholder Input Pattern**: Canvas starts with placeholder tool for better UX
5. **Unified Detection Format**: All detection models output `PortType.DETECTIONS` (PixelFlow Detections objects)
6. **Function-Based Model Tools**: One tool per task type (ObjectDetection, InstanceSegmentation), framework selection via dropdown
   - Simplifies UI: fewer tools, clearer intent
   - Easier experimentation: switch models without rewiring
   - No backward compatibility needed: new library, evolving architecture

## Critical Implementation Details

### Tool System (Python)
- **Base Classes**: `Tool` (general), `InputTool` (input-only)
- **Port Definitions**: Required/optional inputs defined in tool registry metadata
- **Auto-Discovery**: Registry scans for Tool subclasses automatically
- **Connection Validation**: Type checking prevents incompatible connections

### UI Architecture (Svelte)
- **Shared Utilities**: `ui/src/lib/utils.js` contains common functions
- **CSS Strategy**: Bulma framework + `custom.css` for overrides
- **No Inline Styles**: All styling via CSS classes for maintainability
- **Component Structure**:
  - `PlaceholderInputNode.svelte` - Initial canvas state
  - `CustomNode.svelte` - Regular workflow nodes
  - `NodePalette.svelte` - Draggable node library
  - `PropertiesPanel.svelte` - Parameter editing
  - `ResultsPanel.svelte` - Execution results display

### Data Flow
```
UI JSON Workflow → FastAPI → Workflow.execute() → Terminal Tool Results → UI Display
```

### Build Process
- **UI Changes**: Always run `npm run build` before testing
- **Python Changes**: Server restart required
- **Static Files**: UI builds to `agentui/static/` for distribution

## Common Patterns

### Adding New Tool Types
```python
# In agentui/tools/cv_tools.py, pixelflow_tools.py, or model_tools.py
class NewTool(Tool):
    @property
    def tool_type(self) -> str:
        return "NewTool"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"input": Port("input", PortType.IMAGE, "Description")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"output": Port("output", PortType.IMAGE, "Description")}

    def process(self) -> bool:
        # Implementation here
        return True
```

### Adding New Model Tools (Mozo Integration)
```python
# In agentui/tools/model_tools.py
from ..tools.model_tools import MozoModelToolBase

class NewModelTool(MozoModelToolBase):
    @property
    def tool_type(self) -> str:
        return "NewModel"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "detections": Port("detections", PortType.DETECTIONS, "PixelFlow Detections")
        }

    def process(self) -> bool:
        # Get input
        pil_image = self.inputs["image"].data

        # Convert to OpenCV format
        cv2_image = self.pil_to_cv2(pil_image)

        # Get model from Mozo (lazy loads)
        model = self.model_manager.get_model('family', 'variant')

        # Run inference (returns PixelFlow Detections)
        detections = model.predict(cv2_image)

        # Set outputs
        self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)
        return True
```

### Using PixelFlow in Tools
```python
# Example: Integrating pixelflow functions
import pixelflow

class PixelFlowAnnotation(Tool):
    def process(self) -> bool:
        image = self.get_input_data("image")
        detections = self.get_input_data("detections")

        # Use pixelflow for annotation
        result = pixelflow.annotate.box(image, detections,
                                       thickness=self.parameters.get("thickness", 2))

        self.set_output_data("output", result)
        return True
```

### UI Component Patterns
- Import shared utilities: `import { formatParameterLabel } from './utils.js'`
- Use Bulma classes: `<div class="field">`, `<div class="control">`, `<input class="input is-small">`
- No inline styles: Use CSS classes from `custom.css`
- Event handling: Components communicate via Svelte stores and callbacks

### Registry Metadata Structure
```python
# Registry provides UI with:
{
    "ToolType": {
        "category": "Transform",
        "icon": "🔧",
        "description": "Tool description",
        "required_inputs": ["input1"],
        "optional_inputs": ["input2"],
        "ports": {
            "inputs": {"input1": {"type": "image", "description": "..."}},
            "outputs": {"output1": {"type": "image", "description": "..."}}
        }
    }
}
```

## File Structure Context

### Critical Files
- `agentui/core/workflow.py` - Main execution engine with terminal tool logic
- `agentui/core/registry.py` - Tool discovery and metadata
- `agentui/core/tool.py` - Tool base classes and PortType enum
- `agentui/tools/model_tools.py` - ML model tools (Mozo integration)
- `agentui/tools/cv_tools.py` - Image processing tools
- `agentui/tools/pixelflow_tools.py` - Annotation and tracking tools
- `ui/src/App.svelte` - Main UI orchestrator
- `ui/src/lib/utils.js` - Shared UI utilities
- `ui/vite.config.js` - Builds to `../agentui/static/`
- `pixelflow.md` - External pixelflow library integration guide
- `mozo.md` - Model serving library integration guide

### Recent Improvements
- **Model Tool System**: Integrated Mozo library for universal model serving with lazy loading
- **DETECTIONS Port Type**: New dedicated port type for PixelFlow Detections objects
- **Detectron2 Support**: Object detection and instance segmentation via Mozo
- **Depth Estimation**: Monocular depth estimation using Depth Anything models
- **UI Cleanup**: Removed inline styles, consolidated CSS, created shared utilities
- **Placeholder Tools**: Better initial user experience with guided tool placement
- **Terminal Tool Focus**: Results only from workflow endpoints
- **Type Safety**: Enhanced port type validation and visual indicators

### Development Workflow
1. **UI Changes**: Edit Svelte → `npm run build` → restart Python server → test
2. **Python Changes**: Edit Python → restart server → test
3. **Tool Addition**: Add tool class → restart server → appears in palette automatically

The system emphasizes **simplicity over complexity** - tools are straightforward classes, UI uses standard patterns, and the build process is minimal but reliable.

## Model Tools (Mozo Integration)

### Function-Based Architecture

Model tools are organized by **task type** (what they do), not by framework (how they do it).
- Each tool supports multiple frameworks via dropdown selection
- Users select the task first, then choose the specific model/framework
- YOLOv11 is **not supported** due to AGPL-3.0 license restrictions

### Available Model Tools

#### ObjectDetection
**Category**: Models
**Purpose**: Universal object detection supporting multiple frameworks (80 COCO classes)

**Supported Frameworks**:
- **Detectron2**: Faster R-CNN, RetinaNet (various backbones)
- **YOLOv8**: Nano, Small, Medium, Large, XLarge variants

**Inputs**:
- `image` (PortType.IMAGE) - PIL Image

**Outputs**:
- `detections` (PortType.DETECTIONS) - PixelFlow Detections object

**Parameters**:
- `framework`: detectron2 or yolov8
- `model_variant`: Specific model (options change based on framework)
- `confidence_threshold`: Filter detections (0.0-1.0, default 0.5)
- `device`: cpu, cuda, or mps (Apple Silicon)

**Example Workflow**:
```
MediaInput → ObjectDetection → DrawBoundingBoxes → ImageToBase64
```

#### InstanceSegmentation
**Category**: Models
**Purpose**: Universal instance segmentation with pixel-level masks

**Supported Frameworks**:
- **Detectron2**: Mask R-CNN (R50, R101, X101 backbones)
- **YOLOv8**: Segmentation variants (yolov8n-seg through yolov8x-seg)

**Inputs/Outputs**: Same as ObjectDetection
**Special**: Outputs include segmentation masks in addition to bounding boxes

#### DepthEstimation
**Category**: Detection
**Purpose**: Monocular depth estimation using Depth Anything

**Inputs**:
- `image` (PortType.IMAGE) - PIL Image

**Outputs**:
- `depth_map` (PortType.IMAGE) - Grayscale depth map (PIL Image)

**Parameters**:
- `model_variant`: small (~350MB), base (~1.3GB), or large (~1.3GB)
- `device`: cpu, cuda, or mps

### Model Management

**Lazy Loading**: Models load on first use, not at startup
**Thread-Safe**: Multiple concurrent requests handled safely
**Memory Management**: Inactive models automatically cleaned up

**Manual Cleanup** (if needed):
```python
from agentui.tools.model_tools import MozoModelToolBase

# Clean up models inactive for 10 minutes
MozoModelToolBase.cleanup_models(inactive_seconds=600)

# Unload all models
MozoModelToolBase.unload_all_models()
```

### Port Type: DETECTIONS

The new `PortType.DETECTIONS` represents PixelFlow Detections objects:
- Unified format across all detection frameworks
- Method chaining for filtering (`.filter_by_confidence()`, `.filter_by_class_id()`, etc.)
- Compatible with all PixelFlow annotation tools
- Can be converted to dict/JSON for export

**Data Flow**:
```
Model Tool (DETECTIONS) → Annotation Tool (IMAGE) → Output
```

**Filtering Example** (in custom tools):
```python
detections = self.get_input_data("detections")  # PixelFlow Detections object

# Chain filters
filtered = (detections
    .filter_by_confidence(0.7)
    .filter_by_class_id([0, 2, 5])  # person, car, bus
    .filter_by_size(min_area=1000))

# Access individual detections
for det in filtered:
    print(det.class_name, det.confidence, det.bbox)
```

### Tool Categories

- **Models**: `ObjectDetection`, `InstanceSegmentation`, `DepthEstimation`
- **Annotation**: `DrawBoundingBoxes`, `AddLabels`, `DrawMasks` (accept DETECTIONS)
- **Privacy**: `BlurRegions`, `PixelateRegions` (accept DETECTIONS)
- **Tracking**: `ObjectTracker` (accepts DETECTIONS)
- **Analysis**: `ZoneAnalyzer` (accepts DETECTIONS)

### Future Model Tools (Mozo Integration Roadmap)

These model types need to be added to the Mozo library first, then integrated into AgentUI:

#### Classification Models
- ResNet, EfficientNet, ViT, ConvNeXt
- Output: class_name, confidence, top_k predictions

#### OCR (Optical Character Recognition)
- EasyOCR, PaddleOCR, Tesseract, TrOCR
- Output: text, bounding boxes, confidence

#### Semantic Segmentation
- DeepLabV3, SegFormer, SAM2 (with prompts)
- Output: segmentation map (pixel-wise class labels)

#### Keypoint Detection
- MediaPipe Pose, OpenPose, Detectron2 Keypoint R-CNN
- Output: keypoints with confidence scores

#### Other Models (Special Category)
- **SAM/SAM2**: Segment Anything with prompt support (points, boxes)
- **Grounding DINO**: Text-prompted object detection
- **Florence-2**: Multi-modal vision-language model
- **Qwen2-VL**: Vision-language model for VQA and captioning

These require dynamic input ports based on model selection (e.g., SAM needs prompt_points)