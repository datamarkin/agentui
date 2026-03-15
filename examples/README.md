# AgentUI Python API Examples

This directory contains examples demonstrating how to use AgentUI workflows programmatically in Python.

## Quick Start

### 1. Create a Workflow in the UI

1. Start AgentUI: `agentui start`
2. Open browser: `http://localhost:8000`
3. Build your workflow (e.g., MediaInput → Florence2 → DrawBoundingBoxes)
4. Click "Export Workflow" to download `workflow.json`

### 2. Use the Workflow in Python

```python
from agentui import Workflow

# Load workflow
workflow = Workflow.load('workflow.json')

# Process single image
result = workflow.run(image='test.jpg')
print(result['detections'])

# Process batch of images
result = workflow.run(image=['img1.jpg', 'img2.jpg', 'img3.jpg'])
for detections in result['detections']:
    print(f"Found {len(detections)} objects")
```

## Examples

### Single Image Detection

```python
from agentui import Workflow

workflow = Workflow.load('detection_workflow.json')
result = workflow.run(image='photo.jpg')

# Access detections
for det in result['detections']:
    print(f"{det.class_name}: {det.confidence:.2f}")

# Save annotated image
if 'image' in result:
    result['image'].save('annotated.jpg')
```

### Batch Processing (5000 Images)

```python
from agentui import Workflow
from pathlib import Path

workflow = Workflow.load('detection_workflow.json')

# Get all images
image_paths = [str(p) for p in Path('images').glob('*.jpg')]
print(f"Processing {len(image_paths)} images...")

# Batch process - all images processed efficiently
result = workflow.run(image=image_paths)

# Access results
all_detections = result['detections']
print(f"Found {len(all_detections)} detection results")

# Process each result
for i, detections in enumerate(all_detections):
    print(f"Image {i}: {len(detections)} objects")
```

### Using PIL Images Directly

```python
from agentui import Workflow
from PIL import Image

workflow = Workflow.load('workflow.json')

# Load images with PIL
images = [Image.open(f'img{i}.jpg') for i in range(10)]

# Process
result = workflow.run(image=images)

# Get results
for i, dets in enumerate(result['detections']):
    print(f"Image {i}: detected {[d.class_name for d in dets]}")
```

## Available Examples

- **`batch_detection.py`** - Complete examples of single and batch image processing

## API Reference

### `Workflow.load(path)`

Load a workflow from exported JSON file.

**Args:**
- `path` (str): Path to workflow JSON file

**Returns:**
- `Workflow` instance

### `workflow.inputs`

Property that returns list of required input names.

**Returns:**
- `List[str]`: Input names (e.g., `['image']`)

### `workflow.run(**inputs)`

Execute workflow with provided inputs.

**Args:**
- `**inputs`: Named inputs matching workflow.inputs
  - Single value: Process one item
  - List of values: Batch process multiple items

**Returns:**
- `Dict[str, Any]`: All outputs from all tools

**Example:**
```python
# Single
result = workflow.run(image='test.jpg')

# Batch
result = workflow.run(image=['a.jpg', 'b.jpg', 'c.jpg'])
```

## Input Types

The `image` input accepts:

1. **File path (str):** `'test.jpg'`
2. **Path object:** `Path('test.jpg')`
3. **PIL Image:** `Image.open('test.jpg')`
4. **List of any above:** `['a.jpg', 'b.jpg']` or `[img1, img2]`

## Output Structure

Results dictionary contains all outputs from all tools:

```python
{
    'detections': PixelFlow.Detections,  # Detection results
    'image': PIL.Image,                   # Annotated image
    'result': Dict,                       # Text/caption results (if using Florence2 captioning)
}
```

## Performance

The API uses AgentUI's built-in auto-batching:
- Single image: Processed as single item
- List of images: Automatically batched for efficiency
- No manual looping needed
- Memory efficient for large batches

## Error Handling

```python
from agentui import Workflow

try:
    workflow = Workflow.load('workflow.json')
    result = workflow.run(image='test.jpg')
except FileNotFoundError:
    print("Workflow or image file not found")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Next Steps

1. Create a workflow in the UI
2. Export it as JSON
3. Run `python batch_detection.py` to see it in action
4. Modify the example for your use case
