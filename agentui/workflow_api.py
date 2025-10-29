"""
Developer-friendly workflow API for AgentUI.

Provides a simple, Pythonic interface for loading and executing workflows
with manual input injection and easy output access.

Example:
    from agentui import Workflow

    # Load workflow
    workflow = Workflow.load('detection_pipeline.json')

    # See what inputs it needs
    print(workflow.inputs)  # ['image']

    # Run with single image
    result = workflow.run(image='test.jpg')
    print(result['detections'])

    # Run with batch of images
    result = workflow.run(image=['img1.jpg', 'img2.jpg', 'img3.jpg'])
    print(len(result['detections']))  # 3
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union
from PIL import Image

from .core.workflow import WorkflowEngine
from .core.registry import registry
from .core.tool import PortType, ToolOutput


class Workflow:
    """High-level workflow interface for developers."""

    def __init__(self, core_workflow: WorkflowEngine, input_mapping: Dict[str, str]):
        """
        Initialize workflow wrapper.

        Args:
            core_workflow: The underlying WorkflowEngine instance
            input_mapping: Maps input names to tool IDs (e.g., {'image': 'MediaInput-123'})
        """
        self._workflow = core_workflow
        self._input_mapping = input_mapping

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Workflow':
        """
        Load workflow from JSON file.

        Args:
            path: Path to workflow JSON file exported from UI

        Returns:
            Workflow instance ready to execute

        Example:
            workflow = Workflow.load('detection_pipeline.json')
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        # Read JSON
        with open(path, 'r') as f:
            workflow_data = json.load(f)

        # Create workflow engine
        workflow_json = json.dumps(workflow_data)
        core_workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

        # Discover input tools (currently only MediaInput)
        input_mapping = {}
        for node in workflow_data.get('nodes', []):
            node_data = node.get('data', {})
            node_type = node_data.get('nodeType') or node_data.get('toolType')

            # Check if this is an input tool
            if node_type == 'MediaInput':
                # Map 'image' input name to this tool's ID
                input_mapping['image'] = node['id']

        return cls(core_workflow, input_mapping)

    @property
    def inputs(self) -> List[str]:
        """
        Get list of required input names for this workflow.

        Returns:
            List of input names (e.g., ['image'])

        Example:
            workflow = Workflow.load('pipeline.json')
            print(workflow.inputs)  # ['image']
        """
        return list(self._input_mapping.keys())

    def run(self, **inputs) -> Dict[str, Any]:
        """
        Execute workflow with provided inputs.

        Args:
            **inputs: Named inputs (e.g., image='/path/to/img.jpg')
                     Can be file paths (str), PIL Images, or lists of either

        Returns:
            Dictionary of all outputs from all tools

        Example:
            # Single image
            result = workflow.run(image='test.jpg')
            detections = result['detections']

            # Batch of images
            result = workflow.run(image=['img1.jpg', 'img2.jpg'])
            for dets in result['detections']:
                print(f"Found {len(dets)} objects")
        """
        # Validate inputs
        for input_name in inputs.keys():
            if input_name not in self._input_mapping:
                raise ValueError(
                    f"Unknown input '{input_name}'. "
                    f"Workflow expects: {list(self._input_mapping.keys())}"
                )

        # Check for missing required inputs
        for required_input in self._input_mapping.keys():
            if required_input not in inputs:
                raise ValueError(f"Missing required input: '{required_input}'")

        # Inject inputs into tools
        for input_name, input_value in inputs.items():
            tool_id = self._input_mapping[input_name]
            tool = self._workflow.tools[tool_id]

            # Handle different input types
            if input_name == 'image':
                processed_value = self._process_image_input(input_value)
                # Set the output directly (bypass MediaInput processing)
                tool.outputs['image'] = ToolOutput(processed_value, PortType.IMAGE)

        # Execute workflow
        all_results = self._workflow.execute()

        # Collect ALL outputs (not just terminal tools)
        outputs = {}
        for tool_id, result in all_results.items():
            for output_name, output_value in result['outputs'].items():
                # Use output name directly if unique, otherwise prefix with tool type
                if output_name not in outputs:
                    outputs[output_name] = output_value
                else:
                    # Name collision - prefix with tool type
                    prefixed_name = f"{result['type'].lower()}_{output_name}"
                    outputs[prefixed_name] = output_value

        return outputs

    def _process_image_input(self, value: Union[str, Path, Image.Image, List]) -> Union[Image.Image, List[Image.Image]]:
        """
        Process image input - handle file paths, PIL Images, or lists.

        Args:
            value: Image path(s) or PIL Image(s)

        Returns:
            PIL Image or list of PIL Images
        """
        # Handle list of images (batch processing)
        if isinstance(value, list):
            return [self._load_single_image(item) for item in value]

        # Handle single image
        return self._load_single_image(value)

    def _load_single_image(self, value: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load a single image from path or return PIL Image.

        Args:
            value: File path or PIL Image

        Returns:
            PIL Image
        """
        # Already a PIL Image
        if isinstance(value, Image.Image):
            return value

        # File path - load it
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert('RGB')

        raise TypeError(f"Unsupported image type: {type(value)}")
