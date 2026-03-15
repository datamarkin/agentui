"""
Example: Batch image processing with AgentUI workflows

Demonstrates how to:
1. Load a workflow from JSON
2. Process a single image
3. Process multiple images in batch
4. Access detections and annotated images

Prerequisites:
- Create a detection workflow in the UI
- Export it as 'detection_workflow.json'
- Place it in the same directory as this script
"""

from agentui import Workflow
from pathlib import Path

def main():
    # Load workflow from exported JSON
    print("Loading workflow...")
    workflow = Workflow.load('detection_workflow.json')

    # Check what inputs it needs
    print(f"Workflow requires inputs: {workflow.inputs}")

    # Example 1: Process single image
    print("\n=== Example 1: Single Image ===")
    result = workflow.run(image='test_image.jpg')

    print(f"Output keys: {list(result.keys())}")

    if 'detections' in result:
        detections = result['detections']
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"  - {det.class_name}: {det.confidence:.2f} at {det.bbox}")

    if 'image' in result:
        # Save annotated image
        result['image'].save('output_single.jpg')
        print("Saved annotated image to: output_single.jpg")

    # Example 2: Batch process multiple images
    print("\n=== Example 2: Batch Processing ===")

    # Get all images from a directory
    image_dir = Path('images')
    if image_dir.exists():
        image_paths = [str(p) for p in image_dir.glob('*.jpg')]
        print(f"Processing {len(image_paths)} images...")

        # Run workflow with list of images
        batch_result = workflow.run(image=image_paths)

        # Process batch results
        if 'detections' in batch_result:
            all_detections = batch_result['detections']
            print(f"\nBatch processing complete!")
            print(f"Processed {len(all_detections)} images")

            # Summary statistics
            for i, detections in enumerate(all_detections):
                img_path = Path(image_paths[i]).name
                print(f"  {img_path}: {len(detections)} objects detected")

        # Save batch results
        if 'image' in batch_result:
            annotated_images = batch_result['image']
            output_dir = Path('output_batch')
            output_dir.mkdir(exist_ok=True)

            for i, img in enumerate(annotated_images):
                output_path = output_dir / f"annotated_{i:04d}.jpg"
                img.save(output_path)

            print(f"Saved {len(annotated_images)} annotated images to: output_batch/")
    else:
        print(f"Image directory not found: {image_dir}")
        print("Create 'images/' directory and add some .jpg files to test batch processing")

    # Example 3: Process list of PIL Images directly
    print("\n=== Example 3: PIL Images ===")
    from PIL import Image

    # Load images manually
    pil_images = [
        Image.open('test_image.jpg'),
        Image.open('test_image.jpg')  # Same image twice for demo
    ]

    result = workflow.run(image=pil_images)
    print(f"Processed {len(pil_images)} PIL images")
    print(f"Got {len(result['detections'])} detection results")


if __name__ == '__main__':
    main()
