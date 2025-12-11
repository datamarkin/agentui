from typing import Dict, List, Type
from ..core.tool import Tool
from ..tools.base_tools import (
    MediaInputTool,
    ConvertFormatTool,
    SaveImageTool
)
from ..tools.cv_tools import (
    DominantColorTool,
    QualityAnalysisTool,
    VisualizeDetectionsTool,
    BlendImagesTool
)

# Import pixelflow tools if available
try:
    from ..tools.pixelflow_tools import PIXELFLOW_TOOLS
    PIXELFLOW_AVAILABLE = True
except ImportError:
    PIXELFLOW_TOOLS = []
    PIXELFLOW_AVAILABLE = False

# Import model tools if available
try:
    from ..tools.model_tools import MODEL_TOOLS
    MODEL_TOOLS_AVAILABLE = True
except ImportError:
    MODEL_TOOLS = []
    MODEL_TOOLS_AVAILABLE = False

# Import transform tools if available
try:
    from ..tools.transform_tools import TRANSFORM_TOOLS
    TRANSFORM_TOOLS_AVAILABLE = True
except ImportError:
    TRANSFORM_TOOLS = []
    TRANSFORM_TOOLS_AVAILABLE = False


class ToolRegistry:
    """Registry for all available tool types"""

    def __init__(self):
        self._tools: Dict[str, Type[Tool]] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register all built-in tool types"""
        # Input/Output tools
        self.register(MediaInputTool)
        self.register(SaveImageTool)

        # Basic processing
        self.register(ConvertFormatTool)

        # Analysis
        self.register(DominantColorTool)
        self.register(QualityAnalysisTool)

        # Combiners
        self.register(VisualizeDetectionsTool)
        self.register(BlendImagesTool)

        # PixelFlow tools (if available)
        if PIXELFLOW_AVAILABLE:
            for tool_class in PIXELFLOW_TOOLS:
                self.register(tool_class)

        # Model tools (if available)
        if MODEL_TOOLS_AVAILABLE:
            for tool_class in MODEL_TOOLS:
                self.register(tool_class)

        # Transform tools (if available)
        if TRANSFORM_TOOLS_AVAILABLE:
            for tool_class in TRANSFORM_TOOLS:
                self.register(tool_class)

    def register(self, tool_class: Type[Tool]):
        """Register a tool class"""
        tool_instance = tool_class()
        self._tools[tool_instance.tool_type] = tool_class

    def get_tool_class(self, tool_type: str) -> Type[Tool]:
        """Get tool class by type"""
        if tool_type not in self._tools:
            raise ValueError(f"Unknown tool type: {tool_type}")
        return self._tools[tool_type]

    def get_all_types(self) -> Dict[str, Type[Tool]]:
        """Get all registered tool types"""
        return self._tools.copy()

    def get_tool_info(self, tool_type: str) -> Dict[str, any]:
        """Get information about a tool type for the UI"""
        tool_class = self.get_tool_class(tool_type)
        instance = tool_class()

        return {
            'type': tool_type,
            'name': self._get_tool_name(tool_type),
            'inputs': instance.input_types,
            'outputs': instance.output_types,
            'parameters': self._get_default_parameters(tool_type),
            'ports': instance.get_tool_info(),
            'category': self._get_tool_category(tool_type),
            'description': self._get_tool_description(tool_type),
            'parameter_options': self._get_parameter_options(tool_type),
            'required_inputs': self.get_required_inputs(tool_type),
            'optional_inputs': self.get_optional_inputs(tool_type)
        }

    # Unified tool metadata - single source of truth
    TOOL_METADATA = {
        # Input/Output
        'MediaInput': {
            'name': 'Media Input',
            'category': 'Input/Output',
            'description': 'Load media (images, videos) from file or base64 data',
            'required_inputs': [],  # No inputs for input tools
            'optional_inputs': [],
            'parameters': {
                'path': '',
                'data': '',
                'resize_on_load': True,
                'max_width': 1920,
                'max_height': 1080
            }
        },
        'SaveImage': {
            'name': 'Save Image',
            'category': 'Input/Output',
            'description': 'Save image to file',
            'parameters': {
                'path': 'output.jpg',
                'quality': 95,
                'format': 'JPEG',
                'overwrite': True
            },
            'parameter_options': {
                'format': {
                    'type': 'select',
                    'options': [
                        {'value': 'JPEG', 'label': 'JPEG'},
                        {'value': 'PNG', 'label': 'PNG'},
                        {'value': 'WEBP', 'label': 'WebP'},
                        {'value': 'TIFF', 'label': 'TIFF'},
                        {'value': 'BMP', 'label': 'BMP'}
                    ]
                }
            }
        },

        # Adjust
        'ConvertFormat': {
            'name': 'Convert Format',
            'category': 'Adjust',
            'description': 'Convert image color format',
            'parameters': {
                'format': 'RGB',
                'background_color': '#FFFFFF'
            },
            'parameter_options': {
                'format': {
                    'type': 'select',
                    'options': [
                        {'value': 'RGB', 'label': 'RGB'},
                        {'value': 'RGBA', 'label': 'RGBA'},
                        {'value': 'grayscale', 'label': 'Grayscale'},
                        {'value': 'L', 'label': 'Luminance (L)'},
                        {'value': 'CMYK', 'label': 'CMYK'}
                    ]
                }
            }
        },

        # Analysis
        'DominantColor': {
            'name': 'Dominant Color',
            'category': 'Analysis',
            'description': 'Extract dominant color from image',
            'parameters': {
                'num_colors': 5,
                'color_format': 'hex',
                'ignore_white': True,
                'ignore_black': True
            }
        },
        'QualityAnalysis': {
            'name': 'Quality Analysis',
            'category': 'Analysis',
            'description': 'Analyze image quality metrics',
            'parameters': {
                'check_blur': True,
                'check_brightness': True,
                'check_contrast': True,
                'blur_threshold': 100
            }
        },

        # Combine
        'VisualizeDetections': {
            'name': 'Visualize Detections',
            'category': 'Combine',
            'description': 'Overlay detection results on image (supports PixelFlow Detections)',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'box_color': '#00FF00',
                'box_thickness': 2,
                'label_font_size': 12,
                'show_confidence': True
            }
        },
        'BlendImages': {
            'name': 'Blend Images',
            'category': 'Combine',
            'description': 'Blend two images together',
            'required_inputs': ['background', 'foreground'],
            'optional_inputs': [],
            'parameters': {
                'alpha': 0.5,
                'blend_mode': 'normal',
                'resize_to_match': True
            },
            'parameter_options': {
                'blend_mode': {
                    'type': 'select',
                    'options': [
                        {'value': 'normal', 'label': 'Normal'},
                        {'value': 'multiply', 'label': 'Multiply'},
                        {'value': 'screen', 'label': 'Screen'},
                        {'value': 'overlay', 'label': 'Overlay'},
                        {'value': 'soft_light', 'label': 'Soft Light'},
                        {'value': 'hard_light', 'label': 'Hard Light'}
                    ]
                }
            }
        },

        # PixelFlow tools (external library integration)
        'DrawBoundingBoxes': {
            'name': 'Draw Bounding Boxes',
            'category': 'Annotation',
            'description': 'Draw bounding boxes around detected objects using pixelflow',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'thickness': 2,
                'color': [255, 0, 0]
            }
        },
        'AddLabels': {
            'name': 'Add Labels',
            'category': 'Annotation',
            'description': 'Add text labels to detected objects using pixelflow',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'font_size': 12,
                'color': [255, 255, 255],
                'background_color': [0, 0, 0, 128],
                'position': 'top'
            }
        },
        'BlurRegions': {
            'name': 'Blur Regions',
            'category': 'Privacy',
            'description': 'Blur specified regions for privacy protection using pixelflow',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'blur_intensity': 15,
                'kernel_size': 15
            }
        },
        'PixelateRegions': {
            'name': 'Pixelate Regions',
            'category': 'Privacy',
            'description': 'Pixelate specified regions for privacy protection using pixelflow',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'pixel_size': 20
            }
        },
        'DrawMasks': {
            'name': 'Draw Masks',
            'category': 'Annotation',
            'description': 'Draw segmentation masks on image using pixelflow',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'opacity': 0.5,
                'color': [255, 0, 0]
            }
        },
        'DrawPolygons': {
            'name': 'Draw Polygons',
            'category': 'Annotation',
            'description': 'Draw polygon shapes on image using pixelflow',
            'required_inputs': ['image', 'polygons'],
            'optional_inputs': [],
            'parameters': {
                'thickness': 2,
                'color': [0, 255, 0],
                'filled': False
            }
        },
        'DrawKeypoints': {
            'name': 'Draw Keypoints',
            'category': 'Annotation',
            'description': 'Draw colored circles on keypoint locations',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'radius': None,
                'thickness': None,
                'show_names': False
            }
        },
        'DrawKeypointSkeleton': {
            'name': 'Draw Keypoint Skeleton',
            'category': 'Annotation',
            'description': 'Draw lines connecting related keypoints (COCO skeleton by default)',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'thickness': None,
                'skeleton_type': 'coco',
                'custom_connections': ''
            },
            'parameter_options': {
                'skeleton_type': [
                    {'value': 'coco', 'label': 'COCO (Human Pose)'},
                    {'value': 'custom', 'label': 'Custom Connections'}
                ]
            }
        },
        'ObjectTracker': {
            'name': 'Object Tracker',
            'category': 'Tracking',
            'description': 'Track objects across multiple frames using pixelflow',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'max_disappeared': 30,
                'max_distance': 50
            }
        },
        'ZoneAnalyzer': {
            'name': 'Zone Analyzer',
            'category': 'Analysis',
            'description': 'Analyze object presence in predefined zones using pixelflow',
            'required_inputs': ['image', 'detections', 'zone_definitions'],
            'optional_inputs': [],
            'parameters': {}
        },

        # Model tools (Mozo integration)
        'ObjectDetection': {
            'name': 'Object Detection',
            'category': 'Models',
            'description': 'Universal object detection supporting multiple frameworks (Detectron2, YOLOv8)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'framework': 'detectron2',
                'model_variant': 'faster_rcnn_R_50_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'parameter_options': {
                'framework': {
                    'type': 'select',
                    'options': [
                        {'value': 'detectron2', 'label': 'Detectron2'},
                        {'value': 'yolov8', 'label': 'YOLOv8'}
                    ]
                },
                'model_variant': {
                    'type': 'select',
                    'options': [
                        # Detectron2 models
                        {'value': 'faster_rcnn_R_50_FPN_3x', 'label': 'Faster R-CNN (ResNet-50)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_faster_rcnn_r50'},
                        {'value': 'faster_rcnn_R_101_FPN_3x', 'label': 'Faster R-CNN (ResNet-101)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_faster_rcnn_r101'},
                        {'value': 'retinanet_R_50_FPN_3x', 'label': 'RetinaNet (ResNet-50)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_retinanet_r50'},
                        {'value': 'retinanet_R_101_FPN_3x', 'label': 'RetinaNet (ResNet-101)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_retinanet_r101'},
                        # YOLOv8 models
                        {'value': 'yolov8n', 'label': 'YOLOv8 Nano (fastest)', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8n'},
                        {'value': 'yolov8s', 'label': 'YOLOv8 Small', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8s'},
                        {'value': 'yolov8m', 'label': 'YOLOv8 Medium (balanced)', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8m'},
                        {'value': 'yolov8l', 'label': 'YOLOv8 Large', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8l'},
                        {'value': 'yolov8x', 'label': 'YOLOv8 XLarge (most accurate)', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8x'}
                    ]
                },
                'device': {
                    'type': 'select',
                    'options': [
                        {'value': 'cpu', 'label': 'CPU'},
                        {'value': 'cuda', 'label': 'CUDA (GPU)'},
                        {'value': 'mps', 'label': 'MPS (Apple Silicon)'},
                        {'value': 'api', 'label': 'API (Datamarkin Cloud)'}
                    ]
                }
            }
        },
        'InstanceSegmentation': {
            'name': 'Instance Segmentation',
            'category': 'Models',
            'description': 'Universal instance segmentation supporting multiple frameworks (Detectron2, YOLOv8)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'framework': 'detectron2',
                'model_variant': 'mask_rcnn_R_50_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'parameter_options': {
                'framework': {
                    'type': 'select',
                    'options': [
                        {'value': 'detectron2', 'label': 'Detectron2'},
                        {'value': 'yolov8', 'label': 'YOLOv8'}
                    ]
                },
                'model_variant': {
                    'type': 'select',
                    'options': [
                        # Detectron2 models
                        {'value': 'mask_rcnn_R_50_FPN_3x', 'label': 'Mask R-CNN (ResNet-50)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_mask_rcnn_r50'},
                        {'value': 'mask_rcnn_R_101_FPN_3x', 'label': 'Mask R-CNN (ResNet-101)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_mask_rcnn_r101'},
                        {'value': 'mask_rcnn_X_101_32x8d_FPN_3x', 'label': 'Mask R-CNN (ResNeXt-101)', 'framework': 'detectron2', 'training_id': 'PLACEHOLDER_mask_rcnn_x101'},
                        # YOLOv8 segmentation models
                        {'value': 'yolov8n-seg', 'label': 'YOLOv8 Nano Segmentation (fastest)', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8n_seg'},
                        {'value': 'yolov8s-seg', 'label': 'YOLOv8 Small Segmentation', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8s_seg'},
                        {'value': 'yolov8m-seg', 'label': 'YOLOv8 Medium Segmentation (balanced)', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8m_seg'},
                        {'value': 'yolov8l-seg', 'label': 'YOLOv8 Large Segmentation', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8l_seg'},
                        {'value': 'yolov8x-seg', 'label': 'YOLOv8 XLarge Segmentation (most accurate)', 'framework': 'yolov8', 'training_id': 'PLACEHOLDER_yolov8x_seg'}
                    ]
                },
                'device': {
                    'type': 'select',
                    'options': [
                        {'value': 'cpu', 'label': 'CPU'},
                        {'value': 'cuda', 'label': 'CUDA (GPU)'},
                        {'value': 'mps', 'label': 'MPS (Apple Silicon)'},
                        {'value': 'api', 'label': 'API (Datamarkin Cloud)'}
                    ]
                }
            }
        },
        'DepthEstimation': {
            'name': 'Depth Estimation',
            'category': 'Detection',
            'description': 'Estimate depth from single image using Depth Anything',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'model_variant': 'small',
                'device': 'cpu'
            },
            'parameter_options': {
                'model_variant': {
                    'type': 'select',
                    'options': [
                        {'value': 'small', 'label': 'Small (Fast, ~350MB)'},
                        {'value': 'base', 'label': 'Base (Balanced, ~1.3GB)'},
                        {'value': 'large', 'label': 'Large (Best Quality, ~1.3GB)'}
                    ]
                },
                'device': {
                    'type': 'select',
                    'options': [
                        {'value': 'cpu', 'label': 'CPU'},
                        {'value': 'cuda', 'label': 'CUDA (GPU)'},
                        {'value': 'mps', 'label': 'MPS (Apple Silicon)'}
                    ]
                }
            }
        },
        'DatamarkinDetection': {
            'name': 'Datamarkin Detection',
            'category': 'Models',
            'description': 'Cloud-based inference using your custom Datamarkin models (keypoints, detection, segmentation)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'training_id': '',
                'bearer_token': ''
            }
        },

        'OCRDetection': {
            'name': 'OCR Detection',
            'category': 'Models',
            'description': 'Extract text from images using PaddleOCR or EasyOCR (80+ languages supported)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'framework': 'paddleocr',
                'variant': 'mobile',
                'language': 'en'
            },
            'parameter_options': {
                'framework': {
                    'type': 'select',
                    'options': [
                        {'value': 'paddleocr', 'label': 'PaddleOCR (PP-OCRv5)'},
                        {'value': 'easyocr', 'label': 'EasyOCR'}
                    ]
                },
                'variant': {
                    'type': 'select',
                    'options': [
                        {'value': 'mobile', 'label': 'Mobile (Fast)'},
                        {'value': 'server', 'label': 'Server (Accurate)'},
                        {'value': 'mobile-chinese', 'label': 'Mobile Chinese'},
                        {'value': 'server-chinese', 'label': 'Server Chinese'},
                        {'value': 'mobile-multilingual', 'label': 'Mobile Multilingual'},
                        {'value': 'english-light', 'label': 'English Light'},
                        {'value': 'english-full', 'label': 'English Full'},
                        {'value': 'multilingual', 'label': 'Multilingual'},
                        {'value': 'chinese', 'label': 'Chinese'}
                    ]
                }
            }
        },

        'Florence2': {
            'name': 'Florence-2',
            'category': 'Models',
            'description': 'Microsoft Florence-2 multi-task vision model (detection, captioning, OCR, segmentation)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'task': 'detection',
                'prompt': ''
            },
            'parameter_options': {
                'task': {
                    'type': 'select',
                    'options': [
                        {'value': 'detection', 'label': 'Object Detection'},
                        {'value': 'detection_with_caption', 'label': 'Detection + Captions'},
                        {'value': 'captioning', 'label': 'Image Captioning'},
                        {'value': 'detailed_captioning', 'label': 'Detailed Captioning'},
                        {'value': 'more_detailed_captioning', 'label': 'Comprehensive Captioning'},
                        {'value': 'ocr', 'label': 'OCR (Text Extraction)'},
                        {'value': 'ocr_with_region', 'label': 'OCR with Bounding Boxes'},
                        {'value': 'segmentation', 'label': 'Instance Segmentation (needs prompt)'}
                    ]
                }
            }
        },

        'VisualQuestionAnswering': {
            'name': 'Visual Question Answering',
            'category': 'Models',
            'description': 'Answer questions about images using BLIP, Qwen2.5-VL, or Qwen3-VL vision-language models',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'framework': 'blip_vqa',
                'variant': 'base',
                'question': 'What is in this image?'
            },
            'parameter_options': {
                'framework': {
                    'type': 'select',
                    'options': [
                        {'value': 'blip_vqa', 'label': 'BLIP VQA'},
                        {'value': 'qwen2.5_vl', 'label': 'Qwen2.5-VL'},
                        {'value': 'qwen3_vl', 'label': 'Qwen3-VL (with reasoning)'}
                    ]
                },
                'variant': {
                    'type': 'select',
                    'options': [
                        {'value': 'base', 'label': 'Base'},
                        {'value': 'capfilt-large', 'label': 'CapFilt Large'},
                        {'value': '7b-instruct', 'label': '7B Instruct'},
                        {'value': '2b-thinking', 'label': '2B Thinking'}
                    ]
                }
            }
        },

        'StabilityInpainting': {
            'name': 'Stable Diffusion Inpainting',
            'category': 'Models',
            'description': 'Generate and modify image content using text prompts (requires mask image)',
            'required_inputs': ['image', 'mask'],
            'optional_inputs': [],
            'parameters': {
                'prompt': 'high quality, detailed',
                'negative_prompt': 'lowres, bad quality',
                'num_inference_steps': 50,
                'guidance_scale': 7.5
            }
        },

        # Transform tools (pixelflow.transform integration)
        'RotateImage': {
            'name': 'Rotate Image',
            'category': 'Transform',
            'description': 'Rotate image by angle (counter-clockwise)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'angle': 0,
                'center_x': None,
                'center_y': None,
                'fillcolor': None
            }
        },
        'FlipImage': {
            'name': 'Flip Image',
            'category': 'Transform',
            'description': 'Flip image horizontally or vertically',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'direction': 'horizontal'
            },
            'parameter_options': {
                'direction': {
                    'type': 'select',
                    'options': [
                        {'value': 'horizontal', 'label': 'Horizontal'},
                        {'value': 'vertical', 'label': 'Vertical'}
                    ]
                }
            }
        },
        'CropImage': {
            'name': 'Crop Image',
            'category': 'Transform',
            'description': 'Crop image to specified bounding box',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'x1': 0,
                'y1': 0,
                'x2': 100,
                'y2': 100
            }
        },
        'EnhanceCLAHE': {
            'name': 'Enhance CLAHE',
            'category': 'Enhance',
            'description': 'Apply CLAHE contrast enhancement',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'clip_limit': 2.0,
                'tile_size': 8
            }
        },
        'AutoContrast': {
            'name': 'Auto Contrast',
            'category': 'Enhance',
            'description': 'Apply automatic contrast adjustment by stretching histogram',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'cutoff': 1.0
            }
        },
        'GammaCorrection': {
            'name': 'Gamma Correction',
            'category': 'Enhance',
            'description': 'Apply gamma correction (<1 brightens, >1 darkens)',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'gamma': 1.0
            }
        },
        'NormalizeImage': {
            'name': 'Normalize Image',
            'category': 'Enhance',
            'description': 'Normalize image for neural network input',
            'required_inputs': ['image'],
            'optional_inputs': [],
            'parameters': {
                'preset': 'imagenet',
                'mean_r': 0.5,
                'mean_g': 0.5,
                'mean_b': 0.5,
                'std_r': 0.5,
                'std_g': 0.5,
                'std_b': 0.5
            },
            'parameter_options': {
                'preset': {
                    'type': 'select',
                    'options': [
                        {'value': 'imagenet', 'label': 'ImageNet (Standard)'},
                        {'value': 'custom', 'label': 'Custom'}
                    ]
                }
            }
        },
        'RotateWithDetections': {
            'name': 'Rotate With Detections',
            'category': 'Transform',
            'description': 'Rotate image and detections together (modifies detections in-place)',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'angle': 0,
                'center_x': None,
                'center_y': None,
                'fillcolor': None,
                'track_metadata': True
            }
        },
        'FlipWithDetections': {
            'name': 'Flip With Detections',
            'category': 'Transform',
            'description': 'Flip image and detections together (modifies detections in-place)',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'direction': 'horizontal',
                'track_metadata': True
            },
            'parameter_options': {
                'direction': {
                    'type': 'select',
                    'options': [
                        {'value': 'horizontal', 'label': 'Horizontal'},
                        {'value': 'vertical', 'label': 'Vertical'}
                    ]
                }
            }
        },
        'CropWithDetections': {
            'name': 'Crop With Detections',
            'category': 'Transform',
            'description': 'Crop image and filter detections to crop region',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'x1': 0,
                'y1': 0,
                'x2': 100,
                'y2': 100,
                'track_metadata': True
            }
        },
        'CropAroundDetections': {
            'name': 'Crop Around Detections',
            'category': 'Detection',
            'description': 'Extract cropped images around each detection bbox',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'padding_mode': 'uniform',
                'padding': 0.0,
                'padding_left': 0.0,
                'padding_right': 0.0,
                'padding_top': 0.0,
                'padding_bottom': 0.0
            },
            'parameter_options': {
                'padding_mode': {
                    'type': 'select',
                    'options': [
                        {'value': 'uniform', 'label': 'Uniform Padding'},
                        {'value': 'custom', 'label': 'Custom Per Side'}
                    ]
                }
            }
        },
        'AlignDetections': {
            'name': 'Align Detections',
            'category': 'Detection',
            'description': 'Rotate image so two keypoints form specified angle',
            'required_inputs': ['image', 'detections'],
            'optional_inputs': [],
            'parameters': {
                'point1_name': 'p0',
                'point2_name': 'p9',
                'target_angle': 0.0,
                'detection_index': 0
            }
        },
        'UpdateBBoxFromKeypoints': {
            'name': 'Update BBox From Keypoints',
            'category': 'Detection',
            'description': 'Update detection bboxes based on keypoint positions',
            'required_inputs': ['detections'],
            'optional_inputs': [],
            'parameters': {
                'keypoint_mode': 'all',
                'keypoint_names': '',
                'track_metadata': True
            },
            'parameter_options': {
                'keypoint_mode': {
                    'type': 'select',
                    'options': [
                        {'value': 'all', 'label': 'All Keypoints'},
                        {'value': 'custom', 'label': 'Custom List'}
                    ]
                }
            }
        },
        'AddPadding': {
            'name': 'Add Padding',
            'category': 'Detection',
            'description': 'Add padding to all detection bounding boxes',
            'required_inputs': ['detections'],
            'optional_inputs': [],
            'parameters': {
                'padding_mode': 'uniform',
                'padding': 0.1,
                'padding_left': 0.1,
                'padding_right': 0.1,
                'padding_top': 0.1,
                'padding_bottom': 0.1,
                'reference': 'shorter',
                'track_metadata': True
            },
            'parameter_options': {
                'padding_mode': {
                    'type': 'select',
                    'options': [
                        {'value': 'uniform', 'label': 'Uniform Padding'},
                        {'value': 'custom', 'label': 'Custom Per Side'}
                    ]
                },
                'reference': {
                    'type': 'select',
                    'options': [
                        {'value': 'shorter', 'label': 'Shorter Side'},
                        {'value': 'longer', 'label': 'Longer Side'},
                        {'value': 'width', 'label': 'Width'},
                        {'value': 'height', 'label': 'Height'}
                    ]
                }
            }
        }
    }

    def _get_default_parameters(self, tool_type: str) -> Dict[str, any]:
        """Get default parameters for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('parameters', {})

    def _get_tool_category(self, tool_type: str) -> str:
        """Get category for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('category', 'Other')


    def _get_tool_description(self, tool_type: str) -> str:
        """Get description for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('description', 'Process image')

    def _get_tool_name(self, tool_type: str) -> str:
        """Get display name for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('name', tool_type)

    def _get_parameter_options(self, tool_type: str) -> Dict[str, any]:
        """Get parameter options for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('parameter_options', {})

    def get_required_inputs(self, tool_type: str) -> List[str]:
        """Get required inputs for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('required_inputs', [])

    def get_optional_inputs(self, tool_type: str) -> List[str]:
        """Get optional inputs for a tool type"""
        return self.TOOL_METADATA.get(tool_type, {}).get('optional_inputs', [])

    def get_all_tool_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about all tool types"""
        return {tool_type: self.get_tool_info(tool_type) for tool_type in self._tools.keys()}
        # """Get information about all tool types (excludes MediaInput - always on canvas)"""
        # return {
        #     tool_type: self.get_tool_info(tool_type)
        #     for tool_type in self._tools.keys()
        #     if tool_type != 'MediaInput'  # MediaInput is locked on canvas, not in palette
        # }


# Global registry instance
registry = ToolRegistry()