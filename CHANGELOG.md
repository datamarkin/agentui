# Changelog

All notable changes to AgentUI are documented here.

## [0.2.0] - 2026-04-02

### New Features

#### Run / Edit Modes
- Toolbar now has **Edit** and **Run** toggle buttons, switching the canvas between workflow design and execution contexts.
- **Edit mode** — full canvas access: add/connect tools, configure parameters, export/import workflows, run from the canvas.
- **Run mode** — a clean `RunnerView` panel for end-users: upload an image, execute the loaded workflow, and view results — no canvas clutter.

#### Bridge Mode (Embedding Support)
- `agentui.set_header(template_name, context_fn=None)` — inject a custom Jinja2 header into AgentUI's page and hide the default toolbar. Designed for embedding AgentUI inside a Flask application with your own navigation shell.
- `agentui.register_tool(tool_class, metadata)` — register external tool classes at runtime without modifying AgentUI source.
- JS Bridge API: host pages can communicate with the embedded UI via `window.agentui.*` and listen to events (`workflow:loaded`, `workflow:executed`, `workflow:exported`) via `window.addEventListener`.

#### External Tool Registration
- New public API `agentui.register_tool()` allows third-party tools to plug into the palette without forking the library.

#### Explore Workflows Modal
- "Explore" button in the toolbar opens a modal for browsing and loading saved workflows from the server.
- Workflow preview component shows a snapshot before loading.
- App settings support user-configurable options surfaced through this modal.

#### Streaming Workflow Execution
- Workflow execution streams real-time progress updates to the UI.
- Nodes change visual state (pending → running → complete / error) as each tool executes.
- Results render incrementally rather than waiting for the full pipeline to finish.

### UI Improvements

#### Toolbar
- Added Edit / Run mode toggle (button group, highlights active mode).
- Edit mode: shows Run Workflow, Export, Import, and Explore buttons.
- Run mode: shows a single Execute button wired to the RunnerView, disabled until an image is uploaded.
- Toolbar can be hidden entirely when embedding AgentUI via `set_header` / `appConfig.hideToolbar`.

#### Port Type Color System
- Replaced ad-hoc port colors with a semantic, port-type-based CSS system (`--port-image`, `--port-detections`, etc.) for consistent, scalable styling across all tools.

#### CSS
- Significant expansion of `custom.css`: styles for RunnerView, mode toggle, execution state indicators on nodes, and embedding layout support.

### Backend Changes

#### Flask Migration
- Migrated the API server from **FastAPI** to **Flask** (`flask>=3.0.0`, `flask-cors>=4.0.0`).
- All REST routes and workflow execution endpoints updated accordingly.
- Removed FastAPI / Uvicorn dependencies.

#### Hidden Parameters
- Tool registry and properties panel now support `hidden: true` on parameters — used for internal parameters that should not be exposed in the UI.

#### API Additions (Detection Tools)
- Detection model tools expose result data through the REST API for programmatic access.

### Developer Experience

#### Python API Examples
- Added `examples/batch_detection.py` — end-to-end single and batch image processing using `Workflow.load()` and `workflow.run()`.
- Added `examples/README.md` — complete Python API reference with input types, output structure, and batching patterns.

#### Flask Integration Guide
- README expanded with a Flask embedding walkthrough, JS bridge API reference, runner mode details, and event listening examples.

---

## [0.1.1] - 2025-12-05

- Version bump and CLI stabilization.
- Refactored CLI to include `start`, `run`, `version`, and `info` subcommands.
- `MediaInput` nodes made non-deletable on the canvas.
- Improved workflow error messages with detailed `RuntimeError` descriptions.

## [0.1.0] - 2025-11-xx

- Initial release.
- Svelte Flow canvas with drag-and-drop tool palette.
- Python `Workflow` API (`Workflow.load`, `workflow.run`, auto-batching).
- Tool system: ObjectDetection, InstanceSegmentation, DepthEstimation, Florence2, VQA, OCR, StabilityInpainting, DatamarkinDetection.
- Transform tools: Rotate, Flip, Crop (image-only and detection-aware variants), CLAHE, AutoContrast, GammaCorrection, NormalizeImage.
- Annotation tools: DrawBoundingBoxes, AddLabels, DrawMasks, DrawPolygons, DrawKeypoints, DrawKeypointSkeleton.
- Semantic port-type system (`PortType.IMAGE`, `PortType.DETECTIONS`, etc.).
- FastAPI backend serving the UI and REST API from a single process.
