from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, Optional
import base64
import io
import json
import os

from ..core.workflow import WorkflowEngine
from ..core.registry import registry

# Setup Jinja2 templates
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
templates = Jinja2Templates(directory=static_dir)


def serialize_tool_result(tool_id: str, result: dict) -> dict:
    """Convert tool result to JSON-serializable format."""
    serialized = {
        'tool_id': tool_id,
        'type': result['type'],
        'outputs': {},
        'is_terminal': result.get('is_terminal', False)
    }

    for output_name, output_value in result['outputs'].items():
        if hasattr(output_value, 'save'):  # PIL Image
            buffer = io.BytesIO()
            output_value.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            serialized['outputs'][output_name] = f"data:image/jpeg;base64,{img_str}"
        elif hasattr(output_value, 'to_dict'):  # Detections or similar
            serialized['outputs'][output_name] = output_value.to_dict()
        else:
            serialized['outputs'][output_name] = output_value

    return serialized


app = FastAPI(title="AgentUI Workflow API", version="1.0.0")

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WorkflowRequest(BaseModel):
    workflow: Dict[str, Any]


class ExecuteResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = None
    error: str = None


@app.get("/api/")
async def root():
    return {"message": "AgentUI Workflow API"}


@app.get("/api/tools")
async def get_available_tools():
    """Get all available tool types and their information"""
    return registry.get_all_tool_info()


@app.get("/api/workflows")
async def get_workflows():
    """Proxy to fetch workflow templates from external API"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.datamarkin.com/items/workflows")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Proxy to fetch a single workflow from external API"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.datamarkin.com/items/workflows/{workflow_id}")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflow/execute", response_model=ExecuteResponse)
async def execute_workflow(request: WorkflowRequest):
    """Execute a workflow from JSON definition"""
    try:
        # Create workflow engine from JSON
        workflow_json = json.dumps(request.workflow)
        workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

        # Execute workflow
        results = workflow.execute()

        # Convert PIL Images to base64 for JSON serialization
        serializable_results = {}
        for tool_id, result in results.items():
            serializable_results[tool_id] = {
                'type': result['type'],
                'outputs': {}
            }
            for output_name, output_value in result['outputs'].items():
                if hasattr(output_value, 'save'):  # PIL Image
                    # Convert to base64
                    import io
                    buffer = io.BytesIO()
                    output_value.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    serializable_results[tool_id]['outputs'][output_name] = f"data:image/jpeg;base64,{img_str}"
                elif hasattr(output_value, 'to_dict'):  # PixelFlow Detections or similar
                    serializable_results[tool_id]['outputs'][output_name] = output_value.to_dict()
                else:
                    serializable_results[tool_id]['outputs'][output_name] = output_value

        return ExecuteResponse(success=True, results=serializable_results)

    except Exception as e:
        return ExecuteResponse(success=False, error=str(e))


@app.post("/api/workflow/stream")
async def stream_workflow(request: WorkflowRequest):
    """Execute workflow with SSE streaming, yielding results as each tool completes."""

    def event_generator():
        try:
            workflow_json = json.dumps(request.workflow)
            workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

            for tool_id, status, result in workflow.execute_streaming():
                if status == "running":
                    event = {"tool_id": tool_id, "status": "running"}
                elif status == "completed":
                    serialized = serialize_tool_result(tool_id, result)
                    event = {"tool_id": tool_id, "status": "completed", "result": serialized}
                else:  # error
                    event = {"tool_id": tool_id, "status": "error", "error": result.get("error", "Unknown error")}

                yield f"data: {json.dumps(event)}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/workflow/validate")
async def validate_workflow(request: WorkflowRequest):
    """Validate a workflow without executing it"""
    try:
        workflow_json = json.dumps(request.workflow)
        workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

        # Try to get execution order (this validates the DAG)
        execution_order = workflow.get_execution_order()

        return {"valid": True, "execution_order": execution_order}

    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and return base64 encoded data"""
    try:
        contents = await file.read()
        base64_data = base64.b64encode(contents).decode('utf-8')

        return {
            "filename": file.filename,
            "data": base64_data,
            "content_type": file.content_type
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve main app via Jinja2 template (for APP_CONFIG injection)
@app.get("/", response_class=HTMLResponse)
@app.get("/workflows/{workflow_id}", response_class=HTMLResponse)
async def serve_app(request: Request, workflow_id: Optional[str] = None):
    """Serve the main app with injected config"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "deployment_mode": "local",
        "user": None,  # Placeholder - will be set by parent app in deployed mode
        "workflow_id": workflow_id
    })


# Serve logo.png from static directory
@app.get("/logo.png")
async def serve_logo():
    from fastapi.responses import FileResponse
    logo_path = os.path.join(static_dir, "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


# Serve static assets (CSS, JS, images) - MUST be mounted last
if os.path.exists(static_dir):
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


def main(host="0.0.0.0", port=8000, reload=False):
    """
    Start the AgentUI server

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()