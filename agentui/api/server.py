from flask import Blueprint, Flask, jsonify, request, render_template, send_from_directory, Response
from flask_cors import CORS
from typing import Dict, Any
import base64
import io
import json
import os

from ..core.workflow import WorkflowEngine
from ..core.registry import registry

# Setup paths
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
assets_dir = os.path.join(static_dir, "assets")

# Blueprint definition
bp = Blueprint(
    'agentui', __name__,
    template_folder=static_dir,
    static_folder=assets_dir,
    static_url_path='/assets'
)


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


@bp.route("/api/")
def root():
    return jsonify({"message": "AgentUI Workflow API"})


@bp.route("/api/tools")
def get_available_tools():
    """Get all available tool types and their information"""
    return jsonify(registry.get_all_tool_info())


@bp.route("/api/workflows")
def get_workflows():
    """Proxy to fetch workflow templates from external API"""
    import requests as req
    try:
        response = req.get("https://api.datamarkin.com/items/workflows")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/workflows/<workflow_id>")
def get_workflow(workflow_id):
    """Proxy to fetch a single workflow from external API"""
    import requests as req
    try:
        response = req.get(f"https://api.datamarkin.com/items/workflows/{workflow_id}")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/workflow/execute", methods=["POST"])
def execute_workflow():
    """Execute a workflow from JSON definition"""
    try:
        data = request.get_json()
        workflow_data = data.get("workflow", {})

        workflow_json = json.dumps(workflow_data)
        workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

        results = workflow.execute()

        serializable_results = {}
        for tool_id, result in results.items():
            serializable_results[tool_id] = {
                'type': result['type'],
                'outputs': {}
            }
            for output_name, output_value in result['outputs'].items():
                if hasattr(output_value, 'save'):  # PIL Image
                    buffer = io.BytesIO()
                    output_value.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    serializable_results[tool_id]['outputs'][output_name] = f"data:image/jpeg;base64,{img_str}"
                elif hasattr(output_value, 'to_dict'):  # PixelFlow Detections or similar
                    serializable_results[tool_id]['outputs'][output_name] = output_value.to_dict()
                else:
                    serializable_results[tool_id]['outputs'][output_name] = output_value

        return jsonify({"success": True, "results": serializable_results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@bp.route("/api/workflow/stream", methods=["POST"])
def stream_workflow():
    """Execute workflow with SSE streaming, yielding results as each tool completes."""
    data = request.get_json()
    workflow_data = data.get("workflow", {})

    def event_generator():
        try:
            workflow_json = json.dumps(workflow_data)
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

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@bp.route("/api/workflow/validate", methods=["POST"])
def validate_workflow():
    """Validate a workflow without executing it"""
    try:
        data = request.get_json()
        workflow_data = data.get("workflow", {})

        workflow_json = json.dumps(workflow_data)
        workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

        execution_order = workflow.get_execution_order()

        return jsonify({"valid": True, "execution_order": execution_order})

    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@bp.route("/api/upload/image", methods=["POST"])
def upload_image():
    """Upload an image and return base64 encoded data"""
    try:
        file = request.files["file"]
        contents = file.read()
        base64_data = base64.b64encode(contents).decode('utf-8')

        return jsonify({
            "filename": file.filename,
            "data": base64_data,
            "content_type": file.content_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/")
@bp.route("/workflows/<workflow_id>")
def serve_app(workflow_id=None):
    """Serve the main app with injected config"""
    api_base = request.script_root + bp.url_prefix if bp.url_prefix else ""
    return render_template("index.html",
        deployment_mode="local",
        user=None,
        workflow_id=workflow_id,
        api_base=api_base,
    )


@bp.route("/logo.png")
def serve_logo():
    logo_path = os.path.join(static_dir, "logo.png")
    if os.path.exists(logo_path):
        return send_from_directory(static_dir, "logo.png", mimetype="image/png")
    return jsonify({"error": "Logo not found"}), 404


def create_app():
    """Standalone Flask app factory"""
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(bp)
    return app


def main(host="0.0.0.0", port=8000, reload=False):
    """
    Start the AgentUI server

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    app = create_app()
    app.run(host=host, port=port, debug=reload)


if __name__ == "__main__":
    main()
