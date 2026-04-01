__version__ = "0.2.0"
__author__ = "Datamarkin"

from .core.registry import registry
from .core.tool import Tool, Connection
from .workflow_api import Workflow

_header_template: str | None = None
_header_context_fn = None


def set_header(template_name: str, context_fn=None) -> None:
    """Inject a custom header into AgentUI's page and hide the default toolbar.

    template_name: Jinja2 template name resolved from the host app's templates/.
    context_fn:    optional callable() -> dict, called on each request for dynamic data
                   (e.g. a list of saved workflows for a Load dropdown).

    Example::

        import agentui

        agentui.set_header("agentui_header.html",
                           context_fn=lambda: {"saved_workflows": my_app.list_workflows()})
    """
    global _header_template, _header_context_fn
    _header_template = template_name
    _header_context_fn = context_fn


def register_tool(tool_class, metadata: dict = None) -> None:
    """Register an external tool with the global registry.

    ``metadata`` is the same shape as entries in ``ToolRegistry.TOOL_METADATA``::

        agentui.register_tool(MyTool, metadata={
            "name": "My Tool",
            "category": "Custom",
            "description": "Does something useful",
            "parameters": {"threshold": 0.5},
        })

    For dynamic parameter options (e.g. a dropdown populated from a database),
    override ``get_parameter_options()`` on the tool class instead of putting
    options in metadata — they will be fetched fresh on every ``/api/tools`` call.
    """
    registry.register_external(tool_class, metadata or {})


__all__ = ['Workflow', 'registry', 'Tool', 'Connection', 'set_header', 'register_tool']