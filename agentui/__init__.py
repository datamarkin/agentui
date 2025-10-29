__version__ = "0.1.1"
__author__ = "Datamarkin"

from .core.registry import registry
from .core.tool import Tool, Connection
from .workflow_api import Workflow

__all__ = ['Workflow', 'registry', 'Tool', 'Connection']

def hello():
    return "Welcome to AgentUI!"