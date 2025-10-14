__version__ = "0.1.0"
__author__ = "Datamarkin"

from .core.workflow import Workflow
from .core.registry import registry
from .core.node import Node, Connection

__all__ = ['Workflow', 'registry', 'Node', 'Connection']

def hello():
    return "Welcome to AgentUI!"