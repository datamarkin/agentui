import json
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from .node import Node, Connection


class Workflow:
    """Manages and executes a workflow of connected nodes"""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []

    def add_node(self, node: Node):
        """Add a node to the workflow"""
        self.nodes[node.id] = node

    def add_connection(self, connection: Connection):
        """Add a connection between nodes"""
        self.connections.append(connection)

    def get_execution_order(self) -> List[str]:
        """Get nodes in topologically sorted order for execution"""
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all nodes with 0 in-degree
        for node_id in self.nodes:
            in_degree[node_id] = 0

        # Build graph and calculate in-degrees
        for conn in self.connections:
            graph[conn.source_id].append(conn.target_id)
            in_degree[conn.target_id] += 1

        # Topological sort using Kahn's algorithm
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Workflow contains cycles")

        return result

    def get_terminal_nodes(self) -> List[str]:
        """Get nodes that have no outgoing connections (terminal nodes)"""
        nodes_with_outgoing = set()
        for conn in self.connections:
            nodes_with_outgoing.add(conn.source_id)

        return [node_id for node_id in self.nodes.keys() if node_id not in nodes_with_outgoing]

    def execute(self) -> Dict[str, Any]:
        """Execute the workflow and return results from terminal nodes"""
        execution_order = self.get_execution_order()
        all_results = {}
        terminal_nodes = self.get_terminal_nodes()

        for node_id in execution_order:
            node = self.nodes[node_id]

            # Set inputs from connected nodes
            for conn in self.connections:
                if conn.target_id == node_id:
                    source_node = self.nodes[conn.source_id]
                    output = source_node.get_output(conn.source_output)
                    if output:
                        node.set_input(conn.target_input, output.data, output.data_type)

            # Execute the node with auto-batching if available, otherwise use regular process
            if hasattr(node, 'process_with_auto_batching'):
                success = node.process_with_auto_batching()
            else:
                success = node.process()

            if not success:
                raise RuntimeError(f"Node {node_id} ({node.node_type}) failed to execute")

            all_results[node_id] = {
                'type': node.node_type,
                'outputs': {name: output.data for name, output in node.outputs.items()},
                'is_terminal': node_id in terminal_nodes
            }

        # Return all results for UI inspection
        return all_results

    def to_json(self) -> str:
        """Export workflow to JSON (Svelte Flow compatible format)"""
        nodes_data = []
        edges_data = []

        for node in self.nodes.values():
            nodes_data.append({
                'id': node.id,
                'type': node.node_type,
                'data': {
                    'label': node.node_type,
                    'parameters': node.parameters
                },
                'position': {'x': 0, 'y': 0}  # Default position
            })

        for conn in self.connections:
            edges_data.append(conn.to_dict())

        return json.dumps({
            'nodes': nodes_data,
            'edges': edges_data
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str, node_registry: Dict[str, type]) -> 'Workflow':
        """Create workflow from JSON"""
        data = json.loads(json_str)
        workflow = cls()

        # Create nodes
        for node_data in data['nodes']:
            # Check if this is a Svelte Flow format with nodeType in data
            if 'data' in node_data and 'nodeType' in node_data['data']:
                node_type = node_data['data']['nodeType']
                parameters = node_data['data'].get('parameters', {})
            else:
                node_type = node_data['type']
                parameters = node_data.get('parameters', {})

            if node_type not in node_registry:
                raise ValueError(f"Unknown node type: {node_type}")

            node_class = node_registry[node_type]
            node = node_class(node_id=node_data['id'], **parameters)
            workflow.add_node(node)

        # Create connections
        for edge_data in data['edges']:
            connection = Connection.from_dict(edge_data)
            workflow.add_connection(connection)

        return workflow