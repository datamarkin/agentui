"""
AgentUI Command Line Interface

Provides convenient commands for running the AgentUI server and executing workflows.
"""

import argparse
import json
import sys
from . import __version__
from .core.workflow import WorkflowEngine
from .core.registry import registry


def cmd_start(args):
    """Start the AgentUI server"""
    from .api.server import main as server_main

    print(f"🚀 Starting AgentUI Server v{__version__}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    if args.reload:
        print("   Hot reload: enabled")
    print()
    print(f"   Open http://{args.host}:{args.port} in your browser")
    print()

    server_main(host=args.host, port=args.port, reload=args.reload)


def cmd_version(args):
    """Show version information"""
    print(f"AgentUI v{__version__}")


def cmd_info(args):
    """Show system and dependency information"""
    import sys
    import platform

    print(f"AgentUI v{__version__}")
    print()
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print()

    # Check dependencies
    print("Dependencies:")
    dependencies = [
        'fastapi',
        'uvicorn',
        'pillow',
        'numpy',
        'pixelflow',
        'mozo'
    ]

    for dep in dependencies:
        try:
            mod = __import__(dep)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {dep}: {version}")
        except ImportError:
            print(f"  ✗ {dep}: not installed")

    print()
    print(f"Available Tools: {len(registry.get_all_types())}")


def cmd_run(args):
    """Execute a workflow from JSON file"""
    try:
        # Load workflow from JSON file
        with open(args.workflow_file, 'r') as f:
            workflow_data = json.load(f)

        # Create and execute workflow engine
        workflow_json = json.dumps(workflow_data)
        workflow = WorkflowEngine.from_json(workflow_json, registry.get_all_types())

        print(f"Executing workflow with {len(workflow.tools)} tools...")
        results = workflow.execute()

        print("✅ Workflow executed successfully!")

        # Print results
        for tool_id, result in results.items():
            print(f"\n{result['type']} ({tool_id}):")
            for output_name, output_value in result['outputs'].items():
                if hasattr(output_value, '__class__'):
                    print(f"  {output_name}: {output_value.__class__.__name__}")
                else:
                    print(f"  {output_name}: {output_value}")

        # Save results if output file specified
        if args.output:
            # Convert PIL Images to strings for JSON serialization
            json_results = {}
            for tool_id, result in results.items():
                json_results[tool_id] = {
                    'type': result['type'],
                    'outputs': {}
                }
                for output_name, output_value in result['outputs'].items():
                    if hasattr(output_value, 'save'):  # PIL Image
                        json_results[tool_id]['outputs'][output_name] = str(output_value)
                    else:
                        json_results[tool_id]['outputs'][output_name] = output_value

            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    except FileNotFoundError:
        print(f"❌ Error: Workflow file '{args.workflow_file}' not found")
        return 1
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in workflow file")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='agentui',
        description='AgentUI - Visual workflow builder for computer vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agentui start                    Start the server on default port 8000
  agentui start --port 3000        Start the server on port 3000
  agentui start --reload           Start with hot reload (development)
  agentui version                  Show version information
  agentui info                     Show system and dependency info
  agentui run workflow.json        Execute a workflow from file

For more information, visit: https://github.com/datamarkin/agentui
        """
    )

    parser.add_argument('--version', action='version', version=f'AgentUI {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the AgentUI server')
    start_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    start_parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    start_parser.add_argument('--reload', action='store_true', help='Enable auto-reload (development mode)')
    start_parser.set_defaults(func=cmd_start)

    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    version_parser.set_defaults(func=cmd_version)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and dependency information')
    info_parser.set_defaults(func=cmd_info)

    # Run command (execute workflow)
    run_parser = subparsers.add_parser('run', help='Execute a workflow from JSON file')
    run_parser.add_argument('workflow_file', help='Path to workflow JSON file')
    run_parser.add_argument('--output', help='Output file for results (JSON)')
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Execute the command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
