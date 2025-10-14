<script>
    import { onMount, setContext } from 'svelte';
    import { writable } from 'svelte/store';
    import { SvelteFlow, Controls, ControlButton, Background, MiniMap, SvelteFlowProvider } from '@xyflow/svelte';

    import Toolbar from './lib/Toolbar.svelte';
    import CustomNode from './lib/CustomNode.svelte';
    import FlowDropZone from './lib/FlowDropZone.svelte';
    import DrawerSidebar from './lib/DrawerSidebar.svelte';
    import NodePalettePanel from './lib/NodePalettePanel.svelte';
    import { generateNodeClasses } from './lib/utils.js';
    import { openSidebar, closeSidebar, pendingConnection, clearPendingConnection } from './lib/stores.js';

    // Initialize with MediaInput node
    let nodes = writable([{
        id: 'MediaInput-initial',
        type: 'default',
        position: { x: 100, y: 100 },
        data: {
            label: 'Media Input',
            nodeType: 'MediaInput',
            parameters: {}
        },
        class: 'node-category-input'
    }]);
    let edges = writable([]);
    let selectedNode = writable(null);
    let availableNodes = writable([]);
    let executionResults = writable(null);
    let isExecuting = writable(false);

    let svelteFlowInstance;

    // Set context for child components - must be during initialization
    setContext('availableNodes', availableNodes);

    // Define custom node types for SvelteFlow
    const nodeTypes = {
        default: CustomNode
    };

    onMount(async () => {
        console.log('App onMount started');
        // Fetch available tool types
        try {
            const response = await fetch('/api/tools');
            const toolTypes = await response.json();
            console.log('Tool types fetched:', toolTypes);
            availableNodes.set(toolTypes);

            console.log('App initialized successfully');
        } catch (error) {
            console.error('Failed to fetch tool types:', error);
        }
    });

    function initializeCanvas() {
        console.log('initializeCanvas called');
        // Reset to MediaInput node
        const mediaInputNode = {
            id: 'MediaInput-initial',
            type: 'default',
            position: { x: 100, y: 100 },
            data: {
                label: 'Media Input',
                nodeType: 'MediaInput',
                parameters: {}
            },
            class: 'node-category-input'
        };
        nodes.set([mediaInputNode]);
        edges.set([]);
        console.log('Canvas initialized with MediaInput node');
    }

    function handleNodeDrop(event) {
        const { nodeType, position } = event.detail;

        const category = getNodeCategory(nodeType);
        const newNode = {
            id: `${nodeType}-${Date.now()}`,
            type: 'default',
            position,
            data: {
                label: nodeType,
                nodeType: nodeType,
                parameters: getDefaultParameters(nodeType)
            },
            class: generateNodeClasses(nodeType, category),
            origin: [0.5, 0.0]
        };

        nodes.update(n => [...n, newNode]);

        // Check if there's a pending connection to auto-connect
        const pending = $pendingConnection;
        if (pending) {
            // Create auto-connection between pending handle and new node
            createAutoConnection(pending, newNode);
            clearPendingConnection();
        }
    }

    function getDefaultParameters(nodeType) {
        const currentNodes = $availableNodes || {};
        return currentNodes[nodeType]?.parameters || {};
    }

    function getNodeCategory(nodeType) {
        const currentNodes = $availableNodes || {};
        return currentNodes[nodeType]?.category || 'Other';
    }

    function onNodeClick(event) {
        console.log('Node clicked:', event.detail.node.id);
        const node = event.detail.node;

        selectedNode.set(node);
        // Auto-open properties panel when a node is selected
        openSidebar('properties');
    }

    function onPaneClick() {
        // Close drawer when clicking on empty canvas
        closeSidebar();
        // Clear any pending connections
        clearPendingConnection();
    }

    function createAutoConnection(pending, newNode) {
        const { nodeId: pendingNodeId, handleId: pendingHandleId, handleType: pendingHandleType } = pending;

        let sourceNode, targetNode, sourceHandle, targetHandle;

        if (pendingHandleType === 'source') {
            // Pending node has output handle, new node should provide input
            sourceNode = pendingNodeId;
            targetNode = newNode.id;
            sourceHandle = pendingHandleId;
            targetHandle = getDefaultInput(newNode.data.nodeType);
        } else {
            // Pending node has input handle, new node should provide output
            sourceNode = newNode.id;
            targetNode = pendingNodeId;
            sourceHandle = getDefaultOutput(newNode.data.nodeType);
            targetHandle = pendingHandleId;
        }

        // Validate connection before creating
        if (sourceHandle && targetHandle) {
            const sourceNodeType = pendingHandleType === 'source' ?
                $nodes.find(n => n.id === pendingNodeId)?.data?.nodeType :
                newNode.data.nodeType;
            const targetNodeType = pendingHandleType === 'target' ?
                $nodes.find(n => n.id === pendingNodeId)?.data?.nodeType :
                newNode.data.nodeType;

            if (isValidConnection(sourceNodeType, sourceHandle, targetNodeType, targetHandle)) {
                const newEdge = {
                    id: `edge-${Date.now()}`,
                    source: sourceNode,
                    target: targetNode,
                    sourceHandle: sourceHandle,
                    targetHandle: targetHandle
                };

                edges.update(edges => [...edges, newEdge]);
                console.log('Auto-connection created:', newEdge);
            } else {
                console.warn('Auto-connection validation failed');
            }
        } else {
            console.warn('Could not determine handles for auto-connection');
        }
    }

    function onConnect(event) {
        const connection = event.detail.connection;

        // Get node types to determine correct handles
        const sourceNode = $nodes.find(n => n.id === connection.source);
        const targetNode = $nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) {
            console.warn('Could not find source or target node');
            return;
        }

        const sourceNodeType = sourceNode.data?.nodeType;
        const targetNodeType = targetNode.data?.nodeType;

        const sourceHandle = getDefaultOutput(sourceNodeType);
        const targetHandle = getDefaultInput(targetNodeType);

        // Validate connection types
        if (!isValidConnection(sourceNodeType, sourceHandle, targetNodeType, targetHandle)) {
            console.warn(`Invalid connection: ${sourceNodeType}.${sourceHandle} -> ${targetNodeType}.${targetHandle}`);
            alert(`Cannot connect ${sourceNodeType} output to ${targetNodeType} input: incompatible types`);
            return;
        }

        const newEdge = {
            id: `edge-${Date.now()}`,
            source: connection.source,
            target: connection.target,
            sourceHandle: sourceHandle,
            targetHandle: targetHandle
        };

        edges.update(edges => [...edges, newEdge]);
    }

    function isValidConnection(sourceNodeType, sourceHandle, targetNodeType, targetHandle) {
        const sourceNodeInfo = $availableNodes?.[sourceNodeType];
        const targetNodeInfo = $availableNodes?.[targetNodeType];

        if (!sourceNodeInfo || !targetNodeInfo) return false;
        if (!sourceHandle || !targetHandle) return false;

        const sourceOutputType = sourceNodeInfo.outputs?.[sourceHandle];
        const targetInputType = targetNodeInfo.inputs?.[targetHandle];

        if (!sourceOutputType || !targetInputType) return false;

        // Types must match exactly
        return sourceOutputType === targetInputType;
    }

    function getDefaultOutput(nodeType) {
        const nodeInfo = $availableNodes?.[nodeType];
        if (!nodeInfo?.outputs) return 'image';

        // Get first output handle name
        const outputHandles = Object.keys(nodeInfo.outputs);
        return outputHandles[0] || 'image';
    }

    function getDefaultInput(nodeType) {
        const nodeInfo = $availableNodes?.[nodeType];
        if (!nodeInfo?.inputs) return null;

        // Get first input handle name
        const inputHandles = Object.keys(nodeInfo.inputs);
        return inputHandles[0] || null;
    }

    function updateNodeParameters(nodeId, parameters) {
        nodes.update(n =>
            n.map(node =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, parameters } }
                    : node
            )
        );
    }

    async function executeWorkflow() {
        isExecuting.set(true);
        executionResults.set(null);

        try {
            const workflow = {
                nodes: $nodes,
                edges: $edges
            };

            const response = await fetch('/api/workflow/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ workflow })
            });

            const result = await response.json();
            executionResults.set(result);

            // Add 'node-has-output' class to nodes that have outputs
            if (result.success && result.results) {
                nodes.update(n => n.map(node => {
                    const hasOutput = result.results[node.id];
                    const baseClass = node.class.replace(/\s*node-has-output/g, '');
                    return {
                        ...node,
                        class: hasOutput ? `${baseClass} node-has-output` : baseClass
                    };
                }));
            }
        } catch (error) {
            executionResults.set({
                success: false,
                error: error.message
            });
        } finally {
            isExecuting.set(false);
        }
    }

    function exportWorkflow() {
        const workflow = {
            nodes: $nodes,
            edges: $edges
        };

        const dataStr = JSON.stringify(workflow, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = 'workflow.json';
        link.click();

        URL.revokeObjectURL(url);
    }

    function importWorkflow(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const workflow = JSON.parse(e.target.result);
                nodes.set(workflow.nodes || []);
                edges.set(workflow.edges || []);
            } catch (error) {
                alert('Invalid workflow file');
            }
        };
        reader.readAsText(file);
    }

    function clearWorkflow() {
        initializeCanvas();
        selectedNode.set(null);
        executionResults.set(null);
    }
</script>

<SvelteFlowProvider>
            <Toolbar
                    {executeWorkflow}
                    {exportWorkflow}
                    {importWorkflow}
                    {clearWorkflow}
                    isExecuting={$isExecuting}
            />

<!-- Left-side node palette (always visible) -->
<NodePalettePanel {availableNodes} />

<!-- Main app layout without fixed sidebar -->
<div class="main-content">
    <div id="studio">
        <FlowDropZone on:nodedrop={handleNodeDrop}>
            <SvelteFlow
                    {nodes}
                    {edges}
                    {nodeTypes}
                    proOptions={{ hideAttribution: true }}
                    bind:this={svelteFlowInstance}
                    on:nodeclick={onNodeClick}
                    on:paneclick={onPaneClick}
                    on:connect={onConnect}
                    maxZoom={1}
                    fitView
                    fitViewOptions={{
                      maxZoom: 1,      // Prevents zooming in beyond 100%
                      padding: 0.2     // Adds some padding around nodes
                    }}
            >
<!--                <Controls />-->
                <Background variant="dots" />
            </SvelteFlow>
        </FlowDropZone>
    </div>
</div>

<!-- Drawer sidebar for context-aware panels -->
<DrawerSidebar
    {availableNodes}
    selectedNode={$selectedNode}
    {updateNodeParameters}
    executionResults={$executionResults}
/>

</SvelteFlowProvider>