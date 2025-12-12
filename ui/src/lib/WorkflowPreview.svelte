<script>
  import { writable } from 'svelte/store';
  import { SvelteFlow, SvelteFlowProvider } from '@xyflow/svelte';

  export let nodes = [];
  export let edges = [];
  export let height = '150px';

  // Convert arrays to stores for SvelteFlow
  $: nodesStore = writable(nodes);
  $: edgesStore = writable(edges);
</script>

<div class="workflow-preview" style="height: {height};">
  <SvelteFlowProvider>
    <SvelteFlow
      nodes={nodesStore}
      edges={edgesStore}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      panOnDrag={false}
      zoomOnScroll={false}
      zoomOnPinch={false}
      zoomOnDoubleClick={false}
      preventScrolling={true}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.1}
      maxZoom={1}
    />
  </SvelteFlowProvider>
</div>

<style>
  .workflow-preview {
    width: 100%;
    border-radius: 4px;
    overflow: hidden;
    background: #f5f5f5;
    pointer-events: none;
  }

  .workflow-preview :global(.svelte-flow) {
    background: #fafafa;
  }

  .workflow-preview :global(.svelte-flow__node) {
    font-size: 8px;
    padding: 4px 8px;
  }

  .workflow-preview :global(.svelte-flow__attribution) {
    display: none;
  }
</style>
