<script>
  import { isExploreModalOpen, closeExploreModal, workflowTemplates } from './stores.js';
  import WorkflowPreview from './WorkflowPreview.svelte';

  let isLoading = false;
  let error = null;
  let hasFetched = false;

  // Fetch workflows when modal opens (only once)
  $: if ($isExploreModalOpen && !hasFetched) {
    fetchWorkflows();
  }

  async function fetchWorkflows() {
    if (isLoading) return;
    hasFetched = true;
    isLoading = true;
    error = null;
    try {
      const response = await fetch('/api/workflows');
      if (!response.ok) {
        throw new Error('Failed to fetch workflows');
      }
      const data = await response.json();
      console.log('Workflows API response:', data);
      // Handle both { data: [...] } and direct array formats
      const workflows = Array.isArray(data) ? data : (data.data || []);
      console.log('Parsed workflows:', workflows);
      workflowTemplates.set(workflows);
    } catch (e) {
      error = e.message;
      hasFetched = false; // Allow retry on error
      console.error('Error fetching workflows:', e);
    } finally {
      isLoading = false;
    }
  }

  function selectWorkflow(workflowId) {
    // Navigate to workflow page
    window.location.href = `/workflows/${workflowId}`;
  }

  function handleKeydown(event) {
    if (event.key === 'Escape' && $isExploreModalOpen) {
      closeExploreModal();
    }
  }

  function handleBackgroundClick() {
    closeExploreModal();
  }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if $isExploreModalOpen}
<div class="modal is-active">
  <div class="modal-background" on:click={handleBackgroundClick} on:keydown={handleKeydown} role="button" tabindex="-1" aria-label="Close modal"></div>
  <div class="modal-card explore-modal-card">
    <header class="modal-card-head">
      <p class="modal-card-title">Explore Workflows</p>
      <button class="delete" aria-label="close" on:click={closeExploreModal}></button>
    </header>
    <section class="modal-card-body">
      <!-- Loading state -->
      {#if isLoading}
        <div class="has-text-centered py-6">
          <p>Loading workflows...</p>
        </div>
      {:else if error}
        <div class="notification is-danger is-light">
          <p>Error: {error}</p>
          <button class="button is-small mt-2" on:click={fetchWorkflows}>Retry</button>
        </div>
      {:else if $workflowTemplates.length === 0}
        <div class="has-text-centered py-6">
          <p class="has-text-grey">No workflows found</p>
        </div>
      {:else}
        <!-- Grid of workflow cards -->
        <div class="columns is-multiline">
          {#each $workflowTemplates as workflow (workflow.id)}
            <div class="column is-4">
              <div class="card workflow-card" on:click={() => selectWorkflow(workflow.id)} on:keydown={(e) => e.key === 'Enter' && selectWorkflow(workflow.id)} role="button" tabindex="0">
                <div class="card-image">
                  {#if workflow.code && workflow.code.nodes}
                    <WorkflowPreview
                      nodes={workflow.code.nodes}
                      edges={workflow.code.edges || []}
                    />
                  {:else}
                    <div class="preview-placeholder">
                      <span class="has-text-grey">No preview</span>
                    </div>
                  {/if}
                </div>
                <div class="card-content">
                  <p class="title is-6">{workflow.name || 'Untitled'}</p>
                  <p class="subtitle is-7 has-text-grey">{workflow.description || 'No description'}</p>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </section>
  </div>
</div>
{/if}

<style>
  .explore-modal-card {
    width: 90%;
    max-width: 1200px;
    max-height: 85vh;
  }

  .modal-card-body {
    overflow-y: auto;
    max-height: calc(85vh - 120px);
  }

  .workflow-card {
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .workflow-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .workflow-card .card-content {
    padding: 0.75rem;
  }

  .workflow-card .title {
    margin-bottom: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .workflow-card .subtitle {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .preview-placeholder {
    height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f5f5f5;
  }
</style>
