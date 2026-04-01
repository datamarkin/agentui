<script>
  import { openExploreModal, appConfig, viewMode } from './stores.js';

  export let executeWorkflow;
  export let exportWorkflow;
  export let importWorkflow;
  export let isExecuting;
  export let runnerExecute = null;
  export let runnerCanExecute = false;

  let fileInput;

  function handleImportClick() {
    fileInput.click();
  }
</script>

{#if !$appConfig.hideToolbar}
<nav class="navbar is-fixed-top border-bottom" aria-label="main navigation">
  <div class="navbar-brand">
    <div class="navbar-item">
      <img alt="Datamarkin logo" src="/logo.png" width="120" height="24">
    </div>
    <div class="navbar-item">
      <div class="buttons has-addons mb-0">
        <button
          class="button is-small"
          class:is-dark={$viewMode === 'edit'}
          class:is-outlined={$viewMode !== 'edit'}
          on:click={() => viewMode.set('edit')}
        >
          Edit
        </button>
        <button
          class="button is-small"
          class:is-dark={$viewMode === 'run'}
          class:is-outlined={$viewMode !== 'run'}
          on:click={() => viewMode.set('run')}
        >
          Run
        </button>
      </div>
    </div>
  </div>

  <div class="navbar-menu">
    <div class="navbar-end">
      <div class="navbar-item">
        <div class="buttons">
          {#if $viewMode === 'edit'}
            <button class="button is-dark" on:click={executeWorkflow} disabled={isExecuting}>
              {isExecuting ? 'Running...' : 'Run Workflow'}
            </button>

            <button class="button is-dark" on:click={exportWorkflow}>
              Export
            </button>

            <button class="button" on:click={handleImportClick}>
              Import
            </button>

            <button class="button is-info" on:click={openExploreModal}>
              Explore
            </button>
          {:else}
            <button
              class="button is-dark"
              on:click={runnerExecute}
              disabled={isExecuting || !runnerCanExecute}
            >
              {isExecuting ? 'Running...' : 'Execute'}
            </button>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <input
    bind:this={fileInput}
    type="file"
    accept=".json"
    class="is-hidden"
    on:change={importWorkflow}
  />
</nav>
{/if}
