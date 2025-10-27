<script>
  export let executeWorkflow;
  export let exportWorkflow;
  export let importWorkflow;
  // export let clearWorkflow;
  export let isExecuting;

  let fileInput;

  function handleImportClick() {
    fileInput.click();
  }
</script>

<nav class="navbar is-fixed-top border-bottom" aria-label="main navigation">
  <div class="navbar-brand">
    <div class="navbar-item">
      <img alt="Datamarkin logo" src="/logo.png" width="120" height="24">
    </div>
  </div>

  <div class="navbar-menu">
    <div class="navbar-end">
      <div class="navbar-item">
        <div class="buttons">
          <button class="button is-dark" on:click={executeWorkflow} disabled={isExecuting}>
            {isExecuting ? 'Running...' : 'Run'}
          </button>

          <button class="button is-dark" on:click={exportWorkflow}>
            Export
          </button>

          <button class="button" on:click={handleImportClick}>
            Import
          </button>

<!--          <button class="button is-light" on:click={clearWorkflow}>-->
<!--            Clear All-->
<!--          </button>-->
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