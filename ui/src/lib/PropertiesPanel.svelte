<script>
    import { formatParameterLabel, getParameterType, parseValue } from './utils.js';

    export let selectedNode;
    export let updateNodeParameters;
    export let availableNodes;
    export let executionResults;
    export let hideTitle = false;

    let parameters = {};

    $: if (selectedNode) {
        // Get default parameters from availableNodes if node parameters are empty
        const nodeDefaults = availableNodes?.[selectedNode.data.nodeType]?.parameters || {};
        const nodeParameters = selectedNode.data.parameters || {};

        // Merge defaults with actual parameters, preferring actual parameters
        parameters = { ...nodeDefaults, ...nodeParameters };
    }

    function handleParameterChange(key, value) {
        parameters[key] = value;
        updateNodeParameters(selectedNode.id, parameters);
    }

    async function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload/image', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            handleParameterChange('data', result.data);
        } catch (error) {
            console.error('Upload failed:', error);
        }
    }

    function getParameterOptions(nodeType, parameterName) {
        const nodeInfo = availableNodes?.[nodeType];
        return nodeInfo?.parameter_options?.[parameterName];
    }
</script>

{#if !hideTitle}
<div class="block mb-1">
    <strong>Node properties</strong>
</div>
{/if}

<nav class="panel">
    {#if selectedNode}
        <!-- Node Info Block -->
        <div class="panel-block">
            <div>
                <p class="is-size-8 has-text-weight-bold">{selectedNode.data.nodeType}</p>
                <p class="is-size-7 has-text-grey">
                    {availableNodes?.[selectedNode.data.nodeType]?.description || 'Process image data'}
                </p>
            </div>
        </div>

        <!-- Special file upload for MediaInput -->
        {#if selectedNode.data.nodeType === 'MediaInput'}
            <div class="panel-block">
                <div class="field" style="width: 100%;">
                    <label class="label is-small" for="file-upload">Upload Image</label>
                    <div class="control">
                        <input
                                id="file-upload"
                                class="input is-small"
                                type="file"
                                accept="image/*"
                                on:change={handleFileUpload}
                        />
                    </div>
                </div>
            </div>
        {/if}

        <!-- Dynamic parameter forms based on backend data -->
        {#if Object.keys(parameters).length > 0}
            {#each Object.entries(parameters) as [key, value]}
                {@const paramType = getParameterType(value)}
                <div class="panel-block">
                    <div class="field" style="width: 100%;">
                        <label class="label is-small" for={key}>{formatParameterLabel(key)}</label>
                        <div class="control">
                            {#if paramType === 'boolean'}
                                <label class="checkbox">
                                    <input
                                            id={key}
                                            type="checkbox"
                                            bind:checked={parameters[key]}
                                            on:change={(e) => handleParameterChange(key, e.target.checked)}
                                    />
                                    {formatParameterLabel(key)}
                                </label>
                            {:else if paramType === 'number'}
                                <input
                                        id={key}
                                        class="input is-small"
                                        type="number"
                                        step={key.includes('threshold') || key.includes('factor') ? '0.1' : '1'}
                                        value={parameters[key]}
                                        on:input={(e) => handleParameterChange(key, parseValue(e.target.value, 'number'))}
                                />
                            {:else if paramType === 'color'}
                                <input
                                        id={key}
                                        type="color"
                                        value={parameters[key]}
                                        on:input={(e) => handleParameterChange(key, e.target.value)}
                                        style="width: 40px; height: 32px; padding: 1px; border-radius: 4px;"
                                />
                            {:else if getParameterOptions(selectedNode.data.nodeType, key)}
                                {@const paramOptions = getParameterOptions(selectedNode.data.nodeType, key)}
                                <div class="select is-small">
                                    <select
                                            id={key}
                                            value={parameters[key]}
                                            on:change={(e) => handleParameterChange(key, e.target.value)}
                                    >
                                        {#each paramOptions.options as option}
                                            <option value={option.value}>{option.label}</option>
                                        {/each}
                                    </select>
                                </div>
                            {:else}
                                <input
                                        id={key}
                                        class="input is-small"
                                        type="text"
                                        value={parameters[key]}
                                        on:input={(e) => handleParameterChange(key, e.target.value)}
                                        placeholder={key.includes('path') ? 'Enter file path...' : ''}
                                />
                            {/if}
                        </div>
                    </div>
                </div>
            {/each}
        {:else}
            <div class="panel-block">
                <p class="has-text-grey is-italic">No parameters available for this node.</p>
            </div>
        {/if}
    {:else}
        <div class="panel-block">
            <p class="has-text-grey">Select a node to edit its properties</p>
        </div>
    {/if}

    <!-- Error Display Section (Critical Fix) -->
    {#if executionResults && !executionResults.success && executionResults.error}
        <div class="panel-block has-background-danger-light">
            <div class="notification is-danger is-light" style="width: 100%; margin: 0; padding: 0.75rem;">
                <p class="has-text-weight-bold is-size-7">❌ Execution Failed</p>
                <p class="is-size-7" style="margin-top: 0.5rem; white-space: pre-wrap; font-family: monospace;">
                    {executionResults.error}
                </p>
            </div>
        </div>
    {/if}

    <!-- Outputs Section (After Execution) -->
    {#if selectedNode && executionResults?.success && executionResults?.results?.[selectedNode.id]}
        {@const nodeResult = executionResults.results[selectedNode.id]}
        {#if nodeResult.outputs && Object.keys(nodeResult.outputs).length > 0}
            <div class="panel-block">
                <div class="output-section">
                    <div class="label is-small">📊 Outputs</div>
                    {#each Object.entries(nodeResult.outputs) as [outputName, outputValue]}
                        <div class="output-item">
                            <span class="output-name">{outputName}:</span>
                            {#if typeof outputValue === 'string' && outputValue.startsWith('data:image')}
                                <img src={outputValue} alt="Output {outputName}" class="output-image" />
                            {:else if typeof outputValue === 'object'}
                                <pre class="output-data">{JSON.stringify(outputValue, null, 2)}</pre>
                            {:else}
                                <div class="output-text">{outputValue}</div>
                            {/if}
                        </div>
                    {/each}
                </div>
            </div>
        {/if}
    {/if}
</nav>

