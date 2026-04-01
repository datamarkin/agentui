<script>
    import { onMount, onDestroy } from 'svelte';
    import { apiUrl } from './utils.js';

    export let nodes;
    export let edges;
    export let isExecuting;
    export let onExecute;
    export let onCanExecuteChange;

    let uploadedImageData = null;
    let uploadedFileName = '';
    let isDragging = false;
    let fileInput;
    let executionResults = null;
    let executionProgress = [];
    let executionError = null;

    // Find MediaInput nodes in the workflow
    $: mediaInputNodes = $nodes.filter(n => n.data?.nodeType === 'MediaInput');
    $: hasWorkflow = $nodes.length > 0;

    // Notify parent when execute capability changes
    $: canExecute = hasWorkflow && uploadedImageData;
    $: if (onCanExecuteChange) onCanExecuteChange(!!canExecute);

    // Expose execute function to parent
    onMount(() => {
        if (onExecute) onExecute(executeRunner);
    });

    onDestroy(() => {
        if (onExecute) onExecute(null);
        if (onCanExecuteChange) onCanExecuteChange(false);
    });

    async function handleFileUpload(file) {
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(apiUrl('/api/upload/image'), {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            uploadedImageData = result.data;
            uploadedFileName = file.name;
        } catch (error) {
            console.error('Upload failed:', error);
        }
    }

    function onFileInput(event) {
        const file = event.target.files[0];
        handleFileUpload(file);
    }

    function onDrop(event) {
        event.preventDefault();
        isDragging = false;
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileUpload(file);
        }
    }

    function onDragOver(event) {
        event.preventDefault();
        isDragging = true;
    }

    function onDragLeave() {
        isDragging = false;
    }

    function clearImage() {
        uploadedImageData = null;
        uploadedFileName = '';
        executionResults = null;
        executionProgress = [];
        executionError = null;
    }

    async function executeRunner() {
        if (!uploadedImageData || !hasWorkflow) return;

        isExecuting.set(true);
        executionResults = null;
        executionProgress = [];
        executionError = null;

        // Inject uploaded image into MediaInput nodes
        const workflowNodes = $nodes.map(n => {
            if (n.data?.nodeType === 'MediaInput') {
                return {
                    ...n,
                    data: {
                        ...n.data,
                        parameters: { ...n.data.parameters, data: uploadedImageData }
                    }
                };
            }
            return n;
        });

        try {
            const response = await fetch(apiUrl('/api/workflow/stream'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow: { nodes: workflowNodes, edges: $edges } })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            const results = {};

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));

                        if (data.done) {
                            isExecuting.set(false);
                        } else if (data.error && !data.tool_id) {
                            executionError = data.error;
                            isExecuting.set(false);
                        } else if (data.status === 'running') {
                            executionProgress = [...executionProgress, { id: data.tool_id, status: 'running', type: data.tool_type || data.tool_id }];
                        } else if (data.status === 'completed') {
                            executionProgress = executionProgress.map(p =>
                                p.id === data.tool_id ? { ...p, status: 'completed' } : p
                            );
                            if (data.result) {
                                results[data.tool_id] = data.result;
                            }
                        } else if (data.status === 'error') {
                            executionProgress = executionProgress.map(p =>
                                p.id === data.tool_id ? { ...p, status: 'error' } : p
                            );
                            executionError = data.error;
                            isExecuting.set(false);
                        }
                    }
                }
            }

            // Filter to terminal tool results only (tools with no outgoing edges)
            const sourceIds = new Set($edges.map(e => e.source));
            const terminalResults = {};
            for (const [toolId, result] of Object.entries(results)) {
                if (!sourceIds.has(toolId)) {
                    terminalResults[toolId] = result;
                }
            }

            executionResults = Object.keys(terminalResults).length > 0 ? terminalResults : results;
        } catch (error) {
            executionError = error.message;
        } finally {
            isExecuting.set(false);
        }
    }

    function getOutputEntries(results) {
        const entries = [];
        for (const [toolId, result] of Object.entries(results)) {
            if (result.outputs) {
                for (const [name, value] of Object.entries(result.outputs)) {
                    entries.push({ toolId, toolType: result.type, name, value });
                }
            }
        }
        return entries;
    }

    function isImageOutput(value) {
        return typeof value === 'string' && value.startsWith('data:image');
    }

    function isDetectionsOutput(value) {
        return typeof value === 'object' && value !== null && (
            Array.isArray(value) ||
            value.xyxy !== undefined ||
            value.class_id !== undefined ||
            value.confidence !== undefined
        );
    }

    function formatDetections(value) {
        if (Array.isArray(value)) return value;
        // PixelFlow Detections dict format
        const count = value.xyxy ? value.xyxy.length : 0;
        const items = [];
        for (let i = 0; i < count; i++) {
            items.push({
                class: value.class_name?.[i] || value.class_id?.[i] || '-',
                confidence: value.confidence?.[i] != null ? (value.confidence[i] * 100).toFixed(1) + '%' : '-',
                bbox: value.xyxy?.[i] ? value.xyxy[i].map(v => Math.round(v)).join(', ') : '-'
            });
        }
        return items;
    }
</script>

<div class="runner-container">
    <!-- Input Panel -->
    <div class="runner-panel runner-input-panel">
        <div class="runner-panel-header">
            <h2 class="title is-5 mb-0">Input</h2>
        </div>
        <div class="runner-panel-body">
            {#if !uploadedImageData}
                <!-- Upload area -->
                <div
                    class="runner-upload-zone"
                    class:is-dragging={isDragging}
                    on:drop={onDrop}
                    on:dragover={onDragOver}
                    on:dragleave={onDragLeave}
                    on:click={() => fileInput.click()}
                    role="button"
                    tabindex="0"
                    on:keydown={(e) => e.key === 'Enter' && fileInput.click()}
                >
                    <div class="runner-upload-content">
                        <span class="icon is-large has-text-grey-light">
                            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                                <circle cx="8.5" cy="8.5" r="1.5"/>
                                <polyline points="21 15 16 10 5 21"/>
                            </svg>
                        </span>
                        <p class="mt-3 has-text-grey">Drop an image here or click to upload</p>
                    </div>
                </div>
            {:else}
                <!-- Image preview -->
                <div class="runner-preview">
                    <div class="runner-preview-header">
                        <span class="is-size-7 has-text-grey">{uploadedFileName}</span>
                        <button class="delete is-small" on:click={clearImage}></button>
                    </div>
                    <img src={uploadedImageData} alt="Input" class="runner-preview-image" />
                </div>
            {/if}
        </div>
        <input
            bind:this={fileInput}
            type="file"
            accept="image/*"
            class="is-hidden"
            on:change={onFileInput}
        />
    </div>

    <!-- Output Panel -->
    <div class="runner-panel runner-output-panel">
        <div class="runner-panel-header">
            <h2 class="title is-5 mb-0">Output</h2>
        </div>
        <div class="runner-panel-body">
            {#if $isExecuting}
                <!-- Progress -->
                <div class="runner-progress">
                    {#each executionProgress as step}
                        <div class="runner-progress-step">
                            <span class="runner-progress-icon">
                                {#if step.status === 'running'}
                                    <span class="loader is-loading"></span>
                                {:else if step.status === 'completed'}
                                    <span class="has-text-success">&#10003;</span>
                                {:else}
                                    <span class="has-text-danger">&#10007;</span>
                                {/if}
                            </span>
                            <span class="is-size-7">{step.type}</span>
                        </div>
                    {/each}
                </div>
            {:else if executionError}
                <!-- Error -->
                <div class="notification is-danger is-light">
                    <p class="has-text-weight-bold">Execution Failed</p>
                    <p class="is-size-7 mt-2" style="white-space: pre-wrap; font-family: monospace;">{executionError}</p>
                </div>
            {:else if executionResults}
                <!-- Results -->
                <div class="runner-results">
                    {#each getOutputEntries(executionResults) as output}
                        {#if isImageOutput(output.value)}
                            <div class="runner-result-item">
                                <img src={output.value} alt="{output.name}" class="runner-result-image" />
                            </div>
                        {:else if isDetectionsOutput(output.value)}
                            {@const detections = formatDetections(output.value)}
                            <div class="runner-result-item">
                                <p class="has-text-weight-semibold is-size-6 mb-2">
                                    {detections.length} detection{detections.length !== 1 ? 's' : ''} found
                                </p>
                                {#if detections.length > 0}
                                    <table class="table is-narrow is-fullwidth is-size-7">
                                        <thead>
                                            <tr>
                                                <th>#</th>
                                                <th>Class</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {#each detections as det, i}
                                                <tr>
                                                    <td>{i + 1}</td>
                                                    <td>{det.class}</td>
                                                    <td>{det.confidence}</td>
                                                </tr>
                                            {/each}
                                        </tbody>
                                    </table>
                                {/if}
                            </div>
                        {:else if typeof output.value === 'object'}
                            <div class="runner-result-item">
                                <p class="is-size-7 has-text-weight-semibold mb-1">{output.name}</p>
                                <pre class="runner-result-data">{JSON.stringify(output.value, null, 2)}</pre>
                            </div>
                        {:else}
                            <div class="runner-result-item">
                                <p class="is-size-7 has-text-weight-semibold mb-1">{output.name}</p>
                                <div class="runner-result-text">{output.value}</div>
                            </div>
                        {/if}
                    {/each}
                </div>
            {:else}
                <!-- Empty state -->
                <div class="runner-empty">
                    <span class="icon is-large has-text-grey-lighter">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="5 3 19 12 5 21 5 3"/>
                        </svg>
                    </span>
                    <p class="mt-3 has-text-grey-light">Upload an image and click Execute to see results</p>
                </div>
            {/if}
        </div>
    </div>
</div>
