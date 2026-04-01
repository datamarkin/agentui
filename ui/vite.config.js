import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

// Plugin to inject Jinja2 APP_CONFIG template into index.html
const injectJinja2Config = () => ({
  name: 'inject-jinja2-config',
  transformIndexHtml(html) {
    // Inject Jinja2 template before <div id="app">
    const jinja2Script = `
    <script>
      window.APP_CONFIG = {
        apiBase: "{{ api_base }}",
        deploymentMode: "{{ deployment_mode }}",
        {% if user %}
        user: {
          id: "{{ user.id }}",
          name: "{{ user.name }}"
        },
        {% else %}
        user: null,
        {% endif %}
        workflowId: {{ workflow_id | tojson | safe if workflow_id else 'null' }},
        hideToolbar: {{ hide_toolbar | tojson if hide_toolbar is defined else 'false' }},
        viewMode: {{ view_mode | tojson if view_mode is defined else '"editor"' }}
      };
    </script>
    {% if header_template %}{% include header_template %}{% endif %}`;
    return html.replace('<div id="app">', jinja2Script + '\n    <div id="app">');
  }
});

export default defineConfig({
  plugins: [svelte(), injectJinja2Config()],
  // Use relative base for serving from Flask static directory
  base: './',
  build: {
    outDir: '../agentui/static',
    emptyOutDir: true,
    // Let Vite auto-generate asset paths - no manual editing needed
    rollupOptions: {
      output: {
        // Assets will be in static/assets/ with hashed names
        // Vite automatically updates references in index.html
      }
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
});