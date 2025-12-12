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
        deploymentMode: "{{ deployment_mode }}",
        {% if user %}
        user: {
          id: "{{ user.id }}",
          name: "{{ user.name }}"
        },
        {% else %}
        user: null,
        {% endif %}
        workflowId: {{ workflow_id | tojson | safe if workflow_id else 'null' }}
      };
    </script>`;
    return html.replace('<div id="app">', jinja2Script + '\n    <div id="app">');
  }
});

export default defineConfig({
  plugins: [svelte(), injectJinja2Config()],
  build: {
    outDir: '../agentui/static',
    emptyOutDir: true
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
});