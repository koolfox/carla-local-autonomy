import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const operatorApi = process.env.CARLA_OPERATOR_API ?? 'http://127.0.0.1:8765';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: operatorApi,
        changeOrigin: false
      }
    }
  }
});
