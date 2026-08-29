import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: '../carla_vision/operator/console_static',
      assets: '../carla_vision/operator/console_static',
      fallback: 'index.html',
      precompress: false
    })
  }
};

export default config;
