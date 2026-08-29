import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { readFileSync } from 'node:fs';

const packageMetadata = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8')
);

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    version: {
      // SvelteKit otherwise uses the current time, which changes asset hashes
      // on every build and makes the checked-in release bundle unverifiable.
      name: packageMetadata.version
    },
    adapter: adapter({
      pages: '../carla_vision/operator/console_static',
      assets: '../carla_vision/operator/console_static',
      fallback: 'index.html',
      precompress: false
    })
  }
};

export default config;
