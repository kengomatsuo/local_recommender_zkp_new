import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  base: '/local_recommender_zkp_new/',
  root: '.',
  publicDir: 'public',
  plugins: [
    viteStaticCopy({
      targets: [
        { src: 'circuits/auth_js/auth.wasm', dest: 'circuits' },
        { src: 'keys/circuit_final.zkey', dest: 'keys' }
      ]
    })
  ],
  server: {
    port: 5173,
    open: true
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
