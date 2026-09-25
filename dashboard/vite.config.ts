import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

// The built bundle ships inside the thoth-node python package at
// thoth/thoth/dashboard/dist and is served by the node on :80.
export default defineConfig({
  base: './',
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5174,
    // Point the dev server at a live node:
    //   VITE_THOTH_API=http://10.0.0.88:5001 npm run dev
    proxy: {
      '/api': {
        target: process.env.VITE_THOTH_API || 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../thoth/dashboard/dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three', '@react-three/fiber', '@react-three/drei'],
        },
      },
    },
  },
})
