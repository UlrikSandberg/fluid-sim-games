import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Served from https://ulriksandberg.github.io/fluid-sim-games/, so assets
  // need the repo name as their prefix. Keep in sync with the repo name.
  base: process.env.GITHUB_ACTIONS ? '/fluid-sim-games/' : '/',
  plugins: [react()],
})
