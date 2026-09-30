import {defineConfig} from 'vite';
export default defineConfig({root:'demo',base:'./',build:{outDir:'../demo-dist',emptyOutDir:true,rollupOptions:{output:{manualChunks:{three:['three','three/addons/controls/OrbitControls.js','three/addons/utils/BufferGeometryUtils.js']}}}}});
