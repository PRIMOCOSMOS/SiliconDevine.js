import {defineConfig} from 'vite';
import manifest from './package.json';
export default defineConfig({define:{__APP_VERSION__:JSON.stringify(manifest.version)},root:'demo',base:'./',build:{outDir:'../demo-dist',emptyOutDir:true,rollupOptions:{output:{manualChunks:{three:['three','three/addons/controls/OrbitControls.js','three/addons/utils/BufferGeometryUtils.js']}}}}});
