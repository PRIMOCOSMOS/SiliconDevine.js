import {defineConfig} from 'vite';
export default defineConfig({build:{lib:{entry:'src/index.ts',formats:['es'],fileName:()=> 'silicondevine.js'},rollupOptions:{external:id=>id==='three'||id.startsWith('three/')}}});
