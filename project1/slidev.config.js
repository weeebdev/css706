import { defineConfig } from '@slidev/cli'

export default defineConfig({
    theme: 'default',
    vite: {
        optimizeDeps: {
            include: ['lz-string']
        },
        define: {
            'process.env': {}
        }
    }
})
