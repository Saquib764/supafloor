// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['@nuxt/ui', '@nuxt/eslint'],
  css: ['~/assets/main.css'],
  imports: {
    dirs: ['composables', 'types'],
  },
  colorMode: {
    preference: 'dark',
    fallback: 'dark',
    storage: 'localStorage',
    storageKey: 'nuxt-color-mode',
    classSuffix: '',
  },
  runtimeConfig: {
    public: {
      floorplanApi: process.env.NUXT_PUBLIC_FLOORPLAN_API || 'http://localhost:8000',
    },
  },
  routeRules: {
    '/': { ssr: false },
    '/importer': { ssr: false },
  },
})
