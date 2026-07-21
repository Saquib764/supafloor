<script setup lang="ts">
useForcedDarkMode()

useHead({
  meta: [
    { charset: 'utf-8' },
    { name: 'viewport', content: 'width=device-width, initial-scale=1' },
    { key: 'theme-color', name: 'theme-color', content: '#030712' },
  ],
  link: [
    { rel: 'icon', href: '/favicon.ico' },
  ],
  htmlAttrs: {
    lang: 'en',
    class: 'dark',
  },
})
</script>

<template>
  <UApp>
    <NuxtPage />
  </UApp>
</template>
