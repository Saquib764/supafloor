<script setup lang="ts">
const model = defineModel<'simple' | 'realistic'>({ default: 'simple' })
</script>

<template>
  <div class="pointer-events-auto flex rounded-lg bg-gray-950/80 backdrop-blur-md p-0.5 gap-0.5 shrink-0">
    <button
      type="button"
      class="rounded-md px-3 py-1 text-xs font-medium transition-colors cursor-pointer"
      :class="model === 'simple'
        ? 'bg-gray-800 text-white'
        : 'text-gray-400 hover:text-gray-200'"
      @click="model = 'simple'"
    >
      Simple
    </button>
    <button
      type="button"
      class="rounded-md px-3 py-1 text-xs font-medium transition-colors cursor-pointer"
      :class="model === 'realistic'
        ? 'bg-gray-800 text-white'
        : 'text-gray-400 hover:text-gray-200'"
      @click="model = 'realistic'"
    >
      Realistic
    </button>
  </div>
</template>
