<script setup lang="ts">
import { DOOR_COLOR, WINDOW_COLOR } from '~/constants/render'

defineEmits<{
  'reset-view': []
}>()
</script>

<template>
  <div class="pointer-events-none flex flex-col items-stretch gap-2 px-4">
    <div
      class="pointer-events-auto flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 px-3 py-2 rounded-xl border border-gray-800/80 bg-gray-950/90 backdrop-blur-md shadow-lg"
    >
      <div class="flex flex-wrap items-center gap-x-3 gap-y-1.5 min-w-0">
        <span class="text-[10px] uppercase tracking-widest text-gray-500 shrink-0">
          Legend
        </span>
        <div class="flex items-center gap-1.5 text-xs text-gray-300">
          <span
            class="w-5 h-1 rounded-full shrink-0"
            :style="{ backgroundColor: DOOR_COLOR }"
          />
          <span>Door</span>
        </div>
        <div class="flex items-center gap-1.5 text-xs text-gray-300">
          <span
            class="w-5 h-1.5 rounded-full shrink-0"
            :style="{ backgroundColor: WINDOW_COLOR }"
          />
          <span>Window</span>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-2 shrink-0 lg:pl-3 lg:border-l lg:border-gray-800/80">
        <slot name="controls" />
        <UButton
          size="xs"
          color="neutral"
          variant="outline"
          icon="i-lucide-maximize-2"
          @click="$emit('reset-view')"
        >
          Reset view
        </UButton>
      </div>
    </div>
  </div>
</template>
