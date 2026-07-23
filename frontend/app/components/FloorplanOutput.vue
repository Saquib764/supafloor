<script setup lang="ts">
import { buildRoomOutput } from '~/utils/layoutPrompt'

const props = defineProps<{
  plan: FloorPlan | null
}>()

const output = computed(() => {
  if (!props.plan?.rooms.length) return ''
  return buildRoomOutput(props.plan)
})
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="flex-1 overflow-y-auto custom-scrollbar p-4">
      <p
        v-if="!plan"
        class="text-sm text-gray-400"
      >
        Select a floor plan to view room JSON.
      </p>
      <p
        v-else-if="!output"
        class="text-sm text-gray-400"
      >
        This plan has no rooms.
      </p>
      <pre
        v-else
        class="rounded-lg border border-gray-800 bg-gray-950/60 p-3 overflow-x-auto text-xs leading-relaxed text-gray-300"
      ><code>{{ output }}</code></pre>
    </div>
  </div>
</template>
