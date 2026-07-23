<script setup lang="ts">
import { buildLayoutPrompt, buildLayoutSchema } from '~/utils/layoutPrompt'

const props = defineProps<{
  plan: FloorPlan | null
}>()

const prompt = computed(() => {
  if (!props.plan?.rooms.length) return ''
  return buildLayoutPrompt(props.plan)
})

const schema = computed(() => {
  if (!props.plan?.rooms.length) return ''
  return buildLayoutSchema(props.plan)
})
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="flex-1 overflow-y-auto custom-scrollbar p-4">
      <p
        v-if="!plan"
        class="text-sm text-gray-400"
      >
        Select a floor plan to build a layout prompt.
      </p>
      <p
        v-else-if="!prompt"
        class="text-sm text-gray-400"
      >
        This plan has no rooms.
      </p>
      <template v-else>
        <pre class="whitespace-pre-wrap text-sm leading-relaxed text-gray-200 font-sans">{{ prompt }}</pre>
        <div
          v-if="schema"
          class="mt-6 space-y-2"
        >
          <p class="text-xs font-medium uppercase tracking-wide text-gray-500">
            Schema
          </p>
          <pre class="rounded-lg border border-gray-800 bg-gray-950/60 p-3 overflow-x-auto text-xs leading-relaxed text-gray-300"><code>{{ schema }}</code></pre>
        </div>
      </template>
    </div>
  </div>
</template>
