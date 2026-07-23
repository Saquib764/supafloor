<script setup lang="ts">
import { buildLayoutPrompt } from '~/utils/layoutPrompt'

const props = defineProps<{
  plan: FloorPlan | null
}>()

const prompt = ref('')

watch(() => props.plan?.id, () => {
  prompt.value = ''
})

function computePrompt() {
  if (!props.plan) {
    prompt.value = ''
    return
  }
  prompt.value = buildLayoutPrompt(props.plan)
}
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="shrink-0 p-4 border-b border-gray-800/80 space-y-3">
      <UButton
        block
        color="primary"
        variant="soft"
        :disabled="!plan"
        @click="computePrompt"
      >
        Compute prompt
      </UButton>
    </div>

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
        Click compute to describe each room by name and location.
      </p>
      <pre
        v-else
        class="whitespace-pre-wrap text-sm leading-relaxed text-gray-200 font-sans"
      >{{ prompt }}</pre>
    </div>
  </div>
</template>
