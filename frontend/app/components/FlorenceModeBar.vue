<script setup lang="ts">
import type { InferenceMode } from '~/components/FlorenceExtractionPanel.vue'

const mode = defineModel<InferenceMode>({ required: true })

defineProps<{
  inferring?: boolean
  canInfer?: boolean
  pendingCount?: number
}>()

const emit = defineEmits<{
  (e: 'infer'): void
  (e: 'clear'): void
}>()

const modes: { id: InferenceMode, label: string }[] = [
  { id: 'automatic', label: 'Automatic' },
  { id: 'single', label: 'Single point' },
  { id: 'multiple', label: 'Multiple point' },
]
</script>

<template>
  <div class="pointer-events-auto flex items-center gap-2 rounded-lg bg-gray-950/85 p-1 backdrop-blur-md">
    <div class="flex rounded-md bg-gray-900/80 p-0.5">
      <button
        v-for="item in modes"
        :key="item.id"
        type="button"
        class="rounded px-3 py-1 text-xs font-medium transition-colors"
        :class="mode === item.id
          ? 'bg-gray-800 text-white'
          : 'text-gray-400 hover:text-gray-200'"
        :disabled="inferring"
        @click="mode = item.id"
      >
        {{ item.label }}
      </button>
    </div>

    <UButton
      size="xs"
      color="neutral"
      variant="outline"
      :disabled="inferring"
      @click="emit('clear')"
    >
      Clear
    </UButton>

    <UButton
      v-if="mode !== 'single'"
      size="xs"
      color="primary"
      :disabled="inferring || !canInfer"
      @click="emit('infer')"
    >
      {{
        inferring
          ? 'Inferring…'
          : mode === 'multiple' && pendingCount
            ? `Infer (${pendingCount})`
            : 'Infer'
      }}
    </UButton>
  </div>
</template>
