<script setup lang="ts">
type RightPanelTab = 'stats' | 'prompt'

defineProps<{
  scores: ScoreBreakdown | null
  plan: FloorPlan | null
}>()

const tab = ref<RightPanelTab>('stats')

const tabs: Array<{ id: RightPanelTab, label: string }> = [
  { id: 'stats', label: 'Stats' },
  { id: 'prompt', label: 'Prompt' },
]
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="shrink-0 p-4 border-b border-gray-800/80">
      <div class="flex rounded-lg border border-gray-800 p-0.5 gap-0.5">
        <button
          v-for="item in tabs"
          :key="item.id"
          type="button"
          class="flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors"
          :class="tab === item.id
            ? 'bg-primary-500/20 text-primary-400'
            : 'text-gray-400 hover:text-gray-200'"
          @click="tab = item.id"
        >
          {{ item.label }}
        </button>
      </div>
    </div>

    <div class="flex-1 min-h-0">
      <FloorplanScores
        v-show="tab === 'stats'"
        class="h-full"
        hide-title
        :scores="scores"
      />
      <FloorplanPrompt
        v-show="tab === 'prompt'"
        class="h-full"
        :plan="plan"
      />
    </div>
  </div>
</template>
