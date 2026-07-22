<script setup lang="ts">
interface FloorplanAction {
  id: string
  label: string
}

interface FloorplanActionGroup {
  label: string
  actions: FloorplanAction[]
}

const FLOORPLAN_ACTION_GROUPS: FloorplanActionGroup[] = [
  {
    label: 'Plan',
    actions: [
      { id: 'evaluate', label: 'Re-evaluate' },
      { id: 'generate', label: 'Generate new' },
    ],
  },
  {
    label: 'Insert',
    actions: [
      { id: 'insert_balcony', label: 'Balcony' },
      { id: 'insert_toilet', label: 'Toilet' },
      { id: 'insert_closet', label: 'Closet' },
      { id: 'insert_storage', label: 'Storage' },
      { id: 'insert_utility', label: 'Utility' },
    ],
  },
]

defineProps<{
  disabled?: boolean
  busy?: boolean
}>()

const emit = defineEmits<{
  (e: 'action', id: string): void
}>()
</script>

<template>
  <div class="shrink-0 border-t border-gray-800/80">
    <div class="px-4 py-3">
      <h2 class="text-sm font-semibold text-white">
        Actions
      </h2>
      <p class="mt-1 text-xs text-gray-500">
        Generate new creates a fresh plan from the current room count
      </p>
    </div>

    <div class="max-h-56 overflow-y-auto custom-scrollbar px-4 pb-4 space-y-3">
      <div
        v-for="group in FLOORPLAN_ACTION_GROUPS"
        :key="group.label"
      >
        <p class="mb-1.5 text-xs uppercase tracking-wide text-gray-500">
          {{ group.label }}
        </p>
        <div class="flex flex-wrap gap-1.5">
          <UButton
            v-for="action in group.actions"
            :key="action.id"
            size="xs"
            color="neutral"
            variant="outline"
            :disabled="disabled || busy"
            @click="emit('action', action.id)"
          >
            {{ action.label }}
          </UButton>
        </div>
      </div>
    </div>
  </div>
</template>
