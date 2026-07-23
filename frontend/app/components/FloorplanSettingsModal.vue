<script setup lang="ts">
import {
  SCORE_LEVELS,
  type EngineConfig,
  type ScoreLevel,
} from '~/types/floorplan'

const open = defineModel<boolean>('open', { default: false })

const { config, saving, ensureLoaded, save } = useEngineConfig()

const draft = reactive({
  score_level: 'basic' as ScoreLevel,
  score_threshold: 8,
})

const error = ref<string | null>(null)

const levelItems = SCORE_LEVELS.map(l => ({
  label: l.charAt(0).toUpperCase() + l.slice(1),
  value: l,
}))

watch(open, async (isOpen) => {
  if (!isOpen) return
  error.value = null
  await ensureLoaded()
  draft.score_level = config.value.score_level
  draft.score_threshold = config.value.score_threshold
})

async function onSave(close: () => void) {
  error.value = null
  try {
    const updates: Partial<EngineConfig> = {
      score_level: draft.score_level,
      score_threshold: Number(draft.score_threshold),
    }
    await save(updates)
    close()
  }
  catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}
</script>

<template>
  <UModal
    v-model:open="open"
    title="Settings"
    description="Score gates for generation acceptance. Saved to config.yaml and applied immediately."
    :ui="{ footer: 'justify-end gap-2' }"
  >
    <template #body>
      <div class="space-y-4">
        <UFormField
          label="Score level"
          hint="Used for Stats tab and generation acceptance"
        >
          <USelect
            v-model="draft.score_level"
            :items="levelItems"
            value-key="value"
            class="w-full"
          />
        </UFormField>

        <UFormField
          label="Score threshold"
          hint="Regenerate until this level score is met (0–10)"
        >
          <UInput
            v-model.number="draft.score_threshold"
            type="number"
            :min="0"
            :max="10"
            :step="0.1"
            class="w-full"
          />
        </UFormField>

        <UAlert
          v-if="error"
          color="error"
          variant="soft"
          :title="error"
        />
      </div>
    </template>

    <template #footer="{ close }">
      <UButton
        label="Cancel"
        color="neutral"
        variant="outline"
        :disabled="saving"
        @click="close()"
      />
      <UButton
        label="Save"
        color="primary"
        :loading="saving"
        :disabled="saving"
        @click="onSave(close)"
      />
    </template>
  </UModal>
</template>
