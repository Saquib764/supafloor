<script setup lang="ts">
import type { TabsItem } from '@nuxt/ui'
import {
  PIPELINE_PROVIDERS,
  SCORE_LEVELS,
  type EngineConfig,
  type PipelineProvider,
  type ScoreLevel,
} from '~/types/floorplan'

const open = defineModel<boolean>('open', { default: false })

const { config, saving, loading, refresh, save } = useEngineConfig()

const activeTab = ref('score')

const draft = reactive({
  score_level: 'basic' as ScoreLevel,
  score_threshold: 8,
  target_areas: 'deterministic' as PipelineProvider,
  connection_graph: 'deterministic' as PipelineProvider,
  centers: 'deterministic' as PipelineProvider,
})

const error = ref<string | null>(null)

const tabItems: TabsItem[] = [
  { label: 'Score', icon: 'i-lucide-gauge', slot: 'score', value: 'score' },
  { label: 'Pipeline', icon: 'i-lucide-workflow', slot: 'pipeline', value: 'pipeline' },
]

const levelItems = SCORE_LEVELS.map(l => ({
  label: l.charAt(0).toUpperCase() + l.slice(1),
  value: l,
}))

const providerItems = PIPELINE_PROVIDERS.map(p => ({
  label: p === 'llm' ? 'LLM' : 'Deterministic',
  value: p,
}))

function syncDraftFromConfig() {
  draft.score_level = config.value.score_level
  draft.score_threshold = config.value.score_threshold
  draft.target_areas = config.value.target_areas
  draft.connection_graph = config.value.connection_graph
  draft.centers = config.value.centers
}

watch(open, async (isOpen) => {
  if (!isOpen) return
  error.value = null
  activeTab.value = 'score'
  try {
    await refresh()
    syncDraftFromConfig()
  }
  catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

async function onSave(close: () => void) {
  error.value = null
  try {
    const updates: Partial<EngineConfig> = {
      score_level: draft.score_level,
      score_threshold: Number(draft.score_threshold),
      target_areas: draft.target_areas,
      connection_graph: draft.connection_graph,
      centers: draft.centers,
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
    description="Engine settings saved to config.yaml and applied immediately."
    :ui="{ content: 'sm:max-w-xl', body: 'min-h-[22rem]', footer: 'justify-end gap-2' }"
  >
    <template #body>
      <div class="flex min-h-[22rem] flex-col">
        <div
          v-if="loading"
          class="flex flex-1 items-center justify-center text-sm text-muted"
        >
          Loading settings…
        </div>
        <UTabs
          v-else
          v-model="activeTab"
          :items="tabItems"
          orientation="vertical"
          variant="pill"
          color="neutral"
          :unmount-on-hide="false"
          class="w-full flex-1"
          :ui="{
            root: 'flex h-full min-h-[20rem] items-stretch gap-6',
            list: 'flex w-36 shrink-0 flex-col',
            content: 'min-h-[20rem] flex-1 self-stretch pt-0.5',
          }"
        >
          <template #score>
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
            </div>
          </template>

        <template #pipeline>
          <div class="space-y-4">
            <UFormField
              label="Target areas"
              hint="How per-room target areas are chosen before layout"
            >
              <USelect
                v-model="draft.target_areas"
                :items="providerItems"
                value-key="value"
                class="w-full"
              />
            </UFormField>

            <UFormField
              label="Connection graph"
              hint="How room adjacency / circulation links are proposed"
            >
              <USelect
                v-model="draft.connection_graph"
                :items="providerItems"
                value-key="value"
                class="w-full"
              />
            </UFormField>

            <UFormField
              label="Centers"
              hint="How seed points for rectangle growth are placed"
            >
              <USelect
                v-model="draft.centers"
                :items="providerItems"
                value-key="value"
                class="w-full"
              />
            </UFormField>
          </div>
        </template>
        </UTabs>

        <UAlert
          v-if="error && !loading"
          class="mt-4 shrink-0"
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
        :disabled="saving || loading"
        @click="close()"
      />
      <UButton
        label="Save"
        color="primary"
        :loading="saving"
        :disabled="saving || loading"
        @click="onSave(close)"
      />
    </template>
  </UModal>
</template>
