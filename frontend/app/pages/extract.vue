<script setup lang="ts">
const api = useFloorplanApi()

const plan = ref<FloorPlan | null>(null)
const scores = ref<ScoreBreakdown | null>(null)
const svg = ref('')
const extracting = ref(false)
const error = ref<string | null>(null)
const showLabels = ref(true)
const showDimensions = ref(false)
const showTopology = ref(true)
const showCirculation = ref(false)
const canvasRef = ref<{ resetView: () => void } | null>(null)

const layers = computed(() => {
  const list = ['boundary', 'rooms', 'doors', 'windows']
  if (showLabels.value) list.push('labels', 'room_ids')
  if (showDimensions.value) list.push('dimensions')
  if (showTopology.value) list.push('topology')
  if (showCirculation.value) list.push('circulation')
  return list
})

async function refreshRender() {
  if (!plan.value) return
  const rendered = await api.render(plan.value, layers.value)
  svg.value = rendered.svg
}

async function onExtract(payload: { file: File, canvasSize: number }) {
  extracting.value = true
  error.value = null
  try {
    const extracted = await api.extractImage(payload.file, payload.canvasSize)
    const [rendered, evaluated] = await Promise.all([
      api.render(extracted.floor_plan, layers.value),
      api.evaluate(extracted.floor_plan),
    ])
    plan.value = extracted.floor_plan
    svg.value = rendered.svg
    scores.value = evaluated
    await nextTick()
    canvasRef.value?.resetView()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    extracting.value = false
  }
}

watch(layers, () => {
  refreshRender().catch(() => undefined)
})
</script>

<template>
  <div class="relative h-dvh w-full overflow-hidden bg-gray-950 text-white">
    <FloorplanCanvas
      ref="canvasRef"
      class="absolute inset-0 z-0"
      background
      :svg="svg"
      :plan="plan"
    />

    <aside
      class="absolute left-0 top-0 bottom-0 z-10 w-72 flex flex-col border-r border-gray-800/80 bg-gray-950/85 backdrop-blur-md"
    >
      <FloorplanExtractionPanel
        class="min-h-0 flex-1"
        :extracting="extracting"
        @extract="onExtract"
      />
    </aside>

    <aside
      class="absolute right-0 top-0 bottom-0 z-10 w-80 flex flex-col border-l border-gray-800/80 bg-gray-950/85 backdrop-blur-md"
    >
      <FloorplanScores
        :key="scores?.overall ?? 'empty'"
        :scores="scores"
      />
    </aside>

    <FloorplanEditorBar
      class="absolute bottom-4 left-72 right-80 z-10"
      @reset-view="canvasRef?.resetView()"
    >
      <template #controls>
        <UCheckbox
          v-model="showLabels"
          label="Labels"
        />
        <UCheckbox
          v-model="showDimensions"
          label="Dimensions"
        />
        <UCheckbox
          v-model="showTopology"
          label="Topology"
        />
        <UCheckbox
          v-model="showCirculation"
          label="Circulation"
        />
      </template>
    </FloorplanEditorBar>

    <UAlert
      v-if="error"
      color="error"
      variant="soft"
      :title="error"
      class="absolute top-4 left-1/2 z-20 w-[min(28rem,calc(100%-24rem))] -translate-x-1/2"
    />
  </div>
</template>
