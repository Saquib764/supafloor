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
const viewMode = ref<'simple' | 'realistic'>('simple')
const pngB64 = ref('')
const canvasRef = ref<{ resetView: () => void } | null>(null)

const layers = computed(() => {
  const list = ['boundary', 'rooms', 'doors', 'windows']
  if (showLabels.value) list.push('labels', 'room_ids')
  if (showDimensions.value) list.push('dimensions')
  if (showTopology.value) list.push('topology')
  if (showCirculation.value) list.push('circulation')
  return list
})

async function refreshRealistic() {
  if (!plan.value?.rooms.length) {
    pngB64.value = ''
    return
  }
  const exported = await api.exportRealistic(plan.value)
  pngB64.value = exported.png_b64
}

async function refreshView() {
  if (!plan.value?.rooms.length) {
    pngB64.value = ''
    return
  }
  if (viewMode.value === 'realistic') {
    await refreshRealistic()
  } else {
    await refreshRender()
  }
}

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
    if (viewMode.value === 'realistic') {
      await refreshRealistic()
    }
    await nextTick()
    canvasRef.value?.resetView()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    extracting.value = false
  }
}

watch(layers, () => {
  if (viewMode.value === 'simple') {
    refreshRender().catch(() => undefined)
  }
})

watch(viewMode, () => {
  refreshView().catch(() => undefined)
})

watch(plan, () => {
  refreshView().catch(() => undefined)
}, { deep: true })
</script>

<template>
  <div class="relative h-dvh w-full overflow-hidden bg-gray-950 text-white">
    <FloorplanCanvas
      ref="canvasRef"
      class="absolute inset-0 z-0"
      background
      :view-mode="viewMode"
      :svg="svg"
      :png-b64="pngB64"
      :plan="plan"
    />

    <aside class="absolute left-0 top-0 bottom-0 z-20 flex">
      <FloorplanNav />
      <div class="flex w-72 flex-col border-r border-gray-800/80 bg-gray-950/85 backdrop-blur-md">
        <FloorplanExtractionPanel
          class="min-h-0 flex-1"
          :extracting="extracting"
          @extract="onExtract"
        />
      </div>
    </aside>

    <aside
      class="absolute right-0 top-0 bottom-0 z-10 w-80 flex flex-col border-l border-gray-800/80 bg-gray-950/85 backdrop-blur-md"
    >
      <FloorplanRightPanel
        :scores="scores"
        :plan="plan"
      />
    </aside>

    <div class="absolute top-4 left-[21.5rem] right-80 z-10 flex justify-center pointer-events-none">
      <FloorplanViewModeToggle v-model="viewMode" />
    </div>

    <FloorplanEditorBar
      class="absolute bottom-4 left-[21.5rem] right-80 z-10"
      @reset-view="canvasRef?.resetView()"
    >
      <template #controls>
        <template v-if="viewMode === 'simple'">
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
