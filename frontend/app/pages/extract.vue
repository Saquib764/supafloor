<script setup lang="ts">
import type {
  FlorenceRoom,
  FlorenceSample,
  InferenceMode,
} from '~/components/FlorenceExtractionPanel.vue'

const api = useFloorplanApi()

const samples = ref<FlorenceSample[]>([])
const selectedSample = ref<string | null>(null)
const selectedFile = ref<File | null>(null)
const imageUrl = ref<string | null>(null)
const imageSize = ref(768)

const mode = ref<InferenceMode>('automatic')
const rooms = ref<FlorenceRoom[]>([])
const pendingPoints = ref<Array<{ x: number, y: number }>>([])
const plan = ref<FloorPlan | null>(null)
const scores = ref<ScoreBreakdown | null>(null)

const loadingSamples = ref(false)
const inferring = ref(false)
const error = ref<string | null>(null)

const canvasRef = ref<{ resetView: () => void } | null>(null)

const hasImage = computed(() => Boolean(imageUrl.value))
const canInfer = computed(() => {
  if (!hasImage.value || inferring.value) return false
  if (mode.value === 'multiple') return pendingPoints.value.length > 0
  return true
})

function revokeImageUrl() {
  if (imageUrl.value?.startsWith('blob:')) {
    URL.revokeObjectURL(imageUrl.value)
  }
}

function resetInference() {
  rooms.value = []
  pendingPoints.value = []
  plan.value = null
  scores.value = null
  error.value = null
}

function setImageUrl(url: string | null) {
  revokeImageUrl()
  imageUrl.value = url
}

async function loadSamples() {
  loadingSamples.value = true
  try {
    const result = await api.listFlorenceSamples()
    samples.value = result.samples
    if (!selectedSample.value && !selectedFile.value && result.samples.length) {
      await onSelectSample(result.samples[0].name)
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loadingSamples.value = false
  }
}

async function onUpload(file: File) {
  selectedFile.value = file
  selectedSample.value = null
  resetInference()
  setImageUrl(URL.createObjectURL(file))
  await nextTick()
  canvasRef.value?.resetView()
}

async function onSelectSample(name: string) {
  selectedSample.value = name
  selectedFile.value = null
  resetInference()
  setImageUrl(api.florenceSampleUrl(name))
  await nextTick()
  canvasRef.value?.resetView()
}

function imagePayload() {
  if (selectedFile.value) {
    return { file: selectedFile.value }
  }
  if (selectedSample.value) {
    return { sample: selectedSample.value }
  }
  throw new Error('No image selected')
}

async function applyResult(result: Awaited<ReturnType<typeof api.florenceAutomatic>>) {
  rooms.value = result.rooms
  imageSize.value = result.image_size
  plan.value = result.floor_plan
  try {
    scores.value = await api.evaluate(result.floor_plan)
  } catch {
    scores.value = null
  }
}

async function runAutomatic() {
  inferring.value = true
  error.value = null
  try {
    const result = await api.florenceAutomatic(imagePayload())
    await applyResult(result)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    inferring.value = false
  }
}

async function runPoints(points: Array<{ x: number, y: number }>) {
  if (!points.length) return
  inferring.value = true
  error.value = null
  try {
    const result = await api.florencePoints({
      ...imagePayload(),
      points,
      rooms: rooms.value,
    })
    await applyResult(result)
    pendingPoints.value = []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    inferring.value = false
  }
}

async function onInfer() {
  if (mode.value === 'automatic') {
    await runAutomatic()
    return
  }
  if (mode.value === 'multiple') {
    await runPoints([...pendingPoints.value])
  }
}

async function onPoint(point: { x: number, y: number }) {
  if (!hasImage.value || inferring.value) return
  if (mode.value === 'single') {
    pendingPoints.value = [point]
    await runPoints([point])
    return
  }
  if (mode.value === 'multiple') {
    pendingPoints.value.push(point)
  }
}

function onClear() {
  resetInference()
}

watch(mode, () => {
  pendingPoints.value = []
})

onMounted(() => {
  loadSamples()
})

onBeforeUnmount(() => {
  revokeImageUrl()
})
</script>

<template>
  <div class="relative h-dvh w-full overflow-hidden bg-gray-950 text-white">
    <aside class="absolute left-0 top-0 bottom-0 z-20 flex">
      <FloorplanNav />
      <div class="flex w-72 flex-col border-r border-gray-800/80 bg-gray-950/85 backdrop-blur-md">
        <FlorenceExtractionPanel
          class="min-h-0 flex-1"
          :samples="samples"
          :selected-sample="selectedSample"
          :selected-file="selectedFile"
          :loading="loadingSamples"
          :sample-url="api.florenceSampleUrl"
          @upload="onUpload"
          @select-sample="onSelectSample"
        />
      </div>
    </aside>

    <main class="absolute inset-0 z-0 pl-[21.5rem] pr-80">
      <FlorenceInferenceCanvas
        v-if="hasImage && imageUrl"
        ref="canvasRef"
        class="h-full"
        :image-url="imageUrl"
        :image-size="imageSize"
        :rooms="rooms"
        :pending-points="pendingPoints"
        :clickable="mode !== 'automatic' && !inferring"
        @point="onPoint"
      />
      <div
        v-else
        class="grid h-full place-items-center text-sm text-gray-500"
      >
        Upload or choose a test image to begin
      </div>
    </main>

    <div class="absolute top-4 left-[21.5rem] right-80 z-10 flex justify-center pointer-events-none">
      <FlorenceModeBar
        v-model="mode"
        :inferring="inferring"
        :can-infer="canInfer"
        :pending-count="pendingPoints.length"
        @infer="onInfer"
        @clear="onClear"
      />
    </div>

    <aside
      class="absolute right-0 top-0 bottom-0 z-10 w-80 flex flex-col border-l border-gray-800/80 bg-gray-950/85 backdrop-blur-md"
    >
      <FloorplanRightPanel
        :scores="scores"
        :plan="plan"
      />
    </aside>

    <FloorplanEditorBar
      class="absolute bottom-4 left-[21.5rem] right-80 z-10"
      @reset-view="canvasRef?.resetView()"
    />

    <UAlert
      v-if="error"
      color="error"
      variant="soft"
      :title="error"
      class="absolute top-16 left-1/2 z-20 w-[min(28rem,calc(100%-24rem))] -translate-x-1/2"
    />
  </div>
</template>
