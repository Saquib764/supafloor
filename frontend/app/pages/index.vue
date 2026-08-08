<script setup lang="ts">
import type { FloorplanEntry } from '~/components/FloorplanList.vue'
import { FLOORPLAN_DEFAULT_TARGET_AREA_SQFT, FLOORPLAN_TARGET_AREAS_SQFT } from '~/types/floorplan'

const api = useFloorplanApi()

interface LoadedFloorplan {
  id: string
  label: string
  bedroomCount: number
  plan: FloorPlan
  scores: ScoreBreakdown
  svg: string
  loaded: boolean
}

const roomOptions = [1, 2, 3, 4]
const areaOptions = [...FLOORPLAN_TARGET_AREAS_SQFT]
const floorplans = ref<LoadedFloorplan[]>([])
const selectedId = ref<string | null>(null)
const loadingId = ref<string | null>(null)
const deletingId = ref<string | null>(null)
const acting = ref(false)
const selectedRoomId = ref<string | null>(null)
const error = ref<string | null>(null)
const showLabels = ref(true)
const showDimensions = ref(false)
const showTopology = ref(true)
const showCirculation = ref(false)
const viewMode = ref<'simple' | 'realistic'>('simple')
const pngB64 = ref('')
const canvasRef = ref<{ resetView: () => void } | null>(null)

const listItems = computed<FloorplanEntry[]>(() =>
  floorplans.value.map(fp => ({
    id: fp.id,
    label: fp.label,
    bedroomCount: fp.bedroomCount,
    overall: fp.scores?.overall ?? null,
  })),
)

const selected = computed(() =>
  floorplans.value.find(fp => fp.id === selectedId.value) ?? null,
)

const layers = computed(() => {
  const list = ['boundary', 'rooms', 'doors', 'windows']
  if (showLabels.value) list.push('labels', 'room_ids')
  if (showDimensions.value) list.push('dimensions')
  if (showTopology.value) list.push('topology')
  if (showCirculation.value) list.push('circulation')
  return list
})

async function refreshRender(entry: LoadedFloorplan) {
  const rendered = await api.render(entry.plan, layers.value)
  entry.svg = rendered.svg
}

async function refreshRealistic(entry: LoadedFloorplan) {
  const exported = await api.exportRealistic(entry.plan)
  if (selectedId.value === entry.id) pngB64.value = exported.png_b64
}

async function refreshSelectedView() {
  const entry = selected.value
  if (!entry?.plan.rooms?.length) {
    pngB64.value = ''
    return
  }
  if (viewMode.value === 'realistic') {
    await refreshRealistic(entry)
  } else if (!entry.svg) {
    await refreshRender(entry)
  }
}

async function refreshSelectedRender() {
  if (!selected.value) return
  await refreshRender(selected.value)
}

function entryFromSample(sample: Record<string, any>): LoadedFloorplan {
  const requirements = sample.requirements ?? {}
  return {
    id: sample.id,
    label: requirements.label ?? `${requirements.bedroom_count ?? '?'}-room`,
    bedroomCount: Number(requirements.bedroom_count ?? 3),
    plan: sample.floor_plan,
    scores: sample.scores,
    svg: sample.svg || '',
    loaded: true,
  }
}

async function applySample(sample: Record<string, any>, select = true) {
  const entry = entryFromSample(sample)
  const idx = floorplans.value.findIndex(fp => fp.id === entry.id)
  if (idx >= 0) floorplans.value[idx] = entry
  else floorplans.value.unshift(entry)
  if (select) {
    selectedId.value = entry.id
    selectedRoomId.value = entry.plan.rooms[0]?.id ?? null
  }
  if (viewMode.value === 'realistic') await refreshRealistic(entry)
  else if (!entry.svg) await refreshRender(entry)
}

async function refreshList() {
  const plans = await api.listPlans()
  // Keep in-memory scores/svg when possible; rebuild list from disk order
  const byId = new Map(floorplans.value.map(fp => [fp.id, fp]))
  const next: LoadedFloorplan[] = []
  for (const item of plans) {
    const existing = byId.get(item.id)
    if (existing) {
      next.push(existing)
      continue
    }
    next.push({
      id: item.id,
      label: item.label,
      bedroomCount: item.bedroom_count,
      plan: { id: item.id, name: item.label, rooms: [], doors: [], windows: [] } as unknown as FloorPlan,
      scores: { overall: null } as unknown as ScoreBreakdown,
      svg: '',
      loaded: false,
    })
  }
  floorplans.value = next
  if (!selectedId.value && next.length) {
    await onSelect(next[0].id)
  }
}

async function onSelect(id: string) {
  selectedId.value = id
  loadingId.value = id
  error.value = null
  try {
    const entry = floorplans.value.find(fp => fp.id === id)
    if (!entry?.loaded || !entry.plan.rooms?.length) {
      const sample = await api.loadPlan(id)
      await applySample(sample, true)
    } else {
      selectedRoomId.value = entry.plan.rooms[0]?.id ?? null
      await refreshSelectedView()
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loadingId.value = null
  }
}

async function onGenerate(payload?: { bedroomCount?: number, targetArea?: number }) {
  const rooms = payload?.bedroomCount ?? selected.value?.bedroomCount ?? 3
  const targetArea = payload?.targetArea ?? FLOORPLAN_DEFAULT_TARGET_AREA_SQFT
  acting.value = true
  error.value = null
  try {
    const sample = await api.generate(rooms, Date.now() % 100000, targetArea)
    await applySample(sample, true)
    await refreshList()
    selectedId.value = sample.id
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    acting.value = false
  }
}

async function onDelete(id: string) {
  deletingId.value = id
  error.value = null
  try {
    await api.deletePlan(id)
    const wasSelected = selectedId.value === id
    floorplans.value = floorplans.value.filter(fp => fp.id !== id)
    if (wasSelected) {
      selectedRoomId.value = null
      const next = floorplans.value[0]
      selectedId.value = next?.id ?? null
      if (next) await onSelect(next.id)
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    deletingId.value = null
  }
}

async function onAction(action: string) {
  if (action === 'generate') {
    await onGenerate()
    return
  }

  const entry = selected.value
  if (!entry || !entry.plan.rooms?.length) return

  acting.value = true
  error.value = null
  try {
    if (action === 'evaluate') {
      entry.scores = await api.evaluate(entry.plan)
      return
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    acting.value = false
  }
}

watch(layers, () => {
  if (viewMode.value === 'simple') {
    refreshSelectedRender().catch(() => undefined)
  }
})

watch(viewMode, () => {
  refreshSelectedView().catch(() => undefined)
})

watch(() => selected.value?.plan, () => {
  refreshSelectedView().catch(() => undefined)
}, { deep: true })

onMounted(async () => {
  try {
    await refreshList()
    if (!floorplans.value.length) {
      await onGenerate({ bedroomCount: 3, targetArea: FLOORPLAN_DEFAULT_TARGET_AREA_SQFT })
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})
</script>

<template>
  <div class="relative h-dvh w-full overflow-hidden bg-gray-950 text-white">
    <FloorplanCanvas
      ref="canvasRef"
      class="absolute inset-0 z-0"
      background
      :view-mode="viewMode"
      :svg="selected?.svg"
      :png-b64="pngB64"
      :plan="selected?.plan ?? null"
      :highlight-room-id="selectedRoomId"
    />

    <aside
      class="absolute left-0 top-0 bottom-0 z-10 w-72 flex flex-col border-r border-gray-800/80 bg-gray-950/85 backdrop-blur-md"
    >
      <FloorplanList
        class="min-h-0 flex-1"
        :items="listItems"
        :selected-id="selectedId"
        :loading-id="loadingId"
        :deleting-id="deletingId"
        :generating="acting"
        :room-options="roomOptions"
        :area-options="areaOptions"
        @select="onSelect"
        @generate="onGenerate"
        @delete="onDelete"
      />
      <FloorplanActions
        :disabled="!selected"
        :busy="acting"
        @action="onAction"
      />
    </aside>

    <aside
      class="absolute right-0 top-0 bottom-0 z-10 w-80 flex flex-col border-l border-gray-800/80 bg-gray-950/85 backdrop-blur-md"
    >
      <FloorplanRightPanel
        :scores="selected?.scores ?? null"
        :plan="selected?.plan ?? null"
      />
    </aside>

    <div class="absolute top-4 left-72 right-80 z-10 flex justify-center pointer-events-none">
      <FloorplanViewModeToggle v-model="viewMode" />
    </div>

    <FloorplanEditorBar
      class="absolute bottom-4 left-72 right-80 z-10"
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
