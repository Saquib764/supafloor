<script setup lang="ts">
const api = useFloorplanApi()

const templates = ref<{ id: string; name: string }[]>([])
const templateId = ref('compact_2bed')
const plan = ref<FloorPlan | null>(null)
const scores = ref<ScoreBreakdown | null>(null)
const svg = ref('')
const adjacency = ref<Record<string, string[]>>({})
const selectedRoomId = ref<string | null>(null)
const inspection = ref<RoomInspection | null>(null)
const inspecting = ref(false)
const error = ref<string | null>(null)
const showLabels = ref(true)
const showDimensions = ref(false)
const showTopology = ref(true)
const showFurniture = ref(false)
const jsonOpen = ref(true)

const templateItems = computed(() =>
  templates.value.map(t => ({ label: t.name, value: t.id })),
)

const jsonPayload = computed(() =>
  JSON.stringify({ plan: plan.value, scores: scores.value, adjacency: adjacency.value }, null, 2),
)

const layers = computed(() => {
  const list = ['boundary', 'rooms', 'doors', 'windows']
  if (showLabels.value) list.push('labels', 'room_ids')
  if (showDimensions.value) list.push('dimensions')
  if (showTopology.value) list.push('topology')
  if (showFurniture.value) list.push('circulation')
  return list
})

async function loadDemo() {
  error.value = null
  try {
    const data = await api.getDemo(templateId.value)
    plan.value = data.plan
    scores.value = data.scores
    adjacency.value = data.adjacency
    selectedRoomId.value = data.plan.rooms[0]?.id ?? null
    await refreshRender()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

async function refreshRender() {
  if (!plan.value) return
  const rendered = await api.render(plan.value, layers.value)
  svg.value = rendered.svg
}

async function onGenerate() {
  error.value = null
  try {
    const sample = await api.generate(templateId.value, Date.now() % 10000)
    plan.value = sample.floor_plan
    scores.value = sample.scores
    svg.value = sample.svg || ''
    selectedRoomId.value = sample.floor_plan.rooms[0]?.id ?? null
    if (!sample.svg) await refreshRender()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

async function onMutate(op: string) {
  if (!plan.value) return
  const result = await api.mutate(plan.value, op, Date.now() % 1000)
  plan.value = result.plan
  scores.value = await api.evaluate(plan.value)
  await refreshRender()
}

async function selectRoom(roomId: string) {
  selectedRoomId.value = roomId
  if (!plan.value) return
  inspecting.value = true
  try {
    inspection.value = await api.inspect(plan.value, roomId)
  } finally {
    inspecting.value = false
  }
}

watch(layers, () => {
  refreshRender().catch(() => undefined)
})

watch(selectedRoomId, (id) => {
  if (id) selectRoom(id)
})

onMounted(async () => {
  try {
    templates.value = await api.listTemplates()
  } catch {
    templates.value = [{ id: 'compact_2bed', name: 'Compact 2 Bedroom' }]
  }
  await loadDemo()
})
</script>

<template>
  <div class="flex flex-col min-h-screen bg-gray-950 text-white">
    <header class="flex flex-wrap gap-4 items-end justify-between p-4 border-b border-gray-800 bg-gray-900">
      <div>
        <p class="text-xs uppercase tracking-widest text-gray-500">
          Supafloor
        </p>
        <h1 class="text-2xl font-bold">
          FloorPlan Engine
        </h1>
      </div>

      <div class="flex flex-wrap gap-2 items-center">
        <USelect
          v-model="templateId"
          :items="templateItems"
          class="min-w-48"
          @update:model-value="loadDemo"
        />
        <UButton
          variant="outline"
          color="neutral"
          @click="loadDemo"
        >
          Reload
        </UButton>
        <UButton
          color="primary"
          @click="onGenerate"
        >
          Generate
        </UButton>
        <UButton
          variant="ghost"
          color="neutral"
          @click="onMutate('mirror')"
        >
          Mirror
        </UButton>
        <UButton
          variant="ghost"
          color="neutral"
          @click="onMutate('swap_rooms')"
        >
          Swap rooms
        </UButton>
      </div>

      <div class="flex flex-wrap gap-3 items-center">
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
          v-model="showFurniture"
          label="Circulation"
        />
      </div>
    </header>

    <UAlert
      v-if="error"
      color="error"
      variant="soft"
      :title="error"
      class="mx-4 mt-4"
    />

    <div class="flex-1 grid grid-cols-1 lg:grid-cols-[1.4fr_0.7fr_0.8fr] min-h-[520px]">
      <section class="flex flex-col min-w-0">
        <FloorplanCanvas
          :svg="svg"
          :plan="plan"
          :highlight-room-id="selectedRoomId"
          @select-room="selectRoom"
        />
        <div
          v-if="plan"
          class="flex flex-wrap gap-2 p-3 border-t border-gray-800 bg-gray-900"
        >
          <UButton
            v-for="room in plan.rooms"
            :key="room.id"
            size="sm"
            :variant="room.id === selectedRoomId ? 'solid' : 'outline'"
            :color="room.id === selectedRoomId ? 'primary' : 'neutral'"
            @click="selectRoom(room.id)"
          >
            {{ room.name }}
          </UButton>
        </div>
      </section>

      <FloorplanInspector
        :inspection="inspection"
        :loading="inspecting"
      />
      <FloorplanScores :scores="scores" />
    </div>

    <div class="border-t border-gray-800 bg-gray-900">
      <UButton
        variant="ghost"
        color="neutral"
        block
        class="justify-start rounded-none"
        @click="jsonOpen = !jsonOpen; void 0"
      >
        {{ jsonOpen ? 'Hide' : 'Show' }} JSON
      </UButton>
      <pre
        v-if="jsonOpen"
        class="p-4 text-xs overflow-auto max-h-56 text-gray-300 custom-scrollbar"
      >{{ jsonPayload }}</pre>
    </div>
  </div>
</template>
