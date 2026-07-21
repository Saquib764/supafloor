<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import FloorplanCanvas from '../../components/floorplan/FloorplanCanvas.vue'
import FloorplanInspector from '../../components/floorplan/FloorplanInspector.vue'
import FloorplanScores from '../../components/floorplan/FloorplanScores.vue'
import { useFloorplanApi } from '../../composables/floorplan/useFloorplanApi'
import type { FloorPlan, RoomInspection, ScoreBreakdown } from '../../types/floorplan'

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
  <div class="page">
    <header class="top">
      <div class="brand">
        <p class="eyebrow">Supafloor</p>
        <h1>FloorPlan Engine</h1>
      </div>
      <div class="controls">
        <label>
          Template
          <select v-model="templateId" @change="loadDemo">
            <option v-for="t in templates" :key="t.id" :value="t.id">{{ t.name }}</option>
          </select>
        </label>
        <button type="button" @click="loadDemo">Reload</button>
        <button type="button" class="accent" @click="onGenerate">Generate</button>
        <button type="button" @click="onMutate('mirror')">Mirror</button>
        <button type="button" @click="onMutate('swap_rooms')">Swap rooms</button>
      </div>
      <div class="toggles">
        <label><input v-model="showLabels" type="checkbox" /> Labels</label>
        <label><input v-model="showDimensions" type="checkbox" /> Dimensions</label>
        <label><input v-model="showTopology" type="checkbox" /> Topology</label>
        <label><input v-model="showFurniture" type="checkbox" /> Circulation</label>
      </div>
    </header>

    <p v-if="error" class="error">{{ error }}</p>

    <div class="workspace">
      <section class="left">
        <FloorplanCanvas
          :svg="svg"
          :plan="plan"
          :highlight-room-id="selectedRoomId"
          @select-room="selectRoom"
        />
        <div class="room-chips" v-if="plan">
          <button
            v-for="room in plan.rooms"
            :key="room.id"
            type="button"
            :class="{ active: room.id === selectedRoomId }"
            @click="selectRoom(room.id)"
          >
            {{ room.name }}
          </button>
        </div>
      </section>
      <FloorplanInspector :inspection="inspection" :loading="inspecting" />
      <FloorplanScores :scores="scores" />
    </div>

    <section class="json-panel" :class="{ open: jsonOpen }">
      <button type="button" class="json-toggle" @click="jsonOpen = !jsonOpen">
        {{ jsonOpen ? 'Hide' : 'Show' }} JSON
      </button>
      <pre v-if="jsonOpen">{{ JSON.stringify({ plan, scores, adjacency }, null, 2) }}</pre>
    </section>
  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: #2a241c;
  background: #e8e0d4;
}
.top {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: end;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  background:
    linear-gradient(120deg, rgba(42, 36, 28, 0.92), rgba(90, 58, 36, 0.88)),
    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><path d="M0 40h80M40 0v80" stroke="%23ffffff11" stroke-width="1"/></svg>');
  color: #f7f3ec;
}
.brand .eyebrow {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.7rem;
  opacity: 0.75;
}
.brand h1 {
  margin: 0.15rem 0 0;
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(1.6rem, 3vw, 2.2rem);
  font-weight: 700;
}
.controls, .toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  align-items: center;
}
label {
  display: flex;
  gap: 0.35rem;
  align-items: center;
  font-size: 0.85rem;
}
select, button {
  font: inherit;
  border: 1px solid #cbbfac;
  background: #f7f3ec;
  color: #2a241c;
  padding: 0.35rem 0.65rem;
  cursor: pointer;
}
button.accent {
  background: #c45c26;
  border-color: #c45c26;
  color: #fff;
}
.workspace {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(220px, 0.7fr) minmax(240px, 0.8fr);
  min-height: 520px;
}
.left {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.room-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  padding: 0.6rem 0.75rem;
  background: #efe8dc;
  border-top: 1px solid #d9d0c3;
}
.room-chips button.active {
  background: #2a241c;
  color: #f7f3ec;
}
.json-panel {
  border-top: 1px solid #d9d0c3;
  background: #1f1a14;
  color: #e8dfd2;
  max-height: 220px;
  overflow: auto;
}
.json-toggle {
  position: sticky;
  top: 0;
  width: 100%;
  background: #2a241c;
  color: #f7f3ec;
  border: none;
  text-align: left;
  padding: 0.5rem 0.9rem;
}
pre {
  margin: 0;
  padding: 0.75rem 1rem 1.25rem;
  font-size: 0.72rem;
  line-height: 1.4;
}
.error {
  margin: 0;
  padding: 0.6rem 1rem;
  background: #f3d5cf;
  color: #7a2a1e;
}
@media (max-width: 960px) {
  .workspace {
    grid-template-columns: 1fr;
    grid-template-rows: 420px auto auto;
  }
}
</style>
