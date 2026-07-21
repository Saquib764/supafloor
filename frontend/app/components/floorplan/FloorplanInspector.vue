<script setup lang="ts">
import type { RoomInspection } from '../../types/floorplan'

defineProps<{
  inspection: RoomInspection | null
  loading?: boolean
}>()
</script>

<template>
  <aside class="inspector">
    <h2>Inspector</h2>
    <p v-if="loading" class="muted">Loading…</p>
    <p v-else-if="!inspection" class="muted">Select a room to inspect geometry, adjacency, and scores.</p>
    <template v-else>
      <h3>{{ inspection.name }}</h3>
      <dl>
        <div><dt>Type</dt><dd>{{ inspection.room_type }}</dd></div>
        <div><dt>Zone</dt><dd>{{ inspection.zone }}</dd></div>
        <div><dt>Area</dt><dd>{{ inspection.area.toFixed(0) }} sqft</dd></div>
        <div><dt>Aspect</dt><dd>{{ inspection.aspect_ratio.toFixed(2) }}</dd></div>
        <div><dt>Perimeter</dt><dd>{{ inspection.perimeter.toFixed(1) }} ft</dd></div>
        <div><dt>Plan score</dt><dd>{{ inspection.plan_overall.toFixed(2) }}</dd></div>
      </dl>
      <h4>Adjacent rooms</h4>
      <ul>
        <li v-for="id in inspection.adjacent_rooms" :key="id">{{ id }}</li>
        <li v-if="!inspection.adjacent_rooms.length" class="muted">None detected</li>
      </ul>
      <h4>Doors</h4>
      <p>{{ inspection.doors.length }} connected</p>
      <h4>Rule hits</h4>
      <ul class="scores">
        <li v-for="r in inspection.individual_scores" :key="r.rule_id">
          <strong>{{ r.name }}</strong> {{ r.score.toFixed(1) }}
          <span class="muted">{{ r.reason }}</span>
        </li>
        <li v-if="!inspection.individual_scores.length" class="muted">No room-specific hits</li>
      </ul>
    </template>
  </aside>
</template>

<style scoped>
.inspector {
  padding: 1rem;
  background: #f3eee5;
  border-right: 1px solid #d9d0c3;
  overflow: auto;
  height: 100%;
}
h2, h3, h4 {
  font-family: 'Fraunces', Georgia, serif;
  color: #2a241c;
  margin: 0 0 0.5rem;
}
h2 { font-size: 1.15rem; letter-spacing: 0.02em; }
h3 { font-size: 1.35rem; margin-top: 0.5rem; }
h4 { font-size: 0.95rem; margin-top: 1rem; color: #5c5348; }
dl {
  display: grid;
  gap: 0.35rem;
}
dl div {
  display: grid;
  grid-template-columns: 7rem 1fr;
  gap: 0.5rem;
  font-size: 0.9rem;
}
dt { color: #6b6156; }
dd { margin: 0; color: #2a241c; font-weight: 500; }
ul { padding-left: 1.1rem; margin: 0.25rem 0; }
.muted { color: #8a7f72; font-size: 0.88rem; }
.scores li { margin-bottom: 0.45rem; }
</style>
