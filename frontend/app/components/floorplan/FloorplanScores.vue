<script setup lang="ts">
import type { ScoreBreakdown } from '../../types/floorplan'

defineProps<{
  scores: ScoreBreakdown | null
}>()

const categories = [
  'geometry',
  'topology',
  'construction',
  'privacy',
  'circulation',
  'furniture',
  'daylight',
  'architecture',
  'aesthetics',
] as const
</script>

<template>
  <aside class="scores-panel">
    <h2>Rule scores</h2>
    <p v-if="!scores" class="muted">No evaluation yet.</p>
    <template v-else>
      <div class="overall">
        <span>Overall</span>
        <strong>{{ scores.overall.toFixed(2) }}</strong>
      </div>
      <div class="cats">
        <div v-for="c in categories" :key="c" class="cat">
          <span>{{ c }}</span>
          <div class="bar"><i :style="{ width: `${(scores[c] / 10) * 100}%` }" /></div>
          <em>{{ scores[c].toFixed(1) }}</em>
        </div>
      </div>
      <h3>Rules</h3>
      <ul>
        <li
          v-for="r in scores.rules"
          :key="r.rule_id"
          :class="{ fail: r.hard && !r.passed, soft: !r.hard }"
        >
          <div class="row">
            <strong>{{ r.name }}</strong>
            <span>{{ r.score.toFixed(1) }}</span>
          </div>
          <p>{{ r.reason }}</p>
          <p v-if="r.repair_suggestion" class="repair">Repair: {{ r.repair_suggestion }}</p>
        </li>
      </ul>
    </template>
  </aside>
</template>

<style scoped>
.scores-panel {
  padding: 1rem;
  background: #efe8dc;
  overflow: auto;
  height: 100%;
}
h2, h3 {
  font-family: 'Fraunces', Georgia, serif;
  margin: 0 0 0.75rem;
  color: #2a241c;
}
.overall {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.75rem 0;
  border-bottom: 1px solid #d9d0c3;
  margin-bottom: 0.75rem;
  font-family: 'Fraunces', Georgia, serif;
}
.overall strong {
  font-size: 2rem;
  color: #c45c26;
}
.cats { display: grid; gap: 0.4rem; margin-bottom: 1rem; }
.cat {
  display: grid;
  grid-template-columns: 6.5rem 1fr 2rem;
  gap: 0.4rem;
  align-items: center;
  font-size: 0.8rem;
  text-transform: capitalize;
}
.bar {
  height: 6px;
  background: #d9d0c3;
  overflow: hidden;
}
.bar i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #8f4b2a, #c45c26);
}
em { font-style: normal; text-align: right; }
ul { list-style: none; padding: 0; margin: 0; }
li {
  padding: 0.55rem 0;
  border-bottom: 1px solid #e0d7ca;
  font-size: 0.82rem;
}
li.fail { background: rgba(180, 50, 40, 0.08); }
.row { display: flex; justify-content: space-between; gap: 0.5rem; }
p { margin: 0.2rem 0 0; color: #5c5348; }
.repair { color: #8f4b2a; }
.muted { color: #8a7f72; }
</style>
