<script setup lang="ts">
import type { FloorPlan } from '../../types/floorplan'

const props = defineProps<{
  svg?: string
  plan?: FloorPlan | null
  highlightRoomId?: string | null
}>()

const emit = defineEmits<{
  (e: 'select-room', roomId: string): void
}>()

const viewport = ref({ scale: 1, x: 0, y: 0 })
const dragging = ref(false)
const last = ref({ x: 0, y: 0 })

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  viewport.value.scale = Math.min(4, Math.max(0.4, viewport.value.scale * delta))
}

function onPointerDown(e: PointerEvent) {
  dragging.value = true
  last.value = { x: e.clientX, y: e.clientY }
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return
  viewport.value.x += e.clientX - last.value.x
  viewport.value.y += e.clientY - last.value.y
  last.value = { x: e.clientX, y: e.clientY }
}

function onPointerUp() {
  dragging.value = false
}

function resetView() {
  viewport.value = { scale: 1, x: 0, y: 0 }
}

function onSvgClick(e: MouseEvent) {
  if (!props.plan) return
  // Hit-test rooms by finding nearest centroid in screen space is complex;
  // instead use room list selection. Optional: parse data attributes later.
  emit('select-room', props.highlightRoomId || props.plan.rooms[0]?.id)
}
</script>

<template>
  <div class="canvas-wrap">
    <div class="toolbar">
      <button type="button" @click="resetView">Reset view</button>
      <span class="hint">Scroll to zoom · drag to pan</span>
    </div>
    <div
      class="viewport"
      @wheel="onWheel"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
      @click="onSvgClick"
    >
      <div
        class="stage"
        :style="{
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`,
        }"
        v-html="svg || ''"
      />
    </div>
  </div>
</template>

<style scoped>
.canvas-wrap {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 420px;
  background:
    radial-gradient(ellipse at 20% 0%, rgba(196, 92, 38, 0.08), transparent 50%),
    linear-gradient(165deg, #f7f3ec 0%, #ebe4d8 45%, #e2d8c8 100%);
  border-right: 1px solid #d9d0c3;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.6rem 0.9rem;
  border-bottom: 1px solid #d9d0c3;
}
.toolbar button {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  background: #2a241c;
  color: #f7f3ec;
  border: none;
  padding: 0.35rem 0.75rem;
  cursor: pointer;
}
.hint {
  font-size: 0.8rem;
  color: #6b6156;
}
.viewport {
  flex: 1;
  overflow: hidden;
  cursor: grab;
  display: grid;
  place-items: center;
}
.viewport:active {
  cursor: grabbing;
}
.stage {
  transform-origin: center center;
  transition: transform 80ms linear;
}
.stage :deep(svg) {
  max-width: 100%;
  height: auto;
  display: block;
  filter: drop-shadow(0 12px 28px rgba(42, 36, 28, 0.12));
}
</style>
