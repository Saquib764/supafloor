<script setup lang="ts">
import type { FlorenceRoom } from '~/components/FlorenceExtractionPanel.vue'

const props = defineProps<{
  imageUrl: string
  imageSize: number
  rooms: FlorenceRoom[]
  pendingPoints: Array<{ x: number, y: number }>
  clickable?: boolean
}>()

const emit = defineEmits<{
  (e: 'point', point: { x: number, y: number }): void
}>()

const viewport = ref({ scale: 1, x: 0, y: 0 })
const dragging = ref(false)
const last = ref({ x: 0, y: 0 })
const stageRef = ref<HTMLElement | null>(null)

const fillColors = [
  'rgba(232, 213, 183, 0.55)',
  'rgba(212, 230, 195, 0.55)',
  'rgba(201, 214, 223, 0.55)',
  'rgba(184, 197, 214, 0.55)',
  'rgba(240, 237, 230, 0.55)',
  'rgba(221, 213, 200, 0.55)',
  'rgba(168, 197, 184, 0.55)',
  'rgba(229, 229, 229, 0.55)',
]

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  viewport.value.scale = Math.min(4, Math.max(0.4, viewport.value.scale * delta))
}

function onPointerDown(e: PointerEvent) {
  if (props.clickable && e.button === 0) {
    return
  }
  dragging.value = true
  last.value = { x: e.clientX, y: e.clientY }
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return
  const dx = e.clientX - last.value.x
  const dy = e.clientY - last.value.y
  viewport.value.x += dx
  viewport.value.y += dy
  last.value = { x: e.clientX, y: e.clientY }
}

function onPointerUp(e: PointerEvent) {
  dragging.value = false
}

function onStageClick(e: MouseEvent) {
  if (!props.clickable || e.button !== 0) return
  const point = pointerToImage(e.clientX, e.clientY)
  if (point) {
    emit('point', point)
  }
}

function pointerToImage(clientX: number, clientY: number) {
  const stage = stageRef.value
  if (!stage) return null

  const rect = stage.getBoundingClientRect()
  const localX = clientX - rect.left
  const localY = clientY - rect.top
  if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) {
    return null
  }

  const x = (localX / rect.width) * props.imageSize
  const y = (localY / rect.height) * props.imageSize
  return {
    x: Math.max(0, Math.min(props.imageSize - 1, x)),
    y: Math.max(0, Math.min(props.imageSize - 1, y)),
  }
}

function polygonPoints(polygon: number[][]) {
  return polygon.map(([x, y]) => `${x},${y}`).join(' ')
}

function resetView() {
  viewport.value = { scale: 1, x: 0, y: 0 }
}

defineExpose({ resetView })
</script>

<template>
  <div
    class="flex h-full flex-col overflow-hidden bg-gray-950/40"
    @wheel="onWheel"
    @pointerdown="onPointerDown"
    @pointermove="onPointerMove"
    @pointerup="onPointerUp"
    @pointercancel="onPointerUp"
  >
    <div
      class="grid flex-1 place-items-center overflow-hidden"
      :class="clickable ? '' : 'cursor-grab active:cursor-grabbing'"
    >
      <div
        class="origin-center transition-transform duration-75"
        :style="{
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`,
        }"
      >
        <div
          ref="stageRef"
          class="relative shadow-2xl aspect-square w-[min(78vh,calc(100vw-26rem))] max-w-full"
          :class="clickable ? 'cursor-pointer' : ''"
          @click="onStageClick"
        >
          <img
            :src="imageUrl"
            alt="Floor plan"
            class="absolute inset-0 h-full w-full object-fill"
            draggable="false"
          >
          <svg
            class="absolute inset-0 h-full w-full pointer-events-none"
            :viewBox="`0 0 ${imageSize} ${imageSize}`"
            preserveAspectRatio="none"
          >
            <polygon
              v-for="(room, index) in rooms"
              :key="`${room.name}-${index}`"
              :points="polygonPoints(room.polygon)"
              :fill="fillColors[index % fillColors.length]"
              stroke="rgba(44, 44, 44, 0.95)"
              stroke-width="2"
            />
            <circle
              v-for="(point, index) in pendingPoints"
              :key="`pending-${index}`"
              :cx="point.x"
              :cy="point.y"
              r="6"
              fill="rgb(255, 200, 0)"
              stroke="rgba(44, 44, 44, 0.95)"
              stroke-width="2"
            />
          </svg>
        </div>
      </div>
    </div>
  </div>
</template>
