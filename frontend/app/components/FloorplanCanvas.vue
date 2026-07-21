<script setup lang="ts">
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

function onSvgClick() {
  if (!props.plan) return
  emit('select-room', props.highlightRoomId || props.plan.rooms[0]?.id)
}
</script>

<template>
  <div class="flex flex-col h-full min-h-[420px] bg-gray-950 border-r border-gray-800">
    <div class="flex items-center gap-3 p-3 border-b border-gray-800">
      <UButton
        size="sm"
        color="neutral"
        variant="outline"
        @click="resetView"
      >
        Reset view
      </UButton>
      <span class="text-sm text-gray-500">
        Scroll to zoom · drag to pan
      </span>
    </div>
    <div
      class="flex-1 overflow-hidden cursor-grab active:cursor-grabbing grid place-items-center"
      @wheel="onWheel"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
      @click="onSvgClick"
    >
      <div
        class="origin-center transition-transform duration-75 [&_svg]:max-w-full [&_svg]:h-auto [&_svg]:block [&_svg]:drop-shadow-lg"
        :style="{
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`,
        }"
        v-html="svg || ''"
      />
    </div>
  </div>
</template>
