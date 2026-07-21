<script setup lang="ts">
export interface FloorplanEntry {
  id: string
  label: string
  totalRooms: number
  overall: number | null
}

const props = defineProps<{
  items: FloorplanEntry[]
  selectedId: string | null
  loadingId?: string | null
  generating?: boolean
  roomOptions?: number[]
}>()

const emit = defineEmits<{
  (e: 'select', id: string): void
  (e: 'generate', totalRooms: number): void
}>()

const rooms = ref(3)

watch(
  () => props.items.find(i => i.id === props.selectedId)?.totalRooms,
  (value) => {
    if (value != null) rooms.value = value
  },
)
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="shrink-0 p-4 border-b border-gray-800/80 space-y-3">
      <div>
        <p class="text-xs uppercase tracking-widest text-gray-500">
          Supafloor
        </p>
        <h1 class="text-lg font-semibold text-white">
          Floorplans
        </h1>
      </div>

      <div class="flex items-center gap-2">
        <USelect
          v-model="rooms"
          :items="(roomOptions ?? [1, 2, 3, 4]).map(n => ({ label: `${n}-room`, value: n }))"
          value-key="value"
          class="flex-1"
          size="sm"
        />
        <UButton
          size="sm"
          color="primary"
          :loading="generating"
          :disabled="generating"
          @click="emit('generate', Number(rooms))"
        >
          Generate
        </UButton>
      </div>
    </div>

    <ul class="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
      <li
        v-for="item in items"
        :key="item.id"
      >
        <button
          type="button"
          class="w-full text-left rounded-lg px-3 py-2.5 transition-colors"
          :class="item.id === selectedId
            ? 'bg-primary-500/20 ring-1 ring-primary-500/50'
            : 'hover:bg-white/5'"
          @click="emit('select', item.id)"
        >
          <div class="flex items-center justify-between gap-2">
            <span class="text-sm font-medium text-white truncate">
              {{ item.label }}
            </span>
            <span
              v-if="loadingId === item.id"
              class="text-xs text-gray-500 shrink-0"
            >
              …
            </span>
            <UBadge
              v-else-if="item.overall != null"
              color="neutral"
              variant="soft"
              size="sm"
              class="shrink-0"
            >
              {{ item.overall.toFixed(1) }}
            </UBadge>
          </div>
        </button>
      </li>
      <li
        v-if="!items.length"
        class="px-3 py-6 text-sm text-gray-500 text-center"
      >
        No floorplans yet
      </li>
    </ul>
  </div>
</template>
