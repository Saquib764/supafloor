<script setup lang="ts">
import { FLOORPLAN_AREA_UNIT, FLOORPLAN_TARGET_AREAS_SQFT } from '~/types/floorplan'

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
  deletingId?: string | null
  generating?: boolean
  roomOptions?: number[]
  areaOptions?: number[]
}>()

const emit = defineEmits<{
  (e: 'select', id: string): void
  (e: 'generate', payload: { totalRooms: number, targetArea: number }): void
  (e: 'delete', id: string): void
}>()

const rooms = ref(3)
const targetArea = ref<number>(FLOORPLAN_TARGET_AREAS_SQFT[1])
const settingsOpen = ref(false)

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
      <div class="flex items-start gap-2">
        <FloorplanNav class="flex-1" />
        <UButton
          size="sm"
          color="neutral"
          variant="ghost"
          icon="i-lucide-settings"
          aria-label="Settings"
          @click="settingsOpen = true"
        />
      </div>

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
          @click="emit('generate', { totalRooms: Number(rooms), targetArea: Number(targetArea) })"
        >
          Generate
        </UButton>
      </div>

      <USelect
        v-model="targetArea"
        :items="(areaOptions ?? [...FLOORPLAN_TARGET_AREAS_SQFT]).map(a => ({
          label: `${a.toLocaleString()} ${FLOORPLAN_AREA_UNIT}`,
          value: a,
        }))"
        value-key="value"
        size="sm"
      />
    </div>

    <ul class="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
      <li
        v-for="item in items"
        :key="item.id"
      >
        <div
          class="flex items-center gap-2 rounded-lg px-3 py-2.5 transition-colors"
          :class="item.id === selectedId
            ? 'bg-primary-500/20 ring-1 ring-primary-500/50'
            : 'hover:bg-white/5'"
        >
          <button
            type="button"
            class="min-w-0 flex-1 text-left text-sm font-medium text-white truncate"
            @click="emit('select', item.id)"
          >
            {{ item.label }}
          </button>
          <div class="flex items-center gap-1 shrink-0">
            <span
              v-if="loadingId === item.id"
              class="text-xs text-gray-500"
            >
              …
            </span>
            <UBadge
              v-else-if="item.overall != null"
              color="neutral"
              variant="soft"
              size="sm"
            >
              {{ item.overall.toFixed(1) }}
            </UBadge>
            <UButton
              size="xs"
              color="error"
              variant="ghost"
              icon="i-lucide-trash-2"
              :loading="deletingId === item.id"
              :disabled="deletingId === item.id"
              aria-label="Delete floorplan"
              @click.stop="emit('delete', item.id)"
            />
          </div>
        </div>
      </li>
      <li
        v-if="!items.length"
        class="px-3 py-6 text-sm text-gray-500 text-center"
      >
        No floorplans yet
      </li>
    </ul>

    <FloorplanSettingsModal v-model:open="settingsOpen" />
  </div>
</template>
