<script setup lang="ts">
import { FLOORPLAN_LINEAR_UNIT } from '~/types/floorplan'

defineProps<{
  importing?: boolean
}>()

const emit = defineEmits<{
  (e: 'import', payload: { file: File, feetPerPixel: number }): void
}>()

const fileInput = ref<HTMLInputElement | null>(null)
const selectedFile = ref<File | null>(null)
const feetPerPixel = ref(0.05)

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  selectedFile.value = input.files?.[0] ?? null
}

function onImport() {
  if (!selectedFile.value) return
  emit('import', { file: selectedFile.value, feetPerPixel: feetPerPixel.value })
}

function openFilePicker() {
  fileInput.value?.click()
}
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="shrink-0 p-4 border-b border-gray-800/80 space-y-3">
      <FloorplanNav />

      <div>
        <p class="text-xs uppercase tracking-widest text-gray-500">
          Supafloor
        </p>
        <h1 class="text-lg font-semibold text-white">
          Importer
        </h1>
        <p class="mt-1 text-xs text-gray-500 leading-relaxed">
          BW image (black rooms on white) → rough floor plan via OpenCV contours.
        </p>
      </div>

      <input
        ref="fileInput"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        class="hidden"
        @change="onFileChange"
      >

      <UButton
        size="sm"
        color="neutral"
        variant="outline"
        class="w-full justify-center"
        @click="openFilePicker"
      >
        {{ selectedFile ? selectedFile.name : 'Choose image…' }}
      </UButton>

      <label class="block text-xs text-gray-400">
        Scale ({{ FLOORPLAN_LINEAR_UNIT }}/px)
        <UInput
          v-model.number="feetPerPixel"
          type="number"
          step="0.01"
          min="0.01"
          size="sm"
          class="mt-1"
        />
      </label>

      <UButton
        size="sm"
        color="primary"
        class="w-full justify-center"
        :disabled="importing || !selectedFile"
        @click="onImport"
      >
        {{ importing ? 'Importing…' : 'Import' }}
      </UButton>
    </div>

    <div class="flex-1 overflow-y-auto custom-scrollbar p-4 text-xs text-gray-500 space-y-2">
      <p>
        <span class="text-gray-400">POST</span> /floorplan/import
      </p>
      <p>
        Output is rooms and boundary only — no doors or windows. Room types are guessed from geometry.
      </p>
    </div>
  </div>
</template>
