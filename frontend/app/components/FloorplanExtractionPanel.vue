<script setup lang="ts">
defineProps<{
  extracting?: boolean
}>()

const emit = defineEmits<{
  (e: 'extract', payload: { file: File, canvasSize: number }): void
}>()

const fileInput = ref<HTMLInputElement | null>(null)
const selectedFile = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const canvasSize = ref(1024)

function setPreview(file: File | null) {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
  if (file) {
    previewUrl.value = URL.createObjectURL(file)
  }
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] ?? null
  selectedFile.value = file
  setPreview(file)
}

function onExtract() {
  if (!selectedFile.value) return
  emit('extract', { file: selectedFile.value, canvasSize: canvasSize.value })
}

function openFilePicker() {
  fileInput.value?.click()
}

onBeforeUnmount(() => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
})
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
          Extract
        </h1>
        <p class="mt-1 text-xs text-gray-500 leading-relaxed">
          Floor plan image → room polygons via vision LLM extraction.
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
        Canvas size
        <UInput
          v-model.number="canvasSize"
          type="number"
          step="64"
          min="256"
          max="2048"
          size="sm"
          class="mt-1"
        />
      </label>

      <UButton
        size="sm"
        color="primary"
        class="w-full justify-center"
        :disabled="extracting || !selectedFile"
        @click="onExtract"
      >
        {{ extracting ? 'Extracting…' : 'Extract' }}
      </UButton>
    </div>

    <div class="flex-1 min-h-0 flex flex-col p-4 border-t border-gray-800/80">
      <p class="shrink-0 text-xs uppercase tracking-widest text-gray-500 mb-2">
        Preview
      </p>
      <div
        v-if="previewUrl"
        class="flex-1 min-h-0 rounded-md border border-gray-800 bg-gray-900/60 overflow-hidden flex items-center justify-center"
      >
        <img
          :src="previewUrl"
          :alt="selectedFile?.name ?? 'Uploaded floor plan'"
          class="max-w-full max-h-full object-contain"
        >
      </div>
      <p
        v-else
        class="flex-1 flex items-center justify-center rounded-md border border-dashed border-gray-800 text-xs text-gray-600 text-center px-4"
      >
        Choose an image to preview it here
      </p>
    </div>
  </div>
</template>
