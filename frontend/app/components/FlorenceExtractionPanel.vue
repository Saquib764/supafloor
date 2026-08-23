<script setup lang="ts">
export type InferenceMode = 'automatic' | 'single' | 'multiple'

export interface FlorenceSample {
  name: string
}

export interface FlorenceRoom {
  name: string
  polygon: number[][]
  count?: number
}

export interface FlorenceInferResult {
  rooms: FlorenceRoom[]
  image_size: number
  canvas_size: number
  floor_plan: FloorPlan
}

const props = defineProps<{
  samples: FlorenceSample[]
  selectedSample: string | null
  selectedFile: File | null
  loading?: boolean
  sampleUrl: (name: string) => string
}>()

const emit = defineEmits<{
  (e: 'upload', file: File): void
  (e: 'select-sample', name: string): void
}>()

const fileInput = ref<HTMLInputElement | null>(null)

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) {
    emit('upload', file)
  }
}

function openFilePicker() {
  fileInput.value?.click()
}
</script>

<template>
  <div class="flex h-full flex-col">
    <div class="shrink-0 space-y-3 border-b border-gray-800/80 p-4">
      <div>
        <p class="text-xs uppercase tracking-widest text-gray-500">
          Supafloor
        </p>
        <h1 class="text-lg font-semibold text-white">
          Extract
        </h1>
        <p class="mt-1 text-xs leading-relaxed text-gray-500">
          Florence inference on floor plan images.
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
        {{ selectedFile ? selectedFile.name : 'Upload image…' }}
      </UButton>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto p-2">
      <p class="px-2 pb-2 text-xs uppercase tracking-widest text-gray-500">
        Test samples
      </p>
      <p
        v-if="loading && !samples.length"
        class="px-3 py-6 text-center text-sm text-gray-500"
      >
        Loading…
      </p>
      <ul
        v-else
        class="space-y-1"
      >
        <li
          v-for="sample in samples"
          :key="sample.name"
        >
          <button
            type="button"
            class="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors hover:bg-white/5"
            :class="selectedSample === sample.name && !selectedFile
              ? 'bg-primary-500/15 text-white ring-1 ring-primary-500/30'
              : 'text-gray-300'"
            @click="emit('select-sample', sample.name)"
          >
            <img
              :src="sampleUrl(sample.name)"
              :alt="sample.name"
              class="size-10 shrink-0 rounded object-cover bg-gray-900"
            >
            <span class="truncate">{{ sample.name }}</span>
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>
