<script setup lang="ts">
import type { DatasetListEntry } from '~/components/DatasetItemList.vue'

const PAGE_SIZE = 50

const api = useFloorplanApi()

const items = ref<DatasetListEntry[]>([])
const selectedStem = ref<string | null>(null)
const selectedJson = ref<unknown>(null)
const total = ref<number | null>(null)
const loadingList = ref(false)
const loadingMore = ref(false)
const loadingItem = ref(false)
const error = ref<string | null>(null)

const imageUrl = computed(() =>
  selectedStem.value ? api.datasetImageUrl(selectedStem.value) : '',
)

const jsonText = computed(() =>
  selectedJson.value != null ? JSON.stringify(selectedJson.value, null, 2) : '',
)

async function loadMore() {
  if (loadingList.value || loadingMore.value) return
  if (total.value != null && items.value.length >= total.value) return

  const loading = items.value.length === 0
  if (loading) loadingList.value = true
  else loadingMore.value = true

  error.value = null
  try {
    const page = await api.listDatasetItems(items.value.length, PAGE_SIZE)
    items.value.push(...page.items)
    total.value = page.total
    if (!selectedStem.value && page.items.length) {
      await onSelect(page.items[0].stem)
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loadingList.value = false
    loadingMore.value = false
  }
}

async function onSelect(stem: string) {
  if (selectedStem.value === stem && selectedJson.value != null) return

  selectedStem.value = stem
  loadingItem.value = true
  error.value = null
  try {
    const loaded = await api.loadDatasetJson(stem)
    selectedJson.value = loaded.json
  } catch (e) {
    selectedJson.value = null
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loadingItem.value = false
  }
}

onMounted(() => {
  loadMore()
})
</script>

<template>
  <div class="relative h-dvh w-full overflow-hidden bg-gray-950 text-white">
    <aside class="absolute left-0 top-0 bottom-0 z-20 flex">
      <FloorplanNav />
      <div class="flex w-72 flex-col border-r border-gray-800/80 bg-gray-950/85 backdrop-blur-md">
        <DatasetItemList
          class="min-h-0 flex-1"
          :items="items"
          :selected-stem="selectedStem"
          :loading="loadingList"
          :loading-more="loadingMore"
          :total="total ?? undefined"
          @select="onSelect"
          @load-more="loadMore"
        />
      </div>
    </aside>

    <div class="absolute inset-0 left-[21.5rem] right-80 z-0 flex items-center justify-center bg-gray-950 p-6">
      <p
        v-if="!selectedStem && !loadingList"
        class="text-sm text-gray-500"
      >
        Select a dataset item.
      </p>
      <p
        v-else-if="loadingItem"
        class="text-sm text-gray-500"
      >
        Loading item…
      </p>
      <img
        v-else-if="imageUrl"
        :src="imageUrl"
        :alt="selectedStem ?? 'Dataset image'"
        class="max-h-full max-w-full rounded-lg border border-gray-800 object-contain shadow-2xl"
      >
    </div>

    <aside class="absolute right-0 top-0 bottom-0 z-10 flex w-80 flex-col border-l border-gray-800/80 bg-gray-950/85 backdrop-blur-md">
      <div class="shrink-0 border-b border-gray-800/80 p-4">
        <h2 class="text-sm font-semibold text-white">
          JSON
        </h2>
        <p
          v-if="selectedStem"
          class="mt-1 truncate text-xs text-gray-500"
        >
          {{ selectedStem }}.json
        </p>
      </div>
      <div class="custom-scrollbar min-h-0 flex-1 overflow-y-auto p-4">
        <p
          v-if="!selectedStem"
          class="text-sm text-gray-400"
        >
          Select an item to view extraction JSON.
        </p>
        <pre
          v-else-if="jsonText"
          class="overflow-x-auto rounded-lg border border-gray-800 bg-gray-950/60 p-3 text-xs leading-relaxed text-gray-300"
        ><code>{{ jsonText }}</code></pre>
      </div>
    </aside>

    <UAlert
      v-if="error"
      color="error"
      variant="soft"
      :title="error"
      class="absolute top-4 left-1/2 z-20 w-[min(28rem,calc(100%-24rem))] -translate-x-1/2"
    />
  </div>
</template>
