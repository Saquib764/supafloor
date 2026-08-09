<script setup lang="ts">
export interface DatasetListEntry {
  stem: string
}

const props = defineProps<{
  items: DatasetListEntry[]
  selectedStem: string | null
  loading?: boolean
  loadingMore?: boolean
  total?: number
}>()

const emit = defineEmits<{
  (e: 'select', stem: string): void
  (e: 'load-more'): void
}>()

const scrollRoot = ref<HTMLElement | null>(null)
const loadMoreRef = ref<HTMLElement | null>(null)

onMounted(() => {
  const root = scrollRoot.value
  const sentinel = loadMoreRef.value
  if (!root || !sentinel) return

  const observer = new IntersectionObserver(
    (entries) => {
      if (entries[0]?.isIntersecting) {
        emit('load-more')
      }
    },
    { root, rootMargin: '120px' },
  )
  observer.observe(sentinel)
  onBeforeUnmount(() => observer.disconnect())
})
</script>

<template>
  <div class="flex h-full flex-col">
    <div class="shrink-0 space-y-3 border-b border-gray-800/80 p-4">
      <div>
        <p class="text-xs uppercase tracking-widest text-gray-500">
          Supafloor
        </p>
        <h1 class="text-lg font-semibold text-white">
          Dataset
        </h1>
        <p class="mt-1 text-xs leading-relaxed text-gray-500">
          basic_5000 image + extraction JSON pairs.
        </p>
      </div>
      <p
        v-if="total != null"
        class="text-xs text-gray-500"
      >
        {{ items.length.toLocaleString() }} / {{ total.toLocaleString() }}
      </p>
    </div>

    <div
      ref="scrollRoot"
      class="custom-scrollbar min-h-0 flex-1 overflow-y-auto p-2"
    >
      <p
        v-if="loading && !items.length"
        class="px-3 py-6 text-center text-sm text-gray-500"
      >
        Loading…
      </p>

      <ul
        v-else
        class="space-y-1"
      >
        <li
          v-for="item in items"
          :key="item.stem"
        >
          <button
            type="button"
            class="w-full rounded-lg px-3 py-2.5 text-left text-sm font-medium transition-colors truncate"
            :class="item.stem === selectedStem
              ? 'bg-primary-500/20 text-white ring-1 ring-primary-500/50'
              : 'text-gray-300 hover:bg-white/5'"
            @click="emit('select', item.stem)"
          >
            {{ item.stem }}
          </button>
        </li>
      </ul>

      <div
        ref="loadMoreRef"
        class="px-3 py-4 text-center text-xs text-gray-500"
      >
        <span v-if="loadingMore">Loading more…</span>
        <span v-else-if="total != null && items.length >= total">End of list</span>
      </div>
    </div>
  </div>
</template>
