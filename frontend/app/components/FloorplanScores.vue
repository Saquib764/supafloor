<script setup lang="ts">
defineProps<{
  scores: ScoreBreakdown | null
}>()

const categories = [
  'geometry',
  'topology',
  'construction',
  'privacy',
  'circulation',
  'furniture',
  'daylight',
  'architecture',
  'aesthetics',
] as const
</script>

<template>
  <UCard
    :ui="{
      root: 'h-full overflow-auto rounded-none border-0 bg-gray-900 custom-scrollbar',
      body: 'p-4 space-y-4',
    }"
  >
    <h2 class="text-lg font-semibold text-white">
      Rule scores
    </h2>

    <p
      v-if="!scores"
      class="text-sm text-gray-400"
    >
      No evaluation yet.
    </p>

    <template v-else>
      <div class="flex justify-between items-baseline pb-3 border-b border-gray-800">
        <span class="text-gray-400">Overall</span>
        <strong class="text-3xl text-primary-500">{{ scores.overall.toFixed(2) }}</strong>
      </div>

      <div class="grid gap-3">
        <div
          v-for="c in categories"
          :key="c"
          class="grid grid-cols-[6.5rem_1fr_2rem] gap-2 items-center text-xs capitalize"
        >
          <span class="text-gray-400">{{ c }}</span>
          <div class="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              class="h-full bg-primary-500 rounded-full"
              :style="{ width: `${(scores[c] / 10) * 100}%` }"
            />
          </div>
          <span class="text-right text-gray-300">{{ scores[c].toFixed(1) }}</span>
        </div>
      </div>

      <USeparator />

      <h3 class="text-sm font-semibold text-gray-300">
        Rules
      </h3>

      <ul class="space-y-0">
        <li
          v-for="r in scores.rules"
          :key="r.rule_id"
          class="py-3 border-b border-gray-800 text-sm"
          :class="r.hard && !r.passed ? 'bg-error-500/10' : ''"
        >
          <div class="flex justify-between gap-2 items-start">
            <div class="flex items-center gap-2 flex-wrap">
              <strong class="text-white">{{ r.name }}</strong>
              <UBadge
                v-if="r.hard && !r.passed"
                color="error"
                variant="soft"
                size="sm"
              >
                Fail
              </UBadge>
            </div>
            <span class="text-gray-300 shrink-0">{{ r.score.toFixed(1) }}</span>
          </div>
          <p class="text-gray-400 mt-1">
            {{ r.reason }}
          </p>
          <p
            v-if="r.repair_suggestion"
            class="text-primary-400 mt-1"
          >
            Repair: {{ r.repair_suggestion }}
          </p>
        </li>
      </ul>
    </template>
  </UCard>
</template>
