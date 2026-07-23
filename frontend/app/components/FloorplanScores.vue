<script setup lang="ts">
import {
  SCORE_LEVELS,
  activeCategoriesForLevel,
  overallForLevel,
  ruleActiveAtLevel,
  type ScoreLevel,
} from '~/types/floorplan'

const props = defineProps<{
  scores: ScoreBreakdown | null
}>()

const configuredLevel = useRuntimeConfig().public.scoreLevel as ScoreLevel | undefined
const level = ref<ScoreLevel>(
  configuredLevel && SCORE_LEVELS.includes(configuredLevel) ? configuredLevel : 'basic',
)

const hasScores = computed(() =>
  props.scores?.overall != null && Array.isArray(props.scores.rules),
)

const activeCategories = computed(() => activeCategoriesForLevel(level.value))

const categoryScores = computed(() => {
  if (!hasScores.value || !props.scores) return null
  const levelCats = props.scores.level_categories?.[level.value]
  if (levelCats) return levelCats
  return Object.fromEntries(activeCategories.value.map(c => [c, props.scores![c]]))
})

const activeRules = computed(() => {
  if (!hasScores.value || !props.scores?.rules) return []
  return props.scores.rules.filter(r => ruleActiveAtLevel(r.rule_id, level.value))
})

const overall = computed(() => {
  if (!hasScores.value || !props.scores) return null
  return overallForLevel(props.scores, level.value)
})

const levelLabels: Record<ScoreLevel, string> = {
  basic: 'Basic',
  intermediate: 'Intermediate',
  advance: 'Advance',
}
</script>

<template>
  <div class="flex flex-col h-full">
    <div class="shrink-0 p-4 border-b border-gray-800/80 space-y-3">
      <h2 class="text-lg font-semibold text-white">
        Stats
      </h2>

      <div
        v-if="hasScores"
        class="flex rounded-lg border border-gray-800 p-0.5 gap-0.5"
      >
        <button
          v-for="l in SCORE_LEVELS"
          :key="l"
          type="button"
          class="flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors"
          :class="level === l
            ? 'bg-primary-500/20 text-primary-400'
            : 'text-gray-400 hover:text-gray-200'"
          @click="level = l"
        >
          {{ levelLabels[l] }}
        </button>
      </div>
    </div>

    <div class="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
      <p
        v-if="!hasScores"
        class="text-sm text-gray-400"
      >
        No evaluation yet.
      </p>

      <template v-else-if="scores">
        <div class="flex justify-between items-baseline pb-3 border-b border-gray-800">
          <span class="text-gray-400">Overall</span>
          <strong class="text-3xl text-primary-500">{{ overall!.toFixed(2) }}</strong>
        </div>

        <div class="grid gap-3">
          <div
            v-for="c in activeCategories"
            :key="c"
            class="grid grid-cols-[6.5rem_1fr_2rem] gap-2 items-center text-xs capitalize"
          >
            <span class="text-gray-400">{{ c }}</span>
            <div class="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                class="h-full bg-primary-500 rounded-full"
                :style="{ width: `${((categoryScores![c] ?? 0) / 10) * 100}%` }"
              />
            </div>
            <span class="text-right text-gray-300">{{ (categoryScores![c] ?? 0).toFixed(1) }}</span>
          </div>
        </div>

        <USeparator />

        <h3 class="text-sm font-semibold text-gray-300">
          Rules
        </h3>

        <ul class="space-y-0 pb-4">
          <li
            v-for="r in activeRules"
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
    </div>
  </div>
</template>
