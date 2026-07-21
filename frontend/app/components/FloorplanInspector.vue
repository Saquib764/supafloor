<script setup lang="ts">
defineProps<{
  inspection: RoomInspection | null
  loading?: boolean
}>()
</script>

<template>
  <UCard
    :ui="{
      root: 'h-full overflow-auto rounded-none border-0 border-r border-gray-800 bg-gray-900',
      body: 'p-4 space-y-4',
    }"
  >
    <h2 class="text-lg font-semibold text-white">
      Inspector
    </h2>

    <p
      v-if="loading"
      class="text-sm text-gray-400"
    >
      Loading…
    </p>
    <p
      v-else-if="!inspection"
      class="text-sm text-gray-400"
    >
      Select a room to inspect geometry, adjacency, and scores.
    </p>

    <template v-else>
      <div>
        <h3 class="text-xl font-semibold text-white">
          {{ inspection.name }}
        </h3>
      </div>

      <dl class="grid gap-2">
        <div class="grid grid-cols-[7rem_1fr] gap-2 text-sm">
          <dt class="text-gray-500">
            Type
          </dt>
          <dd class="font-medium text-white">
            {{ inspection.room_type }}
          </dd>
        </div>
        <div class="grid grid-cols-[7rem_1fr] gap-2 text-sm">
          <dt class="text-gray-500">
            Zone
          </dt>
          <dd class="font-medium text-white">
            {{ inspection.zone }}
          </dd>
        </div>
        <div class="grid grid-cols-[7rem_1fr] gap-2 text-sm">
          <dt class="text-gray-500">
            Area
          </dt>
          <dd class="font-medium text-white">
            {{ inspection.area.toFixed(2) }} m²
          </dd>
        </div>
        <div class="grid grid-cols-[7rem_1fr] gap-2 text-sm">
          <dt class="text-gray-500">
            Aspect
          </dt>
          <dd class="font-medium text-white">
            {{ inspection.aspect_ratio.toFixed(2) }}
          </dd>
        </div>
        <div class="grid grid-cols-[7rem_1fr] gap-2 text-sm">
          <dt class="text-gray-500">
            Perimeter
          </dt>
          <dd class="font-medium text-white">
            {{ inspection.perimeter.toFixed(2) }} m
          </dd>
        </div>
        <div class="grid grid-cols-[7rem_1fr] gap-2 text-sm">
          <dt class="text-gray-500">
            Plan score
          </dt>
          <dd class="font-medium text-white">
            {{ inspection.plan_overall.toFixed(2) }}
          </dd>
        </div>
      </dl>

      <USeparator />

      <div>
        <h4 class="text-sm font-semibold text-gray-300 mb-2">
          Adjacent rooms
        </h4>
        <ul class="list-disc pl-5 text-sm text-gray-300 space-y-1">
          <li
            v-for="id in inspection.adjacent_rooms"
            :key="id"
          >
            {{ id }}
          </li>
          <li
            v-if="!inspection.adjacent_rooms.length"
            class="list-none -ml-5 text-gray-500"
          >
            None detected
          </li>
        </ul>
      </div>

      <USeparator />

      <div>
        <h4 class="text-sm font-semibold text-gray-300 mb-2">
          Doors
        </h4>
        <p class="text-sm text-gray-300">
          {{ inspection.doors.length }} connected
        </p>
      </div>

      <USeparator />

      <div>
        <h4 class="text-sm font-semibold text-gray-300 mb-2">
          Rule hits
        </h4>
        <ul class="space-y-3">
          <li
            v-for="r in inspection.individual_scores"
            :key="r.rule_id"
            class="text-sm"
          >
            <div class="flex items-center gap-2 flex-wrap">
              <strong class="text-white">{{ r.name }}</strong>
              <UBadge
                color="primary"
                variant="soft"
                size="sm"
              >
                {{ r.score.toFixed(1) }}
              </UBadge>
            </div>
            <p class="text-gray-400 mt-1">
              {{ r.reason }}
            </p>
          </li>
          <li
            v-if="!inspection.individual_scores.length"
            class="text-sm text-gray-500"
          >
            No room-specific hits
          </li>
        </ul>
      </div>
    </template>
  </UCard>
</template>
