<script setup lang="ts">
const settingsOpen = ref(false)

const linkClass =
  'flex h-10 w-10 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-white/5 hover:text-white'
const activeClass = '!bg-primary-500/20 !text-white ring-1 ring-primary-500/40'
const tooltipContent = { side: 'right' as const, sideOffset: 8 }
</script>

<template>
  <nav class="flex h-full w-14 shrink-0 flex-col items-center border-r border-gray-800/80 bg-gray-950/90 py-3 gap-1">
    <UTooltip
      text="Floorplans"
      :content="tooltipContent"
    >
      <NuxtLink
        to="/"
        :class="linkClass"
        :active-class="activeClass"
        aria-label="Floorplans"
      >
        <UIcon
          name="i-lucide-layout-grid"
          class="size-5"
        />
      </NuxtLink>
    </UTooltip>

    <UTooltip
      text="Extract"
      :content="tooltipContent"
    >
      <NuxtLink
        to="/extract"
        :class="linkClass"
        :active-class="activeClass"
        aria-label="Extract"
      >
        <UIcon
          name="i-lucide-scan"
          class="size-5"
        />
      </NuxtLink>
    </UTooltip>

    <UTooltip
      text="Dataset"
      :content="tooltipContent"
    >
      <NuxtLink
        to="/dataset"
        :class="linkClass"
        :active-class="activeClass"
        aria-label="Dataset"
      >
        <UIcon
          name="i-lucide-images"
          class="size-5"
        />
      </NuxtLink>
    </UTooltip>

    <div class="mt-auto">
      <UTooltip
        text="Settings"
        :content="tooltipContent"
      >
        <button
          type="button"
          :class="linkClass"
          aria-label="Settings"
          @click="settingsOpen = true"
        >
          <UIcon
            name="i-lucide-settings"
            class="size-5"
          />
        </button>
      </UTooltip>
    </div>

    <FloorplanSettingsModal v-model:open="settingsOpen" />
  </nav>
</template>
