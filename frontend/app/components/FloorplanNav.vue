<template>
  <nav class="flex gap-1 rounded-lg bg-gray-900/60 p-1">
    <NuxtLink
      to="/"
      class="flex-1 rounded-md px-2 py-1.5 text-center text-xs font-medium text-gray-400 transition-colors hover:text-white"
      active-class="!bg-primary-500/20 !text-white ring-1 ring-primary-500/40"
    >
      Floorplans
    </NuxtLink>
    <NuxtLink
      to="/importer"
      class="flex-1 rounded-md px-2 py-1.5 text-center text-xs font-medium text-gray-400 transition-colors hover:text-white"
      active-class="!bg-primary-500/20 !text-white ring-1 ring-primary-500/40"
    >
      Importer
    </NuxtLink>
  </nav>
</template>
