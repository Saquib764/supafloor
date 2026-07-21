/** Site-wide: always use dark mode. Ignores OS preference. */
export function useForcedDarkMode() {
  const colorMode = useColorMode()

  if (import.meta.client && colorMode.preference !== 'dark') {
    colorMode.preference = 'dark'
  }

  watch(
    () => colorMode.preference,
    (pref) => {
      if (pref !== 'dark') {
        colorMode.preference = 'dark'
      }
    },
    { immediate: true },
  )

  return colorMode
}
