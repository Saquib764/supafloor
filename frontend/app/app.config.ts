export default defineAppConfig({
  ui: {
    colors: {
      primary: 'pink',
      neutral: 'slate',
    },
    notifications: {
      position: 'top-0 bottom-auto',
    },
    button: {
      slots: {
        base: 'cursor-pointer',
      },
    },
    input: {
      defaultVariants: {
        size: 'md',
      },
    },
    card: {
      slots: {
        root: 'rounded-xl',
      },
    },
    header: {
      slots: {
        root: 'bg-gray-900 border-b border-gray-800',
      },
    },
    footer: {
      slots: {
        top: 'border-t border-gray-800 py-8 lg:py-16',
        bottom: 'border-t border-gray-800',
      },
    },
    page: {
      slots: {
        root: 'lg:py-24',
      },
    },
  },
})
