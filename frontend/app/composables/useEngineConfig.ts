import type { EngineConfig, LayoutMode, ScoreLevel } from '~/types/floorplan'
import { LAYOUT_MODES, SCORE_LEVELS } from '~/types/floorplan'

const DEFAULT_CONFIG: EngineConfig = {
  score_level: 'basic',
  score_threshold: 8,
  generate_max_attempts: 10,
  layout_mode: 'cpsat',
}

function normalizeConfig(raw: Partial<EngineConfig> | Record<string, unknown>): EngineConfig {
  const level = raw.score_level as ScoreLevel | undefined
  const mode = raw.layout_mode as LayoutMode | undefined
  return {
    score_level: level && SCORE_LEVELS.includes(level) ? level : DEFAULT_CONFIG.score_level,
    score_threshold: Number(raw.score_threshold ?? DEFAULT_CONFIG.score_threshold),
    generate_max_attempts: Number(raw.generate_max_attempts ?? DEFAULT_CONFIG.generate_max_attempts),
    layout_mode: mode && LAYOUT_MODES.includes(mode) ? mode : DEFAULT_CONFIG.layout_mode,
  }
}

/** Shared reactive engine config from config.yaml (live across the session). */
export function useEngineConfig() {
  const api = useFloorplanApi()
  const config = useState<EngineConfig>('engine-config', () => ({ ...DEFAULT_CONFIG }))
  const loaded = useState<boolean>('engine-config-loaded', () => false)
  const loading = useState<boolean>('engine-config-loading', () => false)
  const saving = useState<boolean>('engine-config-saving', () => false)

  async function refresh() {
    loading.value = true
    try {
      config.value = normalizeConfig(await api.getConfig())
      loaded.value = true
    }
    finally {
      loading.value = false
    }
  }

  async function save(updates: Partial<EngineConfig>) {
    saving.value = true
    try {
      config.value = normalizeConfig(await api.updateConfig(updates))
      loaded.value = true
      return config.value
    }
    finally {
      saving.value = false
    }
  }

  async function ensureLoaded() {
    if (loaded.value || loading.value) return
    try {
      await refresh()
    }
    catch {
      // Keep defaults when API is unavailable.
      loaded.value = true
    }
  }

  return { config, loaded, loading, saving, refresh, save, ensureLoaded }
}
