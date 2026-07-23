import type { EngineConfig, PipelineProvider, ScoreLevel } from '~/types/floorplan'
import { PIPELINE_PROVIDERS, SCORE_LEVELS } from '~/types/floorplan'

const DEFAULT_CONFIG: EngineConfig = {
  score_level: 'basic',
  score_threshold: 8,
  generate_max_attempts: 10,
  target_areas: 'deterministic',
  connection_graph: 'deterministic',
  centers: 'deterministic',
}

function normalizeProvider(value: unknown, fallback: PipelineProvider): PipelineProvider {
  return typeof value === 'string' && PIPELINE_PROVIDERS.includes(value as PipelineProvider)
    ? value as PipelineProvider
    : fallback
}

function pipelineProvider(
  raw: Partial<EngineConfig> | Record<string, unknown>,
  key: 'target_areas' | 'connection_graph' | 'centers',
  fallback: PipelineProvider,
): PipelineProvider {
  if (key in raw && raw[key] != null) {
    return normalizeProvider(raw[key], fallback)
  }
  const pipeline = raw.pipeline as { sample_into_document?: Record<string, unknown> } | undefined
  return normalizeProvider(pipeline?.sample_into_document?.[key], fallback)
}

function normalizeConfig(raw: Partial<EngineConfig> | Record<string, unknown>): EngineConfig {
  const level = raw.score_level as ScoreLevel | undefined
  return {
    score_level: level && SCORE_LEVELS.includes(level) ? level : DEFAULT_CONFIG.score_level,
    score_threshold: Number(raw.score_threshold ?? DEFAULT_CONFIG.score_threshold),
    generate_max_attempts: Number(raw.generate_max_attempts ?? DEFAULT_CONFIG.generate_max_attempts),
    target_areas: pipelineProvider(raw, 'target_areas', DEFAULT_CONFIG.target_areas),
    connection_graph: pipelineProvider(raw, 'connection_graph', DEFAULT_CONFIG.connection_graph),
    centers: pipelineProvider(raw, 'centers', DEFAULT_CONFIG.centers),
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
