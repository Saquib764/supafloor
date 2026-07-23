export const FLOORPLAN_LINEAR_UNIT = 'ft'
export const FLOORPLAN_AREA_UNIT = 'ft²'

export const FLOORPLAN_TARGET_AREAS_SQFT = [150, 400, 600, 900, 1200, 1500, 2000, 2500, 3000, 4000, 4500, 5000] as const

export type RoomType =
  | 'living'
  | 'dining'
  | 'kitchen'
  | 'bedroom'
  | 'master_bedroom'
  | 'bathroom'
  | 'powder'
  | 'hallway'
  | 'closet'
  | 'storage'
  | 'utility'
  | 'balcony'
  | 'entry'
  | 'study'
  | 'laundry'
  | 'other'

export type ZoneType = 'public' | 'private' | 'service' | 'circulation' | 'outdoor' | 'wet'

export interface Point2D {
  x: number
  y: number
}

/** Closed polygon ring (first point equals last). */
export interface Room {
  id: string
  name: string
  polygon: Point2D[]
  room_type?: RoomType
  zone?: ZoneType
  level?: number
  metadata?: Record<string, unknown>
}

/** Door opening as a line segment. */
export interface Door {
  id: string
  start: Point2D
  end: Point2D
  connects: string[]
  is_exterior?: boolean
  swing?: string
}

/** Window opening as a line segment. */
export interface Window {
  id: string
  start: Point2D
  end: Point2D
  room_ids?: string[]
  height?: number
  sill_height?: number
}

export interface Boundary {
  id: string
  exterior: Point2D[]
  holes?: Point2D[][]
  units?: string
}

/** Floor plan = separate room / door / window lists. */
export interface FloorPlan {
  id: string
  name: string
  rooms: Room[]
  doors: Door[]
  windows: Window[]
  boundary?: Boundary | null
  walls?: unknown[]
  zones?: unknown[]
  metadata?: Record<string, unknown>
}

export interface RuleResult {
  rule_id: string
  name: string
  score: number
  reason: string
  severity: string
  repair_suggestion?: string | null
  hard: boolean
  passed: boolean
  details?: Record<string, unknown>
}

export type ScoreLevel = 'basic' | 'intermediate' | 'advance'

export type LayoutMode = 'cpsat' | 'llm'

export type ScoreCategory =
  | 'geometry'
  | 'topology'
  | 'construction'
  | 'privacy'
  | 'circulation'
  | 'daylight'
  | 'architecture'
  | 'aesthetics'

export const SCORE_LEVELS: ScoreLevel[] = ['basic', 'intermediate', 'advance']

export const LAYOUT_MODES: LayoutMode[] = ['cpsat', 'llm']

export interface EngineConfig {
  score_level: ScoreLevel
  score_threshold: number
  generate_max_attempts: number
  layout_mode: LayoutMode
}

export const SCORE_LEVEL_CATEGORY_ADDS: Record<ScoreLevel, readonly ScoreCategory[]> = {
  basic: ['geometry', 'topology', 'construction', 'architecture', 'aesthetics'],
  intermediate: ['privacy', 'circulation'],
  advance: ['daylight'],
}

export const SCORE_LEVEL_RULE_ADDS: Record<ScoreLevel, readonly string[]> = {
  basic: [],
  intermediate: ['reachability', 'bathroom_accessible', 'kitchen_accessible'],
  advance: ['required_rooms', 'windows_on_exterior'],
}

const TIER_GATED_RULES = new Set(
  SCORE_LEVELS.flatMap(level => SCORE_LEVEL_RULE_ADDS[level]),
)

const RULE_CATEGORY: Record<string, ScoreCategory> = {
  polygon_validity: 'geometry',
  room_overlap: 'geometry',
  boundary_containment: 'geometry',
  minimum_area: 'geometry',
  minimum_width: 'geometry',
  rectangularity: 'construction',
  shape_quality: 'construction',
  aspect_quality: 'architecture',
  reachability: 'topology',
  required_rooms: 'topology',
  doors_on_walls: 'topology',
  bathroom_accessible: 'topology',
  kitchen_accessible: 'topology',
  windows_on_exterior: 'daylight',
  daylight_quality: 'daylight',
  natural_light: 'daylight',
  privacy: 'privacy',
  circulation_quality: 'circulation',
  circulation: 'circulation',
  living_size: 'architecture',
  bedroom_ratio: 'architecture',
  wet_core: 'construction',
  storage: 'architecture',
  balcony: 'architecture',
  living_centrality: 'aesthetics',
  kitchen_public_link: 'topology',
  entry_sequence: 'aesthetics',
  public_near_exit: 'aesthetics',
}

function cumulativeUpTo<T extends string>(level: ScoreLevel, adds: Record<ScoreLevel, readonly T[]>): Set<T> {
  const result = new Set<T>()
  for (const tier of SCORE_LEVELS) {
    for (const item of adds[tier]) result.add(item)
    if (tier === level) break
  }
  return result
}

export function activeCategoriesForLevel(level: ScoreLevel): ScoreCategory[] {
  const active = cumulativeUpTo(level, SCORE_LEVEL_CATEGORY_ADDS)
  return (Object.keys(DEFAULT_SCORE_WEIGHTS) as ScoreCategory[]).filter(c => active.has(c))
}

export function ruleActiveAtLevel(ruleId: string, level: ScoreLevel): boolean {
  const activeCats = cumulativeUpTo(level, SCORE_LEVEL_CATEGORY_ADDS)
  const tierRules = cumulativeUpTo(level, SCORE_LEVEL_RULE_ADDS)
  const cat = RULE_CATEGORY[ruleId] ?? 'architecture'
  if (!activeCats.has(cat)) return false
  if (TIER_GATED_RULES.has(ruleId)) return tierRules.has(ruleId)
  return true
}

export const DEFAULT_SCORE_WEIGHTS: Record<ScoreCategory, number> = {
  geometry: 0.15,
  topology: 0.10,
  construction: 0.10,
  privacy: 0.11,
  circulation: 0.10,
  daylight: 0.13,
  architecture: 0.13,
  aesthetics: 0.18,
}

export function overallForLevel(scores: ScoreBreakdown, level: ScoreLevel): number {
  if (scores.levels?.[level] != null) return scores.levels[level]!

  const active = activeCategoriesForLevel(level)
  const weights = scores.weights ?? DEFAULT_SCORE_WEIGHTS
  const levelCats = scores.level_categories?.[level]
  const totalW = active.reduce((sum, c) => sum + (weights[c] ?? DEFAULT_SCORE_WEIGHTS[c]), 0)
  if (totalW <= 0) return scores.overall

  const raw = active.reduce(
    (sum, c) => sum + (levelCats?.[c] ?? scores[c]) * (weights[c] ?? DEFAULT_SCORE_WEIGHTS[c]) / totalW,
    0,
  )
  return Math.max(0, Math.min(10, Math.round(raw * 100) / 100))
}

export interface ScoreBreakdown {
  geometry: number
  topology: number
  construction: number
  privacy: number
  circulation: number
  daylight: number
  architecture: number
  aesthetics: number
  overall: number
  levels?: Partial<Record<ScoreLevel, number>>
  level_categories?: Partial<Record<ScoreLevel, Partial<Record<ScoreCategory, number>>>>
  weights: Record<string, number>
  rules: RuleResult[]
}

export interface RenderResult {
  svg: string
  png: string
  metadata: Record<string, unknown>
}

export interface PlanDocumentUnits {
  length: string
  area: string
}

export interface PlanDocumentSummary {
  id: string
  name: string
  bedroom_count: number
  room_count: number
  total_area_sqft: number
  boundary_area_sqft?: number | null
  target_area_sqft?: number | null
  entry?: { point: Point2D } | null
}

export interface PlanDocumentRoomRequested {
  name: string
  target_area_sqft?: number | null
  room_type: RoomType
  zone: ZoneType
  generated: boolean
}

export interface PlanDocumentConnection {
  from: string
  to: string
}

export interface PlanDocumentRoom {
  id: string
  name: string
  room_type: RoomType
  zone: ZoneType
  area_sqft: number
  bbox: { x: number, y: number, w: number, h: number }
  center: Point2D
  adjacent_to: string[]
  polygon?: Point2D[]
}

export interface PlanDocumentSampling {
  seed: number
  aspect?: number | null
  start_rect?: { x: number, y: number, w: number, h: number } | null
  notches?: Array<{ corner: string, w: number, h: number }>
  space_names?: string[]
  selected_areas?: Record<string, number>
  entry_point?: number[] | null
  entry_edge?: number | null
  centers?: Record<string, number[]>
  rectangles?: Array<Record<string, unknown>>
  connection_graph?: {
    root?: string | null
    edges?: Array<[string, string] | { from: string, to: string }>
    tree?: Array<[string, string] | { from: string, to: string }>
  }
}

export interface PlanDocument {
  schema_version: string
  units: PlanDocumentUnits
  summary: PlanDocumentSummary
  requirements: {
    target_area_sqft?: number | null
    bedroom_count?: number
    boundary_room?: string | null
    rooms_requested: PlanDocumentRoomRequested[]
  }
  sampling: PlanDocumentSampling
  topology: {
    root?: string | null
    connections: PlanDocumentConnection[]
    adjacency: Record<string, string[]>
  }
  rooms: PlanDocumentRoom[]
  openings: {
    doors: Array<{
      id: string
      connects: string[]
      start: Point2D
      end: Point2D
      is_exterior: boolean
    }>
    windows: Array<{
      id: string
      room_names: string[]
      start: Point2D
      end: Point2D
    }>
  }
  boundary: {
    exterior: Point2D[]
  }
  scores?: {
    overall: number
    summary: string
    categories: Record<string, number>
    hard_failures: string[]
  }
}
