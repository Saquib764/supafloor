export const FLOORPLAN_LINEAR_UNIT = 'ft'
export const FLOORPLAN_AREA_UNIT = 'ft²'

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

export interface ScoreBreakdown {
  geometry: number
  topology: number
  construction: number
  privacy: number
  circulation: number
  furniture: number
  daylight: number
  architecture: number
  aesthetics: number
  overall: number
  weights: Record<string, number>
  rules: RuleResult[]
}

export interface RenderResult {
  svg: string
  png: string
  metadata: Record<string, unknown>
}

export interface DemoResponse {
  plan: FloorPlan
  scores: ScoreBreakdown
  render: RenderResult
  adjacency: Record<string, string[]>
}

export interface RoomInspection {
  id: string
  name: string
  room_type: string
  zone: string
  area: number
  aspect_ratio: number
  perimeter: number
  adjacent_rooms: string[]
  doors: Door[]
  windows: Window[]
  individual_scores: RuleResult[]
  plan_overall: number
}
