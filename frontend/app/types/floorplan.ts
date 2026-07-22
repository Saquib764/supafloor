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

export interface FloorPlanLLMUnits {
  length: string
  area: string
}

export interface FloorPlanLLMSummary {
  id: string
  name: string
  bedroom_count: number
  room_count: number
  total_area_sqft: number
  boundary_area_sqft?: number | null
  target_area_sqft?: number | null
  entry?: { point: Point2D } | null
}

export interface FloorPlanLLMRoomRequested {
  name: string
  target_area_sqft?: number | null
  room_type: RoomType
  zone: ZoneType
  generated: boolean
}

export interface FloorPlanLLMConnection {
  from: string
  to: string
}

export interface FloorPlanLLMRoom {
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

export interface FloorPlanLLMView {
  schema_version: string
  units: FloorPlanLLMUnits
  summary: FloorPlanLLMSummary
  requirements: {
    target_area_sqft?: number | null
    bedroom_count?: number
    boundary_room?: string | null
    rooms_requested: FloorPlanLLMRoomRequested[]
  }
  topology: {
    root?: string | null
    connections: FloorPlanLLMConnection[]
    adjacency: Record<string, string[]>
  }
  rooms: FloorPlanLLMRoom[]
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
