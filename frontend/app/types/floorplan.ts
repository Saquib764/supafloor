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

export interface Room {
  id: string
  name: string
  room_type: RoomType
  polygon: Point2D[]
  zone: ZoneType
  level?: number
  metadata?: Record<string, unknown>
}

export interface Door {
  id: string
  wall_id: string
  position: Point2D
  width: number
  connects: string[]
  is_exterior?: boolean
}

export interface Window {
  id: string
  wall_id: string
  position: Point2D
  width: number
  height?: number
}

export interface Boundary {
  id: string
  exterior: Point2D[]
  holes?: Point2D[][]
  units?: string
}

export interface FloorPlan {
  id: string
  name: string
  boundary: Boundary
  rooms: Room[]
  walls: unknown[]
  doors: Door[]
  windows: Window[]
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
