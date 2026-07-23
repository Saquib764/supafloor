import type { FloorPlan } from '~/types/floorplan'

const REGION_BAND = 1 / 3

function regionLabel(nx: number, ny: number): string {
  const band = (value: number): number => {
    if (value < REGION_BAND) return -1
    if (value > 1 - REGION_BAND) return 1
    return 0
  }

  const yb = band(ny)
  const xb = band(nx)
  const ns = yb === -1 ? 'north' : yb === 1 ? 'south' : null
  const ew = xb === -1 ? 'west' : xb === 1 ? 'east' : null

  if (ns && ew) return `${ns}-${ew}`
  if (ns) return xb === 0 ? 'north-center' : ns
  if (ew) return yb === 0 ? 'west-center' : ew
  return 'center'
}

function joinNames(names: string[]): string {
  if (names.length === 1) return names[0]!
  if (names.length === 2) return `${names[0]} and ${names[1]}`
  return `${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`
}

function openRing(polygon: Array<{ x: number, y: number }>): Array<{ x: number, y: number }> {
  return polygon.length >= 2
    && polygon[0]!.x === polygon.at(-1)!.x
    && polygon[0]!.y === polygon.at(-1)!.y
    ? polygon.slice(0, -1)
    : polygon
}

function roundCoord(value: number): number {
  return Math.round(value * 1000) / 1000
}

function polygonArea(polygon: Array<{ x: number, y: number }>): number {
  const ring = openRing(polygon)
  let area = 0
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i]!
    const b = ring[(i + 1) % ring.length]!
    area += a.x * b.y - b.x * a.y
  }
  return Math.abs(area) * 0.5
}

function polygonCentroid(polygon: Array<{ x: number, y: number }>): { x: number, y: number } {
  const ring = openRing(polygon)
  let area = 0
  let cx = 0
  let cy = 0
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i]!
    const b = ring[(i + 1) % ring.length]!
    const cross = a.x * b.y - b.x * a.y
    area += cross
    cx += (a.x + b.x) * cross
    cy += (a.y + b.y) * cross
  }
  if (Math.abs(area) < 1e-9) {
    const xs = ring.map(p => p.x)
    const ys = ring.map(p => p.y)
    return {
      x: (Math.min(...xs) + Math.max(...xs)) / 2,
      y: (Math.min(...ys) + Math.max(...ys)) / 2,
    }
  }
  area *= 0.5
  return { x: cx / (6 * area), y: cy / (6 * area) }
}

function planBounds(plan: FloorPlan): { minx: number, miny: number, maxx: number, maxy: number } {
  const points = plan.boundary?.exterior?.length
    ? plan.boundary.exterior
    : plan.rooms.flatMap(room => room.polygon)
  let minx = Infinity
  let miny = Infinity
  let maxx = -Infinity
  let maxy = -Infinity
  for (const point of points) {
    minx = Math.min(minx, point.x)
    miny = Math.min(miny, point.y)
    maxx = Math.max(maxx, point.x)
    maxy = Math.max(maxy, point.y)
  }
  return { minx, miny, maxx, maxy }
}

function doorNeighbors(plan: FloorPlan): Map<string, Set<string>> {
  const idToName = new Map(plan.rooms.map(room => [room.id, room.name]))
  const names = new Set(plan.rooms.map(room => room.name))
  const neighbors = new Map<string, Set<string>>(
    plan.rooms.map(room => [room.name, new Set<string>()]),
  )
  for (const door of plan.doors) {
    const linked = door.connects
      .map(value => idToName.get(value) ?? value)
      .filter(value => names.has(value))
    for (let i = 0; i < linked.length; i++) {
      for (let j = i + 1; j < linked.length; j++) {
        neighbors.get(linked[i]!)!.add(linked[j]!)
        neighbors.get(linked[j]!)!.add(linked[i]!)
      }
    }
  }
  return neighbors
}

export function buildLayoutPrompt(plan: FloorPlan): string {
  if (!plan.rooms.length) return ''

  const { minx, miny, maxx, maxy } = planBounds(plan)
  const width = Math.max(maxx - minx, 1e-9)
  const height = Math.max(maxy - miny, 1e-9)
  const neighbors = doorNeighbors(plan)

  const rooms = [...plan.rooms].sort((a, b) => {
    const ca = polygonCentroid(a.polygon)
    const cb = polygonCentroid(b.polygon)
    return ca.y - cb.y || ca.x - cb.x
  })

  return rooms.map((room) => {
    const { x, y } = polygonCentroid(room.polygon)
    const region = regionLabel((x - minx) / width, (y - miny) / height)
    let line = `${room.name} is in the ${region}`
    const linked = [...(neighbors.get(room.name) ?? [])].sort()
    if (linked.length) line += `, next to ${joinNames(linked)}`
    return `${line}.`
  }).join('\n')
}

export function buildLayoutSchema(plan: FloorPlan): string {
  if (!plan.rooms.length) return ''

  const rooms = [...plan.rooms]
    .sort((a, b) => a.name.localeCompare(b.name))
    .map(room => ({
      name: room.name,
      actual_area_sqft: roundCoord(polygonArea(room.polygon)),
    }))

  const layoutPolygon = plan.boundary?.exterior?.length
    ? plan.boundary.exterior.map(point => ({
        x: roundCoord(point.x),
        y: roundCoord(point.y),
      }))
    : []

  return JSON.stringify({ rooms, layout_polygon: layoutPolygon }, null, 2)
}

export function buildRoomOutput(plan: FloorPlan): string {
  if (!plan.rooms.length) return '[]'

  const rooms = [...plan.rooms]
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((room) => {
      const row: Record<string, unknown> = {
        name: room.name,
        actual_area_sqft: roundCoord(polygonArea(room.polygon)),
        polygon: room.polygon.map(point => ({
          x: roundCoord(point.x),
          y: roundCoord(point.y),
        })),
      }
      if (room.room_type) row.room_type = room.room_type
      if (room.zone) row.zone = room.zone
      return row
    })

  return JSON.stringify(rooms, null, 2)
}
