const DEFAULT_API = 'http://localhost:8000'

export function useFloorplanApi(baseUrl?: string) {
  const config = useRuntimeConfig()
  const apiBase = baseUrl ?? config.public.floorplanApi ?? DEFAULT_API

  async function render(plan: unknown, layers?: string[]) {
    const res = await fetch(`${apiBase}/floorplan/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, layers }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function evaluate(plan: unknown) {
    const res = await fetch(`${apiBase}/floorplan/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function mutate(plan: unknown, op: string, seed = 0) {
    const res = await fetch(`${apiBase}/floorplan/mutate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, op, seed }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function inspect(plan: unknown, roomId: string) {
    const res = await fetch(`${apiBase}/floorplan/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, room_id: roomId }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function generate(totalRooms: number, seed = 0, targetArea?: number) {
    const res = await fetch(`${apiBase}/floorplan/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        total_rooms: totalRooms,
        target_area: targetArea,
        seed,
        include_renders: true,
        save: true,
      }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function listPlans() {
    const res = await fetch(`${apiBase}/floorplan/plans`)
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<Array<{
      id: string
      label: string
      total_rooms: number
      seed: number
      space_count: number
    }>>
  }

  async function getPlan(planId: string) {
    const res = await fetch(`${apiBase}/floorplan/plans/${encodeURIComponent(planId)}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function loadPlan(planId: string) {
    const res = await fetch(
      `${apiBase}/floorplan/plans/${encodeURIComponent(planId)}/load?include_renders=true`,
      { method: 'POST' },
    )
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function recreate(planId: string) {
    const res = await fetch(`${apiBase}/floorplan/plans/${encodeURIComponent(planId)}/recreate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_renders: true, save: true }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function deletePlan(planId: string) {
    const res = await fetch(`${apiBase}/floorplan/plans/${encodeURIComponent(planId)}`, {
      method: 'DELETE',
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<{ deleted: string }>
  }

  return { render, evaluate, mutate, inspect, generate, listPlans, getPlan, loadPlan, recreate, deletePlan }
}
