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
      room_count: number
    }>>
  }

  async function loadPlan(planId: string) {
    const res = await fetch(
      `${apiBase}/floorplan/plans/${encodeURIComponent(planId)}/load?include_renders=true`,
      { method: 'POST' },
    )
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

  async function importImage(file: File, feetPerPixel = 0.05) {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(
      `${apiBase}/floorplan/import?feet_per_pixel=${encodeURIComponent(String(feetPerPixel))}`,
      { method: 'POST', body: form },
    )
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  return { render, evaluate, generate, listPlans, loadPlan, deletePlan, importImage }
}
