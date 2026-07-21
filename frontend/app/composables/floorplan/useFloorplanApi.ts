const DEFAULT_API = 'http://localhost:8000'

export function useFloorplanApi(baseUrl = DEFAULT_API) {
  async function getDemo(templateId = 'compact_2bed') {
    const res = await fetch(`${baseUrl}/floorplan/demo?template_id=${encodeURIComponent(templateId)}`, {
      method: 'POST',
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function listTemplates() {
    const res = await fetch(`${baseUrl}/floorplan/templates`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function render(plan: unknown, layers?: string[]) {
    const res = await fetch(`${baseUrl}/floorplan/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, layers }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function evaluate(plan: unknown) {
    const res = await fetch(`${baseUrl}/floorplan/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function mutate(plan: unknown, op: string, seed = 0) {
    const res = await fetch(`${baseUrl}/floorplan/mutate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, op, seed }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function inspect(plan: unknown, roomId: string) {
    const res = await fetch(`${baseUrl}/floorplan/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, room_id: roomId }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function generate(templateId: string, seed = 0) {
    const res = await fetch(`${baseUrl}/floorplan/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ template_id: templateId, seed, include_renders: true }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  return { getDemo, listTemplates, render, evaluate, mutate, inspect, generate }
}
