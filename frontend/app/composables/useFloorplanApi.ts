const DEFAULT_API = 'http://localhost:8000'

export function useFloorplanApi(baseUrl?: string) {
  const config = useRuntimeConfig()
  const apiBase = baseUrl ?? config.public.floorplanApi ?? DEFAULT_API

  async function render(plan: unknown, layers?: string[], size = 800) {
    const res = await fetch(`${apiBase}/floorplan/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, layers, size }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<RenderResult>
  }

  async function exportRealistic(plan: FloorPlan, size = 1024) {
    const res = await fetch(`${apiBase}/floorplan/export/realistic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan, size }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<{ png_b64: string, size: number, metadata: Record<string, unknown> }>
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

  async function generate(bedroomCount: number, seed = 0, targetArea?: number) {
    const res = await fetch(`${apiBase}/floorplan/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bedroom_count: bedroomCount,
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
      bedroom_count: number
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

  async function extractImage(file: File, canvasSize = 1024) {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(
      `${apiBase}/floorplan/extract?canvas_size=${encodeURIComponent(String(canvasSize))}`,
      { method: 'POST', body: form },
    )
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function getConfig() {
    const res = await fetch(`${apiBase}/floorplan/config`)
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<EngineConfig>
  }

  async function updateConfig(updates: Partial<EngineConfig>) {
    const res = await fetch(`${apiBase}/floorplan/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<EngineConfig>
  }

  async function listDatasetItems(offset = 0, limit = 50) {
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    })
    const res = await fetch(`${apiBase}/floorplan/dataset/basic5000?${params}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<{
      items: Array<{ stem: string }>
      total: number
      offset: number
      limit: number
      has_more: boolean
    }>
  }

  async function loadDatasetJson(stem: string) {
    const res = await fetch(
      `${apiBase}/floorplan/dataset/basic5000/${encodeURIComponent(stem)}`,
    )
    if (!res.ok) throw new Error(await res.text())
    return res.json() as Promise<{ stem: string, json: unknown }>
  }

  function datasetImageUrl(stem: string) {
    return `${apiBase}/floorplan/dataset/basic5000/${encodeURIComponent(stem)}/image`
  }

  return {
    render,
    exportRealistic,
    evaluate,
    generate,
    listPlans,
    loadPlan,
    deletePlan,
    extractImage,
    getConfig,
    updateConfig,
    listDatasetItems,
    loadDatasetJson,
    datasetImageUrl,
  }
}
