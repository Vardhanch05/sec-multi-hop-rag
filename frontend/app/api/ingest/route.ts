import { NextResponse } from "next/server"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"

    const res = await fetch(`${backendUrl}/api/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to queue ingestion task" }))
      return NextResponse.json(err, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (e) {
    console.error("Ingest API proxy error:", e)
    return NextResponse.json({ error: "Failed to connect to backend ingestion service" }, { status: 500 })
  }
}

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"
    const res = await fetch(`${backendUrl}/api/ingest/tasks`)
    if (!res.ok) {
      throw new Error(`Backend task list status ${res.status}`)
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (e) {
    console.warn("Backend /api/ingest/tasks failed:", e)
    return NextResponse.json([])
  }
}
