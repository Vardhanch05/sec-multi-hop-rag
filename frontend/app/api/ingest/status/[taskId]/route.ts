import { NextResponse } from "next/server"

export async function GET(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  try {
    const { taskId } = await params
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"

    const res = await fetch(`${backendUrl}/api/ingest/status/${taskId}`)
    if (!res.ok) {
      return NextResponse.json({ error: "Task not found" }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (e) {
    console.error("Ingest status proxy error:", e)
    return NextResponse.json({ error: "Failed to fetch task status" }, { status: 500 })
  }
}
