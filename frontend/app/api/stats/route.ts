import { NextResponse } from "next/server"

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"
    const res = await fetch(`${backendUrl}/api/stats`)
    if (!res.ok) {
      throw new Error(`Backend stats returned status ${res.status}`)
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (e) {
    console.warn("Backend /api/stats failed, using mock fallback stats:", e)
    return NextResponse.json({
      totalFilings: 210,
      uniqueTickers: 14,
    })
  }
}
