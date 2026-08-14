import { NextResponse } from "next/server"

const RAGAS_HISTORY = [
  { date: "07/15", faithfulness: 0.78, relevance: 0.81, precision: 0.75, recall: 0.79 },
  { date: "07/20", faithfulness: 0.82, relevance: 0.85, precision: 0.79, recall: 0.82 },
  { date: "07/25", faithfulness: 0.85, relevance: 0.88, precision: 0.82, recall: 0.84 },
  { date: "08/01", faithfulness: 0.88, relevance: 0.90, precision: 0.84, recall: 0.87 },
  { date: "08/08", faithfulness: 0.92, relevance: 0.94, precision: 0.89, recall: 0.91 },
]

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"
    const res = await fetch(`${backendUrl}/api/ragas`)
    if (!res.ok) {
      throw new Error(`Backend ragas returned status ${res.status}`)
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (e) {
    console.warn("Backend /api/ragas failed, using mock fallback ragas history:", e)
    return NextResponse.json(RAGAS_HISTORY)
  }
}
