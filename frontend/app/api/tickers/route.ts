import { NextResponse } from "next/server"

const MOCK_TICKERS = [
  "AAPL",
  "AMZN",
  "CVX",
  "GOOGL",
  "JNJ",
  "JPM",
  "META",
  "MSFT",
  "NFLX",
  "NVDA",
  "PFE",
  "TSLA",
  "UNH",
  "XOM",
]

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"
    const res = await fetch(`${backendUrl}/api/tickers`)
    if (!res.ok) {
      throw new Error(`Backend tickers returned status ${res.status}`)
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (e) {
    console.warn("Backend /api/tickers failed, using mock fallback tickers:", e)
    return NextResponse.json(MOCK_TICKERS)
  }
}
