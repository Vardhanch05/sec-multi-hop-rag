"use client"

import { useState, useEffect } from "react"
import { UploadCloud, CheckCircle2, AlertCircle, Loader2, X } from "lucide-react"

interface IngestModalProps {
  isOpen: boolean
  onClose: () => void
  onIngestComplete?: () => void
}

interface IngestTask {
  task_id: string
  ticker: string
  fiscal_year: number
  status: "pending" | "processing" | "completed" | "failed"
  progress: number
  message: string
}

export function IngestModal({ isOpen, onClose, onIngestComplete }: IngestModalProps) {
  const [ticker, setTicker] = useState("")
  const [fiscalYear, setFiscalYear] = useState("2025")
  const [activeTask, setActiveTask] = useState<IngestTask | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Poll task status if a task is processing
  useEffect(() => {
    if (!activeTask || activeTask.status === "completed" || activeTask.status === "failed") {
      return
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/ingest/status/${activeTask.task_id}`)
        if (res.ok) {
          const updatedTask: IngestTask = await res.json()
          setActiveTask(updatedTask)

          if (updatedTask.status === "completed" && onIngestComplete) {
            onIngestComplete()
          }
        }
      } catch (err) {
        console.error("Failed to poll task status:", err)
      }
    }, 1500)

    return () => clearInterval(interval)
  }, [activeTask, onIngestComplete])

  if (!isOpen) return null

  const handleStartIngestion = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!ticker.trim()) {
      setError("Please enter a valid stock ticker (e.g. AAPL, TSLA)")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: ticker.trim().toUpperCase(),
          fiscal_year: parseInt(fiscalYear, 10),
        }),
      })

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || data.error || "Failed to start background ingestion")
      }

      setActiveTask({
        task_id: data.task_id,
        ticker: data.ticker,
        fiscal_year: data.fiscal_year,
        status: "processing",
        progress: 10,
        message: data.message || "Ingestion task queued...",
      })
    } catch (err: any) {
      setError(err.message || "Failed to queue task")
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setActiveTask(null)
    setTicker("")
    setError(null)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/50 backdrop-blur-sm p-4">
      <div className="bg-white rounded-3xl border border-stone-200 shadow-2xl max-w-md w-full p-6 relative animate-in fade-in zoom-in-95 duration-200">
        <button
          onClick={resetForm}
          className="absolute top-5 right-5 text-stone-400 hover:text-stone-700 p-1 rounded-full hover:bg-stone-100 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-2xl bg-stone-900 text-white flex items-center justify-center shadow-md">
            <UploadCloud className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-stone-800 text-lg leading-tight">Async Background Ingestion</h3>
            <p className="text-xs text-stone-400 font-medium">Fetch SEC filings & update ChromaDB without blocking</p>
          </div>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {activeTask ? (
          <div className="space-y-5 my-2">
            <div className="bg-stone-50 p-4 rounded-2xl border border-stone-150 space-y-3">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-stone-700 font-bold">
                  {activeTask.ticker} (FY{activeTask.fiscal_year})
                </span>
                <span
                  className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                    activeTask.status === "completed"
                      ? "bg-emerald-100 text-emerald-800"
                      : activeTask.status === "failed"
                      ? "bg-rose-100 text-rose-800"
                      : "bg-amber-100 text-amber-800 animate-pulse"
                  }`}
                >
                  {activeTask.status}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-stone-200 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    activeTask.status === "completed"
                      ? "bg-emerald-500"
                      : activeTask.status === "failed"
                      ? "bg-rose-500"
                      : "bg-stone-900"
                  }`}
                  style={{ width: `${activeTask.progress}%` }}
                />
              </div>

              <p className="text-xs text-stone-600 font-medium flex items-center gap-2">
                {activeTask.status === "processing" && <Loader2 className="w-3.5 h-3.5 animate-spin text-stone-700" />}
                {activeTask.status === "completed" && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />}
                <span>{activeTask.message}</span>
              </p>
            </div>

            <div className="flex justify-end">
              <button
                onClick={resetForm}
                className="w-full py-2.5 px-4 rounded-xl bg-stone-900 text-white font-semibold text-xs hover:bg-stone-800 transition-colors"
              >
                {activeTask.status === "completed" ? "Done" : "Close & Run in Background"}
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleStartIngestion} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-stone-700 mb-1">Company Ticker Symbol</label>
              <input
                type="text"
                placeholder="e.g. AAPL, TSLA, NVDA"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-xl border border-stone-200 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-stone-800 uppercase"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-stone-700 mb-1">Fiscal Year</label>
              <select
                value={fiscalYear}
                onChange={(e) => setFiscalYear(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-xl border border-stone-200 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-stone-800 bg-white"
              >
                <option value="2025">2025</option>
                <option value="2024">2024</option>
                <option value="2023">2023</option>
              </select>
            </div>

            <p className="text-[11px] text-stone-400 leading-normal">
              FastAPI BackgroundTasks will execute EDGAR fetching, chunking, and ChromaDB vector indexing asynchronously without delaying API responses.
            </p>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 px-4 rounded-xl bg-stone-900 text-white font-semibold text-xs hover:bg-stone-800 transition-colors flex items-center justify-center gap-2 shadow-md shadow-stone-900/10"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Queueing Task...</span>
                </>
              ) : (
                <span>Start Async Ingestion</span>
              )}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}
