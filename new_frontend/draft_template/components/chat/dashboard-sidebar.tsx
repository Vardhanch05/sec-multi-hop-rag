"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"
import { 
  Database, 
  Filter, 
  TrendingUp, 
  ChevronLeft, 
  ChevronRight, 
  CheckSquare, 
  Square,
  BarChart3,
  Bookmark
} from "lucide-react"
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip,
  CartesianGrid
} from "recharts"

const MOCK_TICKERS = ["AAPL", "AMZN", "CVX", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NFLX", "NVDA", "PFE", "TSLA", "UNH", "XOM"]

const RAGAS_HISTORY = [
  { date: "07/15", faithfulness: 0.78, relevance: 0.81, precision: 0.75, recall: 0.79 },
  { date: "07/20", faithfulness: 0.82, relevance: 0.85, precision: 0.79, recall: 0.82 },
  { date: "07/25", faithfulness: 0.85, relevance: 0.88, precision: 0.82, recall: 0.84 },
  { date: "08/01", faithfulness: 0.88, relevance: 0.90, precision: 0.84, recall: 0.87 },
  { date: "08/08", faithfulness: 0.92, relevance: 0.94, precision: 0.89, recall: 0.91 },
]

interface DashboardSidebarProps {
  selectedTickers: string[]
  onTickerChange: (tickers: string[]) => void
  isCollapsed: boolean
  setIsCollapsed: (collapsed: boolean) => void
  customWidth?: number
  isResizing?: boolean
}

export function DashboardSidebar({
  selectedTickers,
  onTickerChange,
  isCollapsed,
  setIsCollapsed,
  customWidth,
  isResizing
}: DashboardSidebarProps) {
  const [mounted, setMounted] = useState(false)
  const [activeTab, setActiveTab] = useState<"filters" | "ragas">("filters")

  useEffect(() => {
    setMounted(true)
  }, [])

  const toggleTicker = (ticker: string) => {
    const next = selectedTickers.includes(ticker)
      ? selectedTickers.filter(t => t !== ticker)
      : [...selectedTickers, ticker]
    onTickerChange(next)
  }

  const toggleAllTickers = () => {
    if (selectedTickers.length === MOCK_TICKERS.length) {
      onTickerChange([])
    } else {
      onTickerChange([...MOCK_TICKERS])
    }
  }

  if (!mounted) return null

  if (isCollapsed) {
    return (
      <div 
        onClick={() => setIsCollapsed(false)}
        className="w-16 h-full bg-white border-r border-stone-200 flex flex-col items-center py-4 justify-between transition-all duration-300 cursor-pointer hover:bg-stone-50/40"
        title="Click to expand sidebar"
      >
        <div className="flex flex-col items-center gap-6 w-full px-2" onClick={(e) => e.stopPropagation()}>
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-slate-700 to-stone-800 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-slate-500/10">
            S
          </div>
          
          <div className="flex flex-col gap-2 w-full items-center">
            <button 
              onClick={() => {
                setActiveTab("filters")
                setIsCollapsed(false)
              }}
              className={`p-2.5 rounded-xl transition-colors ${
                activeTab === "filters" 
                  ? "bg-stone-100 text-stone-800" 
                  : "text-stone-400 hover:bg-stone-50 hover:text-stone-700"
              }`}
              title="Database Filters"
            >
              <Filter className="w-5 h-5" />
            </button>
            <button 
              onClick={() => {
                setActiveTab("ragas")
                setIsCollapsed(false)
              }}
              className={`p-2.5 rounded-xl transition-colors ${
                activeTab === "ragas" 
                  ? "bg-stone-100 text-stone-800" 
                  : "text-stone-400 hover:bg-stone-50 hover:text-stone-700"
              }`}
              title="RAGAS Evaluation Dashboard"
            >
              <BarChart3 className="w-5 h-5" />
            </button>
          </div>
        </div>
        <button 
          onClick={(e) => {
            e.stopPropagation()
            setIsCollapsed(false)
          }}
          className="p-2.5 rounded-xl hover:bg-stone-100 text-stone-500 transition-colors"
          title="Expand sidebar"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    )
  }

  return (
    <div 
      className={cn(
        "h-full bg-white border-r border-stone-200 flex flex-col justify-between select-none shrink-0",
        !isResizing && "transition-all duration-300"
      )}
      style={{ width: customWidth || 320 }}
    >
      {/* Header */}
      <div className="p-5 border-b border-stone-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-slate-700 to-stone-800 flex items-center justify-center text-white font-bold text-base shadow-sm">
              S
            </div>
            <div>
              <h2 className="font-semibold text-stone-800 text-sm tracking-tight leading-tight">SEC-RAG Analyst</h2>
              <span className="text-[10px] text-stone-600 font-medium bg-stone-100 px-1.5 py-0.5 rounded-md mt-0.5 inline-block">
                Fiscal Standardized
              </span>
            </div>
          </div>
          <button 
            onClick={() => setIsCollapsed(true)}
            className="p-1.5 rounded-lg hover:bg-stone-100 text-stone-400 hover:text-stone-600 transition-all"
            aria-label="Collapse sidebar"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>

        {/* Mini stats cards */}
        <div className="grid grid-cols-2 gap-3 mt-5">
          <div className="bg-stone-50 p-3 rounded-2xl border border-stone-100 flex flex-col">
            <span className="text-[10px] text-stone-400 font-medium">Total Filings</span>
            <span className="text-xl font-bold text-stone-700 mt-0.5">210</span>
            <span className="text-[9px] text-stone-400 mt-1 flex items-center gap-1">
              <Database className="w-2.5 h-2.5" /> 10-K & 10-Q
            </span>
          </div>
          <div className="bg-stone-50 p-3 rounded-2xl border border-stone-100 flex flex-col">
            <span className="text-[10px] text-stone-400 font-medium">Tickers Covered</span>
            <span className="text-xl font-bold text-stone-700 mt-0.5">14</span>
            <span className="text-[9px] text-stone-400 mt-1 flex items-center gap-1">
              <Bookmark className="w-2.5 h-2.5" /> Core coverage
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex px-5 pt-3 border-b border-stone-100">
        <button
          onClick={() => setActiveTab("filters")}
          className={`flex-1 pb-3 text-xs font-semibold text-center transition-all border-b-2 ${
            activeTab === "filters" 
              ? "border-stone-800 text-stone-800" 
              : "border-transparent text-stone-400 hover:text-stone-600"
          }`}
        >
          Filters
        </button>
        <button
          onClick={() => setActiveTab("ragas")}
          className={`flex-1 pb-3 text-xs font-semibold text-center transition-all border-b-2 ${
            activeTab === "ragas" 
              ? "border-stone-800 text-stone-800" 
              : "border-transparent text-stone-400 hover:text-stone-600"
          }`}
        >
          RAGAS Dashboard
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5">
        {activeTab === "filters" ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-stone-600 tracking-wide uppercase">Tickers Filter</span>
              <button 
                onClick={toggleAllTickers}
                className="text-[10px] text-stone-500 hover:text-stone-800 hover:underline font-semibold"
              >
                {selectedTickers.length === MOCK_TICKERS.length ? "Deselect All" : "Select All"}
              </button>
            </div>
            
            <div className="space-y-2 mt-2">
              {MOCK_TICKERS.map(ticker => {
                const isChecked = selectedTickers.includes(ticker)
                return (
                  <div 
                    key={ticker}
                    onClick={() => toggleTicker(ticker)}
                    className={`flex items-center justify-between p-3 rounded-xl border transition-all cursor-pointer select-none ${
                      isChecked 
                        ? "border-stone-900 bg-stone-950 text-white dark:bg-stone-900 dark:border-stone-800" 
                        : "border-stone-150 bg-white text-stone-500 hover:border-stone-300"
                    }`}
                  >
                    <span className="text-xs font-bold tracking-tight">{ticker}</span>
                    {isChecked ? (
                      <CheckSquare className="w-4 h-4 text-white shrink-0" />
                    ) : (
                      <Square className="w-4 h-4 text-stone-300 shrink-0" />
                    )}
                  </div>
                )
              })}
            </div>
            <p className="text-[10px] text-stone-400 leading-normal">
              Selecting specific tickers limits retrieval to those companies' filings during multi-hop reasoning. Leave blank for auto-classification.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-stone-600 tracking-wide uppercase">RAGAS Quality Metrics</span>
            </div>

            {/* Metrics grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-stone-50/80 p-3 rounded-2xl border border-stone-100">
                <span className="text-[10px] text-stone-400 font-semibold block">Faithfulness</span>
                <span className="text-lg font-bold text-stone-700 mt-0.5">0.92</span>
                <span className="text-[9px] text-stone-500 font-semibold block mt-1">Target: &gt;0.85</span>
              </div>
              <div className="bg-stone-50/80 p-3 rounded-2xl border border-stone-100">
                <span className="text-[10px] text-stone-400 font-semibold block">Answer Rel.</span>
                <span className="text-lg font-bold text-stone-700 mt-0.5">0.94</span>
                <span className="text-[9px] text-stone-500 font-semibold block mt-1">Target: &gt;0.85</span>
              </div>
              <div className="bg-stone-50/80 p-3 rounded-2xl border border-stone-100">
                <span className="text-[10px] text-stone-400 font-semibold block">Context Prec.</span>
                <span className="text-lg font-bold text-stone-700 mt-0.5">0.89</span>
                <span className="text-[9px] text-stone-500 font-semibold block mt-1">Target: &gt;0.80</span>
              </div>
              <div className="bg-stone-50/80 p-3 rounded-2xl border border-stone-100">
                <span className="text-[10px] text-stone-400 font-semibold block">Context Recall</span>
                <span className="text-lg font-bold text-stone-700 mt-0.5">0.91</span>
                <span className="text-[9px] text-stone-500 font-semibold block mt-1">Target: &gt;0.80</span>
              </div>
            </div>

            {/* Recharts chart */}
            <div className="bg-white p-3 rounded-2xl border border-stone-150 mt-2">
              <div className="flex items-center gap-1.5 mb-3">
                <BarChart3 className="w-3.5 h-3.5 text-stone-400" />
                <span className="text-[10px] font-bold text-stone-500 uppercase tracking-wider">Evaluation Trend</span>
              </div>
              
              <div className="h-44 w-full">
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={RAGAS_HISTORY} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" vertical={false} />
                      <XAxis dataKey="date" stroke="#a3a3a3" fontSize={9} tickLine={false} axisLine={false} />
                      <YAxis stroke="#a3a3a3" fontSize={9} tickLine={false} axisLine={false} domain={[0.6, 1.0]} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: "#fff", 
                          borderColor: "#e5e5e5", 
                          borderRadius: "12px", 
                          fontSize: "10px", 
                          boxShadow: "0 4px 12px rgba(0,0,0,0.05)" 
                        }} 
                      />
                      <Line type="monotone" dataKey="faithfulness" name="Faithfulness" stroke="#1c1917" strokeWidth={2} dot={{ r: 2 }} />
                      <Line type="monotone" dataKey="relevance" name="Relevance" stroke="#94a3b8" strokeWidth={2} dot={{ r: 2 }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-5 border-t border-stone-100 bg-stone-50/50 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-stone-600 animate-pulse"></span>
          <span className="text-[10px] font-semibold text-stone-600">Local Vector Cache Standby</span>
        </div>
        <span className="text-[9px] text-stone-400 font-mono">ChromaDB + SQLite</span>
      </div>
    </div>
  )
}
