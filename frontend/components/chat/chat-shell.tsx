"use client"

import { useState, useEffect, useCallback } from "react"
import { MessageSquareDashed } from "lucide-react"
import { MessageList } from "./message-list"
import { Composer } from "./composer"
import { Button } from "@/components/ui/button"
import { DashboardSidebar } from "./dashboard-sidebar"

// Data model for messages
export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  createdAt: Date
  imageData?: string
  citations?: {
    filing_type: string
    section: string
    ticker: string
    fiscal_year: number
    accession_number: string
  }[]
  contradictions?: {
    confidence_score: number
    filing_ref_a: string
    claim_a: string
    filing_ref_b: string
    claim_b: string
  }[]
  latency_ms?: number | null
  model_used?: string
  contradiction_detection_skipped?: boolean
}

// localStorage key for persisting messages
const STORAGE_KEY = "chat-messages"

// Generates a unique ID for messages
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

export function ChatShell() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  const [isLoaded, setIsLoaded] = useState(false)
  const [selectedTickers, setSelectedTickers] = useState<string[]>([])
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  
  // Dynamic API states
  const [tickers, setTickers] = useState<string[]>([])
  const [stats, setStats] = useState({ totalFilings: 0, uniqueTickers: 0 })
  const [ragasHistory, setRagasHistory] = useState<any[]>([])

  // Resizable sidebar states
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const [isResizing, setIsResizing] = useState(false)

  const startResizing = useCallback(() => {
    setIsResizing(true)
  }, [])

  const stopResizing = useCallback(() => {
    setIsResizing(false)
  }, [])

  const resize = useCallback((mouseMoveEvent: MouseEvent) => {
    const newWidth = Math.max(240, Math.min(520, mouseMoveEvent.clientX))
    setSidebarWidth(newWidth)
  }, [])

  useEffect(() => {
    if (isResizing) {
      window.addEventListener("mousemove", resize)
      window.addEventListener("mouseup", stopResizing)
    } else {
      window.removeEventListener("mousemove", resize)
      window.removeEventListener("mouseup", stopResizing)
    }
    return () => {
      window.removeEventListener("mousemove", resize)
      window.removeEventListener("mouseup", stopResizing)
    }
  }, [isResizing, resize, stopResizing])

  // Fetch dashboard stats on mount
  useEffect(() => {
    fetch("/api/stats")
      .then((res) => res.json())
      .then((data) => setStats(data))
      .catch((err) => console.error("Error fetching stats:", err))

    fetch("/api/tickers")
      .then((res) => res.json())
      .then((data) => setTickers(data))
      .catch((err) => console.error("Error fetching tickers:", err))

    fetch("/api/ragas")
      .then((res) => res.json())
      .then((data) => setRagasHistory(data))
      .catch((err) => console.error("Error fetching RAGAS metrics:", err))
  }, [])

  // Load messages from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored)
        const messagesWithDates = parsed.map((msg: Message) => ({
          ...msg,
          createdAt: new Date(msg.createdAt),
        }))
        setMessages(messagesWithDates)
      }
    } catch (e) {
      console.error("Failed to load from localStorage:", e)
    } finally {
      setIsLoaded(true)
    }
  }, [])

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    if (!isLoaded) return
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
    } catch (e) {
      console.error("Failed to save messages to localStorage:", e)
    }
  }, [messages, isLoaded])

  // Send a message to the AI
  const sendMessage = useCallback(
    async (content: string, imageData?: string) => {
      if ((!content.trim() && !imageData) || isStreaming) return

      setError(null)

      const userMessage: Message = {
        id: generateId(),
        role: "user",
        content: content.trim() || "Describe this image",
        createdAt: new Date(),
        imageData,
      }

      const assistantMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: "",
        createdAt: new Date(),
      }

      const newMessages = [...messages, userMessage, assistantMessage]
      setMessages(newMessages)
      setIsStreaming(true)

      const controller = new AbortController()
      setAbortController(controller)

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            tickers: selectedTickers,
            messages: [...messages, userMessage].map((m) => ({
              role: m.role,
              content: m.content,
              imageData: m.imageData,
            })),
          }),
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (!reader) {
          throw new Error("No response body")
        }

        let accumulatedContent = ""
        let streamBuffer = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          streamBuffer += chunk

          const lines = streamBuffer.split("\n")
          // Keep the last partial line in the buffer
          streamBuffer = lines.pop() || ""

          for (const line of lines) {
            if (!line.trim()) continue
            try {
              const data = JSON.parse(line)
              if (data.type === "text") {
                accumulatedContent += data.content
                setMessages((prev) =>
                  prev.map((msg) => (msg.id === assistantMessage.id ? { ...msg, content: accumulatedContent } : msg)),
                )
              } else if (data.type === "metadata") {
                const meta = data.payload
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessage.id
                      ? {
                          ...msg,
                          citations: meta.citations,
                          contradictions: meta.contradictions,
                          latency_ms: meta.latency_ms,
                          model_used: meta.model_used,
                          contradiction_detection_skipped: meta.contradiction_detection_skipped,
                        }
                      : msg,
                  ),
                )
              }
            } catch (err) {
              console.warn("Failed to parse stream line, appending raw chunk:", line, err)
              // Safe fallback: if it's not JSON, it might be raw chunk text from legacy code
              accumulatedContent += line
              setMessages((prev) =>
                prev.map((msg) => (msg.id === assistantMessage.id ? { ...msg, content: accumulatedContent } : msg)),
              )
            }
          }
        }
      } catch (e) {
        if (e instanceof Error && e.name === "AbortError") {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessage.id ? { ...msg, content: msg.content || "[Cancelled]" } : msg,
            ),
          )
        } else {
          console.error("Error sending message:", e)
          setError(e instanceof Error ? e.message : "An error occurred")
          setMessages((prev) => prev.filter((msg) => msg.id !== assistantMessage.id))
        }
      } finally {
        setIsStreaming(false)
        setAbortController(null)
      }
    },
    [messages, isStreaming, selectedTickers],
  )

  const retry = useCallback(() => {
    if (messages.length === 0) return
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user")
    if (lastUserMessage) {
      const index = messages.findIndex((m) => m.id === lastUserMessage.id)
      setMessages(messages.slice(0, index))
      setError(null)
      setTimeout(() => sendMessage(lastUserMessage.content, lastUserMessage.imageData), 100)
    }
  }, [messages, sendMessage])

  const stopStreaming = useCallback(() => {
    if (abortController) {
      abortController.abort()
    }
  }, [abortController])

  const clearChat = useCallback(() => {
    setMessages([])
    setError(null)
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-stone-50">
      <DashboardSidebar 
        selectedTickers={selectedTickers}
        onTickerChange={setSelectedTickers}
        isCollapsed={isSidebarCollapsed}
        setIsCollapsed={setIsSidebarCollapsed}
        customWidth={sidebarWidth}
        isResizing={isResizing}
        tickers={tickers}
        stats={stats}
        ragasHistory={ragasHistory}
      />

      {!isSidebarCollapsed && (
        <div
          className="w-1 cursor-col-resize hover:bg-stone-400/50 active:bg-stone-600 bg-stone-200 h-full select-none shrink-0 z-50 transition-colors"
          onMouseDown={startResizing}
        />
      )}

      <div
        className="flex-1 relative h-full overflow-hidden"
        style={{
          boxShadow:
            "rgba(14, 63, 126, 0.04) 0px 0px 0px 1px, rgba(42, 51, 69, 0.04) 0px 1px 1px -0.5px, rgba(42, 51, 70, 0.04) 0px 3px 3px -1.5px, rgba(42, 51, 70, 0.04) 0px 6px 6px -3px, rgba(14, 63, 126, 0.04) 0px 12px 12px -6px, rgba(14, 63, 126, 0.04) 0px 24px 24px -12px",
        }}
      >
        <Button
          onClick={clearChat}
          variant="ghost"
          size="icon"
          className="absolute top-4 right-4 z-20 h-10 w-10 rounded-full bg-zinc-100 hover:bg-zinc-200 text-stone-600"
          aria-label="Reset chat"
        >
          <MessageSquareDashed className="w-5 h-5" />
        </Button>

        <MessageList messages={messages} isStreaming={isStreaming} error={error} onRetry={retry} isLoaded={isLoaded} />

        <Composer
          onSend={sendMessage}
          onStop={stopStreaming}
          isStreaming={isStreaming}
          disabled={!!error}
        />
      </div>
    </div>
  )
}
