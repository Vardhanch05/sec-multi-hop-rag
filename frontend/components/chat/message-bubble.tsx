"use client"

import { cn } from "@/lib/utils"
import type { Message } from "./chat-shell"
import { User } from "lucide-react"
import { MarkdownRenderer } from "./markdown-renderer"
import Image from "next/image"
import { AnimatedOrb } from "./animated-orb"

interface MessageBubbleProps {
  message: Message
  isStreaming?: boolean
}

// Format time for display
function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
}

export function MessageBubble({ message, isStreaming = false }: MessageBubbleProps) {
  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "flex max-w-[90%] md:max-w-[80%] gap-2",
        isUser
          ? "ml-auto flex-row-reverse user-message-enter"
          : "mr-auto animate-in fade-in slide-in-from-bottom-2 duration-300 items-end",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
          isUser ? "bg-white" : "bg-zinc-800",
          !isUser && isStreaming && "sticky bottom-4 self-end transition-all duration-300",
        )}
        style={{
          boxShadow:
            "rgba(14, 63, 126, 0.04) 0px 0px 0px 1px, rgba(42, 51, 69, 0.04) 0px 1px 1px -0.5px, rgba(42, 51, 70, 0.04) 0px 3px 3px -1.5px, rgba(42, 51, 70, 0.04) 0px 6px 6px -3px, rgba(14, 63, 126, 0.04) 0px 12px 12px -6px, rgba(14, 63, 126, 0.04) 0px 24px 24px -12px",
        }}
        aria-hidden="true"
      >
        {isUser ? <User className="w-4 h-4 text-stone-800" /> : <AnimatedOrb className="w-8 h-8 shrink-0" />}
      </div>

      {/* Message content */}
      <div className={cn("flex flex-col w-full", isUser ? "items-end" : "items-start")}>
        {/* Role label (optional, shown on larger screens) */}
        <span className="text-xs text-stone-400 mb-1 hidden sm:block mt-2">{isUser ? "You" : "Assistant"}</span>

        {/* Bubble */}
        <div
          className={cn(
            "rounded-2xl border-none overflow-hidden w-full",
            isUser
              ? "bg-white text-stone-800 border border-stone-200 rounded-br-md max-w-max ml-auto"
              : "bg-transparent text-stone-800 rounded-bl-md",
          )}
          style={{
            boxShadow: isUser
              ? "rgba(14, 63, 126, 0.04) 0px 0px 0px 1px, rgba(42, 51, 69, 0.04) 0px 1px 1px -0.5px, rgba(42, 51, 70, 0.04) 0px 3px 3px -1.5px, rgba(42, 51, 70, 0.04) 0px 6px 6px -3px, rgba(14, 63, 126, 0.04) 0px 12px 12px -6px, rgba(14, 63, 126, 0.04) 0px 24px 24px -12px"
              : "none",
            willChange: isStreaming ? "height" : "auto",
            transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        >
          <div
            className={cn(isUser ? "px-4 py-3" : "py-1")}
            style={{
              transition: "max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease",
            }}
          >
            {isUser ? (
              <div className="flex flex-col gap-2">
                {message.imageData && (
                  <div className="w-20 h-20 rounded-lg overflow-hidden border border-stone-200">
                    <Image
                      src={message.imageData || "/placeholder.svg"}
                      alt="Uploaded image"
                      width={80}
                      height={80}
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}
                <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
              </div>
            ) : (
              <div className="w-full space-y-4">
                <MarkdownRenderer content={message.content || " "} isStreaming={isStreaming} />

                {/* Contradiction Detection Skipped Notice */}
                {message.contradiction_detection_skipped && (
                  <div className="text-[10px] text-amber-700 bg-amber-50/50 border border-amber-200/50 px-3 py-2 rounded-lg font-medium">
                    ⚠️ Contradiction detection skipped due to NLI model execution timeout.
                  </div>
                )}

                {/* Contradictions Cards */}
                {message.contradictions && message.contradictions.length > 0 && (
                  <div className="space-y-3 mt-4">
                    <h4 className="text-[10px] font-bold text-stone-500 uppercase tracking-wider">Contradictions Detected</h4>
                    <div className="space-y-2">
                      {message.contradictions.map((c, i) => {
                        const isHigh = c.confidence_score >= 0.90
                        const bgColor = isHigh ? "bg-red-50/40 border-red-200 text-red-950" : "bg-amber-50/40 border-amber-200 text-amber-950"
                        const badgeColor = isHigh ? "bg-red-100 text-red-800 border-red-200" : "bg-amber-100 text-amber-800 border-amber-200"
                        
                        return (
                          <div key={i} className={cn("p-4 rounded-2xl border text-xs shadow-sm space-y-3", bgColor)}>
                            <div className="flex items-center justify-between border-b border-stone-200/40 pb-2">
                              <span className="font-bold text-stone-850">Contradiction Event #{i + 1}</span>
                              <span className={cn("px-2 py-0.5 rounded-full border font-mono text-[9px] font-bold", badgeColor)}>
                                Confidence: {Number(c.confidence_score).toFixed(2)}
                              </span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="space-y-1">
                                <span className="font-bold text-[9px] uppercase tracking-wider text-stone-500 block">
                                  {c.filing_ref_a}
                                </span>
                                <p className="leading-relaxed text-stone-750 font-medium">{c.claim_a}</p>
                              </div>
                              <div className="space-y-1">
                                <span className="font-bold text-[9px] uppercase tracking-wider text-stone-500 block">
                                  {c.filing_ref_b}
                                </span>
                                <p className="leading-relaxed text-stone-750 font-medium">{c.claim_b}</p>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Citations Badges */}
                {message.citations && message.citations.length > 0 && (
                  <div className="space-y-2 mt-4">
                    <h4 className="text-[10px] font-bold text-stone-500 uppercase tracking-wider">Citations</h4>
                    <div className="flex flex-wrap gap-2">
                      {message.citations.map((cit, i) => (
                        <span 
                          key={i} 
                          className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-stone-100 hover:bg-stone-150 border border-stone-200/50 text-[10px] font-medium text-stone-600 cursor-help transition-all shadow-sm"
                          title={`Accession: ${cit.accession_number}`}
                        >
                          🛡️ {cit.ticker} {cit.fiscal_year} {cit.filing_type} ({cit.section})
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Latency & Model info */}
                {(message.latency_ms !== undefined || message.model_used) && (
                  <div className="flex items-center gap-4 text-[10px] text-stone-400 font-semibold border-t border-stone-100 pt-3 mt-4">
                    {message.latency_ms !== undefined && (
                      <span>Latency: {message.latency_ms !== null ? `${message.latency_ms}ms` : "0ms (Cached)"}</span>
                    )}
                    {message.model_used && (
                      <span>Model: {message.model_used}</span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Timestamp */}
        <span className="text-xs text-stone-400 mt-1">{formatTime(message.createdAt)}</span>
      </div>
    </div>
  )
}
