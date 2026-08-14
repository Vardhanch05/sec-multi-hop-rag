"use client"

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from "@/lib/utils"
import { AlertTriangle, Info, ShieldAlert, Sparkles } from "lucide-react"

interface MarkdownRendererProps {
  content: string
  className?: string
  isStreaming?: boolean
}

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  return (
    <div className={cn("text-sm text-stone-800 leading-relaxed space-y-2 prose max-w-none w-full", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ node, ...props }) => (
            <div className="w-full my-4 overflow-x-auto border border-stone-200 rounded-xl shadow-sm bg-white">
              <table className="min-w-full divide-y divide-stone-200 border-collapse" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => <thead className="bg-stone-50/75" {...props} />,
          tbody: ({ node, ...props }) => <tbody className="divide-y divide-stone-100 bg-white" {...props} />,
          tr: ({ node, ...props }) => <tr className="hover:bg-stone-50/50 transition-colors" {...props} />,
          th: ({ node, ...props }) => (
            <th className="px-4 py-2.5 text-left text-xs font-bold text-stone-500 uppercase tracking-wider border-b border-stone-200" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="px-4 py-2.5 text-sm text-stone-655" {...props} />
          ),
          p: ({ node, ...props }) => <p className="mb-3 last:mb-0 text-stone-700 leading-relaxed" {...props} />,
          code: ({ node, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '')
            const isInline = !match
            return isInline ? (
              <code className="px-1.5 py-0.5 bg-stone-150 text-stone-800 rounded font-mono text-[12px] font-medium" {...props}>
                {children}
              </code>
            ) : (
              <pre className="p-4 bg-stone-900 text-stone-100 rounded-xl my-4 overflow-x-auto text-xs font-mono border border-stone-800 shadow-md">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            )
          },
          blockquote: ({ node, children }) => {
            // Find text content inside children to check for GitHub alerts
            let alertType: "note" | "warning" | "important" | "tip" | null = null
            let plainText = ""
            
            // Extract text content from React children
            const extractText = (c: any): string => {
              if (typeof c === "string") return c
              if (Array.isArray(c)) return c.map(extractText).join("")
              if (c?.props?.children) return extractText(c.props.children)
              return ""
            }
            
            plainText = extractText(children)
            
            if (plainText.includes("[!NOTE]")) alertType = "note"
            else if (plainText.includes("[!WARNING]")) alertType = "warning"
            else if (plainText.includes("[!IMPORTANT]")) alertType = "important"
            else if (plainText.includes("[!TIP]")) alertType = "tip"
            
            if (alertType) {
              // Clean up the alert tag prefix from children recursively
              const cleanChildren = (child: any): any => {
                if (typeof child === "string") {
                  return child
                    .replace(/\[!NOTE\]\s*/i, "")
                    .replace(/\[!WARNING\]\s*/i, "")
                    .replace(/\[!IMPORTANT\]\s*/i, "")
                    .replace(/\[!TIP\]\s*/i, "")
                }
                if (Array.isArray(child)) {
                  return child.map(cleanChildren)
                }
                if (child?.props?.children) {
                  return {
                    ...child,
                    props: {
                      ...child.props,
                      children: cleanChildren(child.props.children)
                    }
                  }
                }
                return child
              }
              
              const cleaned = cleanChildren(children)
              
              const styles = {
                note: {
                  border: "border-l-4 border-stone-400",
                  bg: "bg-stone-50",
                  text: "text-stone-800",
                  icon: <Info className="w-4 h-4 text-stone-500 shrink-0" />,
                  label: "Note"
                },
                warning: {
                  border: "border-l-4 border-amber-500",
                  bg: "bg-amber-50/30",
                  text: "text-amber-900",
                  icon: <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />,
                  label: "Warning"
                },
                important: {
                  border: "border-l-4 border-red-500",
                  bg: "bg-red-50/30",
                  text: "text-red-900",
                  icon: <ShieldAlert className="w-4 h-4 text-red-500 shrink-0" />,
                  label: "Important"
                },
                tip: {
                  border: "border-l-4 border-stone-400",
                  bg: "bg-stone-50",
                  text: "text-stone-800",
                  icon: <Sparkles className="w-4 h-4 text-stone-500 shrink-0" />,
                  label: "Tip"
                }
              }[alertType]

              return (
                <div className={cn("p-4 rounded-r-xl border my-4 flex gap-3 items-start w-full", styles.border, styles.bg)}>
                  {styles.icon}
                  <div className="flex-1">
                    <span className={cn("text-xs font-bold uppercase tracking-wider block mb-1", styles.text)}>
                      {styles.label}
                    </span>
                    <div className="text-stone-700 text-sm leading-relaxed">{cleaned}</div>
                  </div>
                </div>
              )
            }

            return (
              <blockquote className="border-l-4 border-stone-300 pl-4 py-1 my-4 text-stone-600 italic bg-stone-50/50 pr-4 rounded-r-lg">
                {children}
              </blockquote>
            )
          },
          a: ({ node, ...props }) => (
            <a 
              className="text-stone-800 dark:text-stone-100 hover:text-stone-950 underline underline-offset-2 transition-colors font-semibold"
              target="_blank"
              rel="noopener noreferrer"
              {...props} 
            />
          ),
          h1: ({ node, ...props }) => <h1 className="text-lg font-bold text-stone-850 mt-5 mb-2.5 tracking-tight border-b border-stone-100 pb-1" {...props} />,
          h2: ({ node, ...props }) => <h2 className="text-base font-bold text-stone-800 mt-4 mb-2 tracking-tight" {...props} />,
          h3: ({ node, ...props }) => <h3 className="text-sm font-bold text-stone-800 mt-3.5 mb-1.5" {...props} />,
          ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-3 space-y-1 text-stone-700" {...props} />,
          ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-3 space-y-1 text-stone-700" {...props} />,
          li: ({ node, ...props }) => <li className="text-sm leading-relaxed" {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

