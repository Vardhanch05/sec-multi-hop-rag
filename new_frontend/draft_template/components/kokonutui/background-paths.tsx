"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

function FloatingPaths({ position }: { position: number }) {
  const paths = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    // Scaled down coordinates to make the waves tighter and less overwhelming
    d: `M-${280 - i * 4 * position} -${120 + i * 4}C-${
      280 - i * 4 * position
    } -${120 + i * 4} -${180 - i * 8 * position} ${150 - i * 6} ${
      60 - i * 6 * position
    } ${240 - i * 5}C${400 - i * 10 * position} ${320 - i * 8} ${
      500 - i * 12 * position
    } ${600 - i * 4} ${500 - i * 12 * position} ${600 - i * 4}`,
    width: 0.3 + i * 0.015,
  }))

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg className="w-full h-full text-neutral-950/15 dark:text-white/10" viewBox="0 0 696 316" fill="none">
        <title>Background Paths</title>
        {paths.map((path) => (
          <path
            key={path.id}
            d={path.d}
            stroke="currentColor"
            strokeWidth={path.width}
            strokeOpacity={0.1 + path.id * 0.02}
          />
        ))}
      </svg>
    </div>
  )
}

export default function BackgroundPaths({
  title = "SEC Filings RAG Analyst",
  subtitle = "High-fidelity financial intelligence synthesized across multiple filings and periods with automatic contradiction detection",
}: {
  title?: string
  subtitle?: string
}) {
  const words = title.split(" ")
  const router = useRouter()

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center overflow-hidden bg-white dark:bg-neutral-950">
      <div className="absolute inset-0">
        <FloatingPaths position={1} />
        <FloatingPaths position={-1} />
      </div>

      <div className="relative z-10 container mx-auto px-4 md:px-6 text-center flex flex-col items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2 }}
          className="max-w-4xl mx-auto"
        >
          <h1 className="text-5xl sm:text-7xl md:text-8xl font-bold mb-6 tracking-tighter">
            {words.map((word, wordIndex) => (
              <span key={wordIndex} className="inline-block mr-4 last:mr-0">
                {word.split("").map((letter, letterIndex) => (
                  <motion.span
                    key={`${wordIndex}-${letterIndex}`}
                    initial={{ y: 100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                      delay: wordIndex * 0.1 + letterIndex * 0.03,
                      type: "spring",
                      stiffness: 150,
                      damping: 25,
                    }}
                    className="inline-block text-transparent bg-clip-text 
                                        bg-gradient-to-r from-neutral-900 to-neutral-700/80 
                                        dark:from-white dark:to-white/80"
                  >
                    {letter}
                  </motion.span>
                ))}
              </span>
            ))}
          </h1>

          <motion.p
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 0.6 }}
            transition={{ delay: 0.8, duration: 1 }}
            className="text-lg md:text-xl text-neutral-600 dark:text-neutral-300 max-w-2xl mx-auto mb-10 font-normal leading-relaxed"
          >
            {subtitle}
          </motion.p>

          <div
            className="inline-block group relative bg-gradient-to-b from-zinc-200 to-zinc-300/85 
                        dark:from-zinc-800 to-zinc-700/85 p-px rounded-2xl backdrop-blur-lg 
                        overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300"
          >
            <Button
              onClick={() => router.push("/chat")}
              variant="ghost"
              className="rounded-[1.15rem] px-8 py-6 text-lg font-semibold backdrop-blur-md 
                            bg-white/95 hover:bg-white/100 dark:bg-neutral-900/95 dark:hover:bg-neutral-900/100 
                            text-neutral-800 dark:text-neutral-100 transition-all duration-300 
                            group-hover:-translate-y-0.5 border border-zinc-200 dark:border-zinc-800
                            hover:shadow-md dark:hover:shadow-neutral-900/50"
            >
              <span className="opacity-90 group-hover:opacity-100 transition-opacity">Launch</span>
              <span
                className="ml-3 opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 
                                transition-all duration-300"
              >
                →
              </span>
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}


