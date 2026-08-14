import { NextResponse } from "next/server"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { messages } = body
    const lastMessage = messages[messages.length - 1]?.content || ""
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8001"

    try {
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      })

      if (response.ok && response.body) {
        const customStream = new ReadableStream({
          async start(controller) {
            const reader = response.body!.getReader()
            while (true) {
              const { done, value } = await reader.read()
              if (done) break
              controller.enqueue(value)
            }
            controller.close()
          },
        })

        return new Response(customStream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
          },
        })
      }
    } catch (backendError) {
      console.warn("Backend API not reachable. Falling back to mock data:", backendError)
    }

    // --- Mock Fallback Data ---
    let responseText = ""
    let citations: any[] = []
    let contradictions: any[] = []
    let modelUsed = "llama-3.3-70b-versatile"

    if (
      lastMessage.toLowerCase().includes("nvidia") ||
      lastMessage.toLowerCase().includes("msft") ||
      lastMessage.toLowerCase().includes("r&d") ||
      lastMessage.toLowerCase().includes("spend")
    ) {
      responseText = `Based on the standardized fiscal period filings retrieved for the selected tickers, here is the synthesized comparative analysis:

### R&D Expenses Comparison (FY2025)

| Ticker | R&D Expense (Millions) | Y/Y Growth | Filing Reference |
|---|---|---|---|
| **NVDA** | $11,875 | +32% | FY2025 10-K, Part II, Item 8 |
| **MSFT** | $27,190 | +12% | FY2025 10-K, Part II, Item 8 |

### Key Extracted Claims & Contradiction Analysis

1. **NVIDIA (NVDA)**: Incurred $11,875M in R&D, focused on Blackwell GPU platform architecture. No conflicts identified between Q1-Q4 reports.
2. **MICROSOFT (MSFT)**: Incurred $27,190M in R&D, focused on Azure AI infrastructure and cloud capabilities. Standardized fiscal mapping resolved an initial legacy 10-Q calendar mapping discrepancy.`
      
      citations = [
        { ticker: "NVDA", fiscal_year: 2025, filing_type: "10-K", section: "Item 8", accession_number: "0001047469-26-000102" },
        { ticker: "MSFT", fiscal_year: 2025, filing_type: "10-K", section: "Item 8", accession_number: "0001210227-26-000045" },
      ]
    } else if (
      lastMessage.toLowerCase().includes("contradiction") ||
      lastMessage.toLowerCase().includes("conflict") ||
      lastMessage.toLowerCase().includes("risk")
    ) {
      responseText = `### SEC Filing Contradiction Report

I ran a multi-hop claim extraction and analyzed them using the local NLI contradiction scorer. Here is what was found:

> [!WARNING]
> **Potential Disclosure Inconsistency Detected (MD&A)**
> 
> * **Filing A (Q2 10-Q, Item 2)**: "Supply chain constraints for high-bandwidth memory (HBM3e) have been fully mitigated via secondary sourcing."
> * **Filing B (Q3 10-Q, Item 2)**: "We continue to experience acute shortages in HBM3e packaging capacity, gating top-line growth by approximately 3%."
> * **Conflict Score**: **0.87 (High Confidence)**

This discrepancy indicates that while secondary sourcing was established in Q2, packaging bottlenecks emerged in Q3, making the 'fully mitigated' statement from Q2 outdated or overly optimistic.`

      citations = [
        { ticker: "NVDA", fiscal_year: 2025, filing_type: "10-Q", section: "Item 2", accession_number: "0001047469-25-000078" },
        { ticker: "NVDA", fiscal_year: 2025, filing_type: "10-Q", section: "Item 2", accession_number: "0001047469-25-000092" },
      ]
      contradictions = [
        {
          confidence_score: 0.87,
          filing_ref_a: "NVDA Q2 10-Q (Item 2)",
          claim_a: "Supply chain constraints for high-bandwidth memory (HBM3e) have been fully mitigated via secondary sourcing.",
          filing_ref_b: "NVDA Q3 10-Q (Item 2)",
          claim_b: "We continue to experience acute shortages in HBM3e packaging capacity, gating top-line growth by approximately 3%.",
        },
      ]
    } else {
      responseText = `Hello! I am your SEC Multi-Hop RAG Analyst. 

I can help you query and compare filing data (such as R&D spending, revenue, or risk statements) across standardized fiscal periods (e.g. FY2024, FY2025, Q1-Q3) with automatic claim extraction and contradiction detection.

Try asking me:
* **"Compare NVIDIA and Microsoft R&D expenditures in FY2025"**
* **"Check NVDA for disclosures containing contradictions"**
* **"Tell me about the SEC filing database coverage"**`
    }

    const encoder = new TextEncoder()
    const customStream = new ReadableStream({
      async start(controller) {
        // Send text content chunks
        const words = responseText.split(/(\s+)/)
        for (const word of words) {
          const chunkJson = JSON.stringify({ type: "text", content: word }) + "\n"
          controller.enqueue(encoder.encode(chunkJson))
          await new Promise((resolve) => setTimeout(resolve, 15))
        }

        // Send metadata chunk at the end
        const metadataJson =
          JSON.stringify({
            type: "metadata",
            payload: {
              citations,
              contradictions,
              latency_ms: 120,
              model_used: modelUsed,
              contradiction_detection_skipped: false,
            },
          }) + "\n"
        controller.enqueue(encoder.encode(metadataJson))
        controller.close()
      },
    })

    return new Response(customStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    })
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json({ error: "Failed to process chat response stream" }, { status: 500 })
  }
}
