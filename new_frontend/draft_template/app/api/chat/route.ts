import { NextResponse } from "next/server"

export async function POST(req: Request) {
  try {
    const { messages } = await req.json()
    const lastMessage = messages[messages.length - 1]?.content || ""

    // Create a mock response text based on the query
    let responseText = ""
    
    if (lastMessage.toLowerCase().includes("nvidia") || lastMessage.toLowerCase().includes("msft") || lastMessage.toLowerCase().includes("r&d") || lastMessage.toLowerCase().includes("spend")) {
      responseText = `Based on the standardized fiscal period filings retrieved for the selected tickers, here is the synthesized comparative analysis:

### R&D Expenses Comparison (FY2025)

| Ticker | R&D Expense (Millions) | Y/Y Growth | Filing Reference |
|---|---|---|---|
| **NVDA** | $11,875 | +32% | FY2025 10-K, Part II, Item 8 |
| **MSFT** | $27,190 | +12% | FY2025 10-K, Part II, Item 8 |

### Key Extracted Claims & Contradiction Analysis

1. **NVIDIA (NVDA)**: Incurred $11,875M in R&D, focused on Blackwell GPU platform architecture. No conflicts identified between Q1-Q4 reports.
2. **MICROSOFT (MSFT)**: Incurred $27,190M in R&D, focused on Azure AI infrastructure and cloud capabilities. Standardized fiscal mapping resolved an initial legacy 10-Q calendar mapping discrepancy.

### Citations & Sources
- **NVDA FY25 10-K** (filed Feb 2026), Accession No: \`0001047469-26-000102\`
- **MSFT FY25 10-K** (filed Feb 2026), Accession No: \`0001210227-26-000045\``
    } else if (lastMessage.toLowerCase().includes("contradiction") || lastMessage.toLowerCase().includes("conflict") || lastMessage.toLowerCase().includes("risk")) {
      responseText = `### SEC Filing Contradiction Report

I ran a multi-hop claim extraction and analyzed them using the local NLI contradiction scorer. Here is what was found:

> [!WARNING]
> **Potential Disclosure Inconsistency Detected (MD&A)**
> 
> * **Filing A (Q2 10-Q, Item 2)**: "Supply chain constraints for high-bandwidth memory (HBM3e) have been fully mitigated via secondary sourcing."
> * **Filing B (Q3 10-Q, Item 2)**: "We continue to experience acute shortages in HBM3e packaging capacity, gating top-line growth by approximately 3%."
> * **Conflict Score**: **0.87 (High Confidence)**

This discrepancy indicates that while secondary sourcing was established in Q2, packaging bottlenecks emerged in Q3, making the 'fully mitigated' statement from Q2 outdated or overly optimistic.

### Citations
- **NVDA Q2 10-Q** (filed Aug 2025), Accession No: \`0001047469-25-000078\`
- **NVDA Q3 10-Q** (filed Nov 2025), Accession No: \`0001047469-25-000092\``
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
        // Send the response text in chunks of words
        const words = responseText.split(/(\s+)/)
        for (const word of words) {
          controller.enqueue(encoder.encode(word))
          // Simulate streaming speed
          await new Promise((resolve) => setTimeout(resolve, 30))
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
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json({ error: "Failed to process chat mock stream" }, { status: 500 })
  }
}

