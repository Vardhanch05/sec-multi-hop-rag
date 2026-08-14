import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { IngestModal } from "../components/chat/ingest-modal"

// Mock global fetch
global.fetch = jest.fn()

describe("IngestModal Component", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("does not render when isOpen is false", () => {
    const { container } = render(<IngestModal isOpen={false} onClose={jest.fn()} />)
    expect(container.firstChild).toBeNull()
  })

  it("renders modal title and form when isOpen is true", () => {
    render(<IngestModal isOpen={true} onClose={jest.fn()} />)
    expect(screen.getByText("Async Background Ingestion")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("e.g. AAPL, TSLA, NVDA")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start Async Ingestion" })).toBeInTheDocument()
  })

  it("submits background task request when form is filled", async () => {
    const mockTaskResponse = {
      task_id: "task_TSLA_2025_12345",
      ticker: "TSLA",
      fiscal_year: 2025,
      status: "processing",
      message: "Filing ingestion task queued for TSLA",
    }

    ;(global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockTaskResponse,
    })

    render(<IngestModal isOpen={true} onClose={jest.fn()} />)

    const tickerInput = screen.getByPlaceholderText("e.g. AAPL, TSLA, NVDA")
    fireEvent.change(tickerInput, { target: { value: "tsla" } })

    const submitBtn = screen.getByRole("button", { name: "Start Async Ingestion" })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: "TSLA", fiscal_year: 2025 }),
      })
    })

    expect(await screen.findByText("TSLA (FY2025)")).toBeInTheDocument()
  })
})
