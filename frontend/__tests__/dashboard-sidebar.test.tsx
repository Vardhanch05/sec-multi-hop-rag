import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { DashboardSidebar } from "../components/chat/dashboard-sidebar"

// Mock recharts ResponsiveContainer to avoid JSOM layout sizing issues
jest.mock("recharts", () => {
  const OriginalModule = jest.requireActual("recharts")
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 500, height: 300 }}>{children}</div>
    ),
  }
})

describe("DashboardSidebar Component", () => {
  const defaultProps = {
    selectedTickers: ["AAPL"],
    onTickerChange: jest.fn(),
    isCollapsed: false,
    setIsCollapsed: jest.fn(),
    tickers: ["AAPL", "NVDA", "TSLA"],
    stats: { totalFilings: 210, uniqueTickers: 3 },
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders sidebar title and mini stats cards", () => {
    render(<DashboardSidebar {...defaultProps} />)
    expect(screen.getByText("SEC-RAG Analyst")).toBeInTheDocument()
    expect(screen.getByText("210")).toBeInTheDocument()
    expect(screen.getByText("3")).toBeInTheDocument()
  })

  it("renders ticker checklist and triggers ticker toggle callback", () => {
    render(<DashboardSidebar {...defaultProps} />)
    expect(screen.getByText("AAPL")).toBeInTheDocument()
    expect(screen.getByText("NVDA")).toBeInTheDocument()

    const nvdaItem = screen.getByText("NVDA")
    fireEvent.click(nvdaItem)

    expect(defaultProps.onTickerChange).toHaveBeenCalledWith(["AAPL", "NVDA"])
  })

  it("switches tabs between Filters and RAGAS Dashboard", () => {
    render(<DashboardSidebar {...defaultProps} />)
    const ragasTab = screen.getByRole("button", { name: "RAGAS Dashboard" })
    fireEvent.click(ragasTab)

    expect(screen.getByText("RAGAS Quality Metrics")).toBeInTheDocument()
    expect(screen.getByText("Faithfulness")).toBeInTheDocument()
  })
})
