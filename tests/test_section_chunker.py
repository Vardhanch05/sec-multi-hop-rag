import pytest
from datetime import date
from hypothesis import given, strategies as st
from ingestion.edgar_client import FilingRef
from ingestion.section_chunker import Chunk, chunk_filing, extract_sections, chunk_text_by_tokens

@st.composite
def filing_ref_strategy(draw):
    return FilingRef(
        ticker=draw(st.text(min_size=1, max_size=5)),
        filing_type=draw(st.sampled_from(["10-Q", "10-K"])),
        accession_number=draw(st.text(min_size=5, max_size=20)),
        filing_date=draw(st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 1, 1))),
        source_url=draw(st.text(min_size=10, max_size=50)),
        quarter=draw(st.sampled_from(["Q1", "Q2", "Q3", "Q4", None])),
        fiscal_year=draw(st.integers(min_value=2000, max_value=2030))
    )

@given(
    text=st.text(min_size=1, max_size=5000),
    filing_ref=filing_ref_strategy()
)
def test_property_1_metadata_completeness(text, filing_ref):
    chunks = chunk_filing(text, filing_ref)
    for chunk in chunks:
        assert chunk.ticker == filing_ref.ticker
        assert chunk.filing_type == filing_ref.filing_type
        assert chunk.quarter == filing_ref.quarter
        assert chunk.fiscal_year == filing_ref.fiscal_year
        assert chunk.filing_date == filing_ref.filing_date
        assert chunk.accession_number == filing_ref.accession_number
        assert chunk.source_url == filing_ref.source_url

@given(
    text=st.text(min_size=1, max_size=5000),
    filing_ref=filing_ref_strategy()
)
def test_property_2_section_validity(text, filing_ref):
    valid_sections = {"MD&A", "Risk Factors", "Forward Guidance", "Financial Statements", "Other"}
    chunks = chunk_filing(text, filing_ref)
    for chunk in chunks:
        assert chunk.section_type in valid_sections

@given(
    text=st.text(min_size=1, max_size=500),
    ticker=st.text(min_size=1),
    filing_type=st.text(min_size=1),
    quarter=st.sampled_from(["Q1", None]),
    fiscal_year=st.integers(),
    section_type=st.sampled_from(["MD&A", "Other"]),
    chunk_index=st.integers(),
    filing_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 1, 1)),
    accession_number=st.text(min_size=1),
    source_url=st.text(min_size=1)
)
def test_property_22_serialization_round_trip(
    text, ticker, filing_type, quarter, fiscal_year,
    section_type, chunk_index, filing_date, accession_number, source_url
):
    chunk = Chunk(
        text=text, ticker=ticker, filing_type=filing_type, quarter=quarter,
        fiscal_year=fiscal_year, section_type=section_type, chunk_index=chunk_index,
        filing_date=filing_date, accession_number=accession_number, source_url=source_url
    )
    
    json_str = chunk.to_json()
    round_trip_chunk = Chunk.from_json(json_str)
    
    assert chunk == round_trip_chunk

def test_extract_sections():
    text = "Some intro text. Item 2. Management's Discussion and Analysis. This is MD&A. Item 1A. Risk Factors. This is risk."
    sections = extract_sections(text)
    
    assert len(sections) == 3
    assert sections[0][0] == "Other"
    assert "Some intro text." in sections[0][1]
    
    assert sections[1][0] == "MD&A"
    assert "This is MD&A." in sections[1][1]
    
    assert sections[2][0] == "Risk Factors"
    assert "This is risk." in sections[2][1]

def test_chunk_text_by_tokens():
    words = [f"word{i}" for i in range(2500)]
    text = " ".join(words)
    
    chunks = chunk_text_by_tokens(text, chunk_size=1000, overlap=200)
    
    assert len(chunks) == 3
    
    assert len(chunks[0].split()) == 1000
    assert chunks[0].split()[0] == "word0"
    assert chunks[0].split()[-1] == "word999"
    
    assert len(chunks[1].split()) == 1000
    assert chunks[1].split()[0] == "word800"
    assert chunks[1].split()[-1] == "word1799"
    
    assert len(chunks[2].split()) == 900
    assert chunks[2].split()[0] == "word1600"
    assert chunks[2].split()[-1] == "word2499"
