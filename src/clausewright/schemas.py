"""Domain models for contract document ingestion and layout structure."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Block(BaseModel):
    """A contiguous block of text on a page with layout metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float]
    font_size: float = Field(..., gt=0.0)
    is_bold: bool = False

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = v
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Inverted or degenerate bounding box: ({x0}, {y0}, {x1}, {y1})")
        return v


class Page(BaseModel):
    """A single page of a contract containing layout blocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int = Field(..., ge=1)
    width: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)
    blocks: list[Block] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_block_pages(self) -> Page:
        for block in self.blocks:
            if block.page_number != self.number:
                raise ValueError(
                    f"Block with page_number={block.page_number} assigned to Page {self.number}"
                )
        return self


class ParsedDocument(BaseModel):
    """The complete parsed document consisting of ordered pages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    pages: list[Page] = Field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @model_validator(mode="after")
    def validate_page_numbers(self) -> ParsedDocument:
        if not self.pages:
            return self
        expected_numbers = list(range(1, len(self.pages) + 1))
        actual_numbers = [p.number for p in self.pages]
        if actual_numbers != expected_numbers:
            raise ValueError(
                f"Page numbers must be sequential 1..{len(self.pages)}, got {actual_numbers}"
            )
        return self


class Section(BaseModel):
    """A semantic section or subsection in a contract document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section_number: str = Field(..., min_length=1)
    title: str | None = None
    level: int = Field(default=1, ge=1)
    parent_number: str | None = None
    blocks: list[Block] = Field(default_factory=list)
    page_range: tuple[int, int] = Field(..., description="(start_page, end_page) inclusive")

    @field_validator("page_range")
    @classmethod
    def validate_page_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        start, end = v
        if start < 1 or end < 1:
            raise ValueError(f"Page numbers must be >= 1, got {v}")
        if start > end:
            raise ValueError(f"start_page ({start}) cannot exceed end_page ({end})")
        return v

    @property
    def text(self) -> str:
        """Contiguous text content of all blocks in the section."""
        return "\n\n".join(b.text for b in self.blocks)


class Chunk(BaseModel):
    """A fixed-size text window produced by the chunker for embedding / retrieval.

    Each ``Chunk`` is the atomic unit fed into the vector store.  It carries
    full provenance so retrieved chunks can be traced back to the source
    section and page.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Unique identifier: '<doc_id>-<section_number>-<index>'")
    doc_id: str = Field(..., min_length=1)
    section_number: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0, description="0-based position within the parent section")
    text: str = Field(..., min_length=1)
    page_range: tuple[int, int] = Field(..., description="(start_page, end_page) inclusive")
    word_count: int = Field(..., ge=1)
    breadcrumbs: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Hierarchical lineage path: e.g. ('Article 12: Indemnification', 'Section 12.3: Carve-Outs')",
    )
    contextual_text: str | None = Field(
        default=None,
        description="Chunk text prepended with hierarchical lineage for context-aware retrieval",
    )

    @field_validator("page_range")
    @classmethod
    def validate_page_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        start, end = v
        if start < 1 or end < 1:
            raise ValueError(f"Page numbers must be >= 1, got {v}")
        if start > end:
            raise ValueError(f"start_page ({start}) cannot exceed end_page ({end})")
        return v

