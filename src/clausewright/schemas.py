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
