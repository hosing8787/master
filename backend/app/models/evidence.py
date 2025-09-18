from dataclasses import dataclass
from typing import Optional

@dataclass
class EvidenceItem:
    text: str
    source: str
    title: Optional[str] = None
    fetched_at: Optional[str] = None
    score: Optional[float] = None
