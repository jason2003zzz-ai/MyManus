from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TeamRole(str, Enum):
    GENERAL = "general"
    BROWSER = "browser"
    DATA = "data"


class TeamTask(BaseModel):
    id: str
    title: str
    role: TeamRole
    objective: str
    deliverable: str = ""
    depends_on: List[str] = Field(default_factory=list)


class TeamPlan(BaseModel):
    summary: str
    tasks: List[TeamTask]


class TeamTaskResult(BaseModel):
    task_id: str
    title: str
    role: TeamRole
    status: str
    answer: str = ""
    raw_result: str = ""
    error: Optional[str] = None
    started_at: str
    finished_at: str

    def handoff_text(self, limit: int = 5000) -> str:
        value = self.answer or self.raw_result or self.error or ""
        if len(value) <= limit:
            return value
        head = int(limit * 0.65)
        return value[:head] + "\n...[handoff truncated]...\n" + value[-(limit - head) :]
