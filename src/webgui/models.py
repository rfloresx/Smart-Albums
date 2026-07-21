"""Pydantic models for the web GUI."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    """Status of a pipeline job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobRecord(BaseModel):
    """A pipeline job record."""

    id: str
    pipeline: str
    pipeline_settings: dict[str, Any]
    status: JobStatus = JobStatus.pending
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    log_output: str = ""
    progress: Optional[dict[str, Any]] = None


class ConnectionStatus(BaseModel):
    """Status of a backend connection."""

    connected: bool
    message: str
    url: str
    provider: str = ""


class ScheduleRecord(BaseModel):
    """A cron schedule record for recurring pipeline runs."""

    id: str
    name: str
    pipeline: str
    pipeline_settings: dict[str, Any] = Field(default_factory=dict)
    cron_expression: str
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    last_job_id: Optional[str] = None
