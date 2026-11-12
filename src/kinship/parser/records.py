"""Typed records produced by the parser. All are frozen; blanks are ``None``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class Connection:
    first_name: str | None
    last_name: str | None
    profile_url: str | None
    email: str | None
    company: str | None
    position: str | None
    connected_on: date | None


@dataclass(frozen=True, slots=True)
class Position:
    company_name: str | None
    title: str | None
    description: str | None
    location: str | None
    started_on: date | None
    finished_on: date | None


@dataclass(frozen=True, slots=True)
class Profile:
    """Only these three ``Profile.csv`` columns are kept; the rest is dropped."""

    headline: str | None
    summary: str | None
    industry: str | None


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    text: str | None


@dataclass(frozen=True, slots=True)
class Skill:
    name: str | None


@dataclass(frozen=True, slots=True)
class EndorsementReceived:
    endorsed_at: datetime | None
    skill_name: str | None
    endorser_first_name: str | None
    endorser_last_name: str | None
    endorser_profile_url: str | None
    status: str | None


@dataclass(frozen=True, slots=True)
class EndorsementGiven:
    endorsed_at: datetime | None
    skill_name: str | None
    endorsee_first_name: str | None
    endorsee_last_name: str | None
    endorsee_profile_url: str | None
    status: str | None


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Used for both received and given; the name is the other person's."""

    first_name: str | None
    last_name: str | None
    company: str | None
    job_title: str | None
    text: str | None
    created_at: datetime | None
    status: str | None


@dataclass(frozen=True, slots=True)
class Invitation:
    from_name: str | None
    to_name: str | None
    sent_at: datetime | None
    message: str | None
    direction: str | None
    inviter_profile_url: str | None
    invitee_profile_url: str | None


@dataclass(frozen=True, slots=True)
class Note:
    first_name: str | None
    last_name: str | None
    profile_url: str | None
    note: str | None
    created_at: datetime | None
    edited_at: datetime | None


@dataclass(frozen=True, slots=True)
class CompanyFollow:
    organization: str | None
    followed_at: datetime | None


@dataclass(frozen=True, slots=True)
class Education:
    school_name: str | None
    started_on: date | None
    ended_on: date | None
    notes: str | None
    degree_name: str | None
    activities: str | None


@dataclass(frozen=True, slots=True)
class JobApplication:
    applied_at: datetime | None
    contact_email: str | None
    contact_phone_number: str | None
    company_name: str | None
    job_title: str | None
    job_url: str | None
    resume_name: str | None
    question_and_answers: str | None


@dataclass(frozen=True, slots=True)
class SavedJob:
    saved_at: datetime | None
    job_url: str | None
    job_title: str | None
    company_name: str | None


@dataclass(frozen=True, slots=True)
class Message:
    conversation_id: str | None
    conversation_title: str | None
    from_name: str | None
    sender_profile_url: str | None
    to_names: str | None
    recipient_profile_urls: tuple[str, ...]
    sent_at: datetime | None
    subject: str | None
    content: str | None
    folder: str | None


@dataclass(frozen=True)
class ParsedExport:
    """Everything Kinship ingests from one export.

    ``file_counts`` maps each logical file (the list attribute names below) to
    the number of records parsed from it; absent files count 0.
    """

    source_sha256: str
    exported_at: datetime
    file_counts: dict[str, int]
    warnings: list[str]
    connections: list[Connection] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    profile: Profile | None = None
    profile_summary: list[ProfileSummary] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    endorsements_received: list[EndorsementReceived] = field(default_factory=list)
    endorsements_given: list[EndorsementGiven] = field(default_factory=list)
    recommendations_received: list[Recommendation] = field(default_factory=list)
    recommendations_given: list[Recommendation] = field(default_factory=list)
    invitations: list[Invitation] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    company_follows: list[CompanyFollow] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    job_applications: list[JobApplication] = field(default_factory=list)
    saved_jobs: list[SavedJob] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
