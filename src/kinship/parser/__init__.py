"""CSV parsing layer for LinkedIn data exports.

Independent of the database: ``parse_export`` turns a zip or an extracted
directory into typed, frozen records. Only the allowlisted files in
``docs/linkedin-export-format.md`` are ever opened.
"""

from kinship.parser.dates import parse_date, parse_datetime
from kinship.parser.export import ExportError, parse_export
from kinship.parser.records import (
    CompanyFollow,
    Connection,
    Education,
    EndorsementGiven,
    EndorsementReceived,
    Invitation,
    JobApplication,
    Message,
    Note,
    ParsedExport,
    Position,
    Profile,
    ProfileSummary,
    Recommendation,
    SavedJob,
    Skill,
)
from kinship.parser.urls import canonical_profile_url

__all__ = [
    "CompanyFollow",
    "Connection",
    "Education",
    "EndorsementGiven",
    "EndorsementReceived",
    "ExportError",
    "Invitation",
    "JobApplication",
    "Message",
    "Note",
    "ParsedExport",
    "Position",
    "Profile",
    "ProfileSummary",
    "Recommendation",
    "SavedJob",
    "Skill",
    "canonical_profile_url",
    "parse_date",
    "parse_datetime",
    "parse_export",
]
