# LinkedIn export format

Header rows taken from a real LinkedIn data export (November 2026), with no data
rows. The synthetic generator (#2) and the parser (#3) must match these exactly.
`<memberid>` stands for a numeric member id that LinkedIn appends to some file
names.

## Export variants

LinkedIn sends two archives:

- **Basic** (fast): includes `Jobs/` and `Verifications/` subfolders.
- **Complete** (slower): adds activity files (`Comments_<memberid>.csv`,
  `Reactions_<memberid>.csv`, `Shares_<memberid>.csv`, `Logins.csv`, and others).

The parser must accept either, find files case-insensitively in subfolders, and
concatenate split files such as `Saved Jobs.csv` + `Saved Jobs_1.csv`.

## Files Kinship ingests (allowlist)

The parser reads **only** these files. Everything else in the archive is ignored
and never stored.

| File | Header |
|---|---|
| `Connections.csv` | `First Name,Last Name,URL,Email Address,Company,Position,Connected On` |
| `Positions.csv` | `Company Name,Title,Description,Location,Started On,Finished On` |
| `Profile.csv` (only `Headline`, `Summary`, `Industry`) | `First Name,Last Name,Maiden Name,Address,Birth Date,Headline,Summary,Industry,Zip Code,Geo Location,Twitter Handles,Websites,Instant Messengers` |
| `Profile Summary.csv` | `Profile Summary` |
| `Skills.csv` | `Name` |
| `Endorsement_Received_Info.csv` | `Endorsement Date,Skill Name,Endorser First Name,Endorser Last Name,Endorser Public Url,Endorsement Status` |
| `Endorsement_Given_Info.csv` | `Endorsement Date,Skill Name,Endorsee First Name,Endorsee Last Name,Endorsee Public Url,Endorsement Status` |
| `Recommendations_Received.csv` | `First Name,Last Name,Company,Job Title,Text,Creation Date,Status` |
| `Recommendations_Given.csv` | `First Name,Last Name,Company,Job Title,Text,Creation Date,Status` |
| `Invitations.csv` | `From,To,Sent At,Message,Direction,inviterProfileUrl,inviteeProfileUrl` |
| `Notes.csv` | `Connection First Name,Connection Last Name,Connection Profile URL,Note,Created On,Edited On` |
| `Company Follows.csv` | `Organization,Followed On` |
| `Education.csv` | `School Name,Start Date,End Date,Notes,Degree Name,Activities` |
| `Jobs/Job Applications.csv` (basic only) | `Application Date,Contact Email,Contact Phone Number,Company Name,Job Title,Job Url,Resume Name,Question And Answers` |
| `Jobs/Saved Jobs*.csv` (basic only) | `Saved Date,Job Url,Job Title,Company Name` |

`messages.csv` was not in this export sample. Treat it as optional until a real
archive confirms its header.

## Quirks to handle

- The connection's profile URL column is named `URL` in `Connections.csv`, but
  `Endorser Public Url`, `Connection Profile URL`, `inviterProfileUrl` and
  `inviteeProfileUrl` elsewhere. Normalize all of them to one canonical profile URL
  so endorsements, notes and invitations join to connections.
- Older exports put `Notes:` preamble lines before the header in
  `Connections.csv`. This sample has none, but the parser still skips them.
- A file can hold a placeholder sentence instead of CSV (for example
  `No conversations found`). Treat a file whose first line is not the expected
  header as empty, with a warning.
- Some headers end with a trailing comma (an empty column name), and some repeat
  a column name. Parse by position where names repeat; ignore empty names.
- Files may be split with a numeric suffix (`Saved Jobs_1.csv`).
- Member-id suffixes on file names (`_<memberid>`) vary by person; match by
  prefix.

## Never ingested (privacy)

`Ad_Targeting`, `Ads Clicked`, `Email Addresses`, `Inferences_about_you`,
`LAN Ads Engagement`, `Logins`, `PhoneNumbers`, `Private_identity_asset`,
`Receipts_v2`, `Registration`, `Security Challenges`, `SearchQueries`,
`Verifications/*`, `Job Seeker Preferences`, `Job Applicant Saved*`, and the
activity files (`Comments_*`, `Reactions_*`, `Shares_*`, `Votes_*`,
`InstantReposts_*`, `Member_Follows_*`, `Saved_Items_*`, `Learning*`, `Events`,
`Rich_Media`, `Certifications`, `Languages`, `SavedJobAlerts`). Profile fields
other than `Headline`, `Summary` and `Industry` are dropped too.
