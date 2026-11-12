"""Built-in word lists for the synthetic export generator.

Everything here is invented. Names are whimsical on purpose, companies do not
exist, and all contact details use reserved example domains.
"""

from __future__ import annotations

from dataclasses import dataclass

FIRST_NAMES: tuple[str, ...] = (
    "Avery",
    "Blake",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Greer",
    "Harper",
    "Indira",
    "Jules",
    "Kai",
    "Lennox",
    "Marlow",
    "Noor",
    "Oakley",
    "Paz",
    "Quinn",
    "Reese",
    "Sage",
    "Tatum",
    "Uma",
    "Vesper",
    "Wren",
    "Xiomara",
    "Yael",
    "Zuri",
    "Anouk",
    "Bodhi",
    "Cleo",
    "Dashiell",
    "Elio",
    "Freya",
    "Gideon",
    "Hattie",
    "Ignatius",
    "Juno",
    "Kestrel",
    "Linus",
    "Mireille",
    "Nico",
    "Odessa",
    "Pippa",
    "Rafferty",
    "Soren",
    "Thea",
    "Ulric",
    "Viola",
    "Zoë",
)

# A few names need CSV quoting or URL slugging care (apostrophe, accent, comma).
LAST_NAMES: tuple[str, ...] = (
    "Ashgrove",
    "Bramblewick",
    "Copperpot",
    "Dunmere",
    "Eldridge-Vane",
    "Fennimore",
    "Glasswing",
    "Hollowell",
    "Inkwell",
    "Juniper",
    "Kettleby",
    "Larkspur",
    "Marrowby",
    "Nettlefold",
    "O'Thistle",
    "Pennywhistle",
    "Quince",
    "Rookwood",
    "Saltmere",
    "Thornbury",
    "Umberfield",
    "Vellacott",
    "Wexcombe",
    "Yarrowby",
    "Zennor",
    "Núñez-Lark",
    "Applegarth",
    "Birchall",
    "Cindermoor",
    "Dovecote",
    "Emberly",
    "Foxglove",
    "Greywether",
    "Harrowgate",
    "Ivywood",
    "Jesper",
    "Kingsmoor",
    "Lindenmoor",
    "Mossbank",
    "Northcote",
    "Oakhollow",
    "Pendlewick",
    "Quarrymoor",
    "Ravenscar",
    "Stonebrook, PhD",
    "Tidewell",
    "Underhill",
    "Winterbourne",
)

# Canonical company key -> the raw spellings that appear in the export. The
# first three groups deliberately have several spellings of one company.
COMPANY_GROUPS: dict[str, tuple[str, ...]] = {
    "acme": ("Acme", "Acme, Inc.", "ACME Corp"),
    "gifthealth": ("Gifthealth", "Gift Health, Inc.", "gift health"),
    "quillfeather": ("Quillfeather Systems", "Quillfeather Systems, LLC"),
    "brindlewood": ("Brindlewood Analytics",),
    "tessellate": ("Tessellate Health",),
    "marrowbone": ("Marrowbone Robotics",),
    "copperline": ("Copperline Freight",),
    "saltmarsh": ("Saltmarsh Bank",),
    "juniperwren": ("Juniper & Wren Studio",),
    "halcyon": ("Halcyon Grid",),
    "pemberwick": ("Pemberwick Foods",),
    "lanternfish": ("Lanternfish Games",),
    "oddfellow": ("Oddfellow Security",),
    "thistledown": ("Thistledown Mobile",),
    "vantablue": ("Vantablue Energy",),
    "norwhal": ("Norwhal Labs",),
}

# Companies that never employ a connection (for job applications with no path).
COMPANIES_WITHOUT_CONNECTIONS: tuple[str, ...] = (
    "Zephyrine Aerospace",
    "Mossgiel Publishing",
    "Kelpforest Biotech",
)

SCHOOLS: tuple[str, ...] = ("Example State University", "Faketown Institute of Technology")

LOCATIONS: tuple[str, ...] = (
    "Faketown, Example State",
    "Sampleville, Example State",
    "Remote",
    "Placeholder City, Testland",
)

EVENTS: tuple[str, ...] = (
    "the Faketown data meetup",
    "SampleConf",
    "a Placeholder City hack night",
    "the Testland product summit",
    "an alumni mixer",
    "a workshop on incident reviews",
)

OWNER_SKILLS: tuple[str, ...] = (
    "Python",
    "SQL",
    "Data Pipelines",
    "Apache Airflow",
    "PostgreSQL",
    "Data Modeling",
    "Healthcare Billing",
    "API Design",
    "Mentoring",
    "Incident Response",
    "dbt",
    "Stakeholder Communication",
)


@dataclass(frozen=True)
class Domain:
    """A professional topic. Gives each person distinct, searchable text."""

    key: str
    topic: str
    titles: tuple[str, ...]
    skills: tuple[str, ...]
    praise: str  # recommendation text; ``{name}`` is the person recommended
    ask: str  # first message of a conversation, from the connection
    reply: str  # the owner's answer


DOMAINS: tuple[Domain, ...] = (
    Domain(
        "data-pipelines",
        "data pipelines",
        ("Data Engineer", "Analytics Engineer", "Data Platform Engineer"),
        ("Apache Airflow", "ETL", "Data Warehousing"),
        "{name} rebuilt our nightly data pipelines in Airflow and cut the warehouse load "
        "from six hours to forty minutes.\n\nEvery batch job got alerting and a runbook. "
        "I would work with {name} again without hesitation.",
        "our Airflow backfills keep timing out. How did you structure the batch pipelines?",
        "We split each backfill into daily partitions and made every task idempotent.",
    ),
    Domain(
        "healthcare-billing",
        "healthcare billing",
        ("Billing Systems Analyst", "Revenue Cycle Engineer", "Claims Operations Lead"),
        ("Healthcare Billing", "Claims Adjudication", "HIPAA Compliance"),
        "{name} untangled our pharmacy claims backlog. Rejected insurance claims dropped by "
        "a third once the healthcare billing rules were encoded as tests, and the audit "
        "trail satisfied our HIPAA reviewers.",
        "we are seeing a spike in rejected pharmacy claims. Have you dealt with payer "
        "code changes before?",
        "Yes, keep the payer rules in a versioned table and replay rejected claims nightly.",
    ),
    Domain(
        "mobile-apps",
        "mobile apps",
        ("Mobile Engineer", "iOS Developer", "Android Developer"),
        ("Swift", "Kotlin", "Mobile App Development"),
        "{name} shipped our iOS and Android apps on one shared API and kept the crash-free "
        "rate above 99.8%.\n\nThe offline sync design alone saved the mobile team a quarter "
        "of rework.",
        "we are adding offline mode to the mobile app. Any advice on sync conflicts?",
        "Queue writes locally and let the server resolve conflicts by version number.",
    ),
    Domain(
        "security-audits",
        "security audits",
        ("Security Engineer", "Application Security Analyst", "Compliance Auditor"),
        ("Penetration Testing", "SOC 2", "Threat Modeling"),
        "{name} led our SOC 2 security audit from scoping to the final report. The "
        "penetration test findings were triaged within a week and the access-review "
        "process now runs itself.",
        "our SOC 2 auditors want evidence of access reviews. What did you hand over?",
        "A quarterly export of role grants with sign-off from each team lead was enough.",
    ),
    Domain(
        "machine-learning",
        "machine learning models",
        ("Machine Learning Engineer", "Data Scientist", "Applied Scientist"),
        ("Machine Learning", "Model Evaluation", "Feature Engineering"),
        "{name} built the churn prediction model that our retention team still uses. "
        "Careful feature engineering and honest offline evaluation meant the model held "
        "up when it met production traffic.",
        "our churn model looks great offline but poor in production. Where would you look?",
        "Check for label leakage first, then compare feature distributions week by week.",
    ),
    Domain(
        "payments",
        "payment systems",
        ("Payments Engineer", "Fintech Product Engineer", "Billing Platform Engineer"),
        ("Payment Processing", "PCI DSS", "Ledger Design"),
        "{name} designed the double-entry ledger behind our card payments and made "
        "reconciliation with the processor a daily non-event. Refund and chargeback "
        "handling finally matched what finance expected.",
        "we need idempotent retries for card payments. How did you key the requests?",
        "One idempotency key per checkout attempt, stored alongside the ledger entry.",
    ),
    Domain(
        "devops",
        "Kubernetes infrastructure",
        ("Site Reliability Engineer", "DevOps Engineer", "Platform Engineer"),
        ("Kubernetes", "Terraform", "Observability"),
        "{name} migrated forty services to Kubernetes with Terraform and no customer-facing "
        "downtime.\n\nOn-call pages fell by half after the new dashboards and deploy "
        "pipeline landed.",
        "we are planning a Kubernetes migration. Did you move the stateful services too?",
        "We moved stateless services first and kept the databases on managed instances.",
    ),
    Domain(
        "ux-research",
        "user research",
        ("UX Researcher", "Product Designer", "Design Researcher"),
        ("User Interviews", "Usability Testing", "Journey Mapping"),
        "{name} ran the user interviews and usability tests that reshaped our onboarding "
        "flow. The research readouts were short, specific and impossible to ignore.",
        "we are recruiting participants for usability tests. How many sessions were enough?",
        "Five sessions per segment surfaced nearly every onboarding problem we later fixed.",
    ),
    Domain(
        "product-management",
        "product roadmaps",
        ("Product Manager", "Group Product Manager", "Technical Product Manager"),
        ("Product Roadmapping", "Prioritization", "A/B Testing"),
        "{name} turned a vague growth goal into a product roadmap with clear bets, and "
        "then killed two of them when the A/B tests said so. Engineering always knew why "
        "a feature mattered.",
        "I am rewriting our product roadmap for next year. How do you say no to pet features?",
        "Tie every roadmap item to one metric and review the misses in public.",
    ),
    Domain(
        "partnerships",
        "enterprise sales and partnerships",
        ("Account Executive", "Partnerships Manager", "Solutions Consultant"),
        ("Enterprise Sales", "Contract Negotiation", "Channel Partnerships"),
        "{name} closed our first three enterprise sales contracts and built the reseller "
        "partnership programme from nothing. Negotiations stayed friendly even when the "
        "procurement teams did not.",
        "an enterprise prospect wants a pilot before signing. How did you scope yours?",
        "Thirty days, one team, and success criteria agreed in writing before kickoff.",
    ),
    Domain(
        "technical-writing",
        "developer documentation",
        ("Technical Writer", "Documentation Lead", "Developer Advocate"),
        ("Technical Writing", "API Documentation", "Developer Relations"),
        "{name} rewrote our API documentation and quickstart guides. Support tickets about "
        "authentication fell sharply, and new developers now reach a first successful "
        "request in minutes.",
        "we are restructuring the API documentation. Did you organise by task or by endpoint?",
        "By task for the guides, with an endpoint reference generated from the schema.",
    ),
    Domain(
        "recruiting",
        "technical recruiting",
        ("Technical Recruiter", "Talent Partner", "Head of People"),
        ("Technical Recruiting", "Interview Design", "Employer Branding"),
        "{name} rebuilt our engineering hiring loop with structured interviews and a fair "
        "take-home exercise.\n\nTime to hire halved and candidates told us the recruiting "
        "process was the reason they accepted.",
        "we are hiring two senior engineers. Would structured interviews slow us down?",
        "They sped us up, because every interviewer knew exactly what to assess.",
    ),
    Domain(
        "embedded",
        "embedded firmware",
        ("Firmware Engineer", "Embedded Systems Engineer", "Hardware Test Engineer"),
        ("Embedded C", "RTOS", "Hardware Debugging"),
        "{name} wrote the firmware for our sensor boards and tracked down a power "
        "regression with an oscilloscope and a great deal of patience. Battery life "
        "doubled after the RTOS scheduling fix.",
        "our sensor firmware drains the battery overnight. How did you profile sleep states?",
        "Log every wake source and measure current draw on the bench for each one.",
    ),
    Domain(
        "accessibility",
        "web accessibility",
        ("Frontend Engineer", "Accessibility Specialist", "UI Engineer"),
        ("Web Accessibility", "React", "Design Systems"),
        "{name} made our React design system meet WCAG AA and taught the whole frontend "
        "team to test with a screen reader. Accessibility bugs stopped reaching customers.",
        "we failed an accessibility review on our forms. Where should we start?",
        "Start with labels and focus order, then automate the checks in the build.",
    ),
)
