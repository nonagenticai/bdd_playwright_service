"""Named constants for the BDD Playwright API."""

# Health status strings
HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"

# Job statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Stream types
STREAM_QA_SUITE = "qa_suite"
STREAM_ACCESSIBILITY = "accessibility"
STREAM_DOCUMENTATION = "documentation"
STREAM_DEMO_VIDEO = "demo_video"

# Input types
INPUT_WEB_URL = "web_url"
INPUT_MOBILE_APP = "mobile_app"
INPUT_GHERKIN_UPLOAD = "gherkin_upload"
INPUT_INSPECTION_PASSTHROUGH = "inspection_passthrough"

# Narration tones
TONE_TUTORIAL = "tutorial"
TONE_DEMO = "demo"
TONE_MARKETING = "marketing"
TONE_TECHNICAL = "technical"

# Error message truncation
MAX_ERROR_LENGTH = 4000

# Cleanup service defaults
STUCK_JOB_TIMEOUT_MINUTES = 60
# ⚠️ FAR longer than STUCK_JOB_TIMEOUT_MINUTES, and NOT a copy of it. `pending` is a
# legitimate queued state; a 60-minute clock here would fail out live queued work.
# This is the backstop for a row whose dispatched task died before mark_processing.
PENDING_JOB_TIMEOUT_MINUTES = 360
RETENTION_DAYS = 30
