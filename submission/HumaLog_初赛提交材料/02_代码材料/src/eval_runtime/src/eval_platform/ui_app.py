"""Fresh UI-only process for this fork; GPU manager stays uninterrupted."""
from . import app as platform
from .ui_reports import report,csv_report
platform.report=report
platform.csv_report=csv_report
app=platform.app
