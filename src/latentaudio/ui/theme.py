# theme.py - UI theme and styling constants
"""UI theme constants and styling for LatentAudio."""

from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt

# ============================================================================
# COLORS
# ============================================================================

# Background colors
BG_DARK = QColor(30, 30, 30)
BG_MEDIUM = QColor(45, 45, 45)
BG_LIGHT = QColor(60, 60, 60)

# Accent colors
ACCENT_PRIMARY = QColor(74, 158, 255)    # Blue
ACCENT_SECONDARY = QColor(138, 173, 244) # Light blue
ACCENT_SUCCESS = QColor(76, 175, 80)     # Green
ACCENT_WARNING = QColor(255, 152, 0)     # Orange
ACCENT_ERROR = QColor(244, 67, 54)       # Red

# Text colors
TEXT_PRIMARY = QColor(255, 255, 255)
TEXT_SECONDARY = QColor(158, 158, 158)
TEXT_MUTED = QColor(117, 117, 117)

# Grid and lines
GRID_COLOR = QColor(60, 60, 60)
CENTER_LINE_COLOR = QColor(80, 80, 80)
BORDER_COLOR = QColor(90, 90, 90)

# ============================================================================
# DIMENSIONS
# ============================================================================

# Widget sizes
WAVEFORM_HEIGHT_MIN = 150
WAVEFORM_HEIGHT_MAX = 200
SLIDER_HEIGHT = 400
TAB_HEIGHT = 500
BUTTON_HEIGHT = 30

# Spacing
MARGIN_SMALL = 5
MARGIN_MEDIUM = 10
MARGIN_LARGE = 20

# Font sizes
FONT_SIZE_SMALL = 8
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 12
FONT_SIZE_TITLE = 16

# ============================================================================
# FONTS
# ============================================================================

def get_font(size: int = FONT_SIZE_NORMAL, bold: bool = False) -> QFont:
    """Get application font with specified size and weight."""
    font = QFont("Segoe UI", size)
    if bold:
        font.setBold(True)
    return font

TITLE_FONT = get_font(FONT_SIZE_TITLE, bold=True)
NORMAL_FONT = get_font(FONT_SIZE_NORMAL)
SMALL_FONT = get_font(FONT_SIZE_SMALL)

# ============================================================================
# STYLESHEETS
# ============================================================================

BUTTON_STYLE = f"""
QPushButton {{
    background-color: {BG_MEDIUM.name()};
    color: {TEXT_PRIMARY.name()};
    border: 1px solid {BORDER_COLOR.name()};
    border-radius: 4px;
    padding: 8px 16px;
    font-size: {FONT_SIZE_NORMAL}pt;
}}

QPushButton:hover {{
    background-color: {BG_LIGHT.name()};
    border-color: {ACCENT_PRIMARY.name()};
}}

QPushButton:pressed {{
    background-color: {ACCENT_PRIMARY.name()};
}}

QPushButton:disabled {{
    color: {TEXT_MUTED.name()};
    background-color: {BG_DARK.name()};
}}
"""

SLIDER_STYLE = f"""
QSlider::groove:horizontal {{
    background: {BG_MEDIUM.name()};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT_PRIMARY.name()};
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -6px 0;
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT_SECONDARY.name()};
}}
"""

GROUP_BOX_STYLE = f"""
QGroupBox {{
    font-weight: bold;
    border: 2px solid {BORDER_COLOR.name()};
    border-radius: 5px;
    margin-top: 1ex;
    color: {TEXT_PRIMARY.name()};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: {ACCENT_PRIMARY.name()};
}}
"""

TAB_WIDGET_STYLE = f"""
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR.name()};
    background: {BG_DARK.name()};
}}

QTabBar::tab {{
    background: {BG_MEDIUM.name()};
    color: {TEXT_SECONDARY.name()};
    padding: 8px 16px;
    border: 1px solid {BORDER_COLOR.name()};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background: {BG_LIGHT.name()};
    color: {TEXT_PRIMARY.name()};
    border-bottom: 2px solid {ACCENT_PRIMARY.name()};
}}

QTabBar::tab:hover {{
    color: {TEXT_PRIMARY.name()};
}}
"""

LIST_WIDGET_STYLE = f"""
QListWidget {{
    background-color: {BG_MEDIUM.name()};
    color: {TEXT_PRIMARY.name()};
    border: 1px solid {BORDER_COLOR.name()};
    border-radius: 4px;
    selection-background-color: {ACCENT_PRIMARY.name()};
}}

QListWidget::item:hover {{
    background-color: {BG_LIGHT.name()};
}}
"""

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

WINDOW_TITLE = "LatentAudio: Neural Sound Exploration"
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800

DEFAULT_SPLITTER_SIZES = [600, 600]  # Left/right panel split