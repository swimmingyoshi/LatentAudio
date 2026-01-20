# sound_map.py - 2D Interactive Sound Map Widget
"""Interactive 2D visualization of latent space."""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient, QTransform, QWheelEvent, QMouseEvent
from typing import List, Optional, Tuple

class SoundMapWidget(QWidget):
    """
    Interactive 2D map for navigating latent space.
    
    Displays projected samples as dots and allows clicking anywhere
    to generate audio from that point. Supports zoom and pan.
    """
    
    point_clicked = pyqtSignal(np.ndarray)  # Emits 2D projected coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = np.array([])  # Projected points (N, 2)
        self.labels = []  # Optional labels for points
        self.current_pos = QPointF(0.5, 0.5)  # Normalized 0-1 coordinates
        
        # View state
        self.scale = 1.0
        self.pan_offset = QPointF(0, 0)
        self.last_mouse_pos = QPointF()
        
        self.setMinimumSize(400, 400)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        
        # Style
        self.bg_color = QColor(20, 20, 25)
        self.point_color = QColor(0, 255, 255, 150)
        self.active_color = QColor(255, 0, 255)
        
        self.margin = 40

    def reset_view(self):
        """Reset zoom and pan."""
        self.scale = 1.0
        self.pan_offset = QPointF(0, 0)
        self.update()

    def set_points(self, points: np.ndarray, labels: Optional[List[str]] = None):
        """Set the 2D projected points to display."""
        if points.size == 0:
            return
            
        # Normalize points to 0-1 range for internal storage
        p_min = points.min(axis=0)
        p_max = points.max(axis=0)
        p_range = p_max - p_min
        p_range[p_range == 0] = 1.0
        
        self.points = (points - p_min) / p_range
        self.labels = labels or []
        self.reset_view()

    def get_transform(self):
        """Calculate the transformation matrix for the current view."""
        transform = QTransform()
        
        # 1. Move to center of widget to apply scale
        w_mid = self.width() / 2
        h_mid = self.height() / 2
        
        transform.translate(w_mid + self.pan_offset.x(), h_mid + self.pan_offset.y())
        transform.scale(self.scale, self.scale)
        # 2. Move back to represent normalized space
        transform.translate(-w_mid, -h_mid)
        
        return transform

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), self.bg_color)
        
        if self.points.size == 0:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                             "Map is currently empty.\n\n"
                             "Click 'Data' to project your dataset,\n"
                             "'Model' to restore from cache,\n"
                             "or 'Synth' to hallucinate random points.")
            return

        # Apply view transformation
        transform = self.get_transform()
        painter.setTransform(transform)

        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        
        # Draw points
        painter.setPen(Qt.PenStyle.NoPen)
        for p in self.points:
            px = self.margin + p[0] * w
            py = self.margin + p[1] * h
            
            # Subtle glow - keeps same size regardless of zoom
            # (Math here adjusts for current scale to keep visual size constant)
            glow_size = 4 / self.scale
            painter.setBrush(self.point_color)
            painter.drawEllipse(QPointF(px, py), 1.5 / self.scale, 1.5 / self.scale)

        # Draw crosshair / current position
        cx = self.margin + self.current_pos.x() * w
        cy = self.margin + self.current_pos.y() * h
        
        cross_size = 10 / self.scale
        pen = QPen(self.active_color, 2 / self.scale)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx - cross_size, cy), QPointF(cx + cross_size, cy))
        painter.drawLine(QPointF(cx, cy - cross_size), QPointF(cx, cy + cross_size))

    def wheelEvent(self, a0: Optional[QWheelEvent]):
        """Handle zoom with focus on the selection crosshair."""
        if a0 is None: return
        angle = a0.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        
        new_scale = self.scale * factor
        # Constrain zoom (Increased to 1000x for ultra-deep exploration)
        if 0.1 <= new_scale <= 1000.0:
            w_mid = self.width() / 2.0
            h_mid = self.height() / 2.0
            
            # Find map-space coordinates of current selection
            w = self.width() - 2 * self.margin
            h = self.height() - 2 * self.margin
            cross_x = self.margin + self.current_pos.x() * w
            cross_y = self.margin + self.current_pos.y() * h
            
            # Find screen position of the crosshair
            pivot = self.get_transform().map(QPointF(cross_x, cross_y))
            
            # Update pan to keep crosshair fixed
            self.pan_offset = factor * self.pan_offset + (1.0 - factor) * (pivot - QPointF(w_mid, h_mid))
            self.scale = new_scale
            self.update()

    def mousePressEvent(self, a0: Optional[QMouseEvent]):
        if a0 is None: return
        self.last_mouse_pos = a0.position()
        if a0.button() == Qt.MouseButton.LeftButton:
            self._handle_click(a0.position())

    def mouseMoveEvent(self, a0: Optional[QMouseEvent]):
        if a0 is None: return
        curr_pos = a0.position()
        
        if a0.buttons() & Qt.MouseButton.RightButton:
            delta = curr_pos - self.last_mouse_pos
            self.pan_offset += delta
            self.update()
        elif a0.buttons() & Qt.MouseButton.LeftButton:
            self._handle_click(curr_pos)
            
        self.last_mouse_pos = curr_pos

    def mouseDoubleClickEvent(self, a0: Optional[QMouseEvent]):
        self.reset_view()

    def _handle_click(self, pos):
        """Map screen click coordinates back to normalized space."""
        # Use inverse transform to find where we clicked in the "flat" map
        transform = self.get_transform()
        inv_trans, ok = transform.inverted()
        
        if not ok: return
        
        map_pos = inv_trans.map(pos)
        
        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        
        nx = max(0.0, min(1.0, (map_pos.x() - self.margin) / w))
        ny = max(0.0, min(1.0, (map_pos.y() - self.margin) / h))
        
        self.current_pos = QPointF(nx, ny)
        self.update()
        
        # Emit normalized coordinates
        self.point_clicked.emit(np.array([nx, ny]))

    def set_active_pos(self, normalized_pos: np.ndarray):
        """Set position from outside (e.g. when sliders change)."""
        self.current_pos = QPointF(normalized_pos[0], normalized_pos[1])
        self.update()
