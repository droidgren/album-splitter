"""Debug rendering: photo quads, text boxes and links drawn over a downscaled page."""

from __future__ import annotations

import cv2
import numpy as np

from .imageio import downscale
from .model import DateStamp, Page

# BGR, chosen to stay distinguishable on photos and paper
PALETTE = [
    (40, 40, 230),    # red
    (230, 120, 20),   # blue
    (30, 180, 30),    # green
    (0, 160, 255),    # orange
    (200, 40, 200),   # magenta
    (200, 200, 0),    # teal
    (20, 100, 160),   # brown
    (120, 0, 120),    # purple
]
GREY = (150, 150, 150)


def photo_color(photo_id: int) -> tuple[int, int, int]:
    return PALETTE[(photo_id - 1) % len(PALETTE)]


def render_debug(page: Page, img_bgr: np.ndarray, long_side: int = 1600) -> np.ndarray:
    small, s = downscale(img_bgr, long_side)
    canvas = small.copy()
    # Dim the image so annotations stand out.
    canvas = cv2.addWeighted(canvas, 0.75, np.full_like(canvas, 255), 0.25, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(2, round(long_side / 500))

    def pt(p) -> tuple[int, int]:
        return round(p[0] * s), round(p[1] * s)

    if page.page_quad:
        cv2.polylines(canvas, [np.array([pt(p) for p in page.page_quad], np.int32)], True,
                      (255, 255, 0), 1, cv2.LINE_AA)

    linked: dict[int, list[int]] = {}
    for lk in page.links:
        linked.setdefault(lk.text_id, []).append(lk.photo_id)
    photos = {p.id: p for p in page.photos}

    for t in page.texts:
        pids = linked.get(t.id, [])
        color = photo_color(pids[0]) if pids else GREY
        x0, y0, x1, y1 = t.bbox()
        if t.outline:  # box runs onto a photo: draw the paper part only
            cv2.polylines(canvas, [np.array([pt(p) for p in t.outline], np.int32)], True, color, th,
                          cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, pt((x0, y0)), pt((x1, y1)), color, th, cv2.LINE_AA)
        for pid in pids:
            cv2.line(canvas, pt(t.center()), pt(photos[pid].center()), photo_color(pid), max(1, th - 1),
                     cv2.LINE_AA)
        cv2.putText(canvas, f"T{t.id}", (pt((x0, y0))[0], pt((x0, y0))[1] - 4), font, 0.5, color, 1,
                    cv2.LINE_AA)

    for ph in page.photos:
        color = photo_color(ph.id)
        # Draw the visible outline when another print covers part of this one.
        shape = np.array([pt(p) for p in (ph.outline or ph.corners)], np.int32)
        cv2.polylines(canvas, [shape], True, color, th + 1, cv2.LINE_AA)
        m = cv2.moments(shape)
        c = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) if m["m00"] else pt(ph.center())
        label = f"{ph.id}{'?' if ph.needs_review else ''}"
        cv2.circle(canvas, c, 22, color, cv2.FILLED, cv2.LINE_AA)
        (tw, tht), _ = cv2.getTextSize(label, font, 0.8, 2)
        cv2.putText(canvas, label, (c[0] - tw // 2, c[1] + tht // 2), font, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    for ph in page.photos:
        if ph.stamp:
            _draw_stamp(canvas, ph.stamp, photo_color(ph.id), pt, font)

    summary = f"{page.page_id}: {len(page.photos)} photos, {len(page.texts)} texts, {len(page.links)} links"
    n_stamps = sum(1 for ph in page.photos if ph.stamp)
    if n_stamps:
        summary += f", {n_stamps} date stamps"
    if page.linker.startswith("llm"):
        summary += " (llm)"
    if page.issues:
        # Kinds and photo ids only: notes may hold å/ä/ö, which Hershey fonts cannot draw.
        line = "issues: " + ", ".join(
            issue.kind + (f" (P{', P'.join(map(str, issue.photo_ids))})" if issue.photo_ids else "")
            for issue in page.issues)
        (tw, tht), base = cv2.getTextSize(line, font, 0.6, 2)
        cv2.rectangle(canvas, (4, 4), (tw + 16, tht + base + 14), (40, 40, 200), cv2.FILLED)
        cv2.putText(canvas, line, (10, tht + 9), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, summary, (10, canvas.shape[0] - 12), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, summary, (10, canvas.shape[0] - 12), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _draw_stamp(canvas: np.ndarray, st: DateStamp, color: tuple[int, int, int], pt, font: int) -> None:
    """Thin quad around the date stamp plus its reading: "6 5 '95 = 1995-05-06" (with the
    nearly as likely alternatives, if any), or "... = ?" when no valid date was read."""
    quad = np.array([pt(p) for p in st.corners], np.int32)
    cv2.polylines(canvas, [quad], True, color, 1, cv2.LINE_AA)
    label = f"{st.text} = {st.date or '?'}"
    if st.date_alts:
        label += f" (or {st.date_alts[0]})"
    (tw, th), base = cv2.getTextSize(label, font, 0.45, 1)
    x = int(np.clip(quad[:, 0].max() - tw, 2, canvas.shape[1] - tw - 2))
    y = int(quad[:, 1].min()) - 4
    if y - th < 2:
        y = int(quad[:, 1].max()) + th + 4
    cv2.rectangle(canvas, (x - 2, y - th - 2), (x + tw + 2, y + base), color, cv2.FILLED)
    cv2.putText(canvas, label, (x, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
