"""Heuristic text-block -> photo linking."""

from __future__ import annotations

from .model import Link, Photo, TextBlock

ABOVE_PENALTY = 2.0     # x text height: captions above a photo usually belong to the photo above
DIAGONAL_PENALTY = 1.5  # neither horizontally nor vertically adjacent
TIE_RATIO = 1.15        # also link to photos scoring within 15 % of the best


def _overlap_frac(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap of two 1-D ranges as a fraction of the shorter one."""
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1.0)


def score(text: TextBlock, photo: Photo) -> float:
    """Edge-to-edge gap between text box and photo bbox, lower is better.

    Where the text sits is decided by its centre, so a caption box that slightly overlaps
    a neighbouring photo's edge is still "below photo A", not "on photo B".
    Beside and below cost the gap plus a term for how little they overlap (a caption
    under photo A that grazes photo B's x-range goes to A). Above costs extra, in units of
    text height so it scales with the page. Diagonal placements are penalized.
    """
    tx0, ty0, tx1, ty1 = text.bbox()
    px0, py0, px1, py1 = photo.bbox()
    cx, cy = text.center()
    th = ty1 - ty0
    dx = max(0.0, px0 - tx1, tx0 - px1)
    dy = max(0.0, py0 - ty1, ty0 - py1)
    above = cy < py0
    if py0 <= cy <= py1:  # beside (or on) the photo
        return dx + (1 - _overlap_frac(ty0, ty1, py0, py1)) * th
    if px0 <= cx <= px1:  # directly below or above
        return dy + (1 - _overlap_frac(tx0, tx1, px0, px1)) * th + (ABOVE_PENALTY * th if above else 0.0)
    # Diagonal: measured from the text's centre, so a box that just clips the photo's
    # corner does not count as touching it.
    dxc = max(0.0, px0 - cx, cx - px1)
    dyc = max(0.0, py0 - cy, cy - py1)
    return (dxc * dxc + dyc * dyc) ** 0.5 * DIAGONAL_PENALTY + (ABOVE_PENALTY * th if above else 0.0)


def link_heuristic(photos: list[Photo], texts: list[TextBlock]) -> list[Link]:
    links: list[Link] = []
    if not photos:
        return links
    for t in texts:
        scored = sorted(((score(t, p), p.id) for p in photos))
        best = scored[0][0]
        # small absolute slack so two photos at gap 0 / 5 px still tie
        limit = best * TIE_RATIO + 1.0
        links.extend(Link(text_id=t.id, photo_id=pid) for s, pid in scored if s <= limit)
    return links
