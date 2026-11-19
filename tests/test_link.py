from albumsplit.link import link_heuristic
from albumsplit.model import Photo, TextBlock


def rect_photo(pid: int, x0: float, y0: float, x1: float, y1: float) -> Photo:
    return Photo(id=pid, corners=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def links_for(photos, text):
    return sorted(lk.photo_id for lk in link_heuristic(photos, [text]))


def test_caption_beside_photo():
    photos = [rect_photo(1, 0, 0, 1000, 600), rect_photo(2, 0, 700, 1000, 1300)]
    assert links_for(photos, TextBlock(1, (1050, 100, 300, 120))) == [1]
    assert links_for(photos, TextBlock(1, (1050, 900, 300, 120))) == [2]


def test_caption_below_beats_caption_above():
    # Text sits in the gap: 20 px below photo 1, 20 px above photo 2.
    photos = [rect_photo(1, 0, 0, 1000, 600), rect_photo(2, 0, 700, 1000, 1300)]
    assert links_for(photos, TextBlock(1, (100, 620, 400, 60))) == [1]


def test_caption_below_goes_to_photo_it_overlaps_most():
    # Caption under photo 1 that just grazes photo 2's x-range.
    photos = [rect_photo(1, 0, 0, 500, 500), rect_photo(2, 520, 0, 1000, 500)]
    assert links_for(photos, TextBlock(1, (10, 520, 530, 60))) == [1]


def test_caption_box_grazing_next_photo_stays_with_photo_above():
    # IMG_0055: caption under photo 1 whose box dips 30 px into photo 2's bbox.
    photos = [rect_photo(1, 25, 0, 510, 650), rect_photo(2, 325, 705, 715, 1385)]
    assert links_for(photos, TextBlock(1, (80, 650, 440, 85))) == [1]


def test_caption_beside_beats_caption_above():
    # IMG_0096: caption right of photo 1 that is also just above photo 2.
    photos = [rect_photo(1, 18, 0, 400, 300), rect_photo(2, 155, 315, 605, 620)]
    assert links_for(photos, TextBlock(1, (405, 225, 130, 83))) == [1]


def test_caption_clipping_neighbours_corner_stays_below_its_photo():
    # PXL: caption under photo 1 whose box reaches 5 px past photo 2's left edge.
    photos = [rect_photo(1, 683, 68, 958, 343), rect_photo(2, 1005, 88, 1262, 345)]
    assert links_for(photos, TextBlock(1, (685, 345, 325, 45))) == [1]


def test_equidistant_caption_links_to_both():
    photos = [rect_photo(1, 0, 0, 500, 500), rect_photo(2, 700, 0, 1200, 500)]
    assert links_for(photos, TextBlock(1, (550, 200, 100, 60))) == [1, 2]


def test_no_photos_no_links():
    assert link_heuristic([], [TextBlock(1, (0, 0, 10, 10))]) == []
