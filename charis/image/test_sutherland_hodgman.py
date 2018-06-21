from sutherland_hodgman import *


def test_clip_of_triangle_is_quad():
    subject = [(0, 0), (1, 0), (0, 1)]
    clipper = [(-1, -1), (2, -1), (2, 0.5), (-1, 0.5)]
    assert len(clip(subject, clipper)) == 4


def test_clip_of_triangle_contains_intact_point():
    subject = [(0, 0), (1, 0), (0, 1)]
    clipper = [(-1, -1), (2, -1), (2, 0.5), (-1, 0.5)]
    assert (0, 0) in clip(subject, clipper)


def test_clip_of_triangle_contains_clipped_point():
    subject = [(0, 0), (1, 0), (0, 1)]
    clipper = [(-1, -1), (2, -1), (2, 0.5), (-1, 0.5)]
    assert (0.5, 0.5) in clip(subject, clipper)


def test_empty_clip_is_empty():
    assert clip([], []) == []


def test_non_intersecting_clip_is_empty():
    subject = [(0, 0), (1, 0), (0, 1)]
    clipper = [(2, 0), (3, 0), (2, 1)]
    assert clip(subject, clipper) == []


def test_diamond_squares_form_octagon():
    subject = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    clipper = [(-1.5, 0), (0, -1.5), (1.5, 0), (0, 1.5)]
    assert len(clip(subject, clipper)) == 8


def test_empty_area():
    assert area([]) == 0


def test_single_point_area():
    assert area([(1, 0)]) == 0


def test_line_area():
    assert area([(1, 0), (1, 2)]) == 0


def test_triangle_area():
    assert area([(0, 0), (1, 0), (0, 1)]) == 0.5


def test_shifted_triangle_area():
    assert area([(3, 0), (4, 0), (3, 1)]) == 0.5


def test_quad_area():
    assert area([(0, 0), (1, 0), (1, 1), (0, 1)]) == 1
