"""Tests for scripts/cell2fire/cell2fire_adapter.py.

Covers `read_asc_header`, `perimeter_from_grid`, `period_grids` (the function that
finds/sorts `ForestGrid*.csv` files by period -- the name assumed by the task spec
turned out to be correct), and an end-to-end smoke test of `cell2fire_perimeters`
(also correctly named in the task spec) against synthetic `ForestGrid<NN>.csv` +
`Forest.asc` files in a tmp instance/output directory pair.

NOTE on `perimeter_from_grid` and diagonally-touching cells: the module docstring
says "Diagonally-touching cells are the same fire (connectivity=8)", and
`features.shapes(..., connectivity=8)` is indeed used. BUT the light clean applied
right after (`buffer(+c).buffer(-c).simplify(c)`, a morphological closing at
half-cell resolution) is run on the union of shapes, and two cells that touch only
at a shared corner polygonise into a single *invalid*, self-touching "bowtie"
polygon (pinched to zero width at the shared corner). GEOS silently repairs that
invalid geometry when buffering, which splits it at the pinch point -- so the
FINAL result for two corner-only-touching cells is actually a 2-part MultiPolygon,
not one merged Polygon. Verified interactively (see report) before writing the
assertion below; the test asserts the real observed behaviour, not the naive
reading of the docstring.
"""

import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

import cell2fire_adapter as ca
from tests.helpers.geodata_factories import make_tiny_asc


def _write_asc_header(path, lines, ncols, nrows):
    """Write just a 6-line Arc/Info header (+ 1 dummy body row) -- enough for
    `read_asc_header`, which only ever reads the first 6 lines and never touches
    the body."""
    header = "\n".join([f"ncols {ncols}", f"nrows {nrows}", *lines]) + "\n"
    body = " ".join(["0"] * ncols) + "\n"
    path.write_text(header + body)


class TestReadAscHeader:
    def test_xllcorner_variant(self, tmp_path):
        p = tmp_path / "corner.asc"
        _write_asc_header(
            p,
            ["xllcorner 100.0", "yllcorner 200.0", "cellsize 10.0", "NODATA_value -9999"],
            ncols=4, nrows=3,
        )
        transform, nrows, ncols, nodata = ca.read_asc_header(str(p))

        assert (nrows, ncols) == (3, 4)
        assert transform.a == pytest.approx(10.0)      # x resolution
        assert transform.e == pytest.approx(-10.0)      # y resolution (negative: top-left origin)
        assert transform.c == pytest.approx(100.0)      # left x == xllcorner
        assert transform.f == pytest.approx(230.0)      # top y == yllcorner + nrows*cellsize
        assert nodata == pytest.approx(-9999.0)

    def test_xllcenter_variant_shifts_by_half_a_cell_to_match_the_corner_variant(self, tmp_path):
        """A *center* header whose xllcenter/yllcenter are exactly half a cell
        northeast of a *corner* header's xllcorner/yllcorner must resolve to the
        identical transform -- proving the function shifts center -> corner by
        `cell/2`, per its own docstring."""
        corner = tmp_path / "corner.asc"
        center = tmp_path / "center.asc"
        _write_asc_header(
            corner,
            ["xllcorner 100.0", "yllcorner 200.0", "cellsize 10.0", "NODATA_value -9999"],
            ncols=4, nrows=3,
        )
        _write_asc_header(
            center,
            ["xllcenter 105.0", "yllcenter 205.0", "cellsize 10.0", "NODATA_value -9999"],
            ncols=4, nrows=3,
        )

        t_corner, *_ = ca.read_asc_header(str(corner))
        t_center, *_ = ca.read_asc_header(str(center))

        assert t_center.c == pytest.approx(t_corner.c)
        assert t_center.f == pytest.approx(t_corner.f)
        assert t_center.a == pytest.approx(t_corner.a)
        assert t_center.e == pytest.approx(t_corner.e)


class TestPerimeterFromGrid:
    CELL = 10.0

    @staticmethod
    def _transform():
        # top-left origin (0, 200), 10 m cells -- matches make_tiny_asc/read_asc_header's
        # top-left-origin, negative-y-resolution convention.
        return Affine(10.0, 0, 0, 0, -10.0, 200.0)

    def test_5x5_block_polygonises_to_approximately_its_raster_area(self):
        grid = np.zeros((20, 20))
        grid[5:10, 5:10] = 1  # 5x5 = 25 cells

        poly = ca.perimeter_from_grid(grid, self._transform())

        assert poly is not None
        expected_area = 25 * self.CELL**2
        # The function's own docstring claims the half-cell morphological
        # close + simplify clean loses <1% of the honest raster area.
        assert poly.area == pytest.approx(expected_area, rel=0.01)

    def test_all_zero_grid_returns_none(self):
        grid = np.zeros((20, 20))

        assert ca.perimeter_from_grid(grid, self._transform()) is None

    def test_corner_touching_cells_end_up_as_two_separate_polygon_parts(self):
        """See module docstring: connectivity=8 groups the two corner-touching
        cells into one (self-touching, invalid) shape at the `features.shapes`
        stage, but the subsequent buffer/erode clean repairs the invalid bowtie
        by splitting it at the zero-width pinch point. The real, final output is
        therefore a 2-part MultiPolygon, not one merged Polygon."""
        grid = np.zeros((20, 20))
        grid[5, 5] = 1
        grid[6, 6] = 1  # touches [5, 5] only at the shared corner

        poly = ca.perimeter_from_grid(grid, self._transform())

        assert poly is not None
        assert poly.geom_type == "MultiPolygon"
        assert len(poly.geoms) == 2
        # No information is lost -- the two 1-cell parts are still individually
        # ~1 cell in area.
        for part in poly.geoms:
            assert part.area == pytest.approx(self.CELL**2, rel=0.01)

    def test_single_isolated_cell_still_returns_a_valid_tiny_polygon(self):
        """`perimeter_from_grid` does NOT do speck-filtering itself (that lives in
        `_outer_front`, gated by `SPECK_MIN_M2`, later in the pipeline) -- a lone
        burned cell surrounded by zeros still comes back as a real polygon."""
        grid = np.zeros((20, 20))
        grid[10, 10] = 1

        poly = ca.perimeter_from_grid(grid, self._transform())

        assert poly is not None
        assert poly.geom_type == "Polygon"
        assert poly.area == pytest.approx(self.CELL**2, rel=0.01)


class TestPeriodGrids:
    def test_finds_and_sorts_forestgrid_csvs_by_period_and_skips_non_matching_files(self, tmp_path):
        grids_dir = tmp_path / "output" / "Grids" / "Grids1"
        grids_dir.mkdir(parents=True)

        # Created deliberately OUT OF ORDER on disk: period 2 first.
        (grids_dir / "ForestGrid2.csv").write_text("0,0\n0,0\n")
        # Matches the ForestGrid*.csv glob but has no digits before .csv -- the
        # regex should skip it (not just accidental directory-listing order).
        (grids_dir / "ForestGridInfo.csv").write_text("not a period file\n")
        (grids_dir / "ForestGrid0.csv").write_text("0,0\n0,0\n")
        (grids_dir / "ForestGrid1.csv").write_text("0,0\n0,0\n")

        result = ca.period_grids(str(tmp_path / "output"), sim=1)

        assert [period for period, _path in result] == [0, 1, 2]
        assert len(result) == 3  # ForestGridInfo.csv excluded
        for period, path in result:
            assert path.endswith(f"ForestGrid{period}.csv")


class TestCell2FirePerimetersEndToEnd:
    def test_growing_fire_yields_monotonically_non_decreasing_periods(self, tmp_path):
        instance_dir = tmp_path / "instance"
        output_dir = tmp_path / "output"
        grids_dir = output_dir / "Grids" / "Grids1"
        instance_dir.mkdir(parents=True)
        grids_dir.mkdir(parents=True)

        transform = from_origin(0, 60, 10, 10)  # 10 m cells, 6x6 grid -> 60x60 m extent
        fuel = np.ones((6, 6), dtype=int)  # trivial fuel codes -- values don't matter here
        make_tiny_asc(str(instance_dir / "Forest.asc"), fuel, transform)

        # Strictly growing burned area: 1 cell -> 4 cells -> 9 cells.
        grid0 = np.zeros((6, 6), dtype=int)
        grid0[2, 2] = 1
        grid1 = np.zeros((6, 6), dtype=int)
        grid1[2:4, 2:4] = 1
        grid2 = np.zeros((6, 6), dtype=int)
        grid2[1:4, 1:4] = 1

        for period, arr in ((0, grid0), (1, grid1), (2, grid2)):
            np.savetxt(str(grids_dir / f"ForestGrid{period}.csv"), arr, fmt="%d", delimiter=",")

        perims = ca.cell2fire_perimeters(str(instance_dir), str(output_dir), crs="EPSG:2100", sim=1)

        assert list(perims["period"]) == [0, 1, 2]
        assert list(perims["n_cells"]) == [1, 4, 9]

        n_cells = perims["n_cells"].to_numpy()
        areas = perims.geometry.area.to_numpy()
        assert np.all(np.diff(n_cells) >= 0)
        assert np.all(np.diff(areas) >= 0)
        # Sanity: area should track n_cells * cell_area closely (light clean only).
        cell_area = transform.a * transform.a
        assert areas == pytest.approx(n_cells * cell_area, rel=0.01)
