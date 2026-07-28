"""Count connected water regions in a 2D grid using a Union-Find (DSU) structure.

A cell counts as water if it is ``"W"``, ``"w"``, ``1`` or ``True``. Water cells that
touch orthogonally (up/down/left/right) belong to the same region.

Run directly to execute the demo and the inline test cases::

    python examples/union_find_water_grids.py
"""

from typing import Any, Sequence

Grid = Sequence[Sequence[Any]]

WATER_VALUES = frozenset({"W", "w", 1, True})


class UnionFind:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._parent = list(range(size))
        self._rank = [0] * size
        self.components = size

    def find(self, node: int) -> int:
        """Return the representative of ``node``'s set, compressing the path."""
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of ``a`` and ``b``; return False if already merged."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


def is_water(cell: Any) -> bool:
    # `1 in WATER_VALUES` is True for True as well, so booleans are handled by the set.
    return cell in WATER_VALUES


def count_water_regions(grid: Grid) -> int:
    """Return the number of orthogonally connected water regions in ``grid``."""
    return len(find_water_regions(grid))


def find_water_regions(grid: Grid) -> list[list[tuple[int, int]]]:
    """Return each connected water region as a sorted list of ``(row, col)`` cells.

    Regions themselves are returned in row-major order of their first cell.
    """
    if not grid or not grid[0]:
        return []

    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("all grid rows must have the same length")

    dsu = UnionFind(rows * cols)
    for r in range(rows):
        for c in range(cols):
            if not is_water(grid[r][c]):
                continue
            index = r * cols + c
            if r + 1 < rows and is_water(grid[r + 1][c]):
                dsu.union(index, index + cols)
            if c + 1 < cols and is_water(grid[r][c + 1]):
                dsu.union(index, index + 1)

    regions: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            if is_water(grid[r][c]):
                regions.setdefault(dsu.find(r * cols + c), []).append((r, c))
    return list(regions.values())


def _run_tests() -> None:
    assert count_water_regions([]) == 0
    assert count_water_regions([[]]) == 0
    assert count_water_regions([["L", "L"], ["L", "L"]]) == 0
    assert count_water_regions([["W"]]) == 1

    # Diagonal touches do not connect.
    assert count_water_regions([["W", "L"], ["L", "W"]]) == 2

    # A single U-shaped region.
    assert count_water_regions([["W", "L", "W"], ["W", "W", "W"]]) == 1

    # Integer grids behave the same as character grids.
    assert count_water_regions([[1, 0, 1], [1, 1, 1]]) == 1

    regions = find_water_regions([["W", "L"], ["L", "W"]])
    assert sorted(regions) == [[(0, 0)], [(1, 1)]], regions

    dsu = UnionFind(4)
    assert dsu.union(0, 1) is True
    assert dsu.union(0, 1) is False
    assert dsu.connected(0, 1) and not dsu.connected(0, 2)
    assert dsu.components == 3

    try:
        count_water_regions([["W", "W"], ["W"]])
    except ValueError:
        pass
    else:  # pragma: no cover - only reached if validation regresses
        raise AssertionError("expected ValueError for ragged grid")

    print("all tests passed")


if __name__ == "__main__":
    demo_grid = [
        ["W", "W", "L", "L", "W"],
        ["W", "L", "L", "W", "W"],
        ["L", "L", "W", "L", "L"],
        ["W", "L", "L", "L", "W"],
    ]

    for row in demo_grid:
        print(" ".join(row))

    print(f"\nwater regions: {count_water_regions(demo_grid)}")
    for i, region in enumerate(find_water_regions(demo_grid), start=1):
        print(f"  region {i}: {region}")

    print()
    _run_tests()
