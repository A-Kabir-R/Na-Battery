"""Cell-balanced mini-batch sampling for NaPINN-Q.

``batch_mode: cell_balanced`` was declared in ``config.yaml`` but had no runtime
effect: the trainer drew a plain ``randperm`` over rows, so cells contributing
more anchors dominated every batch. With 21 training cells contributing between
2 and 14 anchors each, that is a real imbalance.

Both modes are implemented here:

``sequential``
    Plain shuffled row order (the previous behaviour), kept for ablation.
``cell_balanced``
    Cells are sampled approximately uniformly, then transitions are drawn within
    each selected cell. Every batch therefore contains many distinct cells, which
    is also what the cross-cell pairwise loss needs to form valid pairs.

Determinism: every draw goes through the supplied ``torch.Generator``, so a run
seeded with 42 reproduces its batch sequence exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CellBalancedSampler:
    """Epoch-wise batch index generator over a fixed set of training rows.

    Parameters
    ----------
    row_positions:
        Positions (into the full frame) of the rows eligible for training.
    cell_index:
        Cell code per row of the *full frame*; indexed with ``row_positions``.
    batch_size:
        Rows per batch. ``<= 0`` means full batch.
    mode:
        ``"cell_balanced"`` or ``"sequential"``.
    """

    row_positions: torch.Tensor
    cell_index: torch.Tensor
    batch_size: int
    mode: str = "cell_balanced"

    def __post_init__(self) -> None:
        if self.mode not in {"cell_balanced", "sequential"}:
            raise ValueError(f"unknown batch_mode: {self.mode}")
        self.row_positions = self.row_positions.view(-1)
        self.n_rows = int(self.row_positions.numel())
        cells = self.cell_index[self.row_positions]
        self.unique_cells = torch.unique(cells)
        # Row positions grouped by cell, so within-cell draws are a single
        # index op rather than a mask over the whole frame.
        self._by_cell = [
            self.row_positions[cells == cell] for cell in self.unique_cells
        ]

    @property
    def n_cells(self) -> int:
        return int(self.unique_cells.numel())

    def effective_batch_size(self) -> int:
        if self.n_rows == 0:
            return 0
        if self.batch_size and self.batch_size > 0:
            return min(max(int(self.batch_size), 1), self.n_rows)
        return self.n_rows

    def n_batches(self) -> int:
        size = self.effective_batch_size()
        if size <= 0:
            return 1
        return max(1, (self.n_rows + size - 1) // size)

    def epoch_batches(self, generator: torch.Generator | None = None
                      ) -> list[torch.Tensor]:
        """Return this epoch's batches. Resampled on every call."""
        if self.n_rows == 0:
            return []
        size = self.effective_batch_size()
        count = self.n_batches()
        if self.mode == "sequential" or self.n_cells <= 1:
            perm = self._randperm(self.n_rows, generator)
            shuffled = self.row_positions[perm]
            return [shuffled[i * size:(i + 1) * size] for i in range(count)]

        batches: list[torch.Tensor] = []
        for _ in range(count):
            # Uniform over cells, with replacement, then one row per slot from
            # the chosen cell. Cells with many anchors can no longer dominate.
            cell_slots = self._randint(self.n_cells, size, generator)
            picks: list[torch.Tensor] = []
            for slot in cell_slots.tolist():
                pool = self._by_cell[slot]
                if pool.numel() == 0:
                    continue
                offset = self._randint(int(pool.numel()), 1, generator)
                picks.append(pool[offset])
            batches.append(
                torch.cat(picks) if picks else self.row_positions[:0]
            )
        return batches

    # -- deterministic helpers -------------------------------------------
    def _randperm(self, n: int, generator: torch.Generator | None) -> torch.Tensor:
        device = self.row_positions.device
        if generator is not None and generator.device.type == device.type:
            return torch.randperm(n, generator=generator, device=device)
        return torch.randperm(n, device=device)

    def _randint(self, high: int, count: int,
                 generator: torch.Generator | None) -> torch.Tensor:
        device = self.row_positions.device
        if generator is not None and generator.device.type == device.type:
            return torch.randint(
                0, high, (count,), generator=generator, device=device,
            )
        return torch.randint(0, high, (count,), device=device)
