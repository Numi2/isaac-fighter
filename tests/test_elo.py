from __future__ import annotations

from isaac_fight.tasks.direct.unitree_1v1.elo import EloTable


def test_elo_update_win_loss_draw():
    table = EloTable(k_factor=32)
    ra, rb = table.update("a", "b", 1.0)
    assert ra > 1000
    assert rb < 1000
    table.update("a", "b", 0.5)
    assert table.records["a"].games == 2
    assert table.records["b"].draws == 1
