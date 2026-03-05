import numpy as np
from algoquantengine.opt.risk import portfolio_pnl_from_scenarios, var_cvar, max_drawdown


def test_var_cvar_basic():
    pnl = np.array([0.01, -0.02, 0.00, 0.03, -0.01, 0.02, -0.03, 0.01, 0.0, 0.02])
    v, c = var_cvar(pnl, alpha=0.9)
    assert v >= 0.0
    assert c >= v


def test_portfolio_pnl_shapes():
    w = np.array([0.6, 0.4])
    scen2d = np.array([[0.01, 0.02], [-0.01, 0.00]])
    pnl2d = portfolio_pnl_from_scenarios(scen2d, w)
    assert pnl2d.shape == (2,)

    scen3d = np.array([[[0.01, 0.0], [0.0, 0.02]]])
    pnl3d = portfolio_pnl_from_scenarios(scen3d, w)
    assert pnl3d.shape == (1,)


def test_max_drawdown():
    equity = np.array([1.0, 1.2, 1.1, 1.3, 0.9, 1.0])
    mdd = max_drawdown(equity)
    assert 0.0 <= mdd <= 1.0
    assert mdd > 0.0