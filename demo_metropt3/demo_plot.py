import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import demo_logic as L


def desenhar(hs: pd.DataFrame, alerta: pd.Series, lim: float, t, falha, mostrar_falha: bool):
    """hs: escores horários do cenário; só o que vem até `t` é desenhado (replay causal)."""
    t = pd.Timestamp(t)
    vis = hs.loc[hs.index <= t].reindex(hs.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.5]})
    s = vis["sistema"]
    a1.plot(s.index, s.values, ls="none", marker=".", ms=5, color="tab:blue", label="escore do sistema")
    em_alerta = (alerta & (alerta.index <= t)).astype(bool)
    al = s[em_alerta]
    a1.plot(al.index, al.values, ls="none", marker="o", ms=6, color="darkorange", label="alerta")
    a1.axhline(lim, color="red", ls=":", label=f"limiar ({lim:.1f})")
    a1.axvline(t, color="gray", lw=1)
    maximo = float(np.nanmax(hs["sistema"].to_numpy(dtype=float))) if hs["sistema"].notna().any() else lim
    a1.set_yscale("log")
    a1.set_ylim(0.2, max(lim * 3, maximo * 1.5))
    a1.set_ylabel("erro relativo (1 = típico)")
    a1.legend(loc="upper left", fontsize=8)

    mat = np.log10(np.clip(vis[L.COMPONENTES].to_numpy(dtype=float).T, 0.1, None))
    bordas = hs.index.append(pd.DatetimeIndex([hs.index[-1] + pd.Timedelta(hours=1)]))
    a2.pcolormesh(mdates.date2num(bordas), np.arange(len(L.COMPONENTES) + 1),
                  np.ma.masked_invalid(mat), cmap="YlOrRd", vmin=np.log10(0.7), vmax=np.log10(30))
    a2.set_yticks(np.arange(len(L.COMPONENTES)) + 0.5, [L.NOMES[c] for c in L.COMPONENTES])
    a2.invert_yaxis()
    a2.set_title("Desvio por componente (mais escuro = mais longe do normal; branco = sem dado)",
                 fontsize=9, loc="left")
    if mostrar_falha and falha:
        for ax in (a1, a2):
            ax.axvspan(falha[0], falha[1], color="red", alpha=0.15)
    a2.xaxis.set_major_locator(mdates.DayLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    a1.set_xlim(hs.index[0], hs.index[-1] + pd.Timedelta(hours=1))
    fig.tight_layout()
    return fig
