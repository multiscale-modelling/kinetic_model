import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_ts_states(s):
    """Split TS.state on comma and strip—preserves labels like \"3'\"."""
    return [x.strip() for x in str(s).split(',')]

def plot_parabola(ax, x1, x2, E1, E2, x_ts, E_ts, color):
    """Fit and plot a smooth red parabola between two intermediates via TS."""
    def fit_quad(xs, ys):
        A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
        return np.linalg.lstsq(A, ys, rcond=None)[0]

    # left: E1→Ets→E1, right: E2→Ets→E2
    a0, b0, c0 = fit_quad(np.array([x1, x_ts, x2]), np.array([E1, E_ts, E1]))
    a1, b1, c1 = fit_quad(np.array([x1, x_ts, x2]), np.array([E2, E_ts, E2]))

    xs0 = np.linspace(x1, x_ts, 100)
    ys0 = a0*xs0**2 + b0*xs0 + c0
    xs1 = np.linspace(x_ts, x2, 100)
    ys1 = a1*xs1**2 + b1*xs1 + c1

    ax.plot(np.concatenate([xs0, xs1]), np.concatenate([ys0, ys1]), color=color)

def plot_energy_profile(
    df, ax,
    m=0.8,
    annotate_states=False,
    delta_text_energy=5,
    delta_text_state=-5,
    connections=None,
    pallate = ('#fee6ce','#fdae6b','#e6550d')
):
    """
    Plot an energy profile with:
      - red parabolas for TS (type=='TS')
      - dashed lines for explicit 'connections'
      - optional PNG overlay and state annotations

    df: DataFrame with columns
        'E'    : energy
        'type' : 'I' or 'TS'
        'state': e.g. '2', "3'", or '2,3'
        'x'     : only on rows where type=='I'
    connections: dict mapping state -> list of states to connect with dashes
    """
    # 1) Prepare intermediates
    inter = df[df['type']=='I'].copy().sort_values('x')
    state_to_x      = dict(zip(inter['state'], inter['x']))
    state_to_energy = dict(zip(inter['state'], inter['E']))
    half = m / 2

    # 2) Draw intermediate plateaus
    for _, row in inter.iterrows():
        s, E, x = row['state'], row['E'], row['x']
        ax.hlines(E, x-half, x+half, colors=pallate[1])
        ax.text(x, E + delta_text_energy, f"{E:.1f}", ha='center')
        if annotate_states:
            ax.text(x, E + delta_text_state, s, ha='center')

    # 3) Draw TS parabolas
    for _, row in df[df['type']=='TS'].iterrows():
        s1, s2 = parse_ts_states(row['state'])
        x1, x2 = state_to_x[s1], state_to_x[s2]
        E1, E2 = state_to_energy[s1], state_to_energy[s2]
        Ets     = row['E']
        x_ts    = 0.5 * (x1 + x2)
        plot_parabola(ax, x1+half, x2-half, E1, E2, x_ts, Ets, color=pallate[2])
        ax.text(x_ts, Ets + delta_text_energy, f"{Ets:.1f}", ha='center')

    # 4) Draw dashed connections from `connections`
    if connections:
        # Normalize into set of sorted tuples
        pairs = set()
        for s1, targets in connections.items():
            if not isinstance(targets, (list, tuple, set)):
                targets = [targets]
            for s2 in targets:
                pairs.add(tuple(sorted((s1, s2))))
        # Plot each pair
        for s1, s2 in pairs:
            if s1 in state_to_x and s2 in state_to_x:
                x1, x2 = state_to_x[s1], state_to_x[s2]
                E1, E2 = state_to_energy[s1], state_to_energy[s2]
                ax.plot(
                    [x1+half, x2-half],
                    [E1, E2],
                    linestyle='--',
                    color=pallate[0]
                )

    # 5) Final axes tweaks
    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('Energy (kJ/mol)')
    ax.set_title('Energy Profile')
    ax.set_xticks([])
    ax.set_ylim(df['E'].min() - 15, df['E'].max() + 15)
    ax.set_xlim(inter['x'].min() - m, inter['x'].max() + m)
