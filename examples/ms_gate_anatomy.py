"""Anatomy of a Molmer-Sorensen gate.

Renders a 6-panel animation of a maximally-entangling MS gate on a
5-ion Ca-40 chain, showing every layer of the physics simultaneously:

    +------------------+------------------+------------------+
    | Coulomb crystal  | Wigner W(x, p)   | Phase-space loop |
    | (chain + motion) | of the bus mode  | of the sigma_x   |
    | colored by <s_z> | (3 coherent      | sectors          |
    |                  |  blobs)          |                  |
    +------------------+------------------+------------------+
    | Axial mode       | Two-qubit        | Bell fidelity    |
    | spectrum         | populations      | + geometric      |
    | (bus highlighted)| 00 / 01 / 10 / 11| phase + clock    |
    +------------------+------------------+------------------+

Run with the project venv:

    .venv/bin/python examples/ms_gate_anatomy.py

Output: examples/ms_gate_anatomy.gif
"""

from __future__ import annotations

from pathlib import Path

import logging
import warnings

import matplotlib

matplotlib.use("Agg")
# Brand calls for Colfax; fall back silently when it isn't installed.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="findfont")

import matplotlib.pyplot as plt
import numpy as np
import qutip
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from tiqs import (
    HilbertSpace,
    OperatorFactory,
    PaulTrap,
    SimulationConfig,
    SimulationRunner,
    StateFactory,
    get_species,
    lamb_dicke_parameters,
    normal_modes,
)
from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.analysis.phase_space import motional_wigner
from tiqs.constants import HBAR
from tiqs.gates.molmer_sorensen import ms_gate_duration

TWO_PI = 2 * np.pi

# IonQ brand palette - light theme.
# Brand spec: light backgrounds for research/talks; dark strokes #11191F;
# IonQ Orange (#FF5000) reserved for the ion / highlights / accents.
WHITE = "#FFFFFF"
OFF_WHITE = "#FAFAFA"
LIGHT_GREY = "#E0E0E3"
GREY = "#C0C0C3"
FIELD_GREY = "#909093"
DARK_GREY = "#606063"
INK_GREY = "#404043"
IONQ_DARK = "#11191F"
IONQ_ORANGE = "#FF5000"
IONQ_MED_ORANGE = "#FF8200"
IONQ_LIGHT_ORANGE = "#FFB600"

BG = WHITE
PANEL = OFF_WHITE
GRID = GREY
FG = IONQ_DARK
MUTED = DARK_GREY
SECTOR_COLORS = {
    "+2": IONQ_ORANGE,
    "0": FIELD_GREY,
    "-2": IONQ_MED_ORANGE,
}
BUS_COLOR = IONQ_ORANGE

# Wigner colormap: off-white -> light orange -> IonQ orange.
# All-orange ramp on a white background; high density = saturated orange.
WIGNER_CMAP = LinearSegmentedColormap.from_list(
    "ionq_wigner",
    [OFF_WHITE, IONQ_LIGHT_ORANGE, IONQ_MED_ORANGE, IONQ_ORANGE],
    N=256,
)
# Ion sigma_z colormap: ground |0> (light grey) -> excited |1> (IonQ orange).
ION_CMAP = LinearSegmentedColormap.from_list(
    "ionq_state",
    [LIGHT_GREY, IONQ_LIGHT_ORANGE, IONQ_ORANGE],
    N=256,
)


def configure_style() -> None:
    """Apply the IonQ light-theme matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": FG,
        # Brand calls for Colfax. Fall back to a clean geometric sans on
        # systems without it; matplotlib will pick the first available.
        "font.family": [
            "Colfax",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
        ],
        "font.weight": 400,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": 600,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "grid.color": GREY,
        "grid.linewidth": 0.5,
        "savefig.facecolor": BG,
    })


def build_simulation():
    """Set up the 5-ion chain, gate parameters, and run the MS gate.

    Returns a dictionary of everything the animation needs.
    """
    species = get_species("Ca40")
    trap = PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 1e6,
        species=species,
    )

    n_ions = 5
    gate_ions = [0, n_ions - 1]  # entangle the outermost two
    n_modes_phys = 1  # only the COM bus mode in the Hilbert space
    n_fock = 18

    # Pre-compute physical mode spectrum (all modes) for the spectrum panel.
    masses = np.full(n_ions, species.mass_kg)
    full_modes = normal_modes(n_ions, trap, masses)
    axial_freqs = full_modes.modes["axial"].freqs
    axial_vectors = full_modes.modes["axial"].vectors
    eq_positions = full_modes.positions  # meters

    # Lamb-Dicke parameters for the bus (COM) mode.
    k_eff = TWO_PI / species.qubit_wavelength  # 729 nm optical qubit
    eta_full = lamb_dicke_parameters(full_modes, species, k_eff, "axial")

    # Run the MS gate (sesolve, no noise) on a 1-mode Hilbert space.
    detuning = TWO_PI * 5e3  # 5 kHz sideband detuning -> 200 us gate
    n_steps = 90
    config = SimulationConfig(
        species=species,
        trap=trap,
        n_ions=n_ions,
        n_modes=n_modes_phys,
        n_fock=n_fock,
    )
    runner = SimulationRunner(config)
    result = runner.run_ms_gate(
        ions=gate_ions, mode=0, detuning=detuning, n_steps=n_steps
    )
    states = result.states
    tau = ms_gate_duration(detuning, loops=1)
    tlist = np.linspace(0.0, tau, n_steps)

    # Compute analytical phase-space loops for the sigma_x sectors.
    # alpha_s(t) = -(eta * Omega * s / delta) * (exp(i*delta*t) - 1)
    eta_geom = float(
        np.sqrt(eta_full[gate_ions[0], 0] * eta_full[gate_ions[1], 0])
    )
    Omega = detuning / (4.0 * eta_geom)  # max-entangling, K=1
    t_analytic = np.linspace(0, tau, 240)
    alpha_plus = -(eta_geom * Omega * 2 / detuning) * (
        np.exp(1j * detuning * t_analytic) - 1
    )
    alpha_minus = -(eta_geom * Omega * (-2) / detuning) * (
        np.exp(1j * detuning * t_analytic) - 1
    )
    # x = sqrt(2) Re(alpha), p = sqrt(2) Im(alpha)
    loop_plus = np.column_stack([
        np.sqrt(2) * np.real(alpha_plus),
        np.sqrt(2) * np.imag(alpha_plus),
    ])
    loop_minus = np.column_stack([
        np.sqrt(2) * np.real(alpha_minus),
        np.sqrt(2) * np.imag(alpha_minus),
    ])

    # Build single-shot operator factory for spin and mode observables.
    hs = HilbertSpace(n_ions=n_ions, n_modes=n_modes_phys, n_fock=n_fock)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    qubit_indices = list(range(n_ions))

    # Per-ion sigma_z operators and a number operator on the bus mode.
    sz_ops = [ops.sigma_z(i) for i in range(n_ions)]
    n_op = ops.number(0)

    # Pre-compute observables vs time.
    sz_t = np.zeros((n_ions, len(states)))
    n_phonon_t = np.zeros(len(states))
    populations_t = np.zeros((4, len(states)))  # |00>, |01>, |10>, |11>
    fidelity_t = np.zeros(len(states))

    # Two-qubit reduced state target indices: ions 0 and last.
    for k, st in enumerate(states):
        for j in range(n_ions):
            sz_t[j, k] = float(np.real(qutip.expect(sz_ops[j], st)))
        n_phonon_t[k] = float(np.real(qutip.expect(n_op, st)))
        rho_two = st.ptrace(gate_ions)
        rho_two_arr = rho_two.full()
        # Basis order from ptrace([0, 4]): |00>, |01>, |10>, |11>.
        for b in range(4):
            populations_t[b, k] = float(np.real(rho_two_arr[b, b]))
        fidelity_t[k] = bell_state_fidelity(rho_two)

    # Pre-compute Wigner functions for every animation frame.
    xvec = np.linspace(-2.5, 2.5, 90)
    wigner_t = np.zeros((len(states), len(xvec), len(xvec)))
    for k, st in enumerate(states):
        wigner_t[k] = motional_wigner(st, 0, qubit_indices, xvec=xvec)

    # Pre-compute conditional <x>,<p> trace from the actual simulation
    # (a sanity ribbon over the analytical loops).
    x_op = ops.position(0)
    p_op = ops.momentum(0)
    x_sim = np.array([float(np.real(qutip.expect(x_op, st))) for st in states])
    p_sim = np.array([float(np.real(qutip.expect(p_op, st))) for st in states])

    # Geometric phase accumulated: phi(t) = 4 K (eta Omega / delta)^2 * pi *
    # (1 - sinc(delta t / 2pi)) ... use exact integral formula.
    # phi(t) = 2 (eta Omega)^2 * (delta t - sin(delta t)) / delta^2
    geom_phase_t = (
        2.0
        * (eta_geom * Omega) ** 2
        * (detuning * tlist - np.sin(detuning * tlist))
        / detuning**2
    )

    return {
        "species": species,
        "trap": trap,
        "n_ions": n_ions,
        "gate_ions": gate_ions,
        "tau": tau,
        "detuning": detuning,
        "Omega": Omega,
        "eta_geom": eta_geom,
        "axial_freqs": axial_freqs,
        "axial_vectors": axial_vectors,
        "eq_positions": eq_positions,
        "tlist": tlist,
        "states": states,
        "sz_t": sz_t,
        "n_phonon_t": n_phonon_t,
        "populations_t": populations_t,
        "fidelity_t": fidelity_t,
        "wigner_t": wigner_t,
        "xvec": xvec,
        "loop_plus": loop_plus,
        "loop_minus": loop_minus,
        "x_sim": x_sim,
        "p_sim": p_sim,
        "geom_phase_t": geom_phase_t,
    }


def make_figure(sim):
    """Build the static figure layout and return artists keyed by name."""
    fig = plt.figure(figsize=(15.5, 8.6), dpi=110)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.07,
        wspace=0.28,
        hspace=0.40,
    )

    artists = {"fig": fig}

    # ----- A: Coulomb chain (top-left) -----
    ax_chain = fig.add_subplot(gs[0, 0])
    ax_chain.set_title("Coulomb crystal  -  $5 \\times {}^{40}$Ca$^+$")
    eq_um = sim["eq_positions"] * 1e6
    span = max(abs(eq_um.min()), abs(eq_um.max())) * 1.4
    ax_chain.set_xlim(-span, span)
    ax_chain.set_ylim(-span * 0.45, span * 0.45)
    ax_chain.set_xlabel("axial position $z$  [$\\mu$m]")
    ax_chain.set_yticks([])
    ax_chain.spines["left"].set_visible(False)
    ax_chain.axhline(0, color=GRID, lw=0.6, zorder=0)

    # Trap rod outlines, just visual flavor.
    rod_y = span * 0.30
    for sign in (-1, 1):
        ax_chain.plot(
            [-span * 0.95, span * 0.95],
            [sign * rod_y, sign * rod_y],
            color=GRID,
            lw=2,
            zorder=0,
        )

    # Ion scatter - we'll update facecolors and x-positions each frame.
    ion_scatter = ax_chain.scatter(
        eq_um,
        np.zeros_like(eq_um),
        s=460,
        c=[LIGHT_GREY] * sim["n_ions"],
        edgecolors=IONQ_DARK,
        linewidths=1.2,
        zorder=5,
    )
    artists["ion_scatter"] = ion_scatter
    artists["eq_um"] = eq_um

    # Highlight ring around the gate participants.
    halo = ax_chain.scatter(
        eq_um[sim["gate_ions"]],
        np.zeros(2),
        s=820,
        facecolors="none",
        edgecolors=IONQ_ORANGE,
        linewidths=1.6,
        zorder=4,
    )
    artists["halo"] = halo

    # Ion labels.
    for i, x in enumerate(eq_um):
        ax_chain.text(
            x,
            -span * 0.30,
            f"#{i}",
            ha="center",
            va="center",
            color=MUTED,
            fontsize=8,
        )

    # Legend strip showing the |0> -> |1> color ramp.
    legend_ax = ax_chain.inset_axes([0.02, 0.85, 0.45, 0.06])
    legend_ax.imshow(
        np.linspace(0, 1, 256).reshape(1, -1),
        cmap=ION_CMAP,
        aspect="auto",
        extent=(0, 1, 0, 1),
    )
    legend_ax.set_xticks([0, 0.5, 1])
    legend_ax.set_xticklabels(
        ["$|0\\rangle$", "mix", "$|1\\rangle$"], fontsize=8
    )
    legend_ax.set_yticks([])
    legend_ax.tick_params(colors=MUTED, length=2)
    for sp in legend_ax.spines.values():
        sp.set_color(GRID)
    legend_ax.set_title("ion qubit state", fontsize=8, pad=2)

    # ----- B: Wigner function (top-middle) -----
    ax_wigner = fig.add_subplot(gs[0, 1])
    ax_wigner.set_title("Bus mode Wigner function $W(x, p)$")
    extent = (sim["xvec"][0], sim["xvec"][-1], sim["xvec"][0], sim["xvec"][-1])
    # Saturate well below the vacuum-Gaussian peak so the three blobs in the
    # mid-gate mixture stand out instead of clipping to white.
    wigner_peak = float(np.max(np.abs(sim["wigner_t"])))
    wigner_im = ax_wigner.imshow(
        sim["wigner_t"][0],
        origin="lower",
        extent=extent,
        cmap=WIGNER_CMAP,
        vmin=0.0,
        vmax=wigner_peak * 0.55,
        interpolation="bilinear",
    )
    ax_wigner.set_xlabel("$x$  (zero-point units)")
    ax_wigner.set_ylabel("$p$  (zero-point units)")
    ax_wigner.set_aspect("equal")
    artists["wigner_im"] = wigner_im
    # Markers at the analytical coherent-state centers (drawn last so they
    # sit on top of the heatmap).
    blob_plus = ax_wigner.scatter(
        [],
        [],
        s=70,
        facecolors="none",
        edgecolors=IONQ_ORANGE,
        linewidths=1.6,
        zorder=4,
    )
    blob_minus = ax_wigner.scatter(
        [],
        [],
        s=70,
        facecolors="none",
        edgecolors=IONQ_MED_ORANGE,
        linewidths=1.6,
        zorder=4,
    )
    blob_zero = ax_wigner.scatter(
        [0],
        [0],
        s=70,
        facecolors="none",
        edgecolors=FIELD_GREY,
        linewidths=1.4,
        zorder=4,
    )
    artists["blob_plus"] = blob_plus
    artists["blob_minus"] = blob_minus
    cbar = plt.colorbar(wigner_im, ax=ax_wigner, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(colors=MUTED, length=2)

    # ----- C: Phase-space loops (top-right) -----
    ax_loop = fig.add_subplot(gs[0, 2])
    ax_loop.set_title("Phase-space loops by $\\sigma_x$ sector")
    R = (
        max(np.abs(sim["loop_plus"]).max(), np.abs(sim["loop_minus"]).max())
        * 1.35
    )
    ax_loop.set_xlim(-R, R)
    ax_loop.set_ylim(-R, R)
    ax_loop.set_aspect("equal")
    ax_loop.set_xlabel("$\\langle x \\rangle$")
    ax_loop.set_ylabel("$\\langle p \\rangle$")
    ax_loop.grid(True, alpha=0.25)
    ax_loop.axhline(0, color=GRID, lw=0.5)
    ax_loop.axvline(0, color=GRID, lw=0.5)

    # Full analytical loops as faint guides.
    ax_loop.plot(
        sim["loop_plus"][:, 0],
        sim["loop_plus"][:, 1],
        color=SECTOR_COLORS["+2"],
        lw=1.0,
        alpha=0.35,
    )
    ax_loop.plot(
        sim["loop_minus"][:, 0],
        sim["loop_minus"][:, 1],
        color=SECTOR_COLORS["-2"],
        lw=1.0,
        alpha=0.35,
    )
    # Zero-displacement sector marker (always at origin).
    ax_loop.scatter([0], [0], color=SECTOR_COLORS["0"], s=80, zorder=4)

    # Animated traces - they will fill up to the current time.
    (line_plus,) = ax_loop.plot([], [], color=SECTOR_COLORS["+2"], lw=2.4)
    (line_minus,) = ax_loop.plot([], [], color=SECTOR_COLORS["-2"], lw=2.4)
    head_plus = ax_loop.scatter(
        [], [], color=SECTOR_COLORS["+2"], s=110, zorder=5
    )
    head_minus = ax_loop.scatter(
        [], [], color=SECTOR_COLORS["-2"], s=110, zorder=5
    )
    artists["line_plus"] = line_plus
    artists["line_minus"] = line_minus
    artists["head_plus"] = head_plus
    artists["head_minus"] = head_minus

    leg = ax_loop.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=SECTOR_COLORS["+2"],
                lw=2.4,
                label=r"$\sigma_x^{tot}=+2$",
            ),
            plt.Line2D(
                [0],
                [0],
                color=SECTOR_COLORS["0"],
                marker="o",
                lw=0,
                markersize=7,
                label=r"$\sigma_x^{tot}=0$",
            ),
            plt.Line2D(
                [0],
                [0],
                color=SECTOR_COLORS["-2"],
                lw=2.4,
                label=r"$\sigma_x^{tot}=-2$",
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=1.6,
    )
    for txt in leg.get_texts():
        txt.set_color(FG)

    # ----- D: Mode spectrum (bottom-left) -----
    ax_modes = fig.add_subplot(gs[1, 0])
    ax_modes.set_title("Axial normal-mode spectrum")
    f_khz = sim["axial_freqs"] / TWO_PI / 1e3
    bar_colors = [BUS_COLOR if k == 0 else GREY for k in range(len(f_khz))]
    bars = ax_modes.bar(
        np.arange(len(f_khz)),
        f_khz,
        color=bar_colors,
        edgecolor=GRID,
        linewidth=0.8,
    )
    ax_modes.set_xticks(np.arange(len(f_khz)))
    ax_modes.set_xticklabels([
        "COM" if k == 0 else f"m{k}" for k in range(len(f_khz))
    ])
    ax_modes.set_ylabel("$\\omega_m / 2\\pi$  [kHz]")
    ax_modes.grid(True, axis="y", alpha=0.2)

    # Annotate frequencies above each bar.
    for bar, f in zip(bars, f_khz):
        ax_modes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{f:.0f}",
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=8,
        )
    # Bus-mode arrow.
    ax_modes.annotate(
        "bus mode\n(MS gate)",
        xy=(0, f_khz[0]),
        xytext=(0.6, f_khz.max() * 0.7),
        color=BUS_COLOR,
        fontsize=8,
        ha="center",
        arrowprops=dict(color=BUS_COLOR, lw=1.0, arrowstyle="->"),
    )

    # ----- E: Two-qubit populations (bottom-middle) -----
    ax_pop = fig.add_subplot(gs[1, 1])
    ax_pop.set_title(
        "Reduced state of ions "
        f"#{sim['gate_ions'][0]} & #{sim['gate_ions'][1]}"
    )
    pop_labels = [
        r"$|00\rangle$",
        r"$|01\rangle$",
        r"$|10\rangle$",
        r"$|11\rangle$",
    ]
    pop_x = np.arange(4)
    pop_bars = ax_pop.bar(
        pop_x,
        sim["populations_t"][:, 0],
        color=[IONQ_ORANGE, LIGHT_GREY, LIGHT_GREY, IONQ_ORANGE],
        edgecolor=IONQ_DARK,
        linewidth=0.8,
    )
    ax_pop.set_xticks(pop_x)
    ax_pop.set_xticklabels(pop_labels)
    ax_pop.set_ylim(0, 1.0)
    ax_pop.set_ylabel("population")
    ax_pop.grid(True, axis="y", alpha=0.2)
    artists["pop_bars"] = pop_bars

    # Bell-state target marker.
    ax_pop.axhline(0.5, color=IONQ_ORANGE, lw=0.8, alpha=0.6, linestyle="--")
    ax_pop.text(
        3.3,
        0.52,
        "Bell target",
        color=IONQ_ORANGE,
        fontsize=8,
        alpha=0.9,
        ha="right",
    )

    # ----- F: Fidelity & metrics (bottom-right) -----
    ax_fid = fig.add_subplot(gs[1, 2])
    ax_fid.set_title("Bell fidelity & geometric phase")
    ax_fid.set_xlim(0, sim["tau"] * 1e6)
    ax_fid.set_ylim(0, 1.05)
    ax_fid.set_xlabel("time  [$\\mu$s]")
    ax_fid.set_ylabel("Bell-state fidelity", color=IONQ_ORANGE)
    ax_fid.tick_params(axis="y", colors=IONQ_ORANGE)
    ax_fid.grid(True, alpha=0.2)

    (fid_line,) = ax_fid.plot(
        [], [], color=IONQ_ORANGE, lw=2.4, label="Bell fidelity"
    )
    (fid_dot,) = ax_fid.plot([], [], "o", color=IONQ_ORANGE, markersize=7)

    ax_phase = ax_fid.twinx()
    ax_phase.set_ylabel(r"geometric phase $\phi / \pi$", color=IONQ_MED_ORANGE)
    ax_phase.tick_params(axis="y", colors=IONQ_MED_ORANGE)
    phase_max = sim["geom_phase_t"][-1] / np.pi * 1.15
    ax_phase.set_ylim(0, max(phase_max, 0.3))
    (phase_line,) = ax_phase.plot(
        [], [], color=IONQ_MED_ORANGE, lw=2.0, linestyle="--"
    )
    (phase_dot,) = ax_phase.plot(
        [], [], "o", color=IONQ_MED_ORANGE, markersize=6
    )
    ax_phase.spines["right"].set_color(IONQ_MED_ORANGE)
    ax_phase.spines["left"].set_color(IONQ_ORANGE)

    artists["fid_line"] = fid_line
    artists["fid_dot"] = fid_dot
    artists["phase_line"] = phase_line
    artists["phase_dot"] = phase_dot

    # Live metrics text box (top-right of fidelity panel).
    metrics_text = ax_fid.text(
        0.03,
        0.97,
        "",
        transform=ax_fid.transAxes,
        ha="left",
        va="top",
        color=IONQ_DARK,
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc=WHITE, ec=GREY, lw=0.8),
    )
    artists["metrics_text"] = metrics_text

    # ----- Suptitle / banner -----
    fig.suptitle(
        "TIQS  -  Anatomy of a Mølmer-Sørensen Gate",
        fontsize=18,
        fontweight=600,
        color=IONQ_DARK,
        y=0.975,
    )
    sub = (
        f"linear Paul trap  -  $\\omega_z/2\\pi$ = "
        f"{sim['axial_freqs'][0] / TWO_PI / 1e6:.2f} MHz  -  "
        f"$\\delta/2\\pi$ = {sim['detuning'] / TWO_PI / 1e3:.0f} kHz  -  "
        f"$\\eta$ = {sim['eta_geom']:.3f}  -  "
        f"$\\tau$ = {sim['tau'] * 1e6:.0f} $\\mu$s"
    )
    fig.text(0.5, 0.935, sub, ha="center", color=MUTED, fontsize=10)

    return artists


def update(frame, sim, artists):
    """Update all dynamic artists for animation frame *frame*."""
    n_steps = len(sim["tlist"])
    k = min(frame, n_steps - 1)
    t = sim["tlist"][k]

    # Index into the densely-sampled analytical loops that matches frame k.
    n_an = sim["loop_plus"].shape[0]
    cut = max(2, int(n_an * (k + 1) / n_steps))

    # --- Ion chain colors and motion ---
    sz_now = sim["sz_t"][:, k]
    # In QuTiP convention sigma_z|0> = +|0> (ground) and sigma_z|1> = -|1>.
    # Map sz=+1 (ground) -> 0 (light grey) and sz=-1 (excited) -> 1 (orange).
    ion_colors = ION_CMAP((1.0 - sz_now) / 2.0)
    artists["ion_scatter"].set_facecolors(ion_colors)

    # The full-state <x> for the bus mode averages to zero across the
    # symmetric +/- sigma_x sectors, so showing it gives a static chain.
    # Display the +2-sector conditional displacement instead - i.e. the
    # branch of the entangled wavepacket that traces the cyan loop.
    b_com = sim["axial_vectors"][:, 0]
    x_zpf = float(
        np.sqrt(HBAR / (2 * sim["species"].mass_kg * sim["axial_freqs"][0]))
    )
    x_cond = sim["loop_plus"][cut - 1, 0]
    amp_factor = 50.0
    dx_um_per_ion = b_com * x_cond * x_zpf * 1e6 * amp_factor
    new_x = artists["eq_um"] + dx_um_per_ion
    coords = np.column_stack([new_x, np.zeros_like(new_x)])
    artists["ion_scatter"].set_offsets(coords)
    artists["halo"].set_offsets(coords[sim["gate_ions"]])

    # --- Wigner function + analytical blob centers ---
    artists["wigner_im"].set_data(sim["wigner_t"][k])
    artists["blob_plus"].set_offsets(sim["loop_plus"][cut - 1 : cut])
    artists["blob_minus"].set_offsets(sim["loop_minus"][cut - 1 : cut])

    # --- Phase-space loops: animate up to time t/tau ---
    artists["line_plus"].set_data(
        sim["loop_plus"][:cut, 0], sim["loop_plus"][:cut, 1]
    )
    artists["line_minus"].set_data(
        sim["loop_minus"][:cut, 0], sim["loop_minus"][:cut, 1]
    )
    artists["head_plus"].set_offsets(sim["loop_plus"][cut - 1 : cut])
    artists["head_minus"].set_offsets(sim["loop_minus"][cut - 1 : cut])

    # --- Populations bar heights ---
    for b, bar in enumerate(artists["pop_bars"]):
        bar.set_height(sim["populations_t"][b, k])

    # --- Fidelity / phase traces ---
    t_us = sim["tlist"][: k + 1] * 1e6
    artists["fid_line"].set_data(t_us, sim["fidelity_t"][: k + 1])
    artists["fid_dot"].set_data([t_us[-1]], [sim["fidelity_t"][k]])
    artists["phase_line"].set_data(t_us, sim["geom_phase_t"][: k + 1] / np.pi)
    artists["phase_dot"].set_data([t_us[-1]], [sim["geom_phase_t"][k] / np.pi])

    # --- Metrics box ---
    metrics = (
        f"t        = {t * 1e6:6.1f} us / {sim['tau'] * 1e6:.0f} us\n"
        f"<n_bus>  = {sim['n_phonon_t'][k]:6.3f}\n"
        f"phi/pi   = {sim['geom_phase_t'][k] / np.pi:6.3f}\n"
        f"F_Bell   = {sim['fidelity_t'][k]:6.4f}"
    )
    artists["metrics_text"].set_text(metrics)

    # Returning the list of dynamic artists keeps blit happy if used.
    return list(artists.values())


def main() -> None:
    configure_style()
    print("[1/3] Building simulation...")
    sim = build_simulation()
    print(
        f"      gate duration = {sim['tau'] * 1e6:.1f} us, "
        f"{len(sim['tlist'])} sim steps, "
        f"final fidelity = {sim['fidelity_t'][-1]:.4f}"
    )

    print("[2/3] Building figure...")
    artists = make_figure(sim)

    n_frames = len(sim["tlist"])
    fps = 22  # ~4s gif
    print(f"[3/3] Rendering animation ({n_frames} frames @ {fps} fps)...")

    anim = FuncAnimation(
        artists["fig"],
        update,
        frames=n_frames,
        fargs=(sim, artists),
        interval=1000 / fps,
        blit=False,
    )

    out_path = Path(__file__).resolve().parent / "ms_gate_anatomy.gif"
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=110)
    plt.close(artists["fig"])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
