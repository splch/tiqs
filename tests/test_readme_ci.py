"""Consistency checks for README claims and CI configuration.

The README duplicates facts that live elsewhere: runnable snippets
against the public API, a species table against the species database,
and lint/coverage promises against
``.github/workflows/tests.yml``. Duplicated facts drift - the test
count in this file's own section was stale by a factor of two before
this suite existed. Every test below re-derives the claim from its
source of truth (the code, the species data, the workflow file, an
exact Lindblad solution) rather than restating the README.
"""

import dataclasses
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
import qutip

from tiqs import (
    HilbertSpace,
    OperatorFactory,
    SimulationConfig,
    StateFactory,
    get_species,
)
from tiqs.constants import TWO_PI
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.photon_scattering import rayleigh_scattering_op

# Private, but it is the only enumeration of the shipped species; the
# README table has to match it exactly or one of them is lying.
from tiqs.species.ion import _SPECIES_DB

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text()
WORKFLOW = (REPO_ROOT / ".github/workflows/tests.yml").read_text()
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
ADDOPTS = PYPROJECT["tool"]["pytest"]["ini_options"]["addopts"]

# Names the README's noise snippet uses but does not define, so that
# the fragment can be executed exactly as printed.
_SNIPPET_PRELUDE = """
import numpy as np
import qutip

from tiqs import HilbertSpace, OperatorFactory, StateFactory
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_gate_hamiltonian,
)

_hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=12)
ops = OperatorFactory(_hs)
_eta = 0.05
_delta = 2 * np.pi * 15e3
H = ms_gate_hamiltonian(
    ops,
    ions=[0, 1],
    mode=0,
    eta=[_eta, _eta],
    rabi_frequency=_delta / (4 * _eta),
    detuning=_delta,
)
initial_state = StateFactory(_hs).ground_state()
tlist = np.linspace(0, ms_gate_duration(_delta), 20)
"""


def _python_blocks(markdown: str) -> list[str]:
    """Return the body of every fenced ``python`` block."""
    return re.findall(r"^```python\n(.*?)^```", markdown, re.M | re.S)


README_PY_BLOCKS = _python_blocks(README)


def _run_snippet(block: str) -> dict[str, object]:
    """Execute a README snippet and return its namespace."""
    namespace: dict[str, object] = {}
    exec(_SNIPPET_PRELUDE + block, namespace)
    return namespace


def _block_containing(*needles: str) -> str:
    """Return the single README snippet containing all ``needles``."""
    matches = [b for b in README_PY_BLOCKS if all(n in b for n in needles)]
    assert len(matches) == 1, f"{needles} matched {len(matches)} blocks"
    return matches[0]


def _decimals(quoted: str) -> int:
    """Number of digits after the decimal point in ``quoted``."""
    _, _, fraction = quoted.partition(".")
    return len(fraction)


def _half_ulp(quoted: str) -> float:
    """Rounding tolerance implied by ``quoted``'s own precision."""
    return 0.5 * 10.0 ** (-_decimals(quoted))


class TestReadmeSnippets:
    """The printed code has to run against the current public API."""

    @pytest.mark.parametrize("index", range(len(README_PY_BLOCKS)))
    def test_snippet_executes(self, index: int) -> None:
        """No README snippet may raise.

        This is the guard against silent API drift: renamed
        parameters and functions that moved break the README long
        before anyone rereads it.
        """
        _run_snippet(README_PY_BLOCKS[index])

    def test_quick_start_reaches_its_advertised_fidelity(self) -> None:
        """The trailing ``# 1.0000`` comment must be the real value.

        Anchored twice over: against the number the README itself
        prints, and against the single-qubit reduced purity of a
        maximally entangled pure state, which is exactly 1/2 and does
        not go through ``bell_state_fidelity``.
        """
        block = _block_containing("SimulationRunner", "run_ms_gate")
        claimed = re.search(r"# (\d\.\d+)\s*$", block, re.M)
        assert claimed is not None, "quick start lost its printed value"
        namespace = _run_snippet(block)

        fidelity = namespace["fid"]
        assert isinstance(fidelity, float)
        digits = _decimals(claimed[1])
        assert round(fidelity, digits) == float(claimed[1])

        rho_spin = namespace["rho_spin"]
        assert isinstance(rho_spin, qutip.Qobj)
        assert rho_spin.ptrace([0]).purity() == pytest.approx(0.5, abs=1e-6)


class TestReadmeSpeciesTable:
    """The species table must agree with the species database."""

    @staticmethod
    def _rows() -> dict[str, list[str]]:
        """Parse the species table into ``{"Yb171": [cells...]}``."""
        section = README.split("## Supported ion species", 1)[1]
        rows: dict[str, list[str]] = {}
        for line in section.splitlines():
            if not line.startswith("|"):
                if rows:
                    break
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] == "Species" or set(cells[0]) <= set("-: "):
                continue
            rows[cells[0].replace("-", "")] = cells
        return rows

    def test_table_lists_every_shipped_species(self) -> None:
        assert set(self._rows()) == set(_SPECIES_DB)

    @pytest.mark.parametrize("name", sorted(_SPECIES_DB))
    def test_splitting_column(self, name: str) -> None:
        """Splittings match the stored data to their quoted precision.

        A precision floor is enforced as well: hyperfine splittings
        are known to well under a hertz, so quoting one to 3 decimal
        GHz places (Ba-137 as "8.038 GHz", 258 kHz low) hides a real
        number behind a rounding.
        """
        quoted = self._rows()[name][2]
        species = get_species(name)

        if (ghz := re.fullmatch(r"([\d.]+) GHz", quoted)) is not None:
            assert species.qubit_type == "hyperfine"
            assert _decimals(ghz[1]) >= 4, (
                f"quote {name} to at least 0.1 MHz, got {quoted}"
            )
            stored = species.qubit_frequency_hz / 1e9
        else:
            nanometres = re.fullmatch(r"([\d.]+) nm", quoted)
            assert nanometres is not None, f"unparseable cell {quoted!r}"
            assert species.qubit_type == "optical"
            assert species.qubit_wavelength is not None
            assert _decimals(nanometres[1]) >= 1, (
                f"quote {name} to at least 0.1 nm, got {quoted}"
            )
            ghz = nanometres
            stored = species.qubit_wavelength * 1e9

        assert abs(float(ghz[1]) - stored) <= _half_ulp(ghz[1])

    @pytest.mark.parametrize("name", sorted(_SPECIES_DB))
    def test_cooling_column(self, name: str) -> None:
        """Cooling wavelength and linewidth match the stored data.

        The linewidth is the total upper-state decay rate over 2*pi,
        so Be-9's is 17.97 MHz (NIST A_ki = 1.1292e8 s^-1), not the
        folkloric 19.4 MHz the table used to carry.
        """
        cell = self._rows()[name][3]
        parsed = re.fullmatch(r"([\d.]+) nm \(([\d.]+) MHz\)", cell)
        assert parsed is not None, f"unparseable cell {cell!r}"
        cooling = get_species(name).cooling_transition

        assert _decimals(parsed[1]) >= 1, f"{name}: {cell}"
        assert abs(float(parsed[1]) - cooling.wavelength * 1e9) <= _half_ulp(
            parsed[1]
        )
        assert abs(
            float(parsed[2]) - cooling.linewidth / TWO_PI / 1e6
        ) <= _half_ulp(parsed[2])

    @pytest.mark.parametrize("name", sorted(_SPECIES_DB))
    def test_raman_column(self, name: str) -> None:
        """Raman wavelengths match, or read "-" when there is none.

        No precision floor here: "355 nm" and "515 nm" are the
        conventional names of those lasers, and the stored values are
        exactly those integers.
        """
        cell = self._rows()[name][4]
        stored = get_species(name).raman_wavelength
        if cell == "-":
            assert stored is None
            return
        parsed = re.fullmatch(r"([\d.]+) nm", cell)
        assert parsed is not None, f"unparseable cell {cell!r}"
        assert stored is not None
        assert abs(float(parsed[1]) - stored * 1e9) <= _half_ulp(parsed[1])


class TestReadmeNoiseExample:
    """The noise snippet's comments must state the real semantics."""

    def test_heating_rate_is_the_linear_slope(self) -> None:
        r"""The quoted rate is d<n>/dt, and <n>(t) = ndot*t exactly.

        The default channel is the infinite-temperature limit
        $L_\pm = \sqrt{\dot n}\,\{a^\dagger, a\}$, whose exact
        solution is a straight line through the origin (Turchette et
        al., *Phys. Rev. A* **62**, 053807 (2000), Sec. III.A.3). A
        single raising operator instead gives
        $\langle n\rangle = e^{\dot n t} - 1$, i.e. 6.39 rather than
        2 at the last point checked here.

        The channel spreads a thermal-like distribution of width
        $\bar n$, so it is truncation-sensitive: stopping at
        $\bar n = 2$ with ``n_fock = 60`` leaves a residual of
        1.3e-10, which is why ``atol`` can stay this tight.
        """
        block = _block_containing("motional_heating_ops", "c_ops = [")
        quoted = re.search(r"heating_rate=([\d.e+]+)", block)
        assert quoted is not None
        n_dot = float(quoted[1])
        assert f"{n_dot:.0e}".startswith("1e+0"), n_dot

        hilbert = HilbertSpace(n_ions=1, n_modes=1, n_fock=60)
        ops = OperatorFactory(hilbert)
        number = ops.number(0)
        tlist = np.linspace(0, 2 / n_dot, 6)
        result = qutip.mesolve(
            0 * number,
            StateFactory(hilbert).ground_state(),
            tlist,
            c_ops=motional_heating_ops(ops, mode=0, heating_rate=n_dot),
            e_ops=[number],
            options={"atol": 1e-14, "rtol": 1e-12},
        )
        np.testing.assert_allclose(result.expect[0], n_dot * tlist, atol=1e-8)

    def test_rayleigh_rate_is_a_decoherence_rate(self) -> None:
        r"""``rate`` is $\Gamma_\text{el}$, not a scattering rate.

        $L = \sqrt{\Gamma_\text{el}/4}\,\sigma_z$ decays coherences
        at exactly $\Gamma_\text{el}/2$ (Uys et al., *Phys. Rev.
        Lett.* **105**, 200401 (2010), Eqs. 6 and 8), and
        $\Gamma_\text{el} \le \Gamma_\text{Rayleigh}$ always - by
        orders of magnitude for a clock qubit. Labelling the argument
        "elastic scattering" therefore invites an overestimate, so
        the comment has to name the decoherence rate.
        """
        block = _block_containing("rayleigh_scattering_op", "c_ops = [")
        comment = re.search(
            r"((?:^\s*#.*\n)+)\s*rayleigh_scattering_op\(", block, re.M
        )
        assert comment is not None, "rayleigh line lost its comment"
        assert "decoherence" in comment[1]

        quoted = re.search(
            r"rayleigh_scattering_op\(.*?rate=([\d.e+]+)", block
        )
        assert quoted is not None
        gamma_el = float(quoted[1])

        hilbert = HilbertSpace(n_ions=1, n_modes=1, n_fock=2)
        ops = OperatorFactory(hilbert)
        superposition = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(superposition, qutip.basis(2, 0))
        tlist = np.linspace(0, 4 / gamma_el, 5)
        result = qutip.mesolve(
            0 * ops.number(0),
            psi0,
            tlist,
            c_ops=[rayleigh_scattering_op(ops, 0, gamma_el)],
            options={"atol": 1e-12, "rtol": 1e-10},
        )
        coherence = np.array([
            abs(rho.ptrace([0]).full()[0, 1]) for rho in result.states
        ])
        np.testing.assert_allclose(
            coherence, 0.5 * np.exp(-gamma_el * tlist / 2), rtol=1e-6
        )


class TestReadmeScopeClaims:
    """Capability claims must match what the code actually offers."""

    def test_scope_section_exists_and_is_linked(self) -> None:
        assert "## Model scope and approximations" in README
        assert "(#model-scope-and-approximations)" in README

    def test_no_configurable_lamb_dicke_order_claim(self) -> None:
        """No runner path honours a Lamb-Dicke order setting.

        The config field was deleted because nothing read it, and
        neither ``carrier_hamiltonian`` nor ``ms_gate_hamiltonian``
        carries an eta^2 term, so the README may not advertise the
        order as configurable.
        """
        fields = {f.name for f in dataclasses.fields(SimulationConfig)}
        assert "lamb_dicke_order" not in fields
        assert "configurable Lamb-Dicke order" not in README

    def test_wineland_title_is_not_truncated(self) -> None:
        """The README title must match the one in trapping.md."""
        title = (
            "Experimental issues in coherent quantum-state "
            "manipulation of trapped atomic ions"
        )
        trapping = (REPO_ROOT / "docs/theory/trapping.md").read_text()
        assert title in " ".join(trapping.split())
        assert title in " ".join(README.split())


class TestPytestConfiguration:
    """``addopts`` applies to CI too, so it must not hide failures."""

    def test_does_not_stop_at_the_first_failure(self) -> None:
        """A branch that breaks many tests must report all of them."""
        assert "-x" not in ADDOPTS
        assert "--exitfirst" not in ADDOPTS
        assert not any(o.startswith("--maxfail") for o in ADDOPTS)

    def test_rejects_mistyped_markers(self) -> None:
        """Without this, ``@pytest.mark.slwo`` is silently inert."""
        assert "--strict-markers" in ADDOPTS


class TestWorkflow:
    """The workflow file is the source of truth for CI promises."""

    @staticmethod
    def _commands() -> list[str]:
        return [
            line.split("- run:", 1)[1].strip()
            for line in WORKFLOW.splitlines()
            if "- run:" in line
        ]

    def _sole_command(self, tool: str) -> str:
        matches = [c for c in self._commands() if tool in c]
        assert len(matches) == 1, f"{tool!r} in {len(matches)} steps"
        return matches[0]

    @pytest.mark.parametrize(
        "tool", ["ruff format --check", "ruff check", "ty check"]
    )
    def test_lints_tests_as_well_as_src(self, tool: str) -> None:
        """``tests/`` is code and carries its own lint config."""
        targets = self._sole_command(tool).split(tool, 1)[1].split()
        assert "src" in targets, tool
        assert "tests" in targets, tool
        assert "." not in targets, (
            "'.' pulls in docs/build_docs.py, whose pdoc import is "
            "unresolvable without the 'docs' extra"
        )

    def test_coverage_gate_is_skipped_on_a_failed_run(self) -> None:
        """Required once ``-x`` is gone from ``addopts``.

        Verified by hand: with three deliberately failing tests and
        no ``-x``, the run additionally reports "Required test
        coverage of 95% not reached. Total coverage: 0.00%", burying
        the real failures. ``--no-cov-on-fail`` suppresses exactly
        that.
        """
        command = self._sole_command("pytest")
        assert "--cov-fail-under" in command
        assert "--no-cov-on-fail" in command

    def test_readme_quotes_the_real_coverage_floor(self) -> None:
        command = self._sole_command("--cov-fail-under")
        floor = re.search(r"--cov-fail-under=(\d+)", command)
        assert floor is not None
        assert f"{floor[1]}% coverage floor" in README


def test_readme_test_count_is_not_stale() -> None:
    """The headline count may not fall below the functions on disk.

    It was "178 tests" against 355 collected for four months. The
    collected count always exceeds the number of ``def test``
    functions (parametrisation only adds cases), so an understatement
    is unambiguous, while the upper bound leaves room for
    parametrisation without pinning a number that has to be
    regenerated on every commit.
    """
    headline = re.search(r"^(\S+) tests, (\S+) line coverage\.", README, re.M)
    assert headline is not None, "testing section lost its headline"
    if headline[1] == "NNN":
        assert headline[2] == "MM%"
        pytest.skip("placeholders await a measured release build")

    defined = sum(
        len(re.findall(r"^\s*def test", path.read_text(), re.M))
        for path in (REPO_ROOT / "tests").glob("test_*.py")
    )
    claimed = int(headline[1].replace(",", ""))
    assert defined <= claimed <= 3 * defined
