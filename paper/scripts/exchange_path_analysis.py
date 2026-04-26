#!/usr/bin/env python
"""
Exchange Path Analysis with Goodenough-Kanamori Rules

This module identifies relevant exchange pathways in crystal structures
using the Goodenough-Kanamori (GK) rules, which predict the sign and
relative strength of superexchange interactions based on:

1. M-O-M bond angle (180° favours AFM, 90° favours FM)
2. Orbital occupancy of the magnetic ions (which d-orbitals are
   half-filled, filled, or empty in the crystal-field basis)
3. Nature of the bridging ligand

The implementation uses a lookup table for the orbital-dependent part
of the GK rules (Step 2 above), keyed on d-electron counts and
coordination environment.  This avoids DFT while still encoding the
essential physics that a pure angle-based heuristic misses.

All distance calculations use the minimum-image convention for periodic
boundary conditions, so exchange paths that cross unit cell boundaries
are correctly identified.

Convention: H = +J Si·Sj, so J > 0 is AFM, J < 0 is FM.

References
----------
- Goodenough, J.B., Phys. Rev. 100, 564 (1955)
- Kanamori, J., J. Phys. Chem. Solids 10, 87 (1959)
- Anderson, P.W., Phys. Rev. 115, 2 (1959)
"""

import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

# Default output directory (../figures for paper repo structure)
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent / 'figures'
DEFAULT_FIGURES_DIR.mkdir(exist_ok=True)


# ── Orbital filling tables ────────────────────────────────────────────
#
# High-spin octahedral crystal-field filling.
# Each entry: (t2g_up, t2g_dn, eg_up, eg_dn)
#   where "up" and "dn" count electrons with spin-up/down.
#   Half-filled means n_up > 0 and n_dn == 0 for that subshell.
#   Filled   means n_up == n_dn and both > 0.
#   Empty    means n_up == 0 and n_dn == 0.

_OCT_FILLING = {
    # d-count: (t2g_up, t2g_dn, eg_up, eg_dn)
    1: (1, 0, 0, 0),   # Ti3+, V4+
    2: (2, 0, 0, 0),   # V3+
    3: (3, 0, 0, 0),   # Cr3+, Mn4+
    4: (3, 0, 1, 0),   # Mn3+ (JT-active)
    5: (3, 0, 2, 0),   # Fe3+, Mn2+  — all half-filled
    6: (3, 1, 2, 0),   # Fe2+
    7: (3, 2, 2, 0),   # Co2+
    8: (3, 3, 2, 0),   # Ni2+  — t2g full, eg half-filled
    9: (3, 3, 2, 1),   # Cu2+  — one eg hole
}

# Tetrahedral: inverted splitting (e below t2)
_TET_FILLING = {
    1: (0, 0, 1, 0),
    2: (0, 0, 2, 0),
    3: (1, 0, 2, 0),
    4: (2, 0, 2, 0),
    5: (3, 0, 2, 0),
    6: (3, 0, 2, 1),
    7: (3, 0, 2, 2),
    8: (3, 1, 2, 2),
    9: (3, 2, 2, 2),
}


def _subshell_status(n_up: int, n_dn: int) -> str:
    """Classify a subshell as 'empty', 'half-filled', 'filled', or 'partial'.

    'partial' covers cases like d6 where t2g has 3 up + 1 down — neither
    purely half-filled nor full.
    """
    if n_up == 0 and n_dn == 0:
        return 'empty'
    if n_dn == 0 and n_up > 0:
        return 'half-filled'
    if n_up == n_dn and n_up > 0:
        return 'filled'
    return 'partial'  # e.g. t2g in d6 (3 up, 1 down)


def orbital_status(d_count: int, coord: str = 'oct') -> Dict[str, str]:
    """Return the status of the σ-bonding (eg) and π-bonding (t2g) subshells.

    Parameters
    ----------
    d_count : int
        Number of d-electrons.
    coord : str
        Coordination environment: 'oct' or 'tet'.

    Returns
    -------
    dict with keys 'eg' and 't2g', values from
    {'empty', 'half-filled', 'filled', 'partial'}.
    """
    table = _OCT_FILLING if coord == 'oct' else _TET_FILLING
    if d_count not in table:
        return {'eg': 'unknown', 't2g': 'unknown'}
    t2g_up, t2g_dn, eg_up, eg_dn = table[d_count]
    return {
        'eg':  _subshell_status(eg_up, eg_dn),
        't2g': _subshell_status(t2g_up, t2g_dn),
    }


# ── GK superexchange lookup ──────────────────────────────────────────
#
# For 180° superexchange (σ-pathway through bridging anion p-orbital):
#   The dominant virtual hop involves eg orbitals on both metals.
#   half-filled + half-filled → AFM (kinetic exchange)
#   half-filled + empty       → FM  (Hund's rule on ligand)
#   filled      + half-filled → FM  (hole transfer channel)
#   filled      + filled      → weakly AFM / negligible
#
# For 90° superexchange:
#   The two M-L bonds use orthogonal p-orbitals on the ligand,
#   so the exchange is FM (Hund's coupling between orthogonal paths).
#   This is weaker than the 180° channel.

_GK_SIGMA = {
    # (eg_status_1, eg_status_2) → (sign, strength_scale)
    ('half-filled', 'half-filled'): ('AFM', 1.0),
    ('half-filled', 'empty'):       ('FM',  0.5),
    ('empty',       'half-filled'): ('FM',  0.5),
    ('half-filled', 'filled'):      ('FM',  0.3),
    ('filled',      'half-filled'): ('FM',  0.3),
    ('filled',      'filled'):      ('AFM', 0.1),
    ('empty',       'empty'):       ('AFM', 0.0),
    ('half-filled', 'partial'):     ('AFM', 0.6),
    ('partial',     'half-filled'): ('AFM', 0.6),
    ('partial',     'partial'):     ('AFM', 0.4),
    ('filled',      'partial'):     ('FM',  0.2),
    ('partial',     'filled'):      ('FM',  0.2),
    ('filled',      'empty'):       ('AFM', 0.0),
    ('empty',       'filled'):      ('AFM', 0.0),
    ('partial',     'empty'):       ('FM',  0.2),
    ('empty',       'partial'):     ('FM',  0.2),
}

# Jahn-Teller-active d-electron counts (octahedral high-spin)
_JT_ACTIVE = {4, 9}  # d4 (Mn3+), d9 (Cu2+)


# ── Dataclass ─────────────────────────────────────────────────────────

@dataclass
class ExchangePath:
    """An exchange pathway between two magnetic sites."""
    site1: int
    site2: int
    site1_element: str
    site2_element: str
    distance: float

    # Image offset that gives the minimum-image distance
    image: Tuple[int, int, int] = (0, 0, 0)

    # Path geometry
    bridging_atoms: List[str] = field(default_factory=list)
    bridging_positions: List[np.ndarray] = field(default_factory=list)
    bond_angle: float = 180.0  # M-O-M angle

    # Orbital info used in GK prediction
    gk_channel: str = ""       # e.g. "eg-eg half-filled/half-filled"

    # Goodenough-Kanamori prediction
    predicted_sign: str = "AFM"
    predicted_strength: float = 1.0
    confidence: float = 0.5

    # Experimental validation
    validated: bool = False
    measured_J: Optional[float] = None
    measured_uncertainty: Optional[float] = None
    calibration_source: Optional[str] = None

    @property
    def path_type(self) -> str:
        if not self.bridging_atoms:
            return "direct"
        elif len(self.bridging_atoms) == 1:
            return "superexchange"
        else:
            return "super-superexchange"


# ── Ion configuration table ───────────────────────────────────────────

ION_CONFIGS = {
    'Fe3+': {'d_electrons': 5, 'spin': 2.5},
    'Fe2+': {'d_electrons': 6, 'spin': 2.0},
    'Fe':   {'d_electrons': 5, 'spin': 2.5},  # default to Fe3+ if no charge
    'Co2+': {'d_electrons': 7, 'spin': 1.5},
    'Co':   {'d_electrons': 7, 'spin': 1.5},
    'Ni2+': {'d_electrons': 8, 'spin': 1.0},
    'Ni':   {'d_electrons': 8, 'spin': 1.0},
    'Mn2+': {'d_electrons': 5, 'spin': 2.5},
    'Mn3+': {'d_electrons': 4, 'spin': 2.0},
    'Mn4+': {'d_electrons': 3, 'spin': 1.5},
    'Mn':   {'d_electrons': 5, 'spin': 2.5},
    'Cr3+': {'d_electrons': 3, 'spin': 1.5},
    'Cr':   {'d_electrons': 3, 'spin': 1.5},
    'Cu2+': {'d_electrons': 9, 'spin': 0.5},
    'Cu':   {'d_electrons': 9, 'spin': 0.5},
    'V3+':  {'d_electrons': 2, 'spin': 1.0},
    'V':    {'d_electrons': 2, 'spin': 1.0},
}


# ── Minimum-image utilities ──────────────────────────────────────────

def _min_image_delta(frac_delta: np.ndarray) -> np.ndarray:
    """Apply minimum image convention to a fractional displacement vector."""
    return frac_delta - np.round(frac_delta)


def _min_image_distance(frac_i: np.ndarray, frac_j: np.ndarray,
                        lattice: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the minimum-image distance between two sites.

    Returns
    -------
    distance : float
    cart_delta : ndarray, shape (3,)
        Cartesian displacement under minimum image.
    image : ndarray of int, shape (3,)
        The image offset applied (for bookkeeping).
    """
    frac_delta = frac_j - frac_i
    mic = _min_image_delta(frac_delta)
    image = np.round(frac_delta - mic).astype(int)
    cart_delta = mic @ lattice
    return float(np.linalg.norm(cart_delta)), cart_delta, tuple(image)


def _all_images_within(frac_i: np.ndarray, frac_j: np.ndarray,
                       lattice: np.ndarray, cutoff: float,
                       max_shell: int = 1) -> List[Tuple[float, np.ndarray, tuple]]:
    """Find all periodic images of site j within *cutoff* of site i.

    Searches over image offsets in [-max_shell, +max_shell]^3.

    Returns list of (distance, cart_delta, image_tuple).
    """
    results = []
    frac_delta_base = frac_j - frac_i
    for na in range(-max_shell, max_shell + 1):
        for nb in range(-max_shell, max_shell + 1):
            for nc in range(-max_shell, max_shell + 1):
                frac_delta = frac_delta_base + np.array([na, nb, nc], dtype=float)
                cart_delta = frac_delta @ lattice
                d = float(np.linalg.norm(cart_delta))
                if 0.01 < d < cutoff:  # exclude self-overlap
                    results.append((d, cart_delta, (na, nb, nc)))
    return results


# ── Main analyzer ─────────────────────────────────────────────────────

class GoodenoughKanamoriAnalyzer:
    """
    Analyze exchange pathways using Goodenough-Kanamori rules.

    The GK rules predict:
    1. 180° M-O-M with half-filled eg orbitals → strong AFM
    2. 90° M-O-M → weak FM (orthogonal p-orbitals on ligand)
    3. Mixed eg occupancy (half-filled / empty) → FM
    4. Direct exchange between same-orbital ions → AFM (kinetic)

    Convention: H = +J Si·Sj  (positive J = AFM).
    """

    CALIBRATION_RULES = (
        {
            'name': 'cuprate_180_superexchange',
            'description': 'Cu-O-Cu 180° superexchange (cuprate reference)',
            'path_type': 'superexchange',
            'elements': {'Cu'},  # only apply to Cu-O-Cu paths
            'ligands': ('O',),
            'angle_min': 150,
            'angle_max': 190,
            'base_strength': 130.0,  # meV — La2CuO4 NN
            'distance_ref': 3.78,    # Å — Cu-Cu in La2CuO4
            'distance_scale': 1.5,
            'confidence': 0.8,
            'force_sign': 'AFM',
        },
    )

    # ── construction ──────────────────────────────────────────────────

    def __init__(self, structure: Dict,
                 calibrations: Optional[List[Dict]] = None,
                 default_oxidation: Optional[Dict[str, str]] = None):
        """
        Parameters
        ----------
        structure : dict
            Crystal structure with keys 'lattice' (3×3), 'species' (list of str),
            'coords' (N×3 fractional).
        default_oxidation : dict, optional
            Map element symbol → oxidation-state key in ION_CONFIGS,
            e.g. {'Fe': 'Fe3+', 'Mn': 'Mn4+'}.  If not given, bare
            element symbols are used as lookup keys (which default to
            common oxidation states via ION_CONFIGS).
        """
        self.structure = structure
        self.lattice = np.array(structure['lattice'], dtype=float)
        self.species = list(structure['species'])
        self.coords = np.array(structure['coords'], dtype=float)
        self.n_atoms = len(self.species)

        self.default_oxidation = default_oxidation or {}

        self.calibration_rules = tuple(calibrations or self.CALIBRATION_RULES)

        # Identify site types
        self.magnetic_sites = []
        self.ligand_sites = []
        for i, elem in enumerate(self.species):
            if self._is_magnetic(elem):
                self.magnetic_sites.append(i)
            elif self._is_ligand(elem):
                self.ligand_sites.append(i)

    # ── element classification ────────────────────────────────────────

    @staticmethod
    def _is_magnetic(element: str) -> bool:
        magnetic_elements = {
            'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'Cu', 'V',
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        }
        return element in magnetic_elements

    @staticmethod
    def _is_ligand(element: str) -> bool:
        ligands = {'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'N'}
        return element in ligands

    # ── coordinate helpers ────────────────────────────────────────────

    def _cart_coords(self) -> np.ndarray:
        """Fractional → Cartesian."""
        return self.coords @ self.lattice

    def _ion_key(self, element: str) -> str:
        """Resolve element symbol to ION_CONFIGS key."""
        return self.default_oxidation.get(element, element)

    def _d_count(self, element: str) -> int:
        key = self._ion_key(element)
        cfg = ION_CONFIGS.get(key)
        if cfg is None:
            return -1
        return cfg['d_electrons']

    def _coord_number(self, site_idx: int, bond_cutoff: float = 2.8) -> int:
        """Count ligand neighbours within bond_cutoff (PBC-aware).

        Searches over periodic images so that a primitive cell with one
        metal and two oxygens still reports CN=4 or 6 correctly.
        """
        frac_i = self.coords[site_idx]
        count = 0
        for k in self.ligand_sites:
            frac_k = self.coords[k]
            images = _all_images_within(frac_i, frac_k, self.lattice,
                                        bond_cutoff)
            count += len(images)
        return count

    def _guess_coordination(self, site_idx: int) -> str:
        """Guess crystal-field environment from coordination number.

        CN ≤ 4 → 'tet' (tetrahedral) by default
        CN ≥ 5 → 'oct' (octahedral)

        Note: square-planar (CN=4) also has octahedral-like eg/t2g
        splitting, but distinguishing it from tetrahedral requires
        checking whether the ligands are coplanar.  For now we check
        planarity: if all ligand neighbours lie within ±0.5 Å of the
        metal's z-coordinate, call it square-planar → 'oct'.
        """
        frac_i = self.coords[site_idx]
        cart_i = frac_i @ self.lattice
        cn = self._coord_number(site_idx)

        if cn >= 5:
            return 'oct'
        if cn <= 3:
            return 'tet'

        # CN == 4: check planarity to distinguish tet vs square-planar
        ligand_z = []
        for k in self.ligand_sites:
            frac_k = self.coords[k]
            images = _all_images_within(frac_i, frac_k, self.lattice, 2.8)
            for _, delta, _ in images:
                cart_k = cart_i + delta
                ligand_z.append(cart_k[2])

        if ligand_z:
            z_spread = max(ligand_z) - min(ligand_z)
            if z_spread < 1.0:
                # Coplanar ligands → square-planar → oct-like splitting
                return 'oct'

        return 'tet'

    # ── path finding (PBC-aware) ──────────────────────────────────────

    def find_exchange_paths(self,
                           max_distance: float = 8.0,
                           max_bridging: int = 2,
                           bond_cutoff: float = 2.8) -> List[ExchangePath]:
        """Find all exchange paths between magnetic sites (PBC-aware).

        Parameters
        ----------
        max_distance : float
            Maximum M-M distance to consider (Å).
        max_bridging : int
            Maximum number of bridging ligands per path.
        bond_cutoff : float
            Maximum M-L distance to count as bonded (Å).
        """
        paths = []
        seen = set()  # avoid duplicating (i,j) pairs at same image

        for idx_i, site_i in enumerate(self.magnetic_sites):
            for idx_j, site_j in enumerate(self.magnetic_sites):
                if idx_j <= idx_i and site_i != site_j:
                    continue
                if site_i == site_j:
                    # same site — only consider non-zero images
                    pass

                frac_i = self.coords[site_i]
                frac_j = self.coords[site_j]

                # Find all images of site_j within max_distance of site_i
                images = _all_images_within(frac_i, frac_j, self.lattice,
                                           max_distance)
                for d_ij, cart_delta_ij, img in images:
                    # Canonical key to avoid double-counting i→j and j→i
                    key = (min(site_i, site_j), max(site_i, site_j), img)
                    rev_img = tuple(-x for x in img)
                    key_rev = (min(site_i, site_j), max(site_i, site_j), rev_img)
                    if key in seen or key_rev in seen:
                        continue
                    seen.add(key)

                    # Cartesian position of image-j
                    cart_i = frac_i @ self.lattice
                    cart_j_img = cart_i + cart_delta_ij

                    # Find bridging ligands
                    bridging = []
                    bridging_pos = []

                    for k in self.ligand_sites:
                        frac_k = self.coords[k]
                        # Check all images of the ligand
                        lig_images = _all_images_within(
                            frac_i, frac_k, self.lattice, bond_cutoff)
                        for d_ik, delta_ik, _ in lig_images:
                            cart_k_img = cart_i + delta_ik
                            d_kj = float(np.linalg.norm(cart_k_img - cart_j_img))
                            if d_kj < bond_cutoff:
                                bridging.append(self.species[k])
                                bridging_pos.append(cart_k_img)

                    # Prune to max_bridging closest to bond midpoint
                    if len(bridging) > max_bridging:
                        midpoint = (cart_i + cart_j_img) / 2
                        dists = [np.linalg.norm(p - midpoint) for p in bridging_pos]
                        order = np.argsort(dists)[:max_bridging]
                        bridging = [bridging[ii] for ii in order]
                        bridging_pos = [bridging_pos[ii] for ii in order]

                    # Bond angle M-L-M (use first bridging atom)
                    if bridging_pos:
                        v1 = cart_i - bridging_pos[0]
                        v2 = cart_j_img - bridging_pos[0]
                        cos_a = np.dot(v1, v2) / (
                            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    else:
                        angle = 180.0

                    path = ExchangePath(
                        site1=site_i,
                        site2=site_j,
                        site1_element=self.species[site_i],
                        site2_element=self.species[site_j],
                        distance=d_ij,
                        image=img,
                        bridging_atoms=bridging,
                        bridging_positions=bridging_pos,
                        bond_angle=angle,
                    )
                    self._apply_gk_rules(path)
                    paths.append(path)

        return paths

    # ── GK rule application ───────────────────────────────────────────

    def _apply_gk_rules(self, path: ExchangePath):
        """Apply Goodenough-Kanamori rules using orbital occupancy.

        The sign and strength depend on:
          - which d-orbitals are occupied (looked up from d-electron count)
          - the M-O-M bond angle (selects σ vs π pathway)
          - the path type (direct / superexchange / super-superexchange)
        """
        angle = path.bond_angle

        # ── Determine orbital status for both ions ────────────────────
        d1 = self._d_count(path.site1_element)
        d2 = self._d_count(path.site2_element)
        coord1 = self._guess_coordination(path.site1)
        coord2 = self._guess_coordination(path.site2)

        orb1 = orbital_status(d1, coord1) if d1 > 0 else {'eg': 'unknown', 't2g': 'unknown'}
        orb2 = orbital_status(d2, coord2) if d2 > 0 else {'eg': 'unknown', 't2g': 'unknown'}

        jt_active = (d1 in _JT_ACTIVE) or (d2 in _JT_ACTIVE)

        if path.path_type == "direct":
            # Direct exchange (no bridging ligand)
            # Kinetic exchange (Pauli exclusion) dominates when the same
            # orbitals overlap → AFM.  FM direct exchange requires
            # orthogonal orbitals, which is unusual for M-M paths.
            path.predicted_sign = "AFM"
            path.predicted_strength = 0.5 * np.exp(-path.distance / 3.0)
            path.confidence = 0.3
            path.gk_channel = f"direct d{d1}-d{d2}"

        elif path.path_type == "superexchange":
            # ── σ-pathway (eg orbitals, dominant near 180°) ───────────
            sigma_key = (orb1['eg'], orb2['eg'])
            sigma_sign, sigma_scale = _GK_SIGMA.get(
                sigma_key, ('AFM', 0.3))  # fallback

            # ── π-pathway (t2g orbitals, contributes at 90°) ─────────
            # 90° superexchange: orthogonal p-orbitals on ligand →
            # Hund's rule coupling → FM, weak.
            pi_sign = 'FM'
            pi_scale = 0.15

            # ── Interpolate σ and π based on angle ────────────────────
            # Near 180°: σ dominates.  Near 90°: π dominates.
            # Weight σ contribution by cos²(180°-θ), π by sin²(θ).
            sigma_weight = np.cos(np.radians(180 - angle)) ** 2
            pi_weight = np.sin(np.radians(angle)) ** 2

            J_sigma = sigma_scale * sigma_weight
            J_pi = pi_scale * pi_weight

            # Determine net sign
            if sigma_sign == 'AFM':
                # σ is AFM, π is FM — they compete
                J_net = J_sigma - J_pi
                if J_net >= 0:
                    path.predicted_sign = 'AFM'
                    path.predicted_strength = abs(J_net) * 10.0  # scale to meV
                else:
                    path.predicted_sign = 'FM'
                    path.predicted_strength = abs(J_net) * 10.0
            else:
                # σ is FM, π is FM — both ferromagnetic
                path.predicted_sign = 'FM'
                path.predicted_strength = (J_sigma + J_pi) * 10.0

            # Distance decay
            path.predicted_strength *= np.exp(-(path.distance - 3.5) / 2.0)

            # Confidence
            if sigma_key[0] == 'unknown' or sigma_key[1] == 'unknown':
                path.confidence = 0.3
            elif jt_active:
                path.confidence = 0.5  # orbital ordering could alter prediction
            else:
                path.confidence = 0.75

            path.gk_channel = (
                f"σ: eg {orb1['eg']}/{orb2['eg']} → {sigma_sign} "
                f"(scale {sigma_scale:.2f}), "
                f"angle {angle:.0f}°"
            )

        else:
            # Super-superexchange: two bridging ligands
            # Generally weak; sign depends on geometry but defaults to AFM
            # for the common edge-sharing / corner-sharing cases.
            path.predicted_sign = "AFM"
            path.predicted_strength = 0.5 * np.exp(-path.distance / 4.0)
            path.confidence = 0.25
            path.gk_channel = f"super-SE d{d1}-d{d2}, {len(path.bridging_atoms)} bridges"

        # Apply calibration overrides if any match
        self._apply_calibration_overrides(path)

    def _apply_calibration_overrides(self, path: ExchangePath):
        """Replace heuristic strengths with calibrated anchors when a rule matches."""
        if not self.calibration_rules:
            return

        ligand_counter = Counter(path.bridging_atoms)

        for rule in self.calibration_rules:
            if rule.get('path_type') and rule['path_type'] != path.path_type:
                continue

            # Element filter: if the rule specifies elements, both
            # magnetic ions must be in that set.
            rule_elements = rule.get('elements')
            if rule_elements:
                if (path.site1_element not in rule_elements or
                        path.site2_element not in rule_elements):
                    continue

            ligands = rule.get('ligands')
            if ligands:
                if Counter(ligands) != ligand_counter:
                    continue

            a = path.bond_angle
            if rule.get('angle_min') and a < rule['angle_min']:
                continue
            if rule.get('angle_max') and a > rule['angle_max']:
                continue

            base = rule.get('base_strength', path.predicted_strength)
            d_ref = rule.get('distance_ref', path.distance)
            d_scale = rule.get('distance_scale', 3.0)

            strength = base * np.exp(-(path.distance - d_ref) / d_scale)
            path.predicted_strength = max(strength, 0.0)
            path.confidence = rule.get('confidence', path.confidence)

            force_sign = rule.get('force_sign')
            if force_sign:
                path.predicted_sign = force_sign

            path.calibration_source = rule.get('name')
            break

    # ── ranking & clustering ──────────────────────────────────────────

    def rank_paths(self, paths: List[ExchangePath]) -> List[ExchangePath]:
        """Rank paths by predicted importance (strength × confidence)."""
        if not paths:
            return []
        for p in paths:
            p._score = p.predicted_strength * p.confidence
        return sorted(paths, key=lambda p: -p._score)

    def cluster_paths(self, paths: List[ExchangePath],
                      distance_tol: float = 0.05,
                      angle_tol: float = 1.0) -> List[List[ExchangePath]]:
        """Group symmetry-equivalent exchange paths by geometry."""
        clusters: List[List[ExchangePath]] = []

        def is_equivalent(p: ExchangePath, q: ExchangePath) -> bool:
            if p.path_type != q.path_type:
                return False
            if abs(p.distance - q.distance) > distance_tol:
                return False
            if abs(p.bond_angle - q.bond_angle) > angle_tol:
                return False
            if Counter(p.bridging_atoms) != Counter(q.bridging_atoms):
                return False
            return True

        for path in self.rank_paths(paths):
            placed = False
            for cluster in clusters:
                if is_equivalent(path, cluster[0]):
                    cluster.append(path)
                    placed = True
                    break
            if not placed:
                clusters.append([path])

        return clusters

    # ── Hamiltonian generation ────────────────────────────────────────

    def generate_hamiltonians(self,
                              paths: List[ExchangePath],
                              max_terms: int = 3) -> List[Dict]:
        """Generate candidate Hamiltonians from ranked/clustered paths.

        Convention: J > 0 is AFM.
        """
        clusters = self.cluster_paths(paths)
        representatives = [c[0] for c in clusters]

        candidates = []

        # Model 1: single strongest path
        if representatives:
            p = representatives[0]
            J1 = p.predicted_strength if p.predicted_sign == "AFM" else -p.predicted_strength
            candidates.append({
                'name': 'Single-J',
                'terms': {'J1': J1},
                'paths_used': [0],
                'multiplicities': [len(clusters[0])],
                'description': f'{p.path_type} via {p.bridging_atoms}, '
                               f'{p.gk_channel}',
                'prior': 0.3,
            })

        # Model 2: two strongest inequivalent paths
        if len(representatives) >= 2:
            p1, p2 = representatives[0], representatives[1]
            J1 = p1.predicted_strength if p1.predicted_sign == "AFM" else -p1.predicted_strength
            J2 = p2.predicted_strength if p2.predicted_sign == "AFM" else -p2.predicted_strength
            candidates.append({
                'name': 'Two-J',
                'terms': {'J1': J1, 'J2': J2},
                'paths_used': [0, 1],
                'multiplicities': [len(clusters[0]), len(clusters[1])],
                'description': f'J1: {p1.path_type} ({p1.gk_channel}), '
                               f'J2: {p2.path_type} ({p2.gk_channel})',
                'prior': 0.5,
            })

        # Model 3: two J plus phenomenological anisotropy
        if len(representatives) >= 2:
            p1, p2 = representatives[0], representatives[1]
            J1 = p1.predicted_strength if p1.predicted_sign == "AFM" else -p1.predicted_strength
            J2 = p2.predicted_strength if p2.predicted_sign == "AFM" else -p2.predicted_strength
            candidates.append({
                'name': 'Two-J + D',
                'terms': {'J1': J1, 'J2': J2, 'D': 0.1},
                'paths_used': [0, 1],
                'multiplicities': [len(clusters[0]), len(clusters[1])],
                'description': 'Two-J model with single-ion anisotropy placeholder',
                'prior': 0.2,
            })

        return candidates

    # ── experimental feedback ─────────────────────────────────────────

    def update_from_experiment(self,
                               path_idx: int,
                               measured_J: float,
                               uncertainty: float,
                               paths: List[ExchangePath]) -> Dict:
        """Record experimental validation against prediction.

        Parameters
        ----------
        path_idx : int
            Index into *paths* list.
        measured_J : float
            Experimentally determined J (positive = AFM).
        uncertainty : float
            Experimental uncertainty on J.

        Returns
        -------
        dict  —  feedback record suitable for GNN training.
        """
        path = paths[path_idx]
        path.validated = True
        path.measured_J = measured_J
        path.measured_uncertainty = uncertainty

        predicted_J = path.predicted_strength
        if path.predicted_sign == "FM":
            predicted_J = -predicted_J

        error = abs(measured_J - predicted_J)

        return {
            'path_type': path.path_type,
            'element_pair': (path.site1_element, path.site2_element),
            'bridging_atoms': path.bridging_atoms,
            'bond_angle': path.bond_angle,
            'distance': path.distance,
            'gk_channel': path.gk_channel,
            'predicted_J': predicted_J,
            'measured_J': measured_J,
            'uncertainty': uncertainty,
            'prediction_error': error,
            'relative_error': error / (abs(measured_J) + 0.1),
        }


# ══════════════════════════════════════════════════════════════════════
#  Demo: realistic perovskite-plane structure
# ══════════════════════════════════════════════════════════════════════

def make_perovskite_plane(a: float = 3.905,
                          c: float = 12.0,
                          buckle: float = 0.02,
                          element: str = 'Fe',
                          oxidation: str = 'Fe3+') -> Tuple[Dict, Dict]:
    """Create a single perovskite FeO2 (or MO2) plane.

    This is one Fe per unit cell with oxygens at the a-edge and b-edge
    midpoints, slightly buckled out of the basal plane along c.

    With a ≈ 3.9 Å:
      Fe-O ≈ a/2 ≈ 1.95 Å  (realistic)
      Fe-O-Fe angle ≈ 162° for buckle=0.02, 180° for buckle=0

    PBC is *required* to find Fe-O-Fe paths — the second Fe is across
    the cell boundary.

    Parameters
    ----------
    a : float
        In-plane lattice parameter (Å).
    c : float
        Out-of-plane lattice parameter (Å).
    buckle : float
        Fractional c-displacement of O atoms (±buckle).
        0 gives 180° Fe-O-Fe; 0.02 gives ~162°.
    element : str
        Magnetic element symbol.
    oxidation : str
        Key into ION_CONFIGS for that element.

    Returns
    -------
    structure : dict
    oxidation_map : dict
    """
    structure = {
        'name': f'Perovskite_{element}O2_plane',
        'lattice': np.array([
            [a,   0,   0],
            [0,   a,   0],
            [0,   0,   c],
        ]),
        'species': [element, 'O', 'O'],
        'coords': np.array([
            [0.0, 0.0, 0.0],           # M at origin
            [0.5, 0.0, buckle],         # O bridging along a, buckled +c
            [0.0, 0.5, -buckle],        # O bridging along b, buckled -c
        ]),
    }
    oxidation_map = {element: oxidation}
    return structure, oxidation_map


def make_2x2_supercell(a: float = 3.905,
                       c: float = 12.0,
                       buckle: float = 0.02,
                       element: str = 'Fe',
                       oxidation: str = 'Fe3+') -> Tuple[Dict, Dict]:
    """Create a 2×2×1 supercell of the perovskite MO2 plane.

    This puts four magnetic sites in the cell so that nearest-neighbour
    Fe-O-Fe paths exist within the cell (in addition to PBC paths).
    Useful for visualisation.
    """
    # Primitive cell fractional coords (in primitive basis)
    prim_species = [element, 'O', 'O']
    prim_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, buckle],
        [0.0, 0.5, -buckle],
    ])

    # Convert to supercell fractional coords: divide a,b by 2
    sc_base = prim_coords / np.array([2, 2, 1])

    species = []
    coords = []
    for ia in range(2):
        for ib in range(2):
            shift = np.array([ia * 0.5, ib * 0.5, 0.0])
            for sp, fc in zip(prim_species, sc_base):
                species.append(sp)
                coords.append(fc + shift)

    structure = {
        'name': f'Supercell_2x2_{element}O2_plane',
        'lattice': np.array([
            [2 * a, 0, 0],
            [0, 2 * a, 0],
            [0, 0, c],
        ]),
        'species': species,
        'coords': np.array(coords),
    }
    oxidation_map = {element: oxidation}
    return structure, oxidation_map


def demo_exchange_path_analysis():
    """Demonstrate exchange path analysis on a perovskite FeO2 plane."""

    print("=" * 60)
    print("Exchange Path Analysis with Goodenough-Kanamori Rules")
    print("=" * 60)

    # ── Build structure ───────────────────────────────────────────────
    # Primitive cell: 1 Fe, 2 O.  PBC finds Fe-O-Fe paths.
    structure, ox_map = make_perovskite_plane(
        a=3.905, buckle=0.02, element='Fe', oxidation='Fe3+')

    analyzer = GoodenoughKanamoriAnalyzer(
        structure, default_oxidation=ox_map)

    print(f"\nStructure: {structure['name']}")
    print(f"Lattice:   a = {analyzer.lattice[0,0]:.3f} Å, "
          f"c = {analyzer.lattice[2,2]:.3f} Å")
    print(f"Magnetic sites: {len(analyzer.magnetic_sites)} "
          f"({[structure['species'][i] for i in analyzer.magnetic_sites]})")
    print(f"Ligand sites:   {len(analyzer.ligand_sites)} "
          f"({[structure['species'][i] for i in analyzer.ligand_sites]})")

    # Check coordination number
    for ms in analyzer.magnetic_sites:
        cn = analyzer._coord_number(ms)
        coord_env = analyzer._guess_coordination(ms)
        dc = analyzer._d_count(structure['species'][ms])
        orb = orbital_status(dc, coord_env)
        print(f"  Site {ms} ({structure['species'][ms]}): CN={cn} → {coord_env}, "
              f"d{dc}, eg={orb['eg']}, t2g={orb['t2g']}")

    # ── Find paths ────────────────────────────────────────────────────
    paths = analyzer.find_exchange_paths(max_distance=7.0)

    print(f"\nFound {len(paths)} exchange pathway(s):")
    print("-" * 60)

    for i, path in enumerate(paths):
        print(f"\nPath {i+1}: {path.site1_element}({path.site1}) — "
              f"{path.site2_element}({path.site2})  image={path.image}")
        print(f"  Type:      {path.path_type}")
        print(f"  Distance:  {path.distance:.3f} Å")
        print(f"  Bridging:  {path.bridging_atoms if path.bridging_atoms else 'none (direct)'}")
        print(f"  Angle:     {path.bond_angle:.1f}°")
        print(f"  GK channel: {path.gk_channel}")
        print(f"  Prediction: {path.predicted_sign}, "
              f"|J| ≈ {path.predicted_strength:.2f} meV  "
              f"(conf {path.confidence:.0%})")
        if path.calibration_source:
            print(f"  Calibrated via: {path.calibration_source}")

    # ── Rank & cluster ────────────────────────────────────────────────
    clusters = analyzer.cluster_paths(paths)
    ranked = [c[0] for c in clusters]

    print("\n" + "=" * 60)
    print("Ranked Exchange Paths (by importance)")
    print("=" * 60)

    for i, path in enumerate(ranked[:5]):
        mult = len(clusters[i])
        print(f"  {i+1}. {path.path_type}: d={path.distance:.3f} Å, "
              f"θ={path.bond_angle:.1f}°, "
              f"{path.predicted_sign} |J|≈{path.predicted_strength:.2f} meV "
              f"(conf={path.confidence:.0%})")
        print(f"     GK: {path.gk_channel}")
        if mult > 1:
            print(f"     ({mult} equivalent paths)")
        if path.calibration_source:
            print(f"     calibrated via {path.calibration_source}")

    # ── Generate Hamiltonians ─────────────────────────────────────────
    candidates = analyzer.generate_hamiltonians(ranked)

    print("\n" + "=" * 60)
    print("Candidate Hamiltonians  [H = +J Si·Sj, J>0 = AFM]")
    print("=" * 60)

    for cand in candidates:
        print(f"\n  {cand['name']} (prior={cand['prior']:.0%}):")
        print(f"    {cand['description']}")
        for term, val in cand['terms'].items():
            print(f"    {term} = {val:+.2f} meV")

    # ── Simulated experimental validation ─────────────────────────────
    print("\n" + "=" * 60)
    print("Simulated Experimental Validation")
    print("=" * 60)

    TRUE_J1 = 5.0   # meV, AFM
    TRUE_J2 = 0.8   # meV, AFM

    print(f"\n  'Measured': J1 = +{TRUE_J1} meV (AFM), "
          f"J2 = +{TRUE_J2} meV (AFM)")

    if len(ranked) >= 2:
        fb1 = analyzer.update_from_experiment(0, TRUE_J1, 0.3, ranked)
        fb2 = analyzer.update_from_experiment(1, TRUE_J2, 0.2, ranked)

        print("\n  Feedback records:")
        print("  " + "-" * 50)
        for fb in [fb1, fb2]:
            print(f"    {fb['path_type']}, θ={fb['bond_angle']:.0f}°, "
                  f"GK: {fb['gk_channel']}")
            print(f"      predicted: {fb['predicted_J']:+.2f} meV")
            print(f"      measured:  {fb['measured_J']:+.2f} ± "
                  f"{fb['uncertainty']:.2f} meV")
            print(f"      error:     {fb['prediction_error']:.2f} meV "
                  f"({fb['relative_error']:.0%})")

    return analyzer, paths, ranked, candidates


# ══════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════

def create_exchange_path_figure(analyzer, paths, ranked, figures_dir: Path = DEFAULT_FIGURES_DIR):
    """Three-panel figure: structure, GK rules, path ranking.

    Panel A draws the 2×2 supercell in fractional coordinates,
    showing Fe-O-Fe superexchange paths through the bridging oxygens.
    """

    # ── Build 2×2 supercell and find its paths ────────────────────────
    a_prim = analyzer.lattice[0, 0]
    elem = analyzer.species[analyzer.magnetic_sites[0]]
    sc_struct, sc_ox = make_2x2_supercell(
        a=a_prim, buckle=0.02, element=elem)
    sc_analyzer = GoodenoughKanamoriAnalyzer(sc_struct, default_oxidation=sc_ox)
    sc_paths = sc_analyzer.find_exchange_paths(max_distance=7.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Panel A: structure with exchange paths (fractional coords) ────
    ax = axes[0]

    frac = sc_analyzer.coords  # fractional coords in supercell
    inv_lat = np.linalg.inv(sc_analyzer.lattice)

    path_colors = {
        'superexchange': '#2060cc',
        'direct': '#55a868',
        'super-superexchange': 'orange',
    }

    # Build the full set of visible atom positions, including periodic
    # images along cell edges.  An atom at frac=0 also appears at frac=1.
    vis_atoms = []  # (species, frac_x, frac_y, is_image)
    margin = 0.02   # tolerance for edge detection
    for i, (sp, fc) in enumerate(zip(sc_analyzer.species, frac)):
        x, y = fc[0], fc[1]
        # Original atom
        vis_atoms.append((sp, x, y, False))
        # If near lower edge, also draw at upper edge (and vice versa)
        near_x0 = abs(x) < margin
        near_y0 = abs(y) < margin
        near_x1 = abs(x - 1.0) < margin
        near_y1 = abs(y - 1.0) < margin
        if near_x0:
            vis_atoms.append((sp, x + 1.0, y, True))
        if near_y0:
            vis_atoms.append((sp, x, y + 1.0, True))
        if near_x0 and near_y0:
            vis_atoms.append((sp, x + 1.0, y + 1.0, True))
        if near_x1 and near_y0:
            vis_atoms.append((sp, x, y + 1.0, True))
        if near_x0 and near_y1:
            vis_atoms.append((sp, x + 1.0, y, True))

    # Draw exchange paths
    # Sort by strength so strongest draw on top
    sorted_paths = sorted(sc_paths, key=lambda p: p.predicted_strength)

    def _draw_path(ax, fi, fj_img, bp_frac_list, path_type, lw, c, alpha):
        """Draw one path (SE through bridging atom, or direct)."""
        if path_type == 'superexchange' and bp_frac_list:
            for bp2d in bp_frac_list:
                ax.plot([fi[0], bp2d[0]], [fi[1], bp2d[1]],
                        color=c, linewidth=lw, alpha=alpha, zorder=1,
                        solid_capstyle='round')
                ax.plot([bp2d[0], fj_img[0]], [bp2d[1], fj_img[1]],
                        color=c, linewidth=lw, alpha=alpha, zorder=1,
                        solid_capstyle='round')
        else:
            ax.plot([fi[0], fj_img[0]], [fi[1], fj_img[1]],
                    color=c, linewidth=lw, alpha=alpha, zorder=1)

    for p in sorted_paths:
        fi = frac[p.site1][:2].copy()
        fj = frac[p.site2][:2].copy()
        fj_img = fj + np.array(p.image[:2], dtype=float)

        lw = 1.0 + p.predicted_strength / 3
        c = path_colors.get(p.path_type, 'gray')
        alpha = 0.7 if p.path_type == 'superexchange' else 0.35

        # Compute bridging atom fractional positions
        bp_frac_list = []
        if p.path_type == 'superexchange' and p.bridging_positions:
            for bp_cart in p.bridging_positions:
                bp_frac_list.append((bp_cart @ inv_lat)[:2])

        # Draw the path as-is
        _draw_path(ax, fi, fj_img, bp_frac_list, p.path_type, lw, c, alpha)

        # If the path goes outside [0,1], also draw a +1-shifted copy
        # so bonds appear on the far cell edges
        shift = np.array([0.0, 0.0])
        needs_copy = False
        for dim in range(2):
            if fj_img[dim] < -0.01:
                shift[dim] = 1.0
                needs_copy = True
        if needs_copy:
            fi_s = fi + shift
            fj_s = fj_img + shift
            bp_s = [bp + shift for bp in bp_frac_list]
            _draw_path(ax, fi_s, fj_s, bp_s, p.path_type, lw, c, alpha)

    # Draw atoms (originals + periodic images)
    for sp, x, y, is_img in vis_atoms:
        alpha_atom = 0.5 if is_img else 1.0
        if sc_analyzer._is_magnetic(sp):
            ax.scatter(x, y, s=400, c='red', edgecolors='black',
                       linewidths=1.5, zorder=3, alpha=alpha_atom)
            ax.annotate(sp, (x, y), ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white',
                        zorder=4, alpha=alpha_atom)
        elif sc_analyzer._is_ligand(sp):
            ax.scatter(x, y, s=180, c='cornflowerblue', marker='s',
                       edgecolors='black', linewidths=1, zorder=2,
                       alpha=alpha_atom)
            ax.annotate('O', (x, y), ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold',
                        zorder=4, alpha=alpha_atom)

    # Unit cell outline
    uc = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    ax.plot(uc[:, 0], uc[:, 1], 'k-', lw=1.5, alpha=0.3)

    # Tick marks at fractional coordinate grid
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=path_colors['superexchange'], lw=2.5,
               label='Superexchange'),
        Line2D([0], [0], color=path_colors['direct'], lw=1.5,
               label='Direct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label=f'{elem}'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='cornflowerblue', markersize=8, label='O'),
    ]
    ax.legend(handles=legend_els, loc='upper right', fontsize=7,
              framealpha=0.9)
    ax.set_xlabel('a (frac.)')
    ax.set_ylabel('b (frac.)')
    ax.set_title('A) Exchange Pathways (2×2 supercell)')
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect('equal')

    # ── Panel B: GK rules visualisation ───────────────────────────────
    ax = axes[1]

    angles = np.linspace(80, 180, 100)
    afm = 10 * np.cos(np.radians(180 - angles)) ** 2
    fm = np.zeros_like(angles)
    mask = angles < 120
    fm[mask] = 2 * np.sin(np.radians(angles[mask])) ** 2

    ax.fill_between(angles, 0, afm, alpha=0.25, color='red', label='AFM (σ)')
    ax.fill_between(angles, 0, -fm, alpha=0.25, color='blue', label='FM (π)')
    ax.plot(angles, afm, 'r-', lw=2)
    ax.plot(angles, -fm, 'b-', lw=2)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(90, color='gray', ls='--', alpha=0.5)
    ax.axvline(180, color='gray', ls='--', alpha=0.5)

    ax.annotate('90°\n(FM)', xy=(90, -1.5), ha='center', fontsize=9)
    ax.annotate('180°\n(AFM)', xy=(180, 8), ha='center', fontsize=9)

    for p in ranked[:3]:
        ax.axvline(p.bond_angle, color='green', ls='-', alpha=0.7, lw=2)

    ax.set_xlabel('M–O–M Bond Angle (°)')
    ax.set_ylabel('Exchange Strength (arb.)')
    ax.set_title('B) Goodenough-Kanamori Rules')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(80, 185)
    ax.set_ylim(-3, 12)

    # ── Panel C: path importance ranking ──────────────────────────────
    ax = axes[2]

    n_show = min(6, len(ranked))
    y_pos = np.arange(n_show)

    strengths = [p.predicted_strength for p in ranked[:n_show]]
    labels = [f"{p.path_type}\n({p.bond_angle:.0f}°, {p.predicted_sign})"
              for p in ranked[:n_show]]
    bar_colors = ['#ff6b6b' if p.predicted_sign == 'AFM' else '#4ecdc4'
                  for p in ranked[:n_show]]

    ax.barh(y_pos, strengths, color=bar_colors, edgecolor='black', alpha=0.8)

    for idx, (s, p) in enumerate(zip(strengths, ranked[:n_show])):
        half_w = s * 0.3 * (1 - p.confidence)
        ax.plot([s - half_w, s + half_w], [idx, idx], 'k-', lw=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted |J| (meV)')
    ax.set_title('C) Ranked Exchange Paths')
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#ff6b6b', edgecolor='black', label='AFM'),
        Patch(facecolor='#4ecdc4', edgecolor='black', label='FM'),
    ], loc='lower right', fontsize=8)

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / 'exchange_path_analysis.png',
                dpi=150, bbox_inches='tight')
    fig.savefig(figures_dir / 'exchange_path_analysis.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {figures_dir / 'exchange_path_analysis.png'}")


def create_feedback_figure(figures_dir: Path = DEFAULT_FIGURES_DIR):
    """Closed-loop feedback diagram (unchanged from original)."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    boxes = [
        (1.5, 6, 2.0, 1.0, 'Crystal\nStructure',       '#E8E8E8'),
        (4.5, 6, 2.0, 1.0, 'GK Rules /\nGNN',           '#FFE4B5'),
        (7.5, 6, 2.0, 1.0, 'Candidate\nHamiltonians',   '#B0E0E6'),
        (7.5, 4, 2.0, 1.0, 'TAS-AI\nMeasurements',      '#98FB98'),
        (7.5, 2, 2.0, 1.0, 'Validated\nParameters',     '#DDA0DD'),
        (4.5, 2, 2.0, 1.0, 'Feedback\nDatabase',        '#F0E68C'),
        (1.5, 4, 2.0, 1.0, 'Updated\nPredictions',      '#FFA07A'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=10, fontweight='bold')

    arrow = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(3.5, 6), xytext=(2.5, 6), arrowprops=arrow)
    ax.annotate('', xy=(6.5, 6), xytext=(5.5, 6), arrowprops=arrow)
    ax.annotate('', xy=(7.5, 5), xytext=(7.5, 5.5), arrowprops=arrow)
    ax.annotate('', xy=(7.5, 3), xytext=(7.5, 3.5), arrowprops=arrow)
    ax.annotate('', xy=(5.5, 2), xytext=(6.5, 2), arrowprops=arrow)

    fb = dict(arrowstyle='->', color='green', lw=2.5)
    ax.annotate('', xy=(4.5, 3),  xytext=(4.5, 2.5), arrowprops=fb)
    ax.annotate('', xy=(2.5, 4),  xytext=(3.5, 4),   arrowprops=fb)
    ax.annotate('', xy=(1.5, 5),  xytext=(1.5, 4.5), arrowprops=fb)
    ax.annotate('', xy=(2.5, 6),  xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5,
                                connectionstyle='arc3,rad=0.3'))

    ax.text(5, 7.2, 'Closed-Loop Exchange Parameter Learning',
            fontsize=14, fontweight='bold', ha='center')
    for x, y, label in [(3, 6.3, '①'), (6, 6.3, '②'),
                         (8, 5.3, '③'), (8, 3.3, '④'), (6, 1.7, '⑤')]:
        ax.text(x, y, label, fontsize=12, ha='center')

    ax.annotate('Feedback\nLoop', xy=(2.5, 4.5), fontsize=10,
                color='green', fontweight='bold', ha='center')

    for x, y, text in [
        (1.5, 0.8, '① Identify\nmagnetic ions\n& ligands'),
        (4.5, 0.8, '② Apply GK rules\n(orbital + angle)\nto predict paths'),
        (7.5, 0.8, '③-④ Measure\n& discriminate\nmodels'),
    ]:
        ax.text(x, y, text, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / 'gnn_feedback_loop.png',
                dpi=150, bbox_inches='tight')
    fig.savefig(figures_dir / 'gnn_feedback_loop.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figures_dir / 'gnn_feedback_loop.png'}")


# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional output directory for exchange-path figures. Defaults to repo figures/.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    figures_dir = Path(args.save_dir) if args.save_dir else DEFAULT_FIGURES_DIR

    analyzer, paths, ranked, candidates = demo_exchange_path_analysis()

    print("\n" + "=" * 60)
    print("Creating Figures")
    print("=" * 60)

    create_exchange_path_figure(analyzer, paths, ranked, figures_dir=figures_dir)
    create_feedback_figure(figures_dir=figures_dir)

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
Goodenough-Kanamori rules identify WHICH exchange paths matter based on:
  1. Bond geometry (M-O-M angle selects σ vs π pathway)
  2. Orbital occupancy (eg half-filled/empty/filled determines sign)
  3. Bridging ligand type and coordination environment

This is MORE INFORMATIVE than:
  - N nearest neighbours (ignores geometry and chemistry)
  - Distance cutoff (ignores orbital physics)

Convention: H = +J Si·Sj, so J > 0 is AFM, J < 0 is FM.

The feedback loop allows experimental validation to improve future
predictions, creating a self-improving autonomous characterization system.
""")
