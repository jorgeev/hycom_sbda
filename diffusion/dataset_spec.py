"""Dataset descriptor: everything the training core needs to know about a store.

The diffusion code used to hardcode one store's layout -- ``dynamic/<var>``,
``static/ocean_mask``, ``coords/time``, ``meta/<var>/{mean,std}``, a sibling
``splits.json`` -- and one store's variable vocabulary. This module turns all of
that into data: a :class:`DatasetSpec` built from a small YAML descriptor, plus
a :class:`StoreFamily` that serves reads against a *virtual* time axis so a
family of stores (e.g. one zarr per month, tied together by a manifest) looks
like a single contiguous record.

Two things make a second dataset structurally different rather than merely
differently named, and both are handled here:

* **Multiple stores.** A manifest lists member stores in order, each with a
  ``global_*_offset`` into the concatenated axis. Reads spanning a boundary are
  stitched transparently.
* **Multiple cadences.** Truth may be hourly while its observations are daily or
  weekly, each in its own group with its own time coordinate. Every variable
  declares which cadence it lives on; :meth:`DatasetSpec.index_map` maps an index
  on the *base* cadence to the matching index on any other, **by timestamp**
  rather than by arithmetic, so it stays correct across irregular calendars and
  store boundaries alike.

``legacy_spec`` reproduces the original single-store, single-cadence assumptions
exactly, and is used whenever a config sets no ``dataset``. Every existing
config therefore keeps behaving identically.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml
import zarr


# ---------------------------------------------------------------------------
# Storage access: local POSIX, or any fsspec URL (s3://, gs://, file://)
# ---------------------------------------------------------------------------
# A path containing "://" is read through fsspec; anything else is a plain
# directory. That one sniff is the whole local-vs-cloud switch, so an on-prem
# store and an S3 store differ by nothing but the path. ``storage_options``
# (e.g. ``{"region_name": "us-east-1"}`` or ``{"anon": True}``) is forwarded to
# the backend; with an EC2 instance role, none are needed.
#
# fsspec is imported lazily inside each branch: a POSIX-only run -- which is
# every run on the cluster -- never needs it installed.


def _is_url(path) -> bool:
    return "://" in str(path)


def _join(base: str, name: str) -> str:
    """Join ``name`` onto ``base``, for a URL or a POSIX path alike."""
    return f"{str(base).rstrip('/')}/{name.lstrip('/')}"


def _parent(path: str) -> str:
    """Directory containing ``path``, for a URL or a POSIX path alike.

    Used for the manifest root. ``os.path.abspath`` cannot be used here: on an
    ``s3://`` URL it prepends the CWD and the member store paths -- which the
    manifest stores *relative* -- would silently resolve to nonsense.
    """
    if _is_url(path):
        return str(path).rstrip("/").rsplit("/", 1)[0]
    return os.path.dirname(os.path.abspath(path))


def _open_store(path, storage_options: dict | None = None):
    """Open a zarr store from a local path or an fsspec URL."""
    # zarr's own PathNotFoundError names the sub-path *inside* the store (''),
    # never the store location, which makes a wrong env var or a forgotten
    # bind-mount close to undiagnosable. Fail with the actual path instead.
    if _is_url(path):
        import fsspec
        try:
            return zarr.open(fsspec.get_mapper(str(path), **(storage_options or {})),
                             mode="r")
        except Exception as exc:
            raise FileNotFoundError(
                f"could not open remote zarr store {path!r} "
                f"(storage_options={storage_options!r}): {exc}"
            ) from exc
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"zarr store not found: {path!r} (cwd={os.getcwd()!r}). In Docker, "
            "check the store is bind-mounted into the container, or point the "
            "store env var (HYCOM_STORE / GULFSTREAM_MANIFEST) at an s3:// URL."
        )
    try:
        return zarr.open(path, mode="r")
    except Exception as exc:
        raise FileNotFoundError(
            f"{path!r} exists but is not a readable zarr group "
            "(no .zgroup/.zarray at its root?): " + str(exc)
        ) from exc


def _read_json(path, storage_options: dict | None = None):
    """Read + parse a JSON file from a local path or an fsspec URL."""
    if _is_url(path):
        import fsspec
        with fsspec.open(str(path), "r", **(storage_options or {})) as fh:
            return json.load(fh)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path!r} not found -- splits sidecars are read from inside the "
            "store directory and the manifest from beside its member stores; "
            "stage them together, not just the arrays."
        )
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Descriptor pieces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariableSpec:
    """One channel: which cadence it lives on and how to transform it."""
    name: str
    cadence: str
    log10: bool = False          # log10-transform before normalising
    units: str = ""


@dataclass(frozen=True)
class CadenceSpec:
    """One index space: a dynamic group plus the time coord and splits beside it."""
    name: str
    dynamic_group: str = "dynamic"
    time_path: str = "coords/time"
    splits_file: str = "splits.json"
    offset_key: str | None = None       # manifest field, e.g. "global_hour_offset"


# ---------------------------------------------------------------------------
# Store family: N stores presented as one virtual time axis per cadence
# ---------------------------------------------------------------------------
class StoreFamily:
    """Opens the member stores once; serves reads by *global* index.

    With a single member this is a thin pass-through, so the original read path
    stays one ``zarr`` call deep and its performance is unchanged.
    """

    def __init__(self, paths: list[str], storage_options: dict | None = None):
        if not paths:
            raise ValueError("StoreFamily needs at least one store path")
        self.paths = list(paths)
        self.storage_options = storage_options
        self.stores = [_open_store(p, storage_options) for p in self.paths]
        self._times: dict[str, np.ndarray] = {}
        self._bounds: dict[str, np.ndarray] = {}

    @property
    def single(self) -> bool:
        return len(self.stores) == 1

    # -- per-cadence geometry ---------------------------------------------
    def _lengths(self, cad: CadenceSpec) -> list[int]:
        out = []
        for s in self.stores:
            try:
                t = s[cad.time_path]
            except KeyError as e:
                raise KeyError(
                    f"cadence {cad.name!r}: no {cad.time_path!r} in this store"
                ) from e
            out.append(int(t.shape[0]))
        return out

    def bounds(self, cad: CadenceSpec) -> np.ndarray:
        """Cumulative start index of each member store, length ``n_stores + 1``."""
        if cad.name not in self._bounds:
            self._bounds[cad.name] = np.concatenate(
                [[0], np.cumsum(self._lengths(cad))]
            ).astype(np.int64)
        return self._bounds[cad.name]

    def n_steps(self, cad: CadenceSpec) -> int:
        return int(self.bounds(cad)[-1])

    def times(self, cad: CadenceSpec) -> np.ndarray:
        """Concatenated time coordinate (unix seconds) across the family."""
        if cad.name not in self._times:
            parts = [np.asarray(s[cad.time_path][:], dtype=np.float64)
                     for s in self.stores]
            self._times[cad.name] = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return self._times[cad.name]

    # -- reads --------------------------------------------------------------
    def read(self, cad: CadenceSpec, var: str, lo: int, hi: int) -> np.ndarray:
        """``[lo, hi)`` of ``<dynamic_group>/<var>`` on the virtual axis."""
        b = self.bounds(cad)
        if self.single:
            return np.asarray(self.stores[0][cad.dynamic_group][var][lo:hi])
        chunks = []
        for i, s in enumerate(self.stores):
            s0, s1 = int(b[i]), int(b[i + 1])
            a, z = max(lo, s0), min(hi, s1)
            if a < z:
                chunks.append(np.asarray(s[cad.dynamic_group][var][a - s0:z - s0]))
        if not chunks:
            raise IndexError(f"empty range [{lo}, {hi}) for {var!r}")
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def take(self, cad: CadenceSpec, var: str, idx: np.ndarray) -> np.ndarray:
        """Arbitrary (sorted) global indices of ``var`` -- used for stats sampling."""
        b = self.bounds(cad)
        if self.single:
            return np.asarray(self.stores[0][cad.dynamic_group][var].oindex[idx])
        out = []
        for i, s in enumerate(self.stores):
            s0, s1 = int(b[i]), int(b[i + 1])
            local = idx[(idx >= s0) & (idx < s1)] - s0
            if local.size:
                out.append(np.asarray(s[cad.dynamic_group][var].oindex[local]))
        return np.concatenate(out, axis=0)

    def shape(self, cad: CadenceSpec, var: str) -> tuple[int, ...]:
        per = self.stores[0][cad.dynamic_group][var].shape
        return (self.n_steps(cad),) + tuple(per[1:])

    def has_var(self, cad: CadenceSpec, var: str) -> bool:
        try:
            return var in self.stores[0][cad.dynamic_group]
        except KeyError:
            return False

    # -- static / coords / sidecars ----------------------------------------
    def static(self, group: str, name: str) -> np.ndarray:
        return np.asarray(self.stores[0][group][name][:])

    def has_static(self, group: str, name: str) -> bool:
        try:
            return name in self.stores[0][group]
        except KeyError:
            return False

    def coord(self, path: str) -> np.ndarray:
        return np.asarray(self.stores[0][path][:])

    def meta_stats(self, group: str, var: str) -> tuple[float, float]:
        g = self.stores[0][group][var]
        return float(g["mean"][()]), float(g["std"][()])

    def has_meta(self, group: str, var: str) -> bool:
        try:
            return var in self.stores[0][group]
        except KeyError:
            return False

    def sidecar(self, filename: str) -> Any:
        """Read a JSON sidecar from the *first* member store."""
        return _read_json(_join(self.paths[0], filename), self.storage_options)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    name: str
    stores: list[str]
    cadences: dict[str, CadenceSpec]
    base_cadence: str
    variables: dict[str, VariableSpec]
    manifest: dict | None = None
    static_group: str = "static"
    mask_var: str | None = "ocean_mask"
    # Some stores carry no usable land mask but do encode land as NaN in a
    # static field (gulfstream's `bathymetry`). Naming that field here ANDs
    # its finite-pixel pattern into the mask.
    mask_from_finite: str | None = None
    meta_group: str | None = "meta"
    stats_source: str = "meta"          # {"meta", "manifest", "compute"}
    splits_source: str = "store"        # {"store", "manifest"}
    log10_vars: frozenset[str] = frozenset()
    storage_options: dict | None = None   # forwarded to fsspec for s3:// stores
    _family: StoreFamily | None = field(default=None, repr=False, compare=False)
    _maps: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    # -- access ------------------------------------------------------------
    @property
    def family(self) -> StoreFamily:
        if self._family is None:
            self._family = StoreFamily(self.stores, self.storage_options)
        return self._family

    @property
    def base(self) -> CadenceSpec:
        return self.cadences[self.base_cadence]

    def var(self, name: str) -> VariableSpec:
        """The spec for ``name``; unlisted variables fall back to the base cadence.

        Falling back rather than raising keeps a descriptor terse: only variables
        that deviate from the default (another cadence, a log10 transform) have to
        be spelled out.
        """
        v = self.variables.get(name)
        if v is not None:
            return v
        return VariableSpec(name=name, cadence=self.base_cadence,
                            log10=name in self.log10_vars)

    def cadence_of(self, name: str) -> CadenceSpec:
        v = self.var(name)
        if v.cadence not in self.cadences:
            raise ValueError(
                f"variable {name!r} declares cadence {v.cadence!r}, "
                f"which is not one of {sorted(self.cadences)}"
            )
        return self.cadences[v.cadence]

    def is_log(self, name: str) -> bool:
        return self.var(name).log10

    def n_steps(self) -> int:
        """Length of the base-cadence axis: the index space samples live in."""
        return self.family.n_steps(self.base)

    # -- cross-cadence index mapping ---------------------------------------
    def index_map(self, cadence: str) -> np.ndarray | None:
        """Base-cadence index -> index on ``cadence``; ``None`` when it is the base.

        Mapping is by *timestamp* -- the last sample of the target cadence at or
        before the base timestamp -- rather than by arithmetic on the index. That
        is what makes it survive irregular calendars (this project's daily store
        has 15 real gaps) and store boundaries without special-casing either.

        Base steps that precede the target cadence's first sample map to ``-1``
        rather than being clamped to index 0. Clamping would hand the model an
        observation from the *future* -- gulfstream's hourly axis opens at 01:00
        while its daily fields are centred at 12:00, so the first 11 hours have
        no daily observation yet. Callers drop those steps instead.
        """
        if cadence == self.base_cadence:
            return None
        if cadence not in self._maps:
            t_base = self.family.times(self.base)
            t_other = self.family.times(self.cadences[cadence])
            idx = np.searchsorted(t_other, t_base, side="right") - 1
            self._maps[cadence] = np.minimum(idx, len(t_other) - 1).astype(np.int64)
        return self._maps[cadence]

    def first_mapped_step(self, name: str) -> int:
        """First base-cadence step at which variable ``name`` has an observation."""
        m = self.index_map(self.var(name).cadence)
        if m is None:
            return 0
        ok = np.flatnonzero(m >= 0)
        if ok.size == 0:
            raise ValueError(
                f"{self.name}: variable {name!r} (cadence "
                f"{self.var(name).cadence!r}) has no sample at or before any "
                "base-cadence step; the two time axes do not overlap."
            )
        return int(ok[0])

    def map_days(self, name: str, days) -> np.ndarray:
        """Base-cadence indices -> the indices to read for variable ``name``.

        May contain ``-1`` for steps before the variable's first sample; use
        :meth:`first_mapped_step` to exclude those up front.
        """
        days = np.asarray(days, dtype=np.int64)
        m = self.index_map(self.var(name).cadence)
        return days if m is None else m[days]

    # -- grid --------------------------------------------------------------
    def dx_m(self) -> float:
        """Grid spacing in metres, from ``coords/x``.

        Uses the median rather than the mean because one store writes ``coords/x``
        as float32, so consecutive differences scatter around the nominal spacing.
        """
        x = np.asarray(self.family.coord("coords/x"), dtype=np.float64)
        return float(np.median(np.abs(np.diff(x))))

    def ocean_mask(self) -> np.ndarray:
        """``(NY, NX)`` float32 mask, 1 = ocean.

        Combines the declared mask array (synthesised as all-ones when the store
        has none) with the finite-pixel pattern of ``mask_from_finite``. The
        second term matters: gulfstream's ``ocean_mask`` is uniformly 1.0 even
        though 17 pixels are NaN in every field, and without it those pixels
        enter the loss as zeros that the model is asked to reproduce.
        """
        fam = self.family
        if self.mask_var and fam.has_static(self.static_group, self.mask_var):
            m = np.asarray(fam.static(self.static_group, self.mask_var),
                           dtype=np.float32)
        else:
            ny = int(fam.coord("coords/y").shape[0])
            nx = int(fam.coord("coords/x").shape[0])
            m = np.ones((ny, nx), dtype=np.float32)
        if self.mask_from_finite:
            if not fam.has_static(self.static_group, self.mask_from_finite):
                raise KeyError(
                    f"{self.name}: mask_from_finite names "
                    f"{self.mask_from_finite!r}, absent from "
                    f"{self.static_group!r}"
                )
            ref = np.asarray(fam.static(self.static_group, self.mask_from_finite))
            m = m * np.isfinite(ref).astype(np.float32)
        return m

    # -- splits ------------------------------------------------------------
    def splits(self) -> dict[str, list[int]]:
        """``{"train": [...], "val": [...]}`` as base-cadence indices."""
        if self.splits_source == "manifest":
            return self._manifest_splits()
        return self._store_splits()

    def _store_splits(self) -> dict[str, list[int]]:
        fam = self.family
        cad = self.base
        if fam.single:
            s = fam.sidecar(cad.splits_file)
            return {k: sorted(int(i) for i in v)
                    for k, v in s.items() if isinstance(v, list)}
        # Multi-store: shift each member's local indices onto the virtual axis.
        b = fam.bounds(cad)
        out: dict[str, list[int]] = {}
        for i, p in enumerate(fam.paths):
            s = _read_json(_join(p, cad.splits_file), self.storage_options)
            for k, v in s.items():
                if isinstance(v, list):
                    out.setdefault(k, []).extend(int(j) + int(b[i]) for j in v)
        return {k: sorted(v) for k, v in out.items()}

    def _manifest_splits(self) -> dict[str, list[int]]:
        if not self.manifest:
            raise ValueError(f"{self.name}: splits_source='manifest' but no manifest")
        spec = self.manifest.get("splits", {})
        key = f"{self.base_cadence}_boundary"
        if key not in spec:
            raise ValueError(
                f"{self.name}: manifest splits has no {key!r} "
                f"(available: {sorted(spec)})"
            )
        n = self.n_steps()
        b = int(spec[key])
        return {"train": list(range(0, min(b, n))), "val": list(range(min(b, n), n))}

    # -- normalisation stats ------------------------------------------------
    def stored_stats(self, var: str) -> tuple[float, float] | None:
        """``(mean, std)`` from the store/manifest, or ``None`` if unavailable."""
        if self.stats_source == "manifest":
            if not self.manifest:
                return None
            e = self.manifest.get("normalization", {}).get(var)
            return (float(e["mean"]), float(e["std"])) if e else None
        if self.stats_source == "meta" and self.meta_group:
            if self.family.has_meta(self.meta_group, var):
                return self.family.meta_stats(self.meta_group, var)
        return None

    # -- cache namespacing --------------------------------------------------
    def cache_key(self, var: str) -> str:
        """Namespace a cache entry by dataset.

        The legacy dataset keeps the bare variable name so the existing
        ``_norm_cache.json`` / ``_clim_cache.npz`` stay valid and are never
        silently recomputed; anything else is prefixed, which is what stops two
        datasets that both call a channel ``sst`` from poisoning each other.
        """
        return var if self.name == LEGACY_NAME else f"{self.name}::{var}"

    # -- validation ---------------------------------------------------------
    def validate(self, needed: list[str], *, patch: int | None = None,
                 downsample: int | None = None) -> None:
        """Fail loudly at startup instead of deep inside the UNet."""
        fam = self.family
        ny, nx = self.ocean_mask().shape
        for v in needed:
            cad = self.cadence_of(v)
            if not fam.has_var(cad, v):
                raise KeyError(
                    f"{self.name}: variable {v!r} not found in group "
                    f"{cad.dynamic_group!r} (cadence {cad.name!r})"
                )
            shp = fam.shape(cad, v)
            if len(shp) != 3 or shp[1:] != (ny, nx):
                raise ValueError(
                    f"{self.name}: {v!r} has shape {shp}, expected (T, {ny}, {nx})"
                )
            if shp[0] != fam.n_steps(cad):
                raise ValueError(
                    f"{self.name}: {v!r} has {shp[0]} steps but cadence "
                    f"{cad.name!r} time axis has {fam.n_steps(cad)}"
                )
        if patch is not None and (patch > ny or patch > nx):
            raise ValueError(
                f"{self.name}: patch={patch} exceeds the {ny}x{nx} domain"
            )
        if downsample and downsample > 1 and (ny % downsample or nx % downsample):
            raise ValueError(
                f"{self.name}: full-frame inference needs H and W divisible by "
                f"{downsample} (the UNet's total downsample factor), got {ny}x{nx}"
            )

    def warn_climatology(self, clim_vars: list[str]) -> None:
        """Warn when a seasonal harmonic is asked of a record too short to fit one.

        The climatology in ``data.py`` is an annual + semiannual harmonic. Fitting
        that to well under a year is not merely noisy, it is unidentifiable: the
        fit absorbs whatever transient signal happens to be present and then
        subtracts it from the field being modelled.
        """
        if not clim_vars:
            return
        t = self.family.times(self.base)
        years = float(t[-1] - t[0]) / (365.2425 * 86400.0)
        if years < 2.0:
            print(f"[warn] {self.name}: clim_vars={clim_vars} requests a seasonal "
                  f"harmonic, but the record spans only {years:.2f} yr. Below ~2 yr "
                  "the annual/semiannual fit is not identifiable and will absorb "
                  "real signal; prefer clim_vars: [] (per-pixel time-mean).")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
LEGACY_NAME = "gom_nemo"

# The original hardcoded layout, as data.
_LEGACY_CADENCE = CadenceSpec(name="daily", dynamic_group="dynamic",
                              time_path="coords/time", splits_file="splits.json")


def legacy_spec(zarr_path: str, log10_vars: frozenset[str] = frozenset(),
                storage_options: dict | None = None) -> DatasetSpec:
    """The pre-refactor assumptions, expressed as a spec.

    Used whenever a config sets no ``dataset``, so every existing experiment and
    checkpoint resolves to exactly the behaviour it was trained with.
    """
    return DatasetSpec(
        name=LEGACY_NAME,
        stores=[zarr_path],
        cadences={"daily": _LEGACY_CADENCE},
        base_cadence="daily",
        variables={},
        static_group="static",
        mask_var="ocean_mask",
        meta_group="meta",
        stats_source="meta",
        splits_source="store",
        log10_vars=log10_vars,
        storage_options=storage_options,
    )


def load_dataset_spec(path: str, *, log10_vars: frozenset[str] = frozenset(),
                      storage_options: dict | None = None) -> DatasetSpec:
    """Build a :class:`DatasetSpec` from a descriptor YAML.

    Every string in the descriptor is env-var-expanded, so ``stores:`` and
    ``manifest:`` can be written ``${VAR:-<on-prem path>}`` and point at an
    s3:// URL in a container without a second copy of the file.
    """
    from .config import expandvars
    with open(path) as f:
        d = expandvars(yaml.safe_load(f) or {})

    manifest = None
    stores: list[str] = []
    if d.get("manifest"):
        mpath = d["manifest"]
        manifest = _read_json(mpath, storage_options)
        root = _parent(mpath)
        # Member order is the manifest's key order, which the builder writes in
        # chronological order; sorting the keys makes that explicit rather than
        # relying on dict insertion order surviving the JSON round-trip.
        for key in sorted(manifest.get("stores", {})):
            entry = manifest["stores"][key]
            p = entry["path"] if isinstance(entry, dict) else entry
            absolute = _is_url(p) or os.path.isabs(p)
            stores.append(p if absolute else _join(root, p))
    if d.get("stores"):
        raw = d["stores"]
        stores = [raw] if isinstance(raw, str) else list(raw)
    if not stores:
        raise ValueError(f"{path}: descriptor names neither 'stores' nor 'manifest'")

    cad_raw = d.get("cadences") or {"daily": {}}
    cadences = {k: CadenceSpec(name=k, **(v or {})) for k, v in cad_raw.items()}
    base = d.get("base_cadence") or next(iter(cadences))
    if base not in cadences:
        raise ValueError(f"{path}: base_cadence {base!r} not in {sorted(cadences)}")

    variables = {}
    for k, v in (d.get("variables") or {}).items():
        v = v or {}
        variables[k] = VariableSpec(
            name=k,
            cadence=v.get("cadence", base),
            log10=bool(v.get("log10", k in log10_vars)),
            units=v.get("units", ""),
        )

    return DatasetSpec(
        name=d.get("name") or os.path.splitext(os.path.basename(path))[0],
        stores=stores,
        cadences=cadences,
        base_cadence=base,
        variables=variables,
        manifest=manifest,
        static_group=d.get("static_group", "static"),
        mask_var=d.get("mask_var", "ocean_mask"),
        mask_from_finite=d.get("mask_from_finite"),
        meta_group=d.get("meta_group", "meta"),
        stats_source=d.get("stats_source", "meta"),
        splits_source=d.get("splits_source", "store"),
        log10_vars=log10_vars,
        storage_options=storage_options,
    )


def _find_descriptor(path: str) -> str:
    """Locate a descriptor named relative to the repo root.

    Configs name descriptors the same way they are passed on the command line
    (``diffusion/datasets/x.yaml``), which only resolves from the repo root.
    A checkpoint carries that string too, so sampling from anywhere else would
    otherwise fail long after training succeeded -- fall back to the path
    relative to this package's parent.
    """
    if os.path.exists(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt = os.path.join(root, path)
    if os.path.exists(alt):
        return alt
    alt2 = os.path.join(root, "diffusion", "datasets", os.path.basename(path))
    if os.path.exists(alt2):
        return alt2
    raise FileNotFoundError(f"dataset descriptor not found: {path!r}")


def resolve_spec(cfg) -> DatasetSpec:
    """The single entry point: a ``Config`` in, a ``DatasetSpec`` out.

    Also writes a single-store family's path back into ``cfg.zarr_path``, because
    the evaluation modules still open the store that way; that keeps them working
    untouched for every single-store dataset.
    """
    from .config import LOG10_VARS
    so = getattr(cfg, "storage_options", None) or None
    if getattr(cfg, "dataset", None):
        spec = load_dataset_spec(_find_descriptor(cfg.dataset),
                                 log10_vars=LOG10_VARS, storage_options=so)
        if len(spec.stores) == 1:
            cfg.zarr_path = spec.stores[0]
    else:
        spec = legacy_spec(cfg.zarr_path, log10_vars=LOG10_VARS, storage_options=so)
    return spec


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest(descriptors: list[str]) -> None:
    from .config import LOG10_VARS
    for path in descriptors:
        print(f"\n=== {path} ===")
        spec = load_dataset_spec(path, log10_vars=LOG10_VARS)
        fam = spec.family
        ny, nx = spec.ocean_mask().shape
        om = spec.ocean_mask()
        print(f"  name={spec.name}  stores={len(spec.stores)}  grid={ny}x{nx}  "
              f"dx={spec.dx_m():.2f} m  ocean_frac={float(om.mean()):.6f} "
              f"({int((om <= 0.5).sum())} masked px)")
        for cname, cad in spec.cadences.items():
            t = fam.times(cad)
            flag = " (base)" if cname == spec.base_cadence else ""
            print(f"  cadence {cname:<7}{flag} n={fam.n_steps(cad):>6}  "
                  f"bounds={fam.bounds(cad).tolist()}")
            if t.size > 1:
                dt = np.diff(t)
                uniq, cnt = np.unique(dt, return_counts=True)
                if uniq.size > 1:
                    print(f"    irregular dt: "
                          + ", ".join(f"{int(u)}s x{int(c)}" for u, c in zip(uniq, cnt)))

        # cross-cadence mapping must land within one step of the base timestamp
        t_base = fam.times(spec.base)
        for cname, cad in spec.cadences.items():
            if cname == spec.base_cadence:
                continue
            m = spec.index_map(cname)
            t_other = fam.times(cad)
            ok = m >= 0
            lag = t_base[ok] - t_other[m[ok]]
            step = float(np.median(np.diff(t_other))) if t_other.size > 1 else np.inf
            assert (lag >= 0).all(), f"{cname}: mapped a future sample"
            assert lag.max() <= step * 1.5 + 1, (
                f"{cname}: max lag {lag.max():.0f}s exceeds one step ({step:.0f}s)")
            # monotone, and every mapped index is a real one
            assert (np.diff(m[ok]) >= 0).all(), f"{cname}: map is not monotone"
            print(f"    {spec.base_cadence}->{cname}: lag 0..{lag.max():.0f}s "
                  f"(step {step:.0f}s), unmapped head={int((~ok).sum())} steps  OK")

        sp = spec.splits()
        print("  splits: " + ", ".join(
            f"{k}={len(v)}" + (f" [{v[0]}..{v[-1]}]" if v else " []")
            for k, v in sorted(sp.items())))

        # A window straddling a store boundary must read exactly what the two
        # member stores hold, in order -- the failure this would otherwise hide
        # is silent duplication or omission of a frame at the seam.
        if len(spec.stores) > 1:
            fam2, cad = spec.family, spec.base
            b = int(fam2.bounds(cad)[1])
            var = next(iter(v for v in spec.variables
                            if spec.var(v).cadence == spec.base_cadence))
            lo, hi = b - 2, b + 2
            got = fam2.read(cad, var, lo, hi)
            left = np.asarray(fam2.stores[0][cad.dynamic_group][var][lo:b])
            right = np.asarray(fam2.stores[1][cad.dynamic_group][var][0:hi - b])
            want = np.concatenate([left, right], axis=0)
            assert got.shape == want.shape, f"seam shape {got.shape} != {want.shape}"
            assert np.array_equal(np.nan_to_num(got), np.nan_to_num(want)), \
                "seam read does not match the per-store reads"
            t = fam2.times(cad)[lo:hi]
            dt = np.diff(t)
            assert (dt > 0).all(), "time is not increasing across the seam"
            print(f"  seam @{b} ({var}): {got.shape} contiguous, "
                  f"dt={[int(x) for x in dt]}  OK")

        declared = sorted(spec.variables) or None
        if declared:
            spec.validate(declared, patch=128, downsample=8)
            print(f"  validate({len(declared)} declared vars, patch=128, /8) OK")
            for v in declared[:4]:
                st = spec.stored_stats(v)
                print(f"    {v:<20} cadence={spec.var(v).cadence:<7} "
                      f"stats={'(%.4g, %.4g)' % st if st else 'None'}")
    print("\n[selftest] OK")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("descriptors", nargs="*",
                    default=["diffusion/datasets/gom_nemo.yaml",
                             "diffusion/datasets/gulfstream.yaml"])
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.descriptors or ["diffusion/datasets/gom_nemo.yaml"])


if __name__ == "__main__":
    main()
