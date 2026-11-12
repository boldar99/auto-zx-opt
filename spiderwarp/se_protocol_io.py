"""Versioned, non-pickle persistence for adaptive SE protocols.

The manifest commits to the exact executable policy, readouts and code. Loading
does not synthesize a policy or enumerate circuit faults. A stored verification
report is provenance, not a new verification of the loaded program.
"""
from __future__ import annotations

from dataclasses import asdict
from collections.abc import Mapping
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import stim

from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptiveVerificationReport, ExtractionAction, FixedCSSDecoder,
    FTContract, PolicyNode,
)
from spiderwarp._se_circuit import PauliSpace, Readout, SEGadget
from spiderwarp.csscode import CSSCode

SCHEMA_VERSION = 1


class ArrayOutcomes(Mapping):
    """Read-only sorted transition view; no Python object per stored edge."""
    __slots__ = ("rows",)

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return ((int(f), int(s)) for _, f, s, _ in self.rows)

    def __getitem__(self, key):
        flags, syndrome = key
        rows = self.rows
        left = int(np.searchsorted(rows[:, 1], flags, side="left"))
        right = int(np.searchsorted(rows[:, 1], flags, side="right"))
        i = left + int(np.searchsorted(rows[left:right, 2], syndrome))
        if i >= right or rows[i, 2] != syndrome:
            raise KeyError(key)
        return int(rows[i, 3])

    def items(self):
        return (((int(f), int(s)), int(child)) for _, f, s, child in self.rows)

    def values(self):
        return (int(i) for i in self.rows[:, 3])


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def versions():
    result = {}
    for name in ("spiderwarp", "numpy", "stim", "guppylang", "hugr", "selene-sim"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_protocol(protocol, directory):
    """Save an executable bundle without importing Guppy or compiling HUGR."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if protocol.code.n > 63:
        raise NotImplementedError("Bundle frame masks support at most 63 data qubits")
    policy = protocol.policy
    names = sorted(policy.actions)
    files = {}

    def write(name, data):
        data = data.encode() if isinstance(data, str) else data
        # Content-addressed filenames keep the previous manifest valid if a
        # rebuild is interrupted before the new manifest is committed.
        sha = hashlib.sha256(data).hexdigest()
        stem, suffix = name.rsplit(".", 1)
        relative = f"{stem}_{sha[:16]}.{suffix}"
        atomic_write(directory / relative, data)
        files[relative] = sha
        return relative

    code_path = write("code.json", canonical_json(protocol.code.to_dict()))
    actions = []
    for i, name in enumerate(names):
        g = policy.actions[name].gadget
        actions.append(dict(name=name, basis=g.basis, distance=g.distance,
                            circuit=write(f"action_{i}.stim", str(g.circuit)),
                            readout=asdict(g.readout),
                            measurement_map=sorted(g.measurement_map.items())))
    kinds = {"reject": 0, "accept": 1, "extract": 2, "frame": 3}
    mask = (1 << protocol.code.n) - 1
    nodes, edges = [], []
    for i, node in enumerate(policy.nodes):
        nodes.append((kinds[node.kind], names.index(node.action) if node.action else -1,
                      node.frame & mask, node.frame >> protocol.code.n,
                      node.next if node.next is not None else -1))
        edges.extend((i, flags, syndrome, child)
                     for (flags, syndrome), child in sorted(node.children.items()))
    from io import BytesIO
    buf = BytesIO()
    np.savez_compressed(buf, nodes=np.asarray(nodes, dtype="<i8").reshape(-1, 5),
                        edges=np.asarray(edges, dtype="<i8").reshape(-1, 4))
    policy_path = write("policy.npz", buf.getvalue())
    core = dict(schema_version=SCHEMA_VERSION, code=code_path, policy=policy_path,
                root=policy.root, actions=actions, files=files,
                contract=asdict(policy.contract), decoder_hash=policy.decoder.fingerprint)
    manifest = dict(core, protocol_hash=digest(core), versions=versions(),
                    verification=asdict(policy.report), optimal=policy.optimal,
                    search_states=policy.search_states,
                    search_limit_reached=policy.search_limit_reached,
                    peak_qubit_limit=policy.peak_qubit_limit,
                    lower_bound_cnots=policy.lower_bound_cnots)
    path = directory / "protocol.json"
    atomic_write(path, json.dumps(manifest, indent=2) + "\n")
    return path


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "protocol.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported SE protocol schema version")
    core_keys = ("schema_version", "code", "policy", "root", "actions", "files", "contract", "decoder_hash")
    if digest({k: manifest[k] for k in core_keys}) != manifest["protocol_hash"]:
        raise ValueError("Protocol manifest hash mismatch")
    referenced = {manifest["code"], manifest["policy"],
                  *(a["circuit"] for a in manifest["actions"])}
    if not referenced <= manifest["files"].keys():
        raise ValueError("Manifest references an unhashed protocol file")
    for name, sha in manifest["files"].items():
        target = (path.parent / name).resolve()
        if target.parent != path.parent or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Protocol file hash mismatch: {name}")
    return path, manifest


def load_protocol(path, *, verify=False):
    """Load a bundle, optionally re-running the expensive independent checker."""
    from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol

    path, m = read_manifest(path)
    raw = json.loads((path.parent / m["code"]).read_text())
    n = raw["n"]
    if type(n) is not int or not 0 < n <= 63:
        raise ValueError("Invalid data block size")
    arrays = {k: np.asarray(raw[k]).reshape(-1, n)
              for k in ("H_x", "H_z", "L_x", "L_z")}
    if any(np.any((a != 0) & (a != 1)) for a in arrays.values()):
        raise ValueError("Code matrices must be binary")
    arrays = {k: a.astype(np.uint8) for k, a in arrays.items()}
    code = CSSCode(n, raw["k"], raw["d"], **arrays,
                   is_self_dual=raw["is_self_dual"], name=raw.get("name"))
    space = PauliSpace(code)
    contract = FTContract(**m["contract"])
    decoder = FixedCSSDecoder(space, contract)
    if decoder.fingerprint != m["decoder_hash"]:
        raise ValueError("Stored decoder differs from the installed decoder")
    actions = {}
    for a in m["actions"]:
        circuit = stim.Circuit.from_file(path.parent / a["circuit"])
        readout = Readout(*(tuple(tuple(p) for p in a["readout"][k])
                            for k in ("flags", "syndromes")))
        if a["basis"] not in {"X", "Z"} or a["distance"] < 1 or a["name"] in actions:
            raise ValueError("Invalid extraction action")
        if any(not 0 <= i < circuit.num_measurements
               for group in (readout.flags, readout.syndromes) for parity in group for i in parity):
            raise ValueError("Readout references an unavailable measurement")
        checks = code.H_x if a["basis"] == "X" else code.H_z
        if len(readout.syndromes) != len(checks):
            raise ValueError("Readout has the wrong syndrome width")
        gadget = SEGadget(a["distance"], a["basis"], circuit, readout, space,
                          dict(a["measurement_map"]))
        _ = gadget.operations  # Validate gate set and ancilla lifetimes.
        actions[a["name"]] = ExtractionAction(a["name"], gadget)
    with np.load(path.parent / m["policy"], allow_pickle=False) as data:
        rows, edges = data["nodes"], data["edges"]
    if (rows.ndim != 2 or rows.shape[1] != 5 or edges.ndim != 2 or edges.shape[1] != 4
            or rows.dtype.kind != "i" or edges.dtype.kind != "i" or not len(rows)):
        raise ValueError("Malformed policy arrays")
    count = len(rows)
    names = list(actions)
    if len(edges):
        if (np.any(edges < 0) or np.any(edges[:, 0] >= count) or np.any(edges[:, 3] >= count)
                or np.any(rows[edges[:, 0], 0] != 2)):
            raise ValueError("Invalid policy edge")
        a, b = edges[:-1], edges[1:]
        ordered = ((b[:, 0] > a[:, 0]) | ((b[:, 0] == a[:, 0]) &
                   ((b[:, 1] > a[:, 1]) | ((b[:, 1] == a[:, 1]) & (b[:, 2] > a[:, 2])))))
        if not np.all(ordered):
            raise ValueError("Duplicate or unsorted policy outcomes")
    rows.flags.writeable = edges.flags.writeable = False
    offsets = np.concatenate(([0], np.cumsum(np.bincount(edges[:, 0], minlength=count))))
    nodes = []
    for i, row in enumerate(rows):
        kind, action, x, z, following = map(int, row)
        if not (0 <= kind <= 3 and 0 <= x < 1 << n and 0 <= z < 1 << n):
            raise ValueError("Invalid policy node/frame")
        if kind == 2 and not 0 <= action < len(names):
            raise ValueError("Invalid policy action")
        if kind == 3 and not 0 <= following < count:
            raise ValueError("Invalid frame successor")
        children = ArrayOutcomes(edges[offsets[i]:offsets[i + 1]])
        if kind == 2 and len(children):
            g = actions[names[action]].gadget
            if (int(children.rows[:, 1].max()).bit_length() > g.num_flags
                    or int(children.rows[:, 2].max()).bit_length() > len(g.readout.syndromes)):
                raise ValueError("Policy outcome exceeds readout width")
        nodes.append(PolicyNode(("reject", "accept", "extract", "frame")[kind],
                                names[action] if kind == 2 else None,
                                x | (z << n), children, following if kind == 3 else None))
    if nodes[0].kind != "reject" or not 0 <= m["root"] < count:
        raise ValueError("Policy must reject unknown outcomes at node zero")
    # Iterative DFS avoids Python recursion limits on large valid graphs.
    colors = bytearray(count)
    stack = [(m["root"], False)]
    while stack:
        i, leaving = stack.pop()
        if leaving:
            colors[i] = 2
            continue
        if colors[i] == 1:
            raise ValueError("Policy does not terminate")
        if colors[i] == 2:
            continue
        colors[i] = 1
        stack.append((i, True))
        node = nodes[i]
        stack.extend((j, False) for j in
                     ([node.next] if node.kind == "frame" else set(node.children.values())))
    report = dict(m["verification"])
    report["contract"] = FTContract(**report["contract"])
    report = AdaptiveVerificationReport(**report)
    peak = max((actions[node.action].qubits for i, node in enumerate(nodes)
                if colors[i] and node.kind == "extract"), default=code.n)
    if report.peak_qubits < peak or report.decoder_hash != decoder.fingerprint or report.contract != contract:
        raise ValueError("Verification report is inconsistent with the protocol")
    policy = AdaptivePolicy(actions, contract, decoder, tuple(nodes), m["root"], report,
                            m["optimal"], m["search_states"], m["peak_qubit_limit"],
                            m["lower_bound_cnots"], m["search_limit_reached"])
    protocol = AdaptiveSEProtocol(code, policy)
    if verify:
        protocol.verify()
    return protocol
