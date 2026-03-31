from __future__ import annotations

from typing import Iterable, Sequence


class Pauli:
    """
    Lightweight Pauli in binary symplectic (z, x) form.
    (z,x) per qubit: (0,0)->I, (1,0)->Z, (0,1)->X, (1,1)->Y
    """
    _to_str = {
        (0, 0): "I",
        (1, 0): "Z",
        (0, 1): "X",
        (1, 1): "Y",
    }
    _from_char = {
        "I": (0, 0),
        "Z": (1, 0),
        "X": (0, 1),
        "Y": (1, 1),
    }

    def __init__(self, z: list[int], x: list[int]):
        if len(z) != len(x):
            raise ValueError("z and x must have same length")
        self.z = [v & 1 for v in z]
        self.x = [v & 1 for v in x]

    @classmethod
    def identity(cls, n: int) -> Pauli:
        return cls([0] * n, [0] * n)

    @classmethod
    def from_str(cls, s: str) -> Pauli:
        z = []
        x = []
        for ch in s:
            if ch.upper() not in cls._from_char:
                raise ValueError(f"Unknown Pauli char: {ch}")
            zz, xx = cls._from_char[ch.upper()]
            z.append(zz)
            x.append(xx)
        return cls(z, x)

    @classmethod
    def from_lists(cls, z: list[int], x: list[int]) -> Pauli:
        return cls(z, x)

    @classmethod
    def from_z_list(cls, z: list[int]) -> Pauli:
        return cls(z, [0] * len(z))

    @classmethod
    def from_x_list(cls, x: list[int]) -> Pauli:
        return cls([0] * len(x), x)

    def to_lists(self) -> tuple[list[int], list[int]]:
        return list(self.z), list(self.x)

    def __str__(self) -> str:
        return "".join(self._to_str[(z, x)] for z, x in zip(self.z, self.x))

    def __repr__(self) -> str:
        return f"Pauli('{str(self)}')"

    def __len__(self) -> int:
        return len(self.z)

    def weight(self) -> int:
        return sum(1 for a, b in zip(self.z, self.x) if (a | b))

    def support(self) -> list[int]:
        return [i for i, (a, b) in enumerate(zip(self.z, self.x)) if (a | b)]

    def tensor(self, other: Pauli) -> Pauli:
        """Concatenate qubits: this ⊗ other (self's qubits first)."""
        return Pauli(self.z + other.z, self.x + other.x)

    def _symplectic_inner(self, other: Pauli) -> int:
        # s = sum_j ( x_self[j]*z_other[j] - z_self[j]*x_other[j] )
        s = 0
        for a_z, a_x, b_z, b_x in zip(self.z, self.x, other.z, other.x):
            s += (a_x * b_z) - (a_z * b_x)
        return s

    def commutes_with(self, other: Pauli) -> bool:
        return (self._symplectic_inner(other) % 2) == 0

    def __mul__(self, other: Pauli) -> tuple[complex, Pauli]:
        """
        Multiply Paulis: returns (phase, Pauli_result).
        phase in {1, 1j, -1, -1j}.
        Result Pauli = self * other (binary vectors XOR).
        """
        if len(self) != len(other):
            raise ValueError("Paulis must act on same number of qubits to multiply")
        s = self._symplectic_inner(other)
        s_mod4 = s % 4
        phase = (1j) ** s_mod4
        z_res = [(a ^ b) for a, b in zip(self.z, other.z)]
        x_res = [(a ^ b) for a, b in zip(self.x, other.x)]
        return phase, Pauli(z_res, x_res)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pauli):
            return False
        return self.z == other.z and self.x == other.x

    def __getitem__(self, i: int) -> str:
        return self._to_str[(self.z[i], self.x[i])]

    def copy(self):
        return Pauli(self.z, self.x)


class CSSEncoder:
    """
    Very small CSS code container.
    - stabs: tuple[Pauli]  (stabilizer generators, both X- and Z-type encoded as Paulis)
    - x_logicals: tuple[Pauli]  (logical X operators)
    - z_logicals: tuple[Pauli]  (logical Z operators)
    """

    def __init__(self, stabs: list[Pauli], x_logicals: list[Pauli], z_logicals: list[Pauli]):
        # convert to tuples for immutability
        self.stabs = stabs
        self.x_logicals = x_logicals
        self.z_logicals = z_logicals
        self.n = len(stabs[0])

    @classmethod
    def from_matrices(cls, H_x: list[list[int]], H_z: list[list[int]], L_x: list[list[int]],
                      L_z: list[list[int]]) -> CSSEncoder:
        """
        Build CSSEncoder from binary matrices (lists of lists).
        H_x: each row is an X-type stabilizer (binary vector)
        H_z: each row is a Z-type stabilizer (binary vector)
        L_x: each row is a logical X (binary vector corresponding to Z=0, X=row)
        L_z: each row is a logical Z (binary vector corresponding to Z=row, X=0)
        """
        z_stabs = [Pauli.from_z_list(row) for row in H_z]
        x_stabs = [Pauli.from_x_list(row) for row in H_x]

        z_logicals = [Pauli.from_z_list(row) for row in L_z]
        x_logicals = [Pauli.from_x_list(row) for row in L_x]

        stabs = z_stabs + x_stabs
        return cls(stabs, x_logicals, z_logicals)

    def __repr__(self) -> str:
        return f"CSSEncoder(n={self.n}, k={len(self.x_logicals)}, stabs={len(self.stabs)})"

    def __str__(self) -> str:
        lines = [f"CSSEncoder: n={self.n}, k={len(self.x_logicals)}", "Stabilizers:"]
        for s in self.stabs:
            lines.append("  " + str(s))
        lines.append("X logicals:")
        for L in self.x_logicals:
            lines.append("  " + str(L))
        lines.append("Z logicals:")
        for L in self.z_logicals:
            lines.append("  " + str(L))
        return "\n".join(lines)

    def copy(self):
        return CSSEncoder(
            [s.copy() for s in self.stabs],
            [s.copy() for s in self.x_logicals],
            [s.copy() for s in self.z_logicals]
        )

    def tensor(self, other: CSSEncoder) -> CSSEncoder:
        """Tensor product of two codes."""
        n_self = self.n
        n_other = other.n
        id_self = Pauli.identity(n_self)
        id_other = Pauli.identity(n_other)

        new_stabs = []
        for s in self.stabs:
            new_stabs.append(s.tensor(id_other))
        for s in other.stabs:
            new_stabs.append(id_self.tensor(s))

        new_x_logicals = []
        for L in self.x_logicals:
            new_x_logicals.append(L.tensor(id_other))
        for L in other.x_logicals:
            new_x_logicals.append(id_self.tensor(L))

        new_z_logicals = []
        for L in self.z_logicals:
            new_z_logicals.append(L.tensor(id_other))
        for L in other.z_logicals:
            new_z_logicals.append(id_self.tensor(L))

        return CSSEncoder(new_stabs, new_x_logicals, new_z_logicals)

    def logical_CNOT(self, control: int, target: int) -> None:
        x_c, x_t = self.x_logicals[control], self.x_logicals[target]
        z_c, z_t = self.z_logicals[control], self.z_logicals[target]

        _, new_x_c = x_c * x_t
        _, new_z_t = z_c * z_t

        self.x_logicals[control] = new_x_c
        self.z_logicals[target] = new_z_t


class CSSState:
    """
    Represents a stabiliser state for CSS codes: a simple container holding a list of Pauli stabiliser generators.
    """

    def __init__(self, stabs: list[Pauli], code: CSSEncoder):
        self.stabs: list[Pauli] = [s.copy() for s in stabs]
        self.code: CSSEncoder = code

    @property
    def nqubits(self) -> int:
        return len(self.stabs[0]) if self.stabs else 0

    def __repr__(self) -> str:
        return f"CSSState(n={self.nqubits}, gens={len(self.stabs)})"

    def __str__(self) -> str:
        if not self.stabs:
            return "<empty CSSState>"
        return "\n".join(str(s) for s in self.stabs)

    def tensor(self, other: CSSState) -> CSSState:
        """Return the tensor product state (concatenate qubit registers).

        The resulting stabilizers are each stabilizer from self tensored with identity on other, and
        each stabilizer from other tensored with identity on self.
        """
        id_self = Pauli.identity(self.nqubits)
        id_other = Pauli.identity(other.nqubits)
        new_stabs: list[Pauli] = []
        for s in self.stabs:
            new_stabs.append(s.tensor(id_other))
        for s in other.stabs:
            new_stabs.append(id_self.tensor(s))
        return CSSState(new_stabs, self.code.tensor(other.code))

    @classmethod
    def logical_all_zero(cls, code: CSSEncoder) -> CSSState:
        return cls([s for s in code.stabs + code.z_logicals], code)

    @classmethod
    def logical_all_plus(cls, code: CSSEncoder) -> CSSState:
        return cls([s for s in code.stabs + code.x_logicals], code)

    @classmethod
    def ket(cls, code: CSSEncoder, state: str) -> CSSState:
        stabs = []
        for s in code.stabs:
            stabs.append(s.copy())
        for i, s in enumerate(state):
            if s == "0":
                stabs.append(code.z_logicals[i].copy())
            if s == "+":
                stabs.append(code.x_logicals[i].copy())
        return cls(stabs, code)

    @classmethod
    def bell_state(cls, A: CSSEncoder, B: CSSEncoder) -> CSSState:
        """
        Build stabilizers for an encoded Bell state between codes A and B.

        Args:
            A, B: CSSEncoder instances (must have the same number of logicals).

        Returns:
            CSSState acting on 2*n qubits (A block followed by B block).
        """
        if len(A.x_logicals) != len(B.x_logicals) or len(A.z_logicals) != len(B.z_logicals):
            raise ValueError("Both codes must have the same number of logical qubits")

        gens: list[Pauli] = []
        for sA in A.stabs:
            gens.append(sA.tensor(Pauli.identity(B.n)))
        for sB in B.stabs:
            gens.append(Pauli.identity(A.n).tensor(sB))

        # logical Bell generators
        for xA, zA, xB, zB in zip(A.x_logicals, A.z_logicals, B.x_logicals, B.z_logicals):
            gens.append(xA.tensor(xB))
            gens.append(zA.tensor(zB))

        return cls(gens, A.tensor(B))

    def check_commuting(self) -> bool:
        """Quick check: ensure all stabilizer generators pairwise commute."""
        for i in range(len(self.stabs)):
            for j in range(i + 1, len(self.stabs)):
                if not self.stabs[i].commutes_with(self.stabs[j]):
                    return False
        return True

    def tolist(self):
        return [str(s) for s in self.stabs]


if __name__ == "__main__":
    H_x_422 = H_z_422 = [[1, 1, 1, 1]]
    L_x_422 = [[1,1,0,0], [1,0,1,0]]
    L_z_422 = [[1,0,1,0],[1,1,0,0]]

    H_x_713 = H_z_713 = [
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1],
    ]
    L_x_713 = L_z_713 = [
        [0, 0, 0, 0, 1, 1, 1],
    ]

    H_x_15_7_3 = H_z_15_7_3 = [
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
    L_x_15_7_3 = L_z_15_7_3 = [
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    ]

    code_422 = CSSEncoder.from_matrices(H_x=H_x_422, H_z=H_z_422, L_x=L_x_422, L_z=L_z_422)
    print(code_422)
    print()

    print("CNOT(0,1)")
    code_422.logical_CNOT(0, 1)
    print(code_422)
    print()

    code_713 = CSSEncoder.from_matrices(H_x=H_x_713, H_z=H_z_713, L_x=L_x_713, L_z=L_z_713)
    print(code_713)
    print()

    code_15_7_3 = CSSEncoder.from_matrices(H_x=H_x_15_7_3, H_z=H_z_15_7_3, L_x=L_x_15_7_3, L_z=L_z_15_7_3)
    print(code_15_7_3)
    print()

    print(CSSState.logical_all_zero(code_15_7_3))
    print()

    state_15_7_3 = CSSState.bell_state(code_15_7_3, code_15_7_3)
    print("[15,7,3] Bell:", state_15_7_3.code, sep="\n")

    code_15_7_3_top = code_15_7_3
    code_15_7_3_bottom = code_15_7_3.copy()

    code_15_7_3_bottom.logical_CNOT(3, 2)
    code_15_7_3_bottom.logical_CNOT(3, 6)

    print("[15,7,3] Bell:", CSSState.bell_state(code_15_7_3_top, code_15_7_3_bottom), sep="\n")

