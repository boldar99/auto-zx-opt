from guppylang.std.platform import result
"""Certified adaptive CSS SE. Corrections are returned as Pauli-frame masks.

se_cycle borrows the data and returns (accepted, frame_x, frame_z). It does not
apply physical correction gates. Callers must compose the returned frame.
Bit i in a frame mask refers to data qubit i.
"""
from guppylang import guppy
from guppylang.std.builtins import array, comptime, exit, owned
from guppylang.std import quantum as _quantum
from guppylang.std.quantum import qubit, h, cx, cz, x, y, z, measure

N_DATA = 32
DECODER_BASELINE_MODE = 'standard-plus-deltas'
CODE_DISTANCE = 4
MANDATORY_ACCEPTANCE_ORDER = 0
CLEAN_INPUT_ACCEPTANCE_ORDER = 0
LOGICAL_FAULT_ORDER = 1
ANALYZED_FAULT_ORDER = 2
ACCEPTED_FILTER_ORDER = 1
INPUT_SCOPE = 'total-order'
VERIFIED_AT_BUILD = True
WORST_CASE_CNOTS = 433
LOWER_BOUND_CNOTS = 0
DISTANCE_LADDER = 'label >= max(1, 2 * maximum_remaining_faults)'
PEAK_QUBITS = 60
MAX_EXTRACTIONS = 3
MINIMAX_OPTIMAL = False
ROOT_NODE = 2
IDEAL_DECODER_HASH = 'c2ac2ba6d1bd79dfa0080e57c091eed7c8316f1a0361855d9218ae10f3910e24'
VERIFICATION_MODEL = 'All 15 nonidentity Paulis per two-qubit gate; X/Y/Z per one-qubit gate; reset and measurement flips; no implicit idles; terminal Pauli frames.'
CIRCUIT_HASHES = {'X_d2': 'f1feb56af968a29ed22dd64ac356a8cf1989f71289fbd63f8bad3d9a4a17964f', 'X_d4': 'b9544737f3b80865d5d53be01b895856ae1d8ee5aa0350879d7da8a23656a4e1', 'Z_d2': 'c93adf624600ef71519ce78f62c753a2b8cca028aebd6f2f80cb8f1bd7e92a5f', 'Z_d4': 'f8ad6f8bc299ca7ec73cf4227c7fa0fbe72a84305e825818ccf42ed9a6853b61'}
ACTION_NAMES = ('X_d2', 'X_d4', 'Z_d2', 'Z_d4')
ACTION_FLAG_WIDTHS = [6, 11, 6, 11]
ACTION_IS_Z = [0, 0, 1, 1]
FRAME_X_CHECK_ROWS = [4294967295, 65535, 16711935, 252645135, 858993459, 1431655765]
FRAME_Z_CHECK_ROWS = [4294967295, 65535, 16711935, 252645135, 858993459, 1431655765]
ROOT_FRAME_X = 0
ROOT_FRAME_Z = 0
N_POLICY_NODES = 256
N_DECISION_NODES = 3099
N_TERMINAL_DECODERS = 110
N_FRAME_BASES = 13
N_CONTROL_DECISIONS_BEFORE_COMPACTION = 2840
N_CONTROL_STATES = 2
N_BASELINE_STATES = 33
BASELINE_X_ROOT = 47
BASELINE_Z_ROOT = 47
DECISION_ROWS_PACKED = True
DECISION_BIT_BITS = 5
DECISION_BIT_MASK = 31
DECISION_REFERENCE_BITS = 12
DECISION_REFERENCE_MASK = 4095
DECISION_HIGH_SHIFT = 17
DECISION_DELTA_SHIFT = 29
POLICY_ROWS_PACKED = True
POLICY_CONTROL_BITS = 7
POLICY_CONTROL_MASK = 127
POLICY_KIND_SHIFT = 0
POLICY_KIND_MASK = 7
POLICY_ACTION_SHIFT = 3
POLICY_ACTION_MASK = 3
POLICY_TRANSITION_SHIFT = 5
POLICY_TRANSITION_MASK = 127
POLICY_BASIS_X_SHIFT = 12
POLICY_BASIS_X_MASK = 15
POLICY_BASIS_Z_SHIFT = 16
POLICY_BASIS_Z_MASK = 15


@guppy.struct
class ExtractionResult:
    flags: int
    syndrome: int


@guppy.struct
class CycleResult:
    accepted: bool
    frame_x: int
    frame_z: int


if hasattr(_quantum, "Measurement"):
    @guppy
    def measure_bit(q: qubit @ owned) -> bool:
        return measure(q).read()
else:
    @guppy
    def measure_bit(q: qubit @ owned) -> bool:
        return measure(q)


@guppy
def append_bit(mask: int, bit: bool, index: int) -> int:
    if bit:
        return mask | (1 << index)
    return mask


def prepare(register, targets):
    for q in targets:
        register[q] = qubit()


def apply(gate, register, targets):
    for target in targets:
        if isinstance(target, tuple):
            gate(*(register[q] for q in target))
        else:
            gate(register[target])


def read(register, measurements, targets):
    measurements.extend(measure_bit(register.pop(q)) for q in targets)


def pack_parities(measurements, groups):
    mask = 0
    for index, group in enumerate(groups):
        parity = False
        for record in group:
            parity ^= measurements[record]
        mask = append_bit(mask, parity, index)
    return mask


@guppy.comptime
def run_action_0(data: array[qubit, comptime(N_DATA)]) -> ExtractionResult:
    register = dict(enumerate(data))
    measurements = []
    prepare(register, (32, 33))
    apply(h, register, (32, 33))
    prepare(register, (34, 35))
    apply(h, register, (35,))
    prepare(register, (36, 37, 38))
    apply(h, register, (38,))
    prepare(register, (39, 40, 41, 42))
    apply(h, register, (42,))
    prepare(register, (43, 44, 45, 46, 47, 48, 49, 50))
    apply(h, register, (50,))
    apply(cx, register, ((33, 1), (35, 34), (38, 37), (42, 41), (50, 49), (34, 36), (37, 39),
                         (41, 43), (42, 44), (32, 34), (37, 40), (41, 45), (42, 46), (34, 40),
                         (41, 47), (42, 48), (40, 12), (45, 40), (34, 45), (49, 40), (45, 20),
                         (49, 33), (40, 28)))
    apply(h, register, (45,))
    read(register, measurements, (45,))
    apply(cx, register, ((33, 42),))
    apply(h, register, (40,))
    read(register, measurements, (40,))
    prepare(register, (40,))
    apply(cx, register, ((42, 17),))
    prepare(register, (45,))
    apply(cx, register, ((35, 40),))
    apply(h, register, (42,))
    read(register, measurements, (42,))
    prepare(register, (42,))
    apply(cx, register, ((38, 42), (38, 45), (33, 38), (38, 48), (33, 35), (50, 48), (38, 9),
                         (35, 45), (48, 25)))
    apply(h, register, (38,))
    read(register, measurements, (38,))
    prepare(register, (38,))
    apply(h, register, (38,))
    apply(cx, register, ((38, 33), (32, 38), (33, 40), (32, 37), (38, 36), (37, 47), (32, 41),
                         (49, 37), (32, 0), (49, 41), (37, 8)))
    apply(h, register, (32, 37))
    read(register, measurements, (32, 37))
    prepare(register, (32, 37))
    apply(cx, register, ((39, 32), (43, 37), (38, 39), (36, 32), (39, 10), (43, 39), (49, 39),
                         (38, 43), (39, 26), (43, 18)))
    apply(h, register, (43,))
    read(register, measurements, (43,))
    prepare(register, (43,))
    apply(cx, register, ((37, 43), (32, 43), (36, 37), (43, 30), (36, 6)))
    apply(h, register, (36,))
    read(register, measurements, (36,))
    prepare(register, (36,))
    apply(cx, register, ((46, 36), (45, 36), (35, 46), (36, 29), (49, 45), (35, 5), (49, 34),
                         (45, 13), (49, 46)))
    apply(h, register, (45,))
    read(register, measurements, (45,))
    apply(cx, register, ((46, 21),))
    prepare(register, (45,))
    apply(h, register, (46,))
    read(register, measurements, (46,))
    prepare(register, (46,))
    apply(h, register, (46,))
    apply(cx, register, ((46, 45),))
    apply(h, register, (45,))
    apply(cx, register, ((47, 46), (34, 45), (47, 24), (35, 45), (34, 4)))
    apply(h, register, (35, 34))
    read(register, measurements, (35, 34))
    prepare(register, (34,))
    apply(h, register, (34,))
    apply(cx, register, ((34, 47),))
    apply(h, register, (34,))
    apply(cx, register, ((41, 34), (48, 34), (41, 16)))
    apply(h, register, (48,))
    read(register, measurements, (48,))
    apply(cx, register, ((39, 34),))
    prepare(register, (35,))
    apply(h, register, (39,))
    read(register, measurements, (39,))
    apply(cx, register, ((42, 35),))
    prepare(register, (39,))
    apply(cx, register, ((40, 35), (33, 42), (44, 39), (35, 15), (39, 35), (40, 39), (50, 40),
                         (50, 32), (40, 7), (50, 37), (32, 14)))
    apply(h, register, (40,))
    read(register, measurements, (40,))
    apply(cx, register, ((50, 35), (37, 22)))
    apply(h, register, (32,))
    read(register, measurements, (32,))
    apply(cx, register, ((50, 38), (35, 47), (37, 45), (38, 2), (35, 31)))
    apply(h, register, (37,))
    read(register, measurements, (37,))
    apply(cx, register, ((39, 45),))
    apply(h, register, (38, 35))
    read(register, measurements, (38, 35))
    prepare(register, (32,))
    apply(h, register, (45,))
    apply(cx, register, ((39, 47), (44, 32), (41, 47), (39, 23), (42, 32), (33, 44)))
    apply(h, register, (41, 39))
    read(register, measurements, (41, 39))
    apply(cx, register, ((32, 27), (50, 42), (33, 3), (32, 46), (50, 44), (42, 11)))
    apply(h, register, (33, 32))
    read(register, measurements, (33, 32))
    apply(cx, register, ((36, 46), (44, 19), (50, 49)))
    apply(h, register, (42, 36))
    read(register, measurements, (42, 36))
    apply(cx, register, ((43, 46), (44, 34)))
    apply(h, register, (50,))
    read(register, measurements, (50, 49))
    apply(h, register, (43,))
    read(register, measurements, (43,))
    apply(cx, register, ((46, 45),))
    apply(h, register, (34, 44, 46))
    read(register, measurements, (44, 46, 45))
    apply(cx, register, ((34, 47),))
    apply(h, register, (34,))
    read(register, measurements, (34, 47))
    flags = pack_parities(measurements, ((26,), (25,), (30,), (29,), (31,), (32,)))
    syndrome = pack_parities(measurements, ((4, 17, 21, 11, 10, 7, 14, 5, 3, 23, 8, 15, 19, 2, 6,
                                             28, 0, 9, 16, 20, 12, 13, 22, 1, 24, 27, 18),
                                            (4, 17, 21, 11, 10, 7, 14, 5, 3, 23, 8, 15),
                                            (4, 17, 21, 11, 10, 7, 14, 19, 2, 6, 28, 0, 9, 16, 20),
                                            (4, 17, 21, 5, 3, 23, 19, 2, 6, 28, 12, 13, 22),
                                            (4, 11, 10, 5, 3, 8, 19, 2, 0, 9, 12, 1, 24),
                                            (4, 17, 11, 7, 5, 15, 19, 6, 0, 16, 13, 1, 27)))
    result("_sw_stage_0", flags)
    result("_sw_syndrome_0", syndrome)
    return ExtractionResult(flags, syndrome)


@guppy.comptime
def run_action_1(data: array[qubit, comptime(N_DATA)]) -> ExtractionResult:
    register = dict(enumerate(data))
    measurements = []
    prepare(register, (32, 33, 34))
    apply(h, register, (32, 33, 34))
    prepare(register, (35, 36, 37))
    apply(h, register, (37,))
    prepare(register, (38, 39, 40, 41))
    apply(h, register, (41,))
    prepare(register, (42, 43, 44, 45, 46))
    apply(h, register, (46,))
    prepare(register, (47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59))
    apply(h, register, (59,))
    apply(cx, register, ((34, 35), (37, 36), (41, 9), (46, 45), (59, 58), (32, 34), (33, 35),
                         (37, 39), (36, 38), (41, 53), (45, 47), (46, 48), (33, 37), (35, 39),
                         (34, 38), (32, 36), (47, 51), (45, 49), (46, 50), (48, 52), (33, 41),
                         (37, 43), (39, 44), (47, 54), (51, 56), (49, 55), (46, 53), (52, 57),
                         (41, 43), (33, 46), (59, 53), (39, 52), (41, 40), (59, 39), (41, 44),
                         (40, 42), (39, 7), (44, 57), (35, 41)))
    apply(h, register, (39,))
    read(register, measurements, (39,))
    prepare(register, (39,))
    apply(cx, register, ((38, 39), (38, 51), (42, 39), (39, 56), (34, 42), (59, 39), (34, 47),
                         (42, 54), (59, 51), (39, 14), (47, 18), (42, 10), (51, 22)))
    apply(h, register, (39,))
    read(register, measurements, (39,))
    prepare(register, (39,))
    apply(cx, register, ((36, 39), (36, 49), (40, 39), (49, 20), (39, 55), (32, 40), (39, 12),
                         (55, 47), (57, 47), (58, 55), (59, 57), (46, 47), (58, 33), (59, 34)))
    read(register, measurements, (47,))
    apply(cx, register, ((46, 17), (33, 1)))
    prepare(register, (47,))
    apply(h, register, (33,))
    read(register, measurements, (33,))
    apply(cx, register, ((50, 47),))
    prepare(register, (33,))
    apply(cx, register, ((37, 50), (43, 47), (45, 33), (47, 29), (40, 33), (32, 45), (58, 40),
                         (32, 0), (58, 45), (40, 8)))
    apply(h, register, (32,))
    read(register, measurements, (32,))
    apply(cx, register, ((58, 54),))
    prepare(register, (32,))
    apply(h, register, (32,))
    apply(cx, register, ((58, 43), (54, 26), (58, 36), (43, 13), (58, 50)))
    apply(h, register, (43,))
    read(register, measurements, (43,))
    prepare(register, (43,))
    apply(cx, register, ((48, 43), (41, 43), (35, 48), (59, 41), (35, 3), (59, 48), (41, 11),
                         (59, 58)))
    apply(h, register, (41, 59))
    read(register, measurements, (41, 59, 58))
    prepare(register, (41,))
    apply(h, register, (41,))
    prepare(register, (58,))
    apply(cx, register, ((41, 58),))
    apply(h, register, (58,))
    apply(cx, register, ((33, 41), (36, 58), (43, 41), (37, 58), (36, 4), (47, 41), (51, 58)))
    apply(h, register, (47,))
    read(register, measurements, (47,))
    apply(cx, register, ((56, 41),))
    apply(h, register, (51,))
    read(register, measurements, (51,))
    apply(cx, register, ((52, 58),))
    apply(h, register, (58,))
    apply(cx, register, ((41, 58),))
    apply(h, register, (41,))
    read(register, measurements, (41, 58))
    prepare(register, (41,))
    apply(h, register, (41,))
    prepare(register, (47,))
    apply(cx, register, ((32, 47),))
    apply(h, register, (47,))
    apply(cx, register, ((33, 32), (33, 24), (57, 32)))
    apply(h, register, (33,))
    read(register, measurements, (33,))
    apply(cx, register, ((57, 31), (52, 32)))
    prepare(register, (33,))
    apply(h, register, (57,))
    read(register, measurements, (57,))
    apply(cx, register, ((52, 23), (41, 33)))
    apply(h, register, (52,))
    read(register, measurements, (52,))
    apply(h, register, (33,))
    apply(cx, register, ((37, 41), (34, 33), (40, 41), (37, 5), (35, 33), (34, 2)))
    apply(h, register, (40,))
    read(register, measurements, (40,))
    apply(cx, register, ((42, 41),))
    apply(h, register, (37, 35, 34, 42))
    read(register, measurements, (37, 35, 34, 42))
    apply(cx, register, ((39, 41),))
    prepare(register, (34,))
    apply(h, register, (34,))
    prepare(register, (35,))
    apply(h, register, (39,))
    read(register, measurements, (39,))
    apply(cx, register, ((34, 35),))
    apply(h, register, (34,))
    apply(cx, register, ((45, 35), (38, 34), (53, 35), (45, 32), (38, 6), (44, 34), (54, 35),
                         (53, 25), (45, 33)))
    apply(h, register, (38,))
    read(register, measurements, (38,))
    apply(cx, register, ((44, 47), (56, 34)))
    apply(h, register, (54,))
    read(register, measurements, (54,))
    apply(cx, register, ((48, 35), (45, 41), (49, 33), (44, 15), (43, 47), (46, 34), (56, 30),
                         (48, 19), (45, 16)))
    apply(h, register, (49, 44))
    read(register, measurements, (49, 44))
    apply(cx, register, ((55, 47), (43, 33)))
    apply(h, register, (46,))
    read(register, measurements, (46,))
    apply(cx, register, ((50, 34),))
    apply(h, register, (45,))
    read(register, measurements, (45,))
    apply(cx, register, ((55, 28), (43, 41), (56, 33), (48, 34), (50, 47)))
    apply(h, register, (55,))
    read(register, measurements, (55,))
    apply(cx, register, ((43, 27),))
    apply(h, register, (33, 56))
    read(register, measurements, (56,))
    apply(h, register, (34, 48))
    read(register, measurements, (48,))
    apply(cx, register, ((36, 47), (50, 21)))
    apply(h, register, (43,))
    read(register, measurements, (43,))
    apply(cx, register, ((41, 33), (34, 35)))
    apply(h, register, (36,))
    read(register, measurements, (36,))
    apply(cx, register, ((53, 47),))
    apply(h, register, (50, 41))
    read(register, measurements, (50, 41, 33, 35))
    apply(h, register, (34,))
    read(register, measurements, (34,))
    apply(h, register, (47, 53))
    read(register, measurements, (53,))
    apply(cx, register, ((32, 47),))
    apply(h, register, (32,))
    read(register, measurements, (32, 47))
    flags = pack_parities(measurements, ((2,), (8,), (7,), (12,), (11,), (37,), (36,), (40,), (39,),
                                         (35,), (34,)))
    syndrome = pack_parities(measurements, ((4, 3, 19, 18, 32, 17, 22, 0, 16, 20, 6, 21, 5, 1, 25,
                                             27, 26, 30, 24, 33, 10, 15, 13, 38, 23, 31, 28, 9, 29,
                                             14),
                                            (4, 3, 19, 18, 32, 17, 22, 0, 16, 20, 6, 21, 5, 1, 25),
                                            (4, 3, 19, 18, 32, 17, 22, 0, 27, 26, 30, 24, 33, 10,
                                             15),
                                            (4, 3, 19, 18, 16, 20, 6, 27, 26, 30, 13, 38, 23, 31),
                                            (4, 3, 32, 17, 16, 21, 5, 27, 26, 24, 33, 13, 38, 28,
                                             9),
                                            (4, 19, 32, 22, 16, 20, 21, 1, 27, 24, 10, 13, 23, 28,
                                             29)))
    result("_sw_stage_1", flags)
    result("_sw_syndrome_1", syndrome)
    return ExtractionResult(flags, syndrome)


@guppy.comptime
def run_action_2(data: array[qubit, comptime(N_DATA)]) -> ExtractionResult:
    register = dict(enumerate(data))
    measurements = []
    prepare(register, (32, 33, 34))
    apply(h, register, (34,))
    prepare(register, (35, 36, 37))
    apply(h, register, (36, 37))
    prepare(register, (38, 39, 40, 41))
    apply(h, register, (39, 40, 41))
    prepare(register, (42, 43, 44, 45, 46, 47, 48, 49))
    apply(h, register, (43, 44, 45, 46, 47, 48, 49))
    prepare(register, (50,))
    apply(cx, register, ((1, 33), (34, 35), (37, 38), (41, 42), (49, 50), (36, 34), (39, 37),
                         (43, 41), (44, 42), (34, 32), (40, 37), (45, 41), (46, 42), (40, 34),
                         (47, 41), (48, 42), (12, 40), (40, 45), (45, 34), (40, 49), (20, 45),
                         (33, 49), (28, 40)))
    read(register, measurements, (45,))
    apply(cx, register, ((42, 33),))
    read(register, measurements, (40,))
    prepare(register, (40,))
    apply(h, register, (40,))
    apply(cx, register, ((17, 42),))
    prepare(register, (45,))
    apply(h, register, (45,))
    apply(cx, register, ((40, 35),))
    read(register, measurements, (42,))
    prepare(register, (42,))
    apply(h, register, (42,))
    apply(cx, register, ((42, 38), (45, 38), (38, 33), (48, 38), (35, 33), (48, 50), (9, 38),
                         (45, 35), (25, 48)))
    read(register, measurements, (38,))
    prepare(register, (38,))
    apply(cx, register, ((33, 38), (38, 32), (40, 33), (37, 32), (36, 38), (47, 37), (41, 32),
                         (37, 49), (0, 32), (41, 49), (8, 37)))
    read(register, measurements, (32, 37))
    prepare(register, (32, 37))
    apply(h, register, (32, 37))
    apply(cx, register, ((32, 39), (37, 43), (39, 38), (32, 36), (10, 39), (39, 43), (39, 49),
                         (43, 38), (26, 39), (18, 43)))
    read(register, measurements, (43,))
    prepare(register, (43,))
    apply(h, register, (43,))
    apply(cx, register, ((43, 37), (43, 32), (37, 36), (30, 43), (6, 36)))
    read(register, measurements, (36,))
    prepare(register, (36,))
    apply(h, register, (36,))
    apply(cx, register, ((36, 46), (36, 45), (46, 35), (29, 36), (45, 49), (5, 35), (34, 49),
                         (13, 45), (46, 49)))
    read(register, measurements, (45,))
    apply(cx, register, ((21, 46),))
    prepare(register, (45,))
    apply(h, register, (45,))
    read(register, measurements, (46,))
    prepare(register, (46,))
    apply(cx, register, ((45, 46),))
    apply(h, register, (45,))
    apply(cx, register, ((46, 47), (45, 34), (24, 47), (45, 35), (4, 34)))
    read(register, measurements, (35, 34))
    prepare(register, (34,))
    apply(cx, register, ((47, 34),))
    apply(h, register, (34,))
    apply(cx, register, ((34, 41), (34, 48), (16, 41)))
    read(register, measurements, (48,))
    apply(cx, register, ((34, 39),))
    prepare(register, (35,))
    apply(h, register, (35,))
    read(register, measurements, (39,))
    apply(cx, register, ((35, 42),))
    prepare(register, (39,))
    apply(h, register, (39,))
    apply(cx, register, ((35, 40), (42, 33), (39, 44), (15, 35), (35, 39), (39, 40), (40, 50),
                         (32, 50), (7, 40), (37, 50), (14, 32)))
    read(register, measurements, (40,))
    apply(cx, register, ((35, 50), (22, 37)))
    read(register, measurements, (32,))
    apply(cx, register, ((38, 50), (47, 35), (45, 37), (2, 38), (31, 35)))
    read(register, measurements, (37,))
    apply(cx, register, ((45, 39),))
    read(register, measurements, (38, 35))
    prepare(register, (32,))
    apply(h, register, (32, 45))
    apply(cx, register, ((47, 39), (32, 44), (47, 41), (23, 39), (32, 42), (44, 33)))
    read(register, measurements, (41, 39))
    apply(cx, register, ((27, 32), (42, 50), (3, 33), (46, 32), (44, 50), (11, 42)))
    read(register, measurements, (33, 32))
    apply(cx, register, ((46, 36), (19, 44), (49, 50)))
    read(register, measurements, (42, 36))
    apply(cx, register, ((46, 43), (34, 44)))
    read(register, measurements, (50,))
    apply(h, register, (49,))
    read(register, measurements, (49, 43))
    apply(cx, register, ((45, 46),))
    apply(h, register, (34,))
    read(register, measurements, (44, 46))
    apply(h, register, (45,))
    read(register, measurements, (45,))
    apply(cx, register, ((47, 34),))
    read(register, measurements, (34,))
    apply(h, register, (47,))
    read(register, measurements, (47,))
    flags = pack_parities(measurements, ((26,), (25,), (30,), (29,), (31,), (32,)))
    syndrome = pack_parities(measurements, ((4, 17, 21, 11, 10, 7, 14, 5, 3, 23, 8, 15, 19, 2, 6,
                                             28, 0, 9, 16, 20, 12, 13, 22, 1, 24, 27, 18),
                                            (4, 17, 21, 11, 10, 7, 14, 5, 3, 23, 8, 15),
                                            (4, 17, 21, 11, 10, 7, 14, 19, 2, 6, 28, 0, 9, 16, 20),
                                            (4, 17, 21, 5, 3, 23, 19, 2, 6, 28, 12, 13, 22),
                                            (4, 11, 10, 5, 3, 8, 19, 2, 0, 9, 12, 1, 24),
                                            (4, 17, 11, 7, 5, 15, 19, 6, 0, 16, 13, 1, 27)))
    result("_sw_stage_2", flags)
    result("_sw_syndrome_2", syndrome)
    return ExtractionResult(flags, syndrome)


@guppy.comptime
def run_action_3(data: array[qubit, comptime(N_DATA)]) -> ExtractionResult:
    register = dict(enumerate(data))
    measurements = []
    prepare(register, (32, 33, 34, 35, 36))
    apply(h, register, (35, 36))
    prepare(register, (37, 38, 39, 40))
    apply(h, register, (38, 39, 40))
    prepare(register, (41, 42, 43, 44, 45))
    apply(h, register, (42, 43, 44, 45))
    prepare(register, (46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58))
    apply(h, register, (47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58))
    prepare(register, (59,))
    apply(cx, register, ((35, 34), (36, 37), (9, 41), (45, 46), (58, 59), (34, 32), (35, 33),
                         (39, 37), (38, 36), (53, 41), (47, 45), (48, 46), (37, 33), (39, 35),
                         (38, 34), (36, 32), (51, 47), (49, 45), (50, 46), (52, 48), (41, 33),
                         (43, 37), (44, 39), (54, 47), (56, 51), (55, 49), (53, 46), (57, 52),
                         (43, 41), (46, 33), (53, 59), (52, 39), (40, 41), (39, 59), (44, 41),
                         (42, 40), (7, 39), (57, 44), (41, 35)))
    read(register, measurements, (39,))
    prepare(register, (39,))
    apply(h, register, (39,))
    apply(cx, register, ((39, 38), (51, 38), (39, 42), (56, 39), (42, 34), (39, 59), (47, 34),
                         (54, 42), (51, 59), (14, 39), (18, 47), (10, 42), (22, 51)))
    read(register, measurements, (39,))
    prepare(register, (39,))
    apply(h, register, (39,))
    apply(cx, register, ((39, 36), (49, 36), (39, 40), (20, 49), (55, 39), (40, 32), (12, 39),
                         (47, 55), (47, 57), (55, 58), (57, 59), (47, 46), (33, 58), (34, 59)))
    apply(h, register, (47,))
    read(register, measurements, (47,))
    apply(cx, register, ((17, 46), (1, 33)))
    prepare(register, (47,))
    apply(h, register, (47,))
    read(register, measurements, (33,))
    apply(cx, register, ((47, 50),))
    prepare(register, (33,))
    apply(h, register, (33,))
    apply(cx, register, ((50, 37), (47, 43), (33, 45), (29, 47), (33, 40), (45, 32), (40, 58),
                         (0, 32), (45, 58), (8, 40)))
    read(register, measurements, (32,))
    apply(cx, register, ((54, 58),))
    prepare(register, (32,))
    apply(cx, register, ((43, 58), (26, 54), (36, 58), (13, 43), (50, 58)))
    read(register, measurements, (43,))
    prepare(register, (43,))
    apply(h, register, (43,))
    apply(cx, register, ((43, 48), (43, 41), (48, 35), (41, 59), (3, 35), (48, 59), (11, 41),
                         (58, 59)))
    read(register, measurements, (41, 59))
    apply(h, register, (58,))
    read(register, measurements, (58,))
    prepare(register, (41, 58))
    apply(h, register, (58,))
    apply(cx, register, ((58, 41),))
    apply(h, register, (58,))
    apply(cx, register, ((41, 33), (58, 36), (41, 43), (58, 37), (4, 36), (41, 47), (58, 51)))
    read(register, measurements, (47,))
    apply(cx, register, ((41, 56),))
    read(register, measurements, (51,))
    apply(cx, register, ((58, 52),))
    apply(h, register, (58,))
    apply(cx, register, ((58, 41),))
    read(register, measurements, (41,))
    apply(h, register, (58,))
    read(register, measurements, (58,))
    prepare(register, (41, 47))
    apply(h, register, (47,))
    apply(cx, register, ((47, 32),))
    apply(h, register, (47,))
    apply(cx, register, ((32, 33), (24, 33), (32, 57)))
    read(register, measurements, (33,))
    apply(cx, register, ((31, 57), (32, 52)))
    prepare(register, (33,))
    apply(h, register, (33,))
    read(register, measurements, (57,))
    apply(cx, register, ((23, 52), (33, 41)))
    read(register, measurements, (52,))
    apply(h, register, (33,))
    apply(cx, register, ((41, 37), (33, 34), (41, 40), (5, 37), (33, 35), (2, 34)))
    read(register, measurements, (40,))
    apply(cx, register, ((41, 42),))
    read(register, measurements, (37, 35, 34, 42))
    apply(cx, register, ((41, 39),))
    prepare(register, (34, 35))
    apply(h, register, (35,))
    read(register, measurements, (39,))
    apply(cx, register, ((35, 34),))
    apply(h, register, (34,))
    apply(cx, register, ((35, 45), (34, 38), (35, 53), (32, 45), (6, 38), (34, 44), (35, 54),
                         (25, 53), (33, 45)))
    read(register, measurements, (38,))
    apply(cx, register, ((47, 44), (34, 56)))
    read(register, measurements, (54,))
    apply(cx, register, ((35, 48), (41, 45), (33, 49), (15, 44), (47, 43), (34, 46), (30, 56),
                         (19, 48), (16, 45)))
    read(register, measurements, (49, 44))
    apply(cx, register, ((47, 55), (33, 43)))
    read(register, measurements, (46,))
    apply(cx, register, ((34, 50),))
    read(register, measurements, (45,))
    apply(cx, register, ((28, 55), (41, 43), (33, 56), (34, 48), (47, 50)))
    read(register, measurements, (55,))
    apply(cx, register, ((27, 43),))
    apply(h, register, (33,))
    read(register, measurements, (56,))
    apply(h, register, (34,))
    read(register, measurements, (48,))
    apply(cx, register, ((47, 36), (21, 50)))
    read(register, measurements, (43,))
    apply(cx, register, ((33, 41), (35, 34)))
    read(register, measurements, (36,))
    apply(cx, register, ((47, 53),))
    read(register, measurements, (50, 41))
    apply(h, register, (33, 35))
    read(register, measurements, (33, 35, 34))
    apply(h, register, (47,))
    read(register, measurements, (53,))
    apply(cx, register, ((47, 32),))
    read(register, measurements, (32,))
    apply(h, register, (47,))
    read(register, measurements, (47,))
    flags = pack_parities(measurements, ((2,), (8,), (7,), (12,), (11,), (37,), (36,), (40,), (39,),
                                         (35,), (34,)))
    syndrome = pack_parities(measurements, ((4, 3, 19, 18, 32, 17, 22, 0, 16, 20, 6, 21, 5, 1, 25,
                                             27, 26, 30, 24, 33, 10, 15, 13, 38, 23, 31, 28, 9, 29,
                                             14),
                                            (4, 3, 19, 18, 32, 17, 22, 0, 16, 20, 6, 21, 5, 1, 25),
                                            (4, 3, 19, 18, 32, 17, 22, 0, 27, 26, 30, 24, 33, 10,
                                             15),
                                            (4, 3, 19, 18, 16, 20, 6, 27, 26, 30, 13, 38, 23, 31),
                                            (4, 3, 32, 17, 16, 21, 5, 27, 26, 24, 33, 13, 38, 28,
                                             9),
                                            (4, 19, 32, 22, 16, 20, 21, 1, 27, 24, 10, 13, 23, 28,
                                             29)))
    result("_sw_stage_3", flags)
    result("_sw_syndrome_3", syndrome)
    return ExtractionResult(flags, syndrome)


@guppy
def execute_action(data: array[qubit, comptime(N_DATA)], action: int) -> ExtractionResult:
    if action == 0:
        return run_action_0(data)
    if action == 1:
        return run_action_1(data)
    if action == 2:
        return run_action_2(data)
    if action == 3:
        return run_action_3(data)
    return ExtractionResult(-1, 0)



@guppy
def lookup(node: int, key: int) -> int:
    """Evaluate a shared decoder, accumulating its small XOR differences."""
    if node < 0:
        return -(node + 1)
    decisions = comptime(DECISION_NODES)
    deltas = comptime(DECISION_DELTAS)
    value = deltas[node >> comptime(DECISION_REFERENCE_BITS)]
    node &= comptime(DECISION_REFERENCE_MASK)
    while node != 0:
        word = decisions[node - 1]
        bit = word & comptime(DECISION_BIT_MASK)
        low = (word >> comptime(DECISION_BIT_BITS)) & comptime(DECISION_REFERENCE_MASK)
        high = (word >> comptime(DECISION_HIGH_SHIFT)) & comptime(DECISION_REFERENCE_MASK)
        delta = word >> comptime(DECISION_DELTA_SHIFT)
        if (key & (1 << bit)) != 0:
            value ^= deltas[delta]
            node = high
        else:
            node = low
    return value


@guppy
def lookup_control(spec_id: int, key: int) -> int:
    specs = comptime(CONTROL_SPECS)
    bits = comptime(CONTROL_BITS)
    accepted, offset, start, count = specs[spec_id]
    if lookup(accepted, key) == 0:
        return 0
    value = offset
    for i in range(start, start + count):
        bit, root = bits[i]
        if lookup(root, key) != 0:
            value ^= 1 << bit
    return value


@guppy
def expand_frame(coefficient: int, basis_id: int) -> int:
    bases = comptime(FRAME_BASIS_SPECS)
    vectors = comptime(FRAME_BASIS_VECTORS)
    start, count = bases[basis_id]
    mask = 0
    for bit in range(count):
        if (coefficient & (1 << bit)) != 0:
            mask ^= vectors[start + bit]
    return mask


@guppy
def frame_x_syndrome(mask: int) -> int:
    rows = comptime(FRAME_X_CHECK_ROWS)
    result = 0
    for i in range(comptime(len(FRAME_X_CHECK_ROWS))):
        parity = False
        for q in range(comptime(N_DATA)):
            if (mask & rows[i] & (1 << q)) != 0:
                parity = not parity
        if parity:
            result |= 1 << i
    return result


@guppy
def frame_z_syndrome(mask: int) -> int:
    rows = comptime(FRAME_Z_CHECK_ROWS)
    result = 0
    for i in range(comptime(len(FRAME_Z_CHECK_ROWS))):
        parity = False
        for q in range(comptime(N_DATA)):
            if (mask & rows[i] & (1 << q)) != 0:
                parity = not parity
        if parity:
            result |= 1 << i
    return result


@guppy
def policy_node(node: int) -> tuple[int, int, int, int, int, int, int]:
    """Read (kind, action, decoder, X offset, Z offset, X basis, Z basis)."""
    nodes = comptime(POLICY_NODES)
    controls = comptime(POLICY_CONTROLS)
    offsets = comptime(POLICY_FRAME_OFFSETS)
    ref = nodes[node]
    word = controls[ref & comptime(POLICY_CONTROL_MASK)]
    frame_x, frame_z = offsets[ref >> comptime(POLICY_CONTROL_BITS)]
    kind = (word >> comptime(POLICY_KIND_SHIFT)) & comptime(POLICY_KIND_MASK)
    action = (word >> comptime(POLICY_ACTION_SHIFT)) & comptime(POLICY_ACTION_MASK)
    transition = (word >> comptime(POLICY_TRANSITION_SHIFT)) & comptime(POLICY_TRANSITION_MASK)
    basis_x = (word >> comptime(POLICY_BASIS_X_SHIFT)) & comptime(POLICY_BASIS_X_MASK)
    basis_z = (word >> comptime(POLICY_BASIS_Z_SHIFT)) & comptime(POLICY_BASIS_Z_MASK)
    return kind, action, transition, frame_x, frame_z, basis_x, basis_z


@guppy
def se_cycle_with_frame(data: array[qubit, comptime(N_DATA)], initial_x: int, initial_z: int) -> CycleResult:
    flag_widths = comptime(ACTION_FLAG_WIDTHS)
    is_z = comptime(ACTION_IS_Z)
    decoders = comptime(TERMINAL_DECODERS)
    control_frames = comptime(CONTROL_FRAMES)
    node = comptime(ROOT_NODE)
    pending_x = initial_x ^ comptime(ROOT_FRAME_X)
    pending_z = initial_z ^ comptime(ROOT_FRAME_Z)
    syndrome_x = frame_x_syndrome(pending_x)
    syndrome_z = frame_z_syndrome(pending_z)
    while True:
        kind, action, transition, frame_x, frame_z, basis_x, basis_z = policy_node(node)
        use_baseline = kind >= 4
        kind &= 3
        if kind == 0:
            return CycleResult(False, 0, 0)
        if kind == 1:
            return CycleResult(True, frame_x ^ pending_x, frame_z ^ pending_z)
        result = execute_action(data, action)
        syndrome = result.syndrome
        if is_z[action] != 0:
            syndrome ^= syndrome_x
        else:
            syndrome ^= syndrome_z
        key = result.flags | (syndrome << flag_widths[action])
        if use_baseline:
            if is_z[action] != 0:
                frame_x ^= lookup(comptime(BASELINE_X_ROOT), syndrome)
            else:
                frame_z ^= lookup(comptime(BASELINE_Z_ROOT), syndrome)
        if kind == 3:
            accepted, xs, zs = decoders[transition]
            if lookup(accepted, key) == 0:
                return CycleResult(False, 0, 0)
            return CycleResult(True, expand_frame(lookup(xs, key), basis_x) ^ frame_x ^ pending_x,
                               expand_frame(lookup(zs, key), basis_z) ^ frame_z ^ pending_z)
        node = lookup_control(transition, key)
        if node == 0:
            return CycleResult(False, 0, 0)
        xs, zs = control_frames[transition]
        update_x = expand_frame(lookup(xs, key), basis_x) ^ frame_x
        update_z = expand_frame(lookup(zs, key), basis_z) ^ frame_z
        pending_x ^= update_x
        pending_z ^= update_z
        syndrome_x ^= frame_x_syndrome(update_x)
        syndrome_z ^= frame_z_syndrome(update_z)


@guppy
def se_cycle(data: array[qubit, comptime(N_DATA)]) -> CycleResult:
    return se_cycle_with_frame(data, 0, 0)


@guppy
def postselected_se_cycle(data: array[qubit, comptime(N_DATA)]) -> CycleResult:
    result = se_cycle(data)
    if not result.accepted:
        exit("SE protocol rejected", 1)
    return result


def compile_se_cycle():
    target = se_cycle
    minimal_opt = getattr(target, "with_minimal_opt", None)
    if minimal_opt is not None:
        target = minimal_opt()
    return target.compile_function()


DIRECT_TABLE_BYTES = 35371
PATCHED_TABLE_BYTES = 35085
POLICY_NODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 137, 10, 11, 268, 397, 526, 15, 649, 16, 773, 17, 914, 19, 1044, 21, 22,
 23, 1176, 25, 26, 27, 28, 29, 1289, 1413, 1566, 31, 1669, 1824, 33, 1925, 2061, 34, 35, 2181, 36,
 2317, 2437, 2573, 2701, 2829, 2949, 3077, 3205, 37, 3341, 38, 3461, 39, 3624, 3753, 3845, 3981,
 4112, 4229, 4394, 4523, 4652, 4745, 525, 4909, 46, 4997, 5125, 5257, 5381, 47, 5509, 5642, 5296,
 49, 946, 5765, 5897, 6021, 6149, 6323, 52, 53, 54, 55, 6456, 57, 58, 1211, 6537, 60, 61, 3721,
 6707, 62, 63, 909, 6413, 269, 1549, 1797, 3597, 6797, 6963, 64, 65, 7091, 66, 7235, 7347, 68, 69,
 70, 7475, 71, 7603, 72, 73, 7731, 74, 7859, 7987, 75, 8115, 76, 8243, 77, 8371, 78, 79, 7184, 8499,
 8627, 80, 8755, 81, 82, 8883, 83, 9011, 84, 4693, 86, 9139, 4627, 9267, 87, 9395, 9498, 88, 89,
 9651, 90, 9779, 9907, 987, 92, 10035, 93, 4374, 10163, 94, 95, 10291, 96, 10419, 609, 98, 99, 100,
 10547, 101, 102, 10675, 103, 104, 10857, 10931, 106, 11059, 11243, 11269, 108, 8581, 9221, 11397,
 9733, 11525, 7557, 11653, 7149, 110, 8069, 11781, 8325, 11909, 12037, 12165, 12293, 111, 12421,
 9605, 12549, 10629, 10117, 8197, 12677, 12805, 12933, 112, 9861, 13061, 13189, 13317, 13445, 7685,
 13573, 13701, 13829, 13957, 14085, 10245, 10373, 14213, 14341, 14469, 10885, 14597, 14725, 14853,
 14981, 15109, 15237, 15365, 15493, 15621, 9093, 7429, 7941, 8709, 8453, 15749, 8965, 11013, 15877,
 16005, 113, 16133, 16261, 16389, 16517]

POLICY_CONTROLS = [0, 1, 26, 65578, 3, 35, 67, 131171, 131, 163, 195, 227, 196867, 291, 131395, 355, 387, 419, 262595,
 483, 515, 547, 579, 611, 328323, 675, 707, 739, 771, 803, 394051, 867, 459651, 931, 963, 995, 1027,
 1059, 1091, 1123, 525443, 591011, 197827, 1251, 656643, 1315, 1347, 1379, 1415, 1443, 263619,
 722415, 1539, 1571, 1603, 1635, 132739, 1699, 1731, 329443, 1795, 1827, 1859, 1891, 1923, 1955,
 1987, 329699, 2051, 2083, 2115, 2147, 2179, 2211, 2243, 2275, 2307, 2339, 2371, 2403, 2435, 2467,
 2507, 2531, 2563, 526883, 2627, 2659, 2691, 2723, 2755, 2787, 2819, 2851, 2883, 2915, 2947, 2979,
 3011, 3043, 3075, 3107, 3139, 3171, 3203, 3235, 3267, 52451, 3331, 23859, 3395, 3427, 3459, 3491]

POLICY_FRAME_OFFSETS = [(0, 0), (0, 54737980), (0, 2), (0, 64), (0, 16), (0, 262144), (0, 35782656), (0, 2048),
 (0, 2105360), (0, 128), (0, 2097152), (0, 54739779), (0, 512), (0, 33563136), (0, 131586),
 (0, 16781568), (0, 54281020), (0, 17106181), (0, 1024), (0, 17895697), (0, 256), (0, 4096),
 (0, 32), (0, 2105376), (0, 17891328), (0, 526344), (0, 4), (0, 4210752), (0, 16384), (0, 4194304),
 (0, 65793), (0, 8), (0, 2052), (0, 71565312), (0, 2171154), (0, 526340), (0, 4720644),
 (0, 1048576), (0, 38291588), (0, 1052688), (0, 263172), (0, 65536), (0, 67126272), (0, 23466342),
 (0, 2170896), (0, 5461), (0, 524288), (0, 17917542), (0, 1376341), (2147483648, 0), (0, 8192),
 (0, 131072), (32768, 0), (0, 1), (8388608, 0), (128, 0), (0, 4737156), (134217728, 0), (2048, 0),
 (524288, 0), (8, 0), (536870912, 0), (8192, 0), (2097152, 0), (32, 0), (33554432, 0), (512, 0),
 (131072, 0), (2, 0), (1073741824, 0), (16384, 0), (4194304, 0), (64, 0), (67108864, 0), (0, 8208),
 (1024, 0), (262144, 0), (4, 0), (268435456, 0), (4096, 0), (1048576, 0), (16, 0), (16777216, 0),
 (256, 0), (0, 2171152), (65536, 0), (1, 0), (322122547, 0), (54737980, 0), (2097168, 0),
 (35782656, 0), (2105360, 0), (54739779, 0), (33563136, 0), (33685506, 0), (16781568, 0),
 (54281020, 0), (17106181, 0), (17895697, 0), (2105376, 0), (17891328, 0), (526344, 0),
 (4210752, 0), (357897557, 0), (353719637, 0), (65793, 0), (2052, 0), (71565312, 0), (69280016, 0),
 (526340, 0), (371270034, 0), (38291588, 0), (1052688, 0), (263172, 0), (67126272, 0),
 (23466342, 0), (2170896, 0), (118425359, 0), (5461, 0), (17917542, 0), (1376341, 0),
 (322114355, 0), (8323199, 0), (131586, 0), (371269906, 0), (4737156, 0), (4720644, 0), (8208, 0),
 (2171154, 0), (2171152, 0)]

TERMINAL_DECODERS = [(6943, -1, -1), (6944, -1, -1), (6954, -1, -1), (6960, -1, 2865), (6964, -1, -1), (2872, -1, -1),
 (2873, -1, -1), (6974, -1, -1), (2882, -1, 885), (2884, -1, -1), (2886, -1, 885), (6986, -1, -1),
 (2893, -1, -1), (6990, -1, -1), (2898, -1, 888), (2899, -1, -1), (2901, -1, -1), (7003, -1, -1),
 (2909, -1, -1), (2910, -1, -1), (2914, -1, 2915), (7020, -1, -1), (2926, -1, -1), (2928, -1, -1),
 (2930, -1, -1), (2932, -1, -1), (2934, -1, 885), (2935, -1, -1), (7034, -1, 857), (7036, -1, -1),
 (7043, -1, -1), (2950, -1, -1), (2951, -1, -1), (7052, -1, -1), (2959, -1, -1), (2960, -1, -1),
 (2961, -1, 1177), (2962, -1, 1177), (2965, -1, 885), (2967, -1, -1), (2971, -1, 2973),
 (2975, -1, -1), (7073, -1, -1), (2979, -1, -1), (2980, -1, -1), (2981, -1, -1), (2984, -1, 888),
 (7086, -1, 3005), (7104, -1, -1), (3009, -1, -1), (7107, -1, -1), (3012, -1, -1), (3013, -1, 885),
 (3014, -1, -1), (3015, -1, -1), (3016, -1, 857), (3018, -1, -1), (3020, -1, -1), (7117, -1, -1),
 (3022, -1, -1), (7122, -1, -1), (3027, -1, -1), (7126, -1, -1), (3031, -1, 857), (7129, -1, -1),
 (7130, -1, -1), (7131, -1, -1), (7134, -1, -1), (7135, -1, -1), (3040, -1, -1), (7139, -1, -1),
 (7141, -1, -1), (7143, -1, -1), (7145, -1, -1), (3050, -1, -1), (7148, -1, -1), (7150, -1, -1),
 (7152, -1, -1), (-1, -1, -1), (7153, -1, -1), (7154, -1, -1), (3060, -1, 1177), (3062, -1, -1),
 (7159, -1, -1), (7160, -1, -1), (3066, -1, -1), (7163, -1, -1), (3070, -1, -1), (7167, -1, -1),
 (3073, -1, -1), (7170, -1, -1), (7171, -1, -1), (7172, -1, -1), (3077, -1, -1), (7174, -1, -1),
 (3079, -1, -1), (7179, -1, -1), (7180, -1, -1), (7181, -1, -1), (7182, -1, -1), (7183, -1, -1),
 (3089, -1, -1), (7187, -1, -1), (7189, 885, -1), (7190, -1, -1), (2914, 2915, -1), (7191, -1, -1),
 (7192, -1, -1), (7194, -1, -1), (7195, -1, -1)]

CONTROL_SPECS = [(4861, 3, 0, 8), (6002, 1, 8, 8)]

CONTROL_FRAMES = [(-1, -1), (-1, 2840)]

CONTROL_BITS = [(0, 1017), (1, 1210), (2, 1377), (3, 1517), (4, 1633), (5, 1727), (6, 1817), (7, 1890), (0, 2026),
 (1, 2142), (2, 2265), (3, 2364), (4, 2457), (5, 2532), (6, 2610), (7, 2660)]

FRAME_BASIS_SPECS = [(0, 0), (0, 32), (32, 1), (33, 1), (34, 1), (35, 1), (36, 1), (37, 1), (38, 1), (39, 1), (40, 2),
 (42, 6), (48, 1)]

FRAME_BASIS_VECTORS = [2147483648, 32768, 8388608, 128, 134217728, 2048, 524288, 8, 536870912, 8192, 2097152, 32,
 33554432, 512, 131072, 2, 1073741824, 16384, 4194304, 64, 67108864, 1024, 262144, 4, 268435456,
 4096, 1048576, 16, 16777216, 256, 65536, 1, 8192, 2, 2048, 128, 512, 514, 16384, 4194304, 16512,
 16384, 2523490710, 65535, 16711935, 252645135, 858993459, 1431655765, 322122547]

DECISION_DELTAS = [0, 1, 2, 3, 4, 5, 8, 10, 16, 17, 32, 34, 64, 68, 128, 136, 192, 256, 257, 512, 514, 1024, 1028,
 2048, 2056, 3072, 4096, 4112, 8192, 8224, 16384, 16448, 32639, 32768, 65536, 65537, 98302, 131072,
 131074, 147457, 262144, 262148, 270337, 524288, 524296, 528385, 1048576, 1048592, 1050625, 2097152,
 2097184, 2098177, 4194304, 4194368, 4194817, 8323199, 8323582, 8388608, 16777216, 16777345,
 16777472, 33554432, 33554497, 33554944, 67108864, 67108897, 67109888, 118425359, 118427422,
 134217728, 268435456, 268435465, 268439552, 322114355, 322122550, 357897557, 357913942, 375985769,
 376007062, 536870912, 1073741824, 2147483648, 3221225472]

DECISION_NODES = [41875931136, 19327352832, 18790744097, 30064771072, 31675383808, 32212910209, 9664462946,
 36507222016, 25769803776, 25234112769, 24159191040, 38117834752, 38656278881, 14497218882,
 4833673443, 39728447488, 22548578304, 22013936129, 27380416512, 34896609280, 35436102241,
 11813913154, 33285996544, 28991029248, 28457304801, 20937965568, 40802189312, 40268858177,
 16646669090, 6983123651, 2688287204, 20404503393, 33825882881, 10741744642, 26846036609,
 39193674273, 15573976162, 5910430787, 23623762305, 35971399969, 12890146018, 29528424609,
 41339191361, 17185506626, 8058832163, 3763995844, 1616643045, 536870912, 6291457, 6293026,
 536872449, 536872546, 6817347, 536870913, 6686402, 6817507, 7341732, 536872642, 536872771,
 543162369, 544736770, 536872867, 8128356, 8259365, 1537, 2082, 545523427, 536873060, 536872451,
 1923, 546048164, 9308293, 9439238, 6817540, 536873285, 536873318, 9963815, 6817379, 10225412,
 6817283, 2115, 547490308, 10750437, 536873412, 2596, 548014725, 11274854, 545523331, 536873732,
 548014885, 536873798, 11930343, 12061096, 545523299, 536872579, 12454852, 536873989, 8650755,
 549719012, 536874085, 13110310, 543686659, 3268, 3301, 3334, 550636711, 543686660, 14027429, 3140,
 13503909, 14421382, 3559, 551554376, 14814121, 536872675, 550243940, 6819461, 549717956, 552340709,
 552472230, 9175044, 15859717, 9175045, 16125766, 553127687, 549325348, 4037, 4070, 3910, 16912391,
 553914280, 547490788, 536875141, 543688771, 4292, 13504741, 554700966, 536873956, 546048772,
 18223429, 536872547, 536875428, 18092485, 18747782, 18878759, 536872580, 549322756, 556143173,
 12848324, 19529733, 556536454, 536875333, 11141125, 556929798, 20189927, 20320808, 20451433,
 20581962, 7079522, 536876003, 536876036, 536876069, 536876102, 543555586, 5251, 5284, 21757957,
 5350, 8519682, 5411, 5444, 22413317, 5510, 22680839, 559682664, 536876521, 2244, 23199749,
 23330822, 5736, 5769, 560600586, 23860171, 543948802, 21501699, 5891, 24385316, 21632836, 24647525,
 22157059, 6084, 21764069, 25171878, 21501219, 21633092, 6245, 6278, 25827367, 21632324, 5252,
 26220773, 22418629, 26482982, 22681959, 26745032, 5285, 5317, 27138502, 21495811, 6692, 6725, 6758,
 27793927, 21626884, 28054885, 22151171, 6916, 6949, 28580582, 7015, 28842664, 28973481, 5956,
 27401189, 28316836, 29491205, 29629446, 7271, 6373, 7334, 7367, 30284936, 27656198, 27131910,
 30678311, 28056133, 28318244, 31064069, 31202694, 7655, 31464808, 31595785, 31726538, 536875334,
 536872804, 536878757, 536878790, 32382599, 548016453, 536878886, 32775815, 32907016, 536878985,
 551162181, 536879046, 570294279, 536875782, 570559783, 33824776, 33947657, 570957738, 571088491,
 571217644, 22419717, 8454, 8487, 34603014, 22420261, 8582, 35266919, 35397960, 27794759, 5477,
 27525125, 35922470, 8807, 36184584, 36315625, 27530597, 21764677, 36709094, 8999, 9032, 36569094,
 28442629, 27141030, 37495687, 22544390, 37755527, 37889000, 38019945, 38150858, 536872581,
 18097318, 536880294, 38675655, 536880392, 536880425, 554958854, 9575, 9608, 9641, 576464202,
 576595083, 536875461, 536880678, 6853, 9830, 577250887, 22282244, 40632325, 9958, 22685447,
 577906344, 28311556, 10085, 41418758, 10151, 35913734, 10215, 41953224, 578955081, 15990790,
 579215399, 536881288, 9126, 579608615, 6791, 579873000, 43133097, 43264074, 580255755, 43525644,
 580526317, 6630, 10727, 22554342, 10791, 44313096, 41425734, 10887, 27394052, 44826629, 35924710,
 11015, 45230760, 45361769, 10949, 27667334, 11175, 11208, 9925, 35793926, 11303, 11336, 46541801,
 46672746, 536875589, 536882374, 536882407, 536879143, 47328520, 536882472, 47590729, 47197480,
 47852905, 47984010, 584985771, 570556423, 11817, 48365577, 48639562, 560211494, 536882855,
 536882888, 41943048, 586165993, 21889030, 12103, 12136, 42991624, 49950601, 586952490, 587083403,
 50343436, 536873124, 18100293, 50598213, 50868326, 536883367, 576205000, 536883433, 21895654,
 12583, 12616, 12649, 588787978, 588906507, 554958853, 589173510, 536883719, 10616836, 52690949,
 12902, 12935, 589967912, 12845060, 13029, 53477382, 13095, 13500421, 53870598, 13191, 54145864,
 591147721, 536883752, 3367, 54657032, 591672329, 54932458, 591921163, 55194060, 55324717,
 592325070, 10760, 10184, 55850281, 13674, 536882440, 536884649, 536884682, 593376651, 558368355,
 536884772, 552744517, 536884838, 536884871, 6405, 57409542, 14055, 594556584, 536885033, 536885066,
 594935819, 58209804, 536883302, 536885191, 536883334, 536885255, 58865640, 9863, 28049413, 14470,
 14503, 59521128, 596523081, 536885514, 596770827, 12453508, 536885605, 536885638, 536885671,
 548016709, 597570022, 14855, 597834184, 536873957, 590363238, 536885895, 14024709, 15046, 15079,
 598751912, 62011977, 47594314, 599130123, 62404940, 62535597, 9609, 15338, 559288773, 19938342,
 33700935, 63050502, 33700999, 63585384, 536886473, 42336263, 16908295, 64109832, 15689, 601242858,
 601373707, 536878919, 569507846, 15847, 601898440, 536886825, 536886858, 602275851, 65551788, 7141,
 16070, 16103, 66076392, 16169, 16202, 536878758, 32391047, 536887208, 47595465, 47595498,
 603996011, 536886310, 536887367, 26476550, 16519, 604651624, 10982, 16615, 68167624, 605176009,
 15656, 16745, 605569354, 605683723, 68960300, 69090989, 69221326, 606221583, 550241924, 536887909,
 536887942, 536887975, 536888008, 47596266, 18089605, 536888102, 536888135, 536888168, 47596425,
 6820165, 536888262, 536888295, 536888328, 47596585, 71582634, 71713547, 536880711, 536888488, 7495,
 17640, 609240265, 8774, 42353991, 17768, 10248, 73024905, 610026794, 610140171, 610288780,
 554971622, 17991, 18025, 39601802, 558502309, 536889030, 536889063, 536889096, 536889129, 16121862,
 74842119, 18312, 18345, 612124490, 612255403, 18090982, 536889383, 576211016, 549978118, 18567,
 18600, 613173353, 13369348, 13519109, 18726, 18759, 18792, 13503141, 77201414, 18887, 18920,
 77613449, 614615274, 614727691, 78006284, 615007789, 552735173, 536890055, 536890088, 6631, 19240,
 615795465, 536890218, 79179115, 536875783, 536890312, 2725, 19462, 19495, 19528, 79691782, 19591,
 19624, 80497769, 617499626, 617611275, 617761708, 584450058, 536878823, 536890761, 536890760,
 536890825, 81677738, 618679659, 11850, 560206918, 536890983, 536891016, 7205, 619451142, 536891111,
 536891144, 82988713, 564270084, 536891237, 560222086, 536891303, 536891336, 572658438, 536891399,
 536891432, 84168681, 84299594, 621301323, 84561452, 621563213, 84822702, 3429, 20774, 20807, 20840,
 20873, 85327880, 85721097, 85873066, 536876135, 8742, 21063, 86387208, 623399465, 15657, 623661738,
 623792651, 3493, 21286, 87313447, 624307496, 550641703, 616846248, 87839625, 536892394, 624951307,
 88232716, 4709, 21606, 21639, 625618216, 536892617, 625888619, 6818244, 552752421, 536892742,
 536892775, 10225284, 560092581, 536892870, 6982, 627070439, 90330504, 3269, 41440870, 27809415,
 86398632, 627856969, 27662822, 22279, 59135784, 11270, 22375, 22408, 91903817, 628905706,
 629014539, 92296460, 92427341, 570556424, 22633, 22666, 33694952, 536892328, 93214921, 622745866,
 630347947, 558368387, 536893796, 536893829, 536893862, 94119495, 585259496, 564667975, 536893992,
 94657033, 16915751, 23176, 27135814, 95158279, 23272, 95443625, 632445546, 632553483, 95836492,
 546048996, 96080197, 96211430, 33709031, 589173062, 33709095, 96754696, 47602793, 18766790,
 97132583, 33709256, 97398121, 97541258, 634519563, 13775655, 23912, 634924393, 13631494, 24007,
 24040, 12869, 24102, 556793862, 98983495, 24200, 99245577, 636247466, 636354571, 99638604,
 99769229, 99899470, 636899599, 100155984, 543950882, 24515, 1538, 536895491, 637689828, 7603716,
 101081157, 536872644, 101343269, 101474438, 536872450, 24835, 536870915, 638869796, 102129895,
 24868, 1859, 536895940, 639525285, 536870914, 25123, 536896069, 640049670, 24901, 640311943,
 103571848, 536896068, 536870916, 103965446, 25415, 638844933, 536870917, 641360774, 641491816,
 104751849, 536895492, 24900, 105145381, 1860, 642252805, 642409574, 640811014, 642671815, 25383,
 106063112, 103441190, 105119749, 536896037, 643458438, 643589479, 640836262, 536870918, 107111943,
 644113896, 644244809, 107504650, 536870919, 644638410, 107898539, 1539, 6317860, 1540, 645293893,
 2213, 645556102, 103966663, 25381, 641361926, 26663, 109340648, 25382, 26759, 536896422, 646736040,
 646867049, 103940101, 536897830, 640837960, 103965606, 647521992, 647653737, 110913802, 543187879,
 104464391, 648178184, 644612104, 111569482, 111700459, 648701708, 641360807, 648962728, 536870920,
 112356105, 104464390, 107085831, 644639624, 112880489, 649882442, 536895812, 536898566, 638872615,
 536898634, 650537963, 25127, 107769000, 536898761, 26312, 114294793, 114453738, 114584716,
 114715341, 536897064, 114950153, 643956745, 652111306, 536895816, 536899113, 28234, 652635659,
 536870922, 25126, 28361, 536870921, 653291242, 653422251, 116682380, 536872773, 536899462,
 536899496, 107114442, 536897065, 654339850, 654471148, 654602093, 117861774, 28427, 25157, 28872,
 655257772, 536899464, 112357673, 641335302, 655778600, 536896294, 102003112, 656175497, 536900074,
 119566668, 656568589, 536896934, 536900199, 536899943, 120222344, 536900298, 657326091, 104464392,
 657617676, 536897224, 536896423, 29577, 29610, 121402219, 641361320, 536900617, 536900650,
 121926636, 122057549, 122188366, 122318991, 113246217, 29930, 536898565, 536900905, 25993,
 659977546, 660108555, 536896041, 30154, 660501932, 25124, 536901157, 536901194, 660996107,
 536897736, 640841385, 30410, 661550732, 124810765, 653262858, 25125, 125199782, 536901513,
 662337355, 28682, 662599628, 543162378, 662831115, 103940102, 126248489, 6291458, 536901796,
 536901830, 107116777, 126908554, 127008779, 127170636, 127301645, 127432494, 639787560, 649230793,
 112359914, 664825515, 651192937, 128219692, 649227050, 665321483, 536896035, 536902340, 536902373,
 536900903, 129137416, 114326345, 28073, 666401642, 666532524, 129792621, 644640009, 536902666,
 667025419, 638870055, 536902760, 107773065, 536897225, 130841770, 130972748, 647128486, 536902951,
 536902984, 536903017, 125173766, 131727367, 32200, 131989513, 669023626, 107741193, 644612105,
 132546122, 132677164, 132807949, 669809646, 133069231, 133199056, 5091, 639926084, 544474628,
 133726053, 105406757, 133988262, 536895747, 102006788, 134381543, 536903684, 113279077, 536895813,
 134905990, 104497351, 135168072, 31428, 103973158, 536904007, 101974021, 641368454, 135954792,
 672956681, 24836, 102007301, 536872772, 136577029, 136741414, 126255751, 536896295, 674005672,
 113246213, 33542, 137523559, 32934, 107119463, 137921352, 674923241, 675054058, 639762439,
 644645896, 644645930, 675578859, 543187268, 536872452, 675972229, 103449798, 34023, 536896390,
 34087, 139756808, 29095, 643990920, 677021033, 640812549, 103974374, 640845320, 107112136,
 140805673, 140936650, 6316967, 111314600, 536905418, 678332043, 678462572, 641335303, 141852840,
 114329449, 107111335, 34728, 679241257, 679380874, 101974023, 34858, 142903307, 536896039,
 536905864, 112363689, 536905930, 680429676, 680560429, 107085832, 680787977, 115247466, 28201,
 536906154, 681347467, 536899594, 681574411, 144869868, 643984170, 112356777, 536906378, 682265196,
 145525325, 682527022, 102002443, 649235244, 29481, 107111208, 638871688, 146443145, 35786,
 683576172, 146836301, 649226858, 147090219, 641335304, 147360876, 543162373, 36039, 36073,
 653298954, 147980299, 536896936, 36201, 536907146, 685411660, 148671661, 148802606, 685804303,
 641990661, 536907337, 149321034, 686323531, 116035244, 27434, 102891524, 640847621, 36650,
 150245099, 26918, 36744, 640847785, 641372106, 687771500, 151031501, 662044678, 643993673,
 151388171, 125079692, 536872456, 37066, 688783371, 536896038, 663130409, 536901158, 37225,
 689475914, 689569803, 152867084, 152998061, 153129006, 536896040, 112366153, 536908394, 690487307,
 107741192, 649237193, 154047148, 536870923, 31461, 658019144, 536897063, 37768, 691704681,
 653300682, 155095852, 692097805, 536898567, 536908872, 37993, 536908938, 155844619, 101999143,
 38120, 680039689, 107085833, 693409066, 156669132, 647127974, 536909223, 26823, 694064584, 38409,
 150470663, 38472, 694550537, 694720042, 653301420, 158111117, 695112750, 158372399, 695373360,
 101977060, 139106149, 673480709, 695900038, 671520711, 650149989, 134912006, 641374247, 159684584,
 536900007, 112629896, 160077929, 536896293, 39142, 536910087, 674011432, 643465990, 697722407,
 536903846, 643996071, 161257864, 161388873, 698390730, 102891527, 107780680, 107780714, 698915371,
 139067397, 104504006, 113285863, 536904103, 699570952, 649239369, 1541, 536910726, 103979944,
 114334665, 163486570, 25511, 536910890, 163879947, 164010668, 641360551, 701104136, 643956744,
 164535497, 25510, 643956743, 536911176, 701930793, 165190922, 116039083, 536898728, 40425, 26313,
 165846538, 165977548, 702979213, 643984169, 643982889, 703372970, 101974024, 166723593, 703760810,
 703897323, 116391946, 167247883, 167419756, 1861, 7602181, 167813094, 536911912, 107126858, 26152,
 536912009, 536912042, 168599660, 705601485, 705732238, 638870184, 169082892, 104492936, 649240969,
 640286726, 706505736, 29096, 169910761, 41514, 170172844, 707174765, 29031, 170524680, 41674,
 170786827, 641377036, 1543, 536912713, 116433770, 708341003, 26024, 649241545, 41962, 708879276,
 172139309, 172270222, 709271855, 25513, 172621834, 172783435, 28426, 709928172, 655753224,
 536913226, 640836389, 640853386, 173712747, 26630, 42472, 42505, 536913450, 174368204, 174499117,
 104494950, 107128489, 711720971, 26153, 116434698, 175285996, 536901798, 643999593, 689481610,
 712638475, 712769548, 176072525, 176203406, 37449, 536913994, 713424907, 112328713, 176859276,
 649199627, 536908616, 649242889, 536914218, 714385644, 177645773, 650117127, 536914344, 644655561,
 43498, 178257931, 650810633, 693414474, 715565612, 157155336, 43689, 125200678, 179175431, 43784,
 716308489, 179612362, 536899338, 716745580, 180005517, 180136334, 180267055, 717268112, 638844932,
 536915013, 536903685, 717794406, 134655143, 536904454, 641379559, 718318792, 649243945, 536910054,
 44391, 44424, 107125159, 719098760, 182365609, 182496586, 644651626, 182758955, 26151, 33191,
 183152264, 112373449, 640811013, 536915718, 44840, 649244489, 720809706, 6291463, 111325096, 25512,
 184463306, 721465227, 721596012, 649239753, 649226441, 185118826, 27687, 45258, 722383019,
 644639912, 112374057, 651342154, 722907404, 186167373, 649234761, 652128714, 45547, 125042699,
 186823180, 41000, 45674, 116427402, 187347596, 724349517, 187609518, 638873355, 703640364,
 649229097, 103415814, 725090312, 169915305, 46026, 188658540, 188789581, 27495, 189005832, 46186,
 189295403, 649245868, 39815, 536917225, 116438282, 726823787, 536914090, 727102796, 190362829,
 190493742, 190624527, 650144777, 116438538, 116421962, 191149611, 702705418, 728282732, 39146,
 191655595, 26760, 536917769, 536917802, 192198380, 729200301, 106592102, 47017, 192719755,
 653305610, 192985068, 1544, 536918090, 730202123, 543162370, 536918182, 644004041, 107114921,
 730904810, 730988555, 194295948, 731297837, 194557838, 102891528, 47561, 47594, 195035147,
 114322025, 195344940, 112328714, 732430347, 107778888, 112376521, 536898345, 196131562, 733133484,
 196393581, 650144775, 536918920, 48041, 536918986, 197001227, 34856, 143178793, 702725194,
 734444556, 694026248, 48297, 116440266, 132513802, 735100140, 735231117, 735361902, 735492527,
 735622640, 180879366, 536919495, 536915175, 736148968, 112377385, 182190088, 736523113, 736673354,
 107769034, 736935595, 139092773, 48902, 48935, 162709320, 39048, 200851305, 536905190, 49096,
 112377833, 738246570, 163753896, 49226, 738639915, 738770668, 690789673, 739027114, 180617223,
 536920330, 202555627, 34952, 112378217, 165855626, 203080012, 203210925, 653306026, 638844936,
 203554825, 203721130, 740737547, 28331, 740999820, 40934, 49896, 49930, 649227049, 116441930,
 204915500, 741917389, 205177326, 638844939, 536896424, 742441964, 112358185, 112368873, 50282,
 206095436, 206226477, 46252, 40490, 206599019, 43178, 206882092, 207013101, 207144142, 207274959,
 172752907, 653313516, 536906058, 536896036, 103990853, 640861802, 745063979, 536896425, 208451370,
 745457324, 208717325, 30566, 107136841, 745930763, 116430154, 746243980, 47178, 746455051,
 536918217, 536921670, 51273, 747030570, 747110411, 210421772, 210552781, 210683694, 149815307,
 112355945, 211077420, 29991, 211288072, 112380329, 41097, 211732938, 211863852, 211994989, 29545,
 51818, 212336651, 27816, 143182537, 116443882, 749783724, 536909256, 536922953, 536922986,
 116444044, 750439213, 213699150, 213829903, 213960144, 199229448, 536923209, 536900457, 751357034,
 644649674, 214748331, 162660360, 536909960, 215141641, 733007178, 163743400, 184470922, 752536939,
 215796972, 536903848, 536923657, 649723913, 753192490, 180644903, 52874, 753585771, 27913,
 112355017, 753979114, 754110156, 217370093, 653310730, 166889898, 754634635, 175243275, 754896844,
 41034, 51690, 218419244, 755421197, 218681198, 32936, 755807020, 107111496, 536900008, 219336969,
 53578, 219574028, 219730157, 112370444, 47882, 220069900, 220254669, 220385710, 220516527,
 172782859, 667709194, 220910220, 536911082, 53995, 653291274, 221434636, 221565645, 156530507,
 653301066, 221959052, 115998731, 693399562, 222298123, 222483436, 222614477, 222745454, 25129,
 112383146, 759983787, 27241, 760272108, 112328715, 211325768, 54633, 196138378, 224056652,
 761058605, 27656, 54793, 54826, 224657419, 38153, 115267210, 225105516, 536897735, 55016, 55049,
 536925994, 167302988, 225892045, 762893806, 226153615, 226284144, 214433802, 107766312, 644667434,
 763680779, 536919814, 650172551, 725932200, 215144649, 49129, 227596522, 34506, 227858731,
 764860524, 536896167, 765100200, 55753, 536911145, 228645354, 536916170, 765778475, 121005769,
 766032650, 766171756, 229431693, 35145, 116448010, 35242, 766827307, 653291530, 230162443,
 767220588, 536898985, 175168490, 230727788, 767744973, 231004910, 101999528, 768132076, 169632840,
 56521, 56554, 231735308, 231922861, 112374890, 232128523, 56716, 27433, 536927690, 232698412,
 232840621, 232971598, 233102447, 30539, 167304844, 118882312, 57034, 640855818, 233889515,
 208461258, 234151724, 234282669, 30886, 57257, 156557258, 234749963, 234881036, 235069101,
 235200398, 639762440, 57481, 112386218, 235695787, 57580, 195559435, 196083722, 236249388,
 236380429, 113246215, 57768, 57801, 536928746, 236978187, 48169, 116449866, 237429292, 167296236,
 237691533, 237822350, 237953135, 238083696, 536912265, 536898568, 219341673, 58250, 238740300,
 775684109, 536901478, 536929287, 154723368, 112387146, 776339467, 742450316, 536872455, 536929481,
 536929514, 776994827, 536921097, 536929610, 777446700, 240706733, 240837614, 240910351, 536898569,
 536929802, 208434506, 241493547, 191424108, 112590856, 59050, 536921669, 640870122, 779151051,
 536897672, 536930121, 536930154, 779675436, 779806349, 107132841, 243182475, 536897066, 243460108,
 536907978, 780533771, 107114185, 107137097, 244115626, 244187147, 781248652, 781379661, 244639694,
 112366186, 781713419, 121006889, 782035372, 129143624, 536930825, 230615594, 245683532, 782690797,
 536925705, 536930986, 246153227, 143168777, 702737162, 246606572, 536925928, 60265, 536931210,
 179891116, 247262029, 784263822, 784394607, 784524784, 22118, 60519, 60552, 607055017, 584510666,
 608627947, 73461004, 78179629, 584450057, 584510826, 616099211, 80932268, 84733389, 249490766,
 100068879, 786580048, 665586212, 101117541, 640835877, 787148422, 61127, 640843877, 640347910,
 640347943, 787803880, 103971558, 61319, 641527720, 788328297, 101974020, 788562437, 642445350,
 642707527, 106098792, 150512071, 107126055, 252506280, 789508233, 789639146, 107741194, 253030699,
 6291460, 108458373, 103477670, 104001991, 189067752, 109876104, 253948425, 640837160, 164757512,
 254341737, 254472778, 111542282, 254735051, 254865772, 641368231, 255091624, 649261929, 40264,
 792357737, 255652746, 49418, 255914987, 644638408, 62537, 176878698, 256439340, 256570157,
 703389130, 62699, 186840332, 107085834, 232715596, 257357101, 257488078, 231211020, 134873094,
 257818632, 203617801, 63018, 258255020, 258405837, 657063944, 536934058, 258736139, 184481516,
 39977, 63274, 259260427, 42154, 259585900, 259716877, 259847822, 259978671, 116391948, 639762436,
 63557, 536934506, 260571147, 640811017, 709556426, 261028012, 261158957, 107122761, 261357579,
 209385836, 209584139, 105801257, 107112585, 262076874, 262144011, 262338988, 262470029, 262601006,
 154063852, 748158984, 114358985, 64234, 263192588, 263387821, 32330, 263585803, 650772489,
 116456362, 264043404, 110231558, 64519, 64552, 64585, 157679625, 264830058, 263650476, 265092077,
 265222990, 265353871, 265484304, 24579, 265682948, 670629253, 640842437, 803012006, 180682215,
 536902341, 803367206, 134938183, 266796552, 164802184, 267058825, 788579909, 183566054, 107151111,
 107806504, 643459366, 267809127, 25254, 107151271, 268238728, 805240649, 805371594, 37578,
 268763179, 6316356, 65669, 26405, 269287590, 65767, 107806984, 267059497, 103979910, 536936808,
 109838344, 807076233, 270336330, 111280136, 111608330, 270729707, 270860396, 121135784, 657916553,
 29576, 271346537, 808387242, 66315, 176866122, 271909676, 272040557, 643982888, 272263977,
 107152330, 166854666, 272696299, 167445548, 204865548, 273089613, 273220494, 218919084, 112616520,
 115016937, 66826, 273852588, 274007245, 691535880, 649266570, 274333707, 147391948, 36106,
 274726923, 536900393, 536938058, 275187244, 275318253, 275449198, 275580079, 745275402, 275775499,
 173082412, 641360809, 276197130, 276299788, 276498253, 536909130, 276756844, 6291466, 276955147,
 30884, 67686, 107153545, 277505578, 277610507, 277809228, 277940237, 278071246, 649199626,
 278265867, 657887945, 278595980, 107766696, 44680, 278989289, 129567274, 279241004, 279382477,
 693370890, 279576588, 179437577, 279903338, 669417066, 280169228, 280300237, 280431246, 280561999,
 280692464, 1667, 128781284, 139136005, 638214437, 281349158, 717556839, 134936039, 818613384,
 643983048, 818875593, 101999397, 641404198, 107154759, 644681064, 137522023, 26119, 282922408,
 283053449, 820055306, 644647048, 644681290, 283577899, 61829, 103485094, 104009415, 107810536,
 282005257, 536911144, 647495688, 284626761, 284757802, 648180616, 644637608, 822022090, 822153131,
 822283916, 228221608, 282005609, 644641672, 285897065, 286068874, 717515815, 536940810, 823333099,
 536916265, 766054762, 823726412, 823857229, 164495369, 653332970, 703753802, 824381963, 536899273,
 653333098, 824705035, 288035404, 536911846, 536941288, 536941322, 653319146, 288690988, 825692877,
 825823694, 638909992, 826053420, 649228585, 725115688, 649270313, 536941642, 289870860, 826872813,
 536900232, 112399562, 827195403, 641360808, 827528460, 649235817, 536941930, 827909420, 291181901,
 291312814, 828314543, 686293003, 660543020, 103966122, 828768267, 536915304, 536942249, 276240074,
 829363852, 292623949, 662306827, 681645900, 26761, 830018154, 830078987, 293398668, 293541741,
 293672750, 649237065, 536942666, 830865419, 657877609, 831199372, 686686219, 644649800, 657922313,
 666442026, 831854828, 295114957, 644650089, 536943018, 832307211, 644647081, 165878282, 832772588,
 801668490, 116424298, 296295020, 833296973, 296556942, 296687663, 833689104, 665608516, 536943429,
 803412838, 72583, 281746344, 25350, 536943591, 536943624, 834870217, 536939814, 644029543, 72840,
 103973030, 107158727, 298782120, 298917033, 299048010, 214768971, 139133061, 269294982, 73127,
 536898407, 299834824, 278994441, 642646024, 300199753, 300359210, 184484362, 300621451, 300752236,
 228196360, 657923849, 107770760, 301229609, 301408042, 73611, 114321097, 301765354, 301932460,
 302063341, 649199625, 653336650, 272703595, 222240908, 40232, 107159754, 302457068, 303112365,
 303243310, 102006952, 303467308, 188219400, 168370601, 74186, 303955980, 304161165, 649245802,
 304349195, 104538732, 649244649, 536945322, 304936812, 305078925, 305209902, 305340751, 686161930,
 305528843, 191439756, 104003653, 104014794, 306053131, 536907689, 813114410, 306521100, 306652077,
 643457958, 107160745, 306970635, 173090028, 29097, 307402058, 307494923, 307625996, 307832077,
 307963022, 37577, 308218252, 278985417, 211756586, 308611468, 308749837, 115998732, 264765450,
 653295210, 309274316, 309405357, 309536398, 309667279, 309797712, 265707844, 536946597, 803416006,
 75751, 75784, 121055273, 638875173, 75878, 107161735, 536946856, 536946889, 848242762, 644612106,
 311634187, 139129221, 269298022, 104016263, 189082024, 121055689, 536872453, 536947206, 76328,
 76361, 312945130, 254749323, 313207116, 121035209, 753216234, 255929099, 112356617, 217131850,
 313994028, 314125005, 112363849, 653339594, 76779, 221380619, 314780684, 190259436, 315042893,
 315173806, 101974027, 638870440, 315567308, 112385225, 77098, 315926700, 316091661, 184492652,
 6291461, 77255, 77289, 77322, 316801035, 112378889, 536948330, 317271628, 317402541, 317533582,
 317664431, 667680778, 317924236, 103967018, 318111755, 36777, 653340586, 318582668, 318713677,
 74921, 318898187, 653290986, 319238220, 30953, 319465226, 319553547, 319752268, 319893645,
 320024590, 649242794, 320208907, 120979465, 320549260, 211759530, 320733196, 320942541, 652147626,
 321190796, 653295178, 321388556, 321598061, 321729070, 321859919, 321990416, 64869, 322250694,
 78695, 78728, 78761, 536898440, 160854439, 298791944, 323171305, 323302346, 253047915, 104013223,
 189084840, 79049, 536907656, 105775112, 861091081, 324351210, 285081610, 324613483, 324744332,
 255908395, 229166826, 325137900, 325268941, 272685867, 167458412, 73930, 711983114, 325924524,
 326055565, 326186574, 638877864, 326446284, 231946093, 536917098, 326762507, 184498124, 50185,
 79882, 327364172, 327497709, 327628686, 327759663, 130809866, 328009516, 536897834, 328204299,
 208482218, 328546604, 328677613, 40233, 328859659, 167328204, 156542730, 329252875, 329383948,
 329595373, 329726350, 657850377, 329962476, 666002344, 80617, 196164362, 330486764, 330644173,
 680001545, 130890634, 330956812, 279838730, 167328748, 331430861, 331561838, 331692687, 331823280,
 101343557, 332005382, 180698311, 118265862, 671825159, 869481704, 72679, 81256, 869875017, 75911,
 81352, 182268936, 870399465, 870530474, 536908490, 333921867, 25669, 103497382, 104021703, 33158,
 536952583, 334708456, 536908680, 334970697, 536947240, 536952745, 335363978, 285147658, 335626219,
 335756940, 536923690, 255934571, 229166218, 336281740, 336412749, 536918794, 272711947, 144703499,
 336937260, 336666860, 337199469, 337330414, 536896168, 337539244, 115003657, 82474, 337946796,
 338117133, 112382956, 336592908, 338510509, 338641550, 338772431, 179905420, 179885036, 339166029,
 107111337, 339345419, 276775884, 126903850, 339738635, 339937356, 340083693, 340214670, 190185483,
 121017033, 340608172, 334957033, 230638858, 340975916, 341132525, 536903242, 341375884, 179899148,
 341657005, 341788014, 341918863, 342049584, 638214725, 879175622, 180700807, 869484200, 869877449,
 870532842, 333924107, 108731782, 104023879, 104497542, 536954759, 343623528, 334972873, 335366122,
 335628299, 344147756, 336414797, 117488394, 344520012, 344670573, 344803438, 216042284, 338118925,
 205584108, 240580460, 345459021, 345590062, 345721071, 536945706, 345979180, 346112845, 662070182,
 107170377, 346423307, 179915404, 340085421, 346901038, 536908489, 347161772, 748184488, 334973769,
 230640490, 347657196, 347818797, 536935338, 348060556, 341658637, 348343246, 348474095, 348604880,
 536905898, 885653515, 1073741829, 1073827047, 1073827080, 1073827114, 1423179787, 349654220,
 2147483651, 2147569060, 2147569094, 2147569127, 2147569162, 2497839115, 3221225474, 3221311079,
 3221311112, 3221311145, 3221311178, 3572452940, 351358349, 4294967299, 4295053128, 4295053161,
 4295053194, 4646895627, 5368709122, 5368795110, 5368795145, 5368795178, 5721292811, 352800716,
 6442450945, 6442537122, 6442537157, 6442537190, 6442537224, 6442537258, 7516192774, 7516279144,
 7516279177, 7516279210, 8944177484, 6796824717, 354504494, 9126805507, 9481486347, 10200547330,
 10200633987, 10200634021, 10200634056, 10555970156, 11274289153, 11274375970, 11274376005,
 11274376040, 11274376073, 12348030979, 12348117956, 12348117992, 12348118025, 12348118058,
 13778637740, 11631285005, 13958643714, 13958730917, 13958730950, 13958730983, 13958731016,
 13958731050, 14316470283, 15032385541, 15032472968, 15390692716, 16106127360, 16106214885,
 16106214919, 16106214953, 16106214986, 16465133579, 17716740098, 17716827813, 17716827848,
 17716827881, 17716827914, 18076620428, 360011213, 360141966, 360272431, 18253611011, 18253699012,
 18253699045, 18253699081, 18253699114, 18614583307, 19864223746, 19864311945, 19864311978,
 20225808492, 21474836482, 21474924804, 21474924837, 21474924874, 21836988427, 23085449220,
 23085537701, 23085537734, 23085537768, 23085537801, 23085537834, 23448607116, 363288813,
 24696061954, 24696150693, 24696150726, 24696150761, 25059917835, 26306674694, 26306763593,
 26306763626, 26671143724, 27917287424, 27917376456, 27917376490, 28282191883, 30601641986,
 30601731142, 30601731177, 30601731210, 30967201803, 365780012, 365910957, 366041742, 31138512898,
 31138602312, 31138602345, 31138602378, 31505121291, 32749125640, 32749215209, 33116216780,
 34359738376, 34359828042, 34727133195, 37044092930, 37044182691, 37044182724, 37044182757,
 37044182792, 37044182825, 37044182858, 37412626060, 368664109, 37580963843, 37581053892,
 37581053927, 37581053960, 37581053993, 37581054026, 37950455819, 42412802050, 42412892327,
 42412892360, 42412892393, 42412892426, 42783170700, 42949672962, 42949763429, 42949763462,
 42949763495, 42949763528, 42949763561, 42949763594, 43486543881, 43486634570, 44395094572,
 43321483597, 371941294, 372071727, 372201392, 536882537, 536961834, 24167, 372637704, 372768777,
 91050, 909992779, 536961867, 98959367, 910306600, 91176, 910648393, 373424136, 374039657,
 911041674, 91241, 91402, 911434987, 536879144, 536962409, 90984, 91561, 912090506, 536962539,
 26314, 91017, 91722, 912745291, 584187912, 91817, 91850, 91883, 375783435, 909651305, 909862730,
 91305, 92042, 914057067, 11816, 92137, 372992010, 92203, 92170, 92267, 375220234, 378142731,
 536962890, 375088297, 92426, 915630315, 374997001, 92522, 379060235, 914057451, 48365576,
 372861417, 92682, 92715, 91083, 248905738, 92811, 635835143, 909601480, 536963817, 536963850,
 378929162, 917859115, 91594, 93067, 380986987, 372649512, 93161, 93194, 380988459, 536898347,
 635838503, 536964232, 536964265, 375088009, 919563466, 373515112, 375090473, 382992394, 920087819,
 375128074, 383385611, 376963082, 383647755, 375743754, 93739, 375652362, 93803, 379021322, 93867,
 93003, 536882473, 536964874, 536964907, 536964298, 917860203, 911879529, 913928106, 373424137,
 372728872, 385967209, 386101226, 923103179, 373686281, 386400266, 386531339, 93675, 919743849,
 913928458, 374039977, 375746890, 924283179, 373948425, 94634, 387710987, 381157387, 381249643,
 381249259, 379453449, 388457930, 94859, 60778, 388759563, 48456552, 389021705, 95018, 389374923,
 1610612744, 926416907, 909246474, 389808139, 372994474, 927035339, 372899850, 390332427, 390424683,
 390424971, 375089129, 95434, 95467, 909377547, 928211788, 928216397, 928216430, 928216463,
 928216496, 1073779500, 2147483660, 2539746797, 3221225484, 3221321293, 3613881902, 4294967308,
 4295063213, 4295063246, 4688148111, 5368709132, 5368805165, 5368805198, 5368805231, 5762545424,
 536890345, 536967114, 536967147, 91435, 387671306, 931489003, 373030923, 92331, 377094155,
 380895243, 93227, 373780618, 395444235, 375746986, 395801035, 536964971, 92555, 909520168,
 536967657, 536967690, 536967723, 91755, 536962121, 536967818, 536967851, 389283851, 536965034,
 536967947, 927300363, 920347467, 910568809, 536968074, 536968107, 536963307, 94667, 910830953,
 536968234, 536968267, 911303466, 536968331, 912614186, 536968395, 913924906, 536968459, 95307,
 915890986, 536968555, 909861674, 536968619, 924541738, 536968683, 918125323, 911434571, 94826,
 401040331, 379020554, 97963, 914056011, 916022091, 94346, 401735691, 938697547, 92649, 98186,
 98219, 924672843, 374341642, 402653195, 939615051, 927294283, 921133899, 377880587, 920609611,
 384172043, 909528008, 536969481, 536969514, 536969547, 917857099, 917862379, 920353995, 923493195,
 388366346, 98827, 387579914, 941974347, 912089898, 536969867, 536962923, 920616299, 932406091,
 375743946, 942898251, 918119243]

EXPERIMENT_CYCLES = 10
EXPERIMENT_PLUS = False

from guppylang.std.platform import result
from qtm_platform.ops import sleep
from guppylang.std.quantum import s

@guppy.comptime
def prepare_experiment(data: array[qubit, comptime(N_DATA)]) -> None:
    apply(h, data, (0, 1, 2, 4, 8, 16))
    apply(cx, data, ((0, 3), (0, 5), (0, 6), (0, 9), (0, 10), (0, 12), (0, 15), (0, 17), (0, 18), (0, 20), (0, 23), (0, 24), (0, 27), (0, 29), (0, 30), (1, 3), (1, 5), (1, 7), (1, 9), (1, 11), (1, 13), (1, 15), (1, 17), (1, 19), (1, 21), (1, 23), (1, 25), (1, 27), (1, 29), (1, 31), (2, 3), (2, 6), (2, 7), (2, 10), (2, 11), (2, 14), (2, 15), (2, 18), (2, 19), (2, 22), (2, 23), (2, 26), (2, 27), (2, 30), (2, 31), (4, 5), (4, 6), (4, 7), (4, 12), (4, 13), (4, 14), (4, 15), (4, 20), (4, 21), (4, 22), (4, 23), (4, 28), (4, 29), (4, 30), (4, 31), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30), (8, 31), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28), (16, 29), (16, 30), (16, 31)))

@guppy
def main() -> None:
    data = array(qubit() for _ in range(comptime(N_DATA)))
    prepare_experiment(data)
    synchronize_input(data)
    sleep(data, 1152921504606846977)
    accepted = True
    frame_x = 0
    frame_z = 0
    completed = 0
    for _ in range(comptime(EXPERIMENT_CYCLES)):
        if accepted:
            result("_sw_cycle", 1)
            outcome = se_cycle_with_frame(data, frame_x, frame_z)
            accepted = outcome.accepted
            frame_x = outcome.frame_x
            frame_z = outcome.frame_z
            if accepted:
                completed += 1
    sleep(data, 1152921504606846978)
    result("accepted", accepted)
    result("frame_x", frame_x)
    result("frame_z", frame_z)
    result("completed", completed)
    if comptime(EXPERIMENT_PLUS):
        for i in range(comptime(N_DATA)):
            h(data[i])
    bits = array(measure_bit(q) for q in data)
    result("measurements", bits)

@guppy
def check_input(observed: bool, expected: bool) -> None:
    if observed != expected:
        exit("Ideal Sol input synchronization failed")

@guppy.comptime
def synchronize_input(data: array[qubit, comptime(N_DATA)]) -> None:
    probe = qubit()
    h(probe)
    cx(probe, data[0])
    cx(probe, data[1])
    cx(probe, data[2])
    cx(probe, data[3])
    cx(probe, data[4])
    cx(probe, data[5])
    cx(probe, data[6])
    cx(probe, data[7])
    cx(probe, data[8])
    cx(probe, data[9])
    cx(probe, data[10])
    cx(probe, data[11])
    cx(probe, data[12])
    cx(probe, data[13])
    cx(probe, data[14])
    cx(probe, data[15])
    cx(probe, data[16])
    cx(probe, data[17])
    cx(probe, data[18])
    cx(probe, data[19])
    cx(probe, data[20])
    cx(probe, data[21])
    cx(probe, data[22])
    cx(probe, data[23])
    cx(probe, data[24])
    cx(probe, data[25])
    cx(probe, data[26])
    cx(probe, data[27])
    cx(probe, data[28])
    cx(probe, data[29])
    cx(probe, data[30])
    cx(probe, data[31])
    h(probe)
    observed = measure_bit(probe)
    check_input(observed, False)
