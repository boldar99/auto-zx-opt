import stim
from collections import defaultdict


class QGate:
    """A class to represent gates."""
    def __init__(self, name: str, targets: list[int], tag: str='', arg: list[float] = None, classical_reg: list[int]= None ):
        self.name = name
        self.targets = targets
        self.tag = tag
        self.arg= arg
        self.classical_reg= classical_reg

class CustomErrorModel:
    '''Note: errors in the error model are probabiliteis, not infedilities'''
    def __init__(self,*,p_1q: float=0.0,p_2q: float=0.0,p_meas: float =0.0, p_init: float=0.0, p_linear_mem: float=0.0, p_coherent_mem: float=0, p_leakage: float=0.0):
        self.p_1q=p_1q
        self.p_2q=p_2q
        self.p_meas=p_meas
        self.p_init=p_init
        self.p_linear_mem=p_linear_mem
        self.p_coherent_mem=p_coherent_mem
        self.p_leakage=p_leakage


def add_noise_to_ideal_stim_circuit(circ: stim.Circuit, errorModel: CustomErrorModel):
    absteract_circ=stim_circuit_to_abstract_circuit(circ)
    absteract_circ_w_idling=add_idling_gates(absteract_circ)
    return abstract_circuit_to_noisy_stim_circuit(absteract_circ_w_idling, errorModel)

def stim_circuit_to_abstract_circuit(circ: stim.Circuit):
    abst_circ=[]
    for g in circ:
        match g.name:
            case 'I':
                abst_circ.extend([QGate(g.name,[i.value],g.tag,1) for i in g.targets_copy()])
            case 'X'|'Y'|'Z'|'S'|'S_DAG'|'H'|'R'|'RZ'|'M'|'MZ':
                abst_circ.extend([QGate(g.name,[i.value],g.tag) for i in g.targets_copy()])
            case 'CNOT'|'CX'|'CY'|'CZ'|'SWAP':
                abst_circ.extend([QGate(g.name,[i.value, j.value],g.tag) for i,j in zip(g.targets_copy()[::2],g.targets_copy()[1::2])])
            case _:
                abst_circ.append(QGate(g.name,[i.value for i in g.targets_copy()],g.tag,g.gate_args_copy()))

    #fix classical feedback gates
    for g in abst_circ:
        if g.name not in ['DETECTOR','OBSERVABLE_INCLUDE'] and g.targets[0]<0:
            #it's a classical feedback gate
            g.name='C'+g.name
            g.classical_reg=[g.targets[0]]
            del g.targets[0]

    return abst_circ

def abstract_circuit_to_noisy_stim_circuit(abs_circ: list[QGate], errorModel: CustomErrorModel = None):
    if errorModel is None:
        errorModel=CustomErrorModel()

    circ=stim.Circuit()
    for g in abs_circ:
        if g.tag=='perfect':
            if g.name in ['CCNOT','CCX','CCZ']:
                # print(g.classical_reg)
                circ.append(g.name[1:],[stim.target_rec(g.classical_reg[0])]+g.targets,g.arg,tag=g.tag)
            else:
                circ.append(g.name,g.targets,g.arg,tag=g.tag)
            continue
        match g.name:
            case 'X'|'Y'|'Z'|'S'|'S_DAG'|'H':
                circ.append(g.name,g.targets,tag=g.tag)
                if errorModel.p_1q>0:
                    circ.append('DEPOLARIZE1',g.targets,errorModel.p_1q)
            case 'CNOT'|'CX'|'CY'|'CZ'|'SWAP':
                circ.append(g.name,g.targets,tag=g.tag)
                if errorModel.p_2q>0:
                    circ.append('DEPOLARIZE2',g.targets,errorModel.p_2q)
            case 'CCNOT'|'CCX'|'CCZ':
                circ.append(g.name[1:],[stim.target_rec(g.classical_reg[0])]+g.targets,g.arg,tag=g.tag)
                if errorModel.p_1q>0:
                    circ.append('DEPOLARIZE1',g.targets[0],errorModel.p_1q)
            case 'R'|'RZ':
                circ.append(g.name,g.targets,tag=g.tag)
                if errorModel.p_init>0:
                    circ.append('X_ERROR',g.targets,errorModel.p_init)
            case 'M'|'MZ':
                circ.append(g.name,g.targets,errorModel.p_meas,tag=g.tag)
            case 'HERALDED_ERASE':
                p = 0.001 if g.arg is None else g.arg
                circ.append(g.name, g.targets, p, tag=g.tag)
            case 'IDLE':
                if g.arg>0:
                    if errorModel.p_linear_mem>0:
                        circ.append('Z_ERROR',g.targets,min(.5,g.arg*errorModel.p_linear_mem))
                    if errorModel.p_coherent_mem>0:
                        circ.append('Z_ERROR',g.targets,min(.5,(g.arg**2)*errorModel.p_coherent_mem))
                    if errorModel.p_leakage>0:
                        circ.append('DEPOLARIZE1',g.targets,min(0.5,g.arg*errorModel.p_leakage))
            case 'DETECTOR':
                circ.append(g.name,[stim.target_rec(i) for i in g.targets],tag=g.tag)
            case 'OBSERVABLE_INCLUDE':
                circ.append(g.name,[stim.target_rec(i) for i in g.targets],[int(i) for i in g.arg], tag=g.tag)
            case _:
                raise ValueError(f'gate {g.name} is not supported in the abstract circuit')
    return circ


def add_idling_gates(circ: list[QGate], annotations: list[str]=['DETECTOR','OBSERVABLE_INCLUDE', 'HERALDED_ERASE']):
    '''given a circuit as a list of QGates, return a circuit which adds IDLE gates with arg equal to how many
    depth-1 are there between two consecutive gates on each qubit (or untill the end of the circuit if it's the last gate).
    In a fully dense circuit, there are 1 depth-1 idling between each 2-qubit gate.
    It ignores any gate whose name is included in the annotations argument when calculating idling times between
    consecutive gates'''

    new_circ=[[] for _ in range(len(circ))]
    circ_w_indexed_tags=[QGate(g.name,g.targets,g.tag+f'__{t}',g.arg,g.classical_reg) for t,g in enumerate(circ)]


    for g in circ_w_indexed_tags[::-1]:
        if g.name in annotations:
            t=int(g.tag.split('__')[-1])
            original_tag=g.tag[:g.tag.rfind('__')]
            new_circ[t].append(QGate(g.name,g.targets,original_tag,g.arg,g.classical_reg))
            del circ_w_indexed_tags[t]

    circ_partitioned=partition_circ(circ_w_indexed_tags)

    #remove identity gates from slices (they have served their purpose in shaping circuit slices)
    circ_partitioned=[[g for g in part if g.name!='I'] for part in circ_partitioned]

    for d,part in enumerate(circ_partitioned):
        for g in part:
            t=int(g.tag.split('__')[-1])
            original_tag=g.tag[:g.tag.rfind('__')]
            new_circ[t].append(QGate(g.name,g.targets,original_tag,g.arg,g.classical_reg))
            for q in g.targets:
                dd=d+1
                while dd<len(circ_partitioned):
                    if q not in [i for g in circ_partitioned[dd] for i in g.targets]:
                        dd+=1
                    else:
                        break

                new_circ[t].append(QGate('IDLE',[q],'',dd-d-1))
    return [g for gs in new_circ for g in gs]



def partition_circ(C: list[QGate] ):
    '''
    given a circuit as an array of instructions
    C=[(gate_name_1: str,gate_support_1: List[int]),(gate_name_2: str,gate_support_2: List[int]), ...]
    return C_partitioned = [C_1,C_2,...] such that each C_i is a subset of C, all gates in C_i can be applied simultaneously,
    and applying C1 then C2 then C3 etc will result in the same cirucit as C. For example the depth of C = len(C_partitioned).
    this is a naive implimentation and is not necessarily optimal. It first pushes back everything so every gate is applied as soon as possible.
    then it push forward everything except measurements to happen as late as possible (helps with memory error. e.g. better to reset a qubit as late as possible)
    '''
    CC=C.copy()

    C_partitioned=[]
    while CC:
        partition=[]
        engaged_qubits=set()

        idx=0
        while idx<len(CC):
            gate=CC[idx]
            if not(engaged_qubits & set(gate.targets)):
                partition.append(gate)
                engaged_qubits.update(gate.targets)
                del CC[idx]
            else:
                engaged_qubits.update(gate.targets)
                idx+=1

        C_partitioned.append(partition)

    depth=len(C_partitioned)

    #postpone everything except measurements to as late as possible
    for i in range(depth-2,-1,-1):
        idx=0
        while idx<len(C_partitioned[i]):
            gate=C_partitioned[i][idx]
            if gate.name=='M':
                idx+=1
                continue
            #at the end of the while loop, j would be the index of the latest depth
            #that this gate can be pushed to.
            j=i
            while j<depth-1:
                if any([q in gate1.targets for q in gate.targets for gate1 in C_partitioned[j+1]]):
                    break
                j+=1
            if j==i:
                idx+=1
            else:
                del C_partitioned[i][idx]
                C_partitioned[j].append(gate)

    return C_partitioned

def abst_circ_diagram(abst_circ: list[QGate]):
    def asBlock(s,filler='-',length=8):
        s=s[:length-1]
        return s+filler*(length-len(s))

    pc=partition_circ(abst_circ)
    qs=set(sorted([q for gs in pc for g in gs for q in g.targets]))
    for q in qs:
        wire=asBlock(str(q)+':',' ',4)
        for gs in pc:
            gn=next((g.name for g in gs if q in g.targets),'')
            wire+=asBlock(gn)
        print(wire)



def default_noise_model() -> CustomErrorModel:
    """Return a fresh instance of the default hardware noise model."""
    return CustomErrorModel(
        p_1q=2.5e-5,
        p_2q=8e-4,
        p_meas=1e-6,
        p_init=8e-4,
        p_linear_mem=6.3e-4 * 0.02,
    )


def apply_noise(circuit: stim.Circuit, error_model: CustomErrorModel | None = None) -> stim.Circuit:
    """Apply gate and inferred idle noise, using the default model if omitted."""
    if error_model is None:
        error_model = default_noise_model()
    return add_noise_to_ideal_stim_circuit(circuit, error_model)
