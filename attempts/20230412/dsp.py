from typing import *
from qiskit.circuit import Gate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from utils import *
import numpy as np

def make_connection_order(z_index: int, connection_graph: List[List[int]]):
    """
    # TODO
    CURRENTLY IDENTITY: TO BE REFINED!!!!
    """
    ret = connection_graph
    return ret


def update_observable_dsp(observable: Union[QuantumCircuit, Gate, str], z_index: int, connection_graph: List[List[int]]) -> QuantumCircuit:
    """
    Take Pauli string and return the corrsponding Gate operation
    observable should be little endian
    """
    if isinstance(observable, QuantumCircuit):  # do nothing
        return observable
    elif isinstance(observable, Gate):  # convert to quantum circuit
        qr_gate2qc = QuantumRegister(observable.num_qubits)
        qc_gate2qc = QuantumCircuit(qr_gate2qc, name="observable")
        qc_gate2qc.append(observable, qr_gate2qc)
        return qc_gate2qc
    elif isinstance(observable, str):  # make a corresponding quantum circuit
        observable = list(observable)
        qr_observable = QuantumRegister(len(observable))
        qc_observable = QuantumCircuit(qr_observable)
        for i, pauli in enumerate(observable):
            if pauli == "I" or pauli == "Z":  # add X measurement base change gates
                continue
            elif pauli == "X":
                qc_observable.h(qr_observable[i])
                observable[i] = "Z"
            elif pauli == "Y":  # add Y measurement base change gates
                qc_observable.sdg(qr_observable[i])
                qc_observable.h(qr_observable[i])
                observable[i] = "Z"
            else:
                raise Exception
        # transform Z to I except for z_index under the constraints of qubit connection
        connection_order = make_connection_order(z_index, connection_graph)
        for edge in connection_order[::-1]:
            # swap: IZ -> ZI
            if observable[edge[0]] == "I" and observable[edge[1]] == "Z": # and not (edge[0] == z_index and observable[z_index] == "Z"):
                qc_observable.cx(edge[0], edge[1])
                qc_observable.cx(edge[1], edge[0])
                observable[edge[0]], observable[edge[1]] = observable[edge[1]], observable[edge[0]]
            # remove: ZZ -> ZI
            elif observable[edge[0]] == "Z" and observable[edge[1]] == "Z":
                qc_observable.cx(edge[1], edge[0])
                observable[edge[1]] = "I"
        return qc_observable
    else:
        raise Exception
        
        
def make_qc_dsp_ancilla(qc_u: QuantumCircuit = None, 
                        qc_v_dual: QuantumCircuit = None, 
                        qc_v_inv: QuantumCircuit = None,
                        observable: Union[QuantumCircuit, Gate, str] = None, 
                        qc_name: str = None, 
                        measurement: bool = True,
                        z_index: int = None,
                        connection_graph: List[List[int]] = None,
                        barrier: bool = False) -> QuantumCircuit:
    n = qc_u.num_qubits
    observable = update_observable_dsp(observable, z_index, connection_graph)
    if qc_name == "denominator":
        qc_name = "X"
    elif qc_name == "numerator":
        qc_name = "Z"

    qr = QuantumRegister(n, name="qr")
    qr_anc = QuantumRegister(1, name="qr_anc")
    cr = ClassicalRegister(n, name="cr")
    cr_anc = ClassicalRegister(1, name="cr_anc")
    qc_dsp = QuantumCircuit(qr_anc, qr, cr_anc, cr, name="qc_dsp_"+qc_name)

    # add the target quantum circuit
    qc_dsp.compose(qc_u, qr, inplace=True)
    if barrier:
        qc_dsp.barrier()

    # add the observable circuit
    qc_dsp.compose(observable, qr, inplace=True)
    if barrier:
        qc_dsp.barrier()

    # add the CNOT gate corresponding to the intermediate measurement
    if z_index >= 0:  # z_index == -1 means the observable with all "I"
        qc_dsp.cx(qr[z_index], qr_anc)
    if qc_name == "Y":
        qc_dsp.sdg(qr_anc)
    if qc_name == "X" or qc_name == "Y":
        qc_dsp.h(qr_anc)
    if barrier:
        qc_dsp.barrier()

    # add the inverse of the observable circuit
    qc_dsp.compose(observable.inverse(), qr, inplace=True)
    if barrier:
        qc_dsp.barrier()

    # add the inverse of the target (dual) quantum circuit
    if qc_v_inv is not None: # directly add inverse operation of the dual circuit
        qc_dsp.compose(qc_v_inv, qr, inplace=True)
    elif qc_v_dual is not None: # add the inversed dual circuit
        qc_dsp.compose(qc_v_dual.inverse(), qr, inplace=True)
    else: # add the inversed original circuit as the dual circuit
        qc_dsp.compose(qc_u.inverse(), qr, inplace=True)
    if barrier:
        qc_dsp.barrier()

    # measure all
    if measurement:
        qc_dsp.measure(qr_anc, cr_anc)
        qc_dsp.measure(qr, cr)

    return qc_dsp


from qiskit.result import Result, Counts
import qiskit.ignis.mitigation as mit

def make_conditioned_hist(hist: Dict[str, Union[int, float]], condition_state: str, qr_poses: List[int]) -> Dict[str, Union[int, float]]:
    """
    Return the histogram conditioned by the conditioned state and its position
    all data here are supposed to be in little endian
    """
    n = len(next(iter(hist)))
    conditioned_hist = dict()
    # strlen = len(next(iter(hist)))
    for state, count in hist.items():
        state_conditioning = "".join([state[qr_pos] for qr_pos in qr_poses]).replace(" ", "")
        state_conditioned = "".join([state[qr_pos] for qr_pos in range(
            n) if qr_pos not in qr_poses]).replace(" ", "")
        if state_conditioning == condition_state:
            # only works for 1-qubit case. the ancilla qubit is assumed to be in the head of string
            conditioned_hist[state_conditioned] = count
    return conditioned_hist


def remove_space_from_hist_states(hist: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    """
    ### correctness is verified
    remove the space from the labels of histogram
    """
    replaced_hist = dict()
    for state, count in hist.items():
        replaced_hist[state.replace(" ", "")] = count
    return replaced_hist


def reshape_hist(hist: Dict[str, Union[int, float]]) -> List[Dict[str, Union[int, float]]]:
    return {k[::-1]: v for k, v in remove_space_from_hist_states(hist).items()}


def compute_trace_dsp_ancilla_numerator(n: int, hist: Dict[str, Union[int, float]]) -> float:
    za_0, _ = mit.expectation_value(make_conditioned_hist(hist, "0" * n, range(1, n + 1)))
    zp_0_tilde = (hist.get("0" + "0" * n, 0) + hist.get("1" + "0" * n, 0)) / sum(list(hist.values()))
    return za_0 * zp_0_tilde

def compute_trace_dsp_ancilla_denominator(n: int, hist: Dict[str, Union[int, float]]) -> float:
    xa_0, _ = mit.expectation_value(make_conditioned_hist(hist, "0" * n, range(1, n + 1)))
    xp_0_tilde = (hist.get("0" + "0" * n, 0) + hist.get("1" + "0" * n, 0)) / sum(list(hist.values()))
    return (1 + xa_0) * xp_0_tilde