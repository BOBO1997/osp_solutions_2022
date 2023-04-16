from typing import *
from utils import *
from dsp import *
import copy
import numpy as np
import scipy as spy
import qiskit.ignis.mitigation as mit


def make_qc_with_observable(qc: QuantumCircuit, 
                            observable: str):
    """
    observable is in little endian
    """
    num_qubits = qc.num_qubits
    num_clbits = num_qubits - observable.count("I")
    qr_ret = QuantumRegister(num_qubits, name="qr")
    cr_ret = ClassicalRegister(num_clbits, name="cr")
    qc_ret = QuantumCircuit(qr_ret, cr_ret)

    qc_ret.compose(qc, qubits=qr_ret, inplace=True)

    cr_index = 0
    for qr_index, pauli_char in enumerate(observable):
        if pauli_char == "X":
            qc_ret.h(qr_index)
            qc_ret.measure(qr_index, cr_index)
            cr_index += 1
        elif pauli_char == "Y":
            qc_ret.sdg(qr_index) # delete the factor of sdg by s, i.e., HSdg Y SH = Z 
            qc_ret.h(qr_index)
            qc_ret.measure(qr_index, cr_index)
            cr_index += 1
        elif pauli_char == "Z":
            qc_ret.measure(qr_index, cr_index)
            cr_index += 1
        else:
            continue
    return qc_ret


def make_qcs_dgse_ps(H: Hamiltonian,
                     qc_rho: QuantumCircuit,
                     power_of_H: int,
                     measurement: bool = True,
                     z_index: int = None,
                     connection_graph: List[List[int]] = None,
                     barrier: bool = False,
                     flatten_qcs: bool = False) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
    """
    this function prepares the quantum circuits for the subspace {I, \rho H^k} of Dual-GSE.
    """
    # define the size of H matrix and S matrix: I and H^0, H^1, ..., H^k 
    num_qubits = qc_rho.num_qubits
    size_matrix = 2 + power_of_H
    qcs_H_tilde = dict()
    for i in range(size_matrix):
        for j in range(size_matrix):
            qcs_element = []
            if i == 0 and j == 0: # degree of rho == 0
                continue # qcs_element should be empty
            elif i == 0 or j == 0: # degree of rho == 1
                for observable, _ in (H ** (i + j)).items():
                    if not observable == "I" * num_qubits:
                        qcs_element.append(make_qc_with_observable(qc=qc_rho,
                                                                   observable=observable))
            else: # degree of rho == 2
                for observable, _ in (H ** (i + j - 1)).items():
                    if observable == "I" * num_qubits:
                        qcs_element.append(make_qc_dsp_ancilla(qc_u=qc_rho,
                                                               qc_v_dual=qc_rho,
                                                               qc_v_inv=None,
                                                               observable=observable, 
                                                               qc_name="denominator", # if observable is I
                                                               measurement=measurement,
                                                               z_index=z_index,
                                                               connection_graph=connection_graph,
                                                               barrier=barrier))  # constant power: 2
                    else:
                        qcs_element.append(make_qc_dsp_ancilla(qc_u=qc_rho,
                                                               qc_v_dual=qc_rho,
                                                               qc_v_inv=None,
                                                               observable=observable,
                                                               qc_name="numerator",
                                                               measurement=measurement,
                                                               z_index=z_index,
                                                               connection_graph=connection_graph,
                                                               barrier=barrier))  # constant power: 2
            qcs_H_tilde[(i, j)] = qcs_element

    qcs_S_tilde = dict()
    for i in range(size_matrix):
        for j in range(size_matrix):
            qcs_element = []
            if i == 0 and j == 0:  # degree of rho == 0
                continue  # qcs_element should be empty
            elif i == 0 or j == 0:  # degree of rho == 1
                for observable, _ in (H ** (i + j - 1)).items():
                    if not observable == "I" * num_qubits:
                        qcs_element.append(make_qc_with_observable(qc=qc_rho,
                                                                   observable=observable))
            else:  # degree of rho == 2
                for observable, _ in (H ** (i + j - 2)).items():
                    if observable == "I" * num_qubits:
                        qcs_element.append(make_qc_dsp_ancilla(qc_u=qc_rho,
                                                               qc_v_dual=qc_rho,
                                                               qc_v_inv=None,
                                                               observable=observable,
                                                               qc_name="denominator",  # if observable is I
                                                               measurement=measurement,
                                                               z_index=z_index,
                                                               connection_graph=connection_graph,
                                                               barrier=barrier))  # constant power: 2
                    else:
                        qcs_element.append(make_qc_dsp_ancilla(qc_u=qc_rho,
                                                               qc_v_dual=qc_rho,
                                                               qc_v_inv=None,
                                                               observable=observable,
                                                               qc_name="numerator",
                                                               measurement=measurement,
                                                               z_index=z_index,
                                                               connection_graph=connection_graph,
                                                               barrier=barrier))  # constant power: 2
            qcs_S_tilde[(i, j)] = qcs_element

    if flatten_qcs:
        qcs_H_tilde_flattened = []
        for _, qcs in qcs_H_tilde.items():
            qcs_H_tilde_flattened += qcs
        qcs_S_tilde_flattened = []
        for _, qcs in qcs_S_tilde.items():
            qcs_S_tilde_flattened += qcs
        return qcs_H_tilde_flattened, qcs_S_tilde_flattened
    else:
        return qcs_H_tilde, qcs_S_tilde


def compute_energy_dgse_ps(hists_H_tilde: List[Counts],
                           hists_S_tilde: List[Counts],
                           H: Hamiltonian,
                           qc_rho: QuantumCircuit,
                           power_of_H: int,
                           hist_type: str = "raw",
                           return_all: bool = False) -> Tuple[complex, np.array]:
    num_qubits = qc_rho.num_qubits
    size_matrix = power_of_H + 2
    if hist_type == "raw":
        hists_H_tilde = [reshape_hist(hist) for hist in hists_H_tilde]
        hists_S_tilde = [reshape_hist(hist) for hist in hists_S_tilde]

    counter = 0
    H_tilde = np.zeros((size_matrix, size_matrix), dtype="complex")
    for i in range(size_matrix):
        for j in range(size_matrix):
            if i == 0 and j == 0:
                H_tilde[i, j] = 1
            elif i == 0 or j == 0:
                for observable, sign in (H ** (i + j)).items():
                    if not observable == "I" * num_qubits:
                        sign *= mit.expectation_value(counts=hists_H_tilde[counter])[0]
                        counter += 1
                    H_tilde[i, j] += sign
            else:
                for observable, sign in (H ** (i + j - 1)).items():
                    if observable == "I" * num_qubits:
                        sign *= compute_trace_dsp_ancilla_denominator(num_qubits, 
                                                                      hists_H_tilde[counter])
                    else:
                        sign *= compute_trace_dsp_ancilla_numerator(num_qubits, 
                                                                    hists_H_tilde[counter])
                    counter += 1
                    H_tilde[i, j] += sign
    assert counter == len(hists_H_tilde)

    counter = 0
    S_tilde = np.zeros((size_matrix, size_matrix), dtype="complex")
    for i in range(size_matrix):
        for j in range(size_matrix):
            if i == 0 and j == 0:
                S_tilde[i, j] = 1 << num_qubits
            elif i == 0 or j == 0:
                for observable, sign in (H ** (i + j - 1)).items():
                    if not observable == "I" * num_qubits:
                        sign *= mit.expectation_value(counts=hists_S_tilde[counter])[0]
                        counter += 1
                    S_tilde[i, j] += sign
            else:
                for observable, sign in (H ** (i + j - 2)).items():
                    if observable == "I" * num_qubits:
                        sign *= compute_trace_dsp_ancilla_denominator(num_qubits,
                                                                      hists_S_tilde[counter])
                    else:
                        sign *= compute_trace_dsp_ancilla_numerator(num_qubits,
                                                                    hists_S_tilde[counter])
                    S_tilde[i, j] += sign
                    counter += 1
    assert counter == len(hists_S_tilde)

    eig_vals, eig_vecs = spy.linalg.eig(H_tilde, S_tilde)
    if return_all:
        return eig_vals, eig_vecs, H_tilde, S_tilde
    max_energy = len(H)
    energy = np.sort(eig_vals.real[- max_energy < eig_vals.real]
                     [eig_vals.real[- max_energy < eig_vals.real] < max_energy])[0]
    eig_vecs = eig_vecs[:, np.argsort(
        eig_vals.real[- max_energy < eig_vals.real][eig_vals.real[- max_energy < eig_vals.real] < max_energy])]
    return energy, eig_vecs[:, 0]
