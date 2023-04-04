from typing import *
from utils import *
from dsp import *
import numpy as np
import scipy as spy

def make_qcs_gse_dsp_ancilla(hamiltonian: Hamiltonian = None,
                             sigma_list: List[Tuple[QuantumCircuit, int]] = None,
                             sigma_list_inv: List[Tuple[QuantumCircuit, int]] = None,
                             pauli_list: List[Tuple[Hamiltonian, int]] = None,
                             measurement: bool = True,
                             z_index: int = None, 
                             connection_graph: List[List[int]] = None,
                             barrier: bool = False) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
    qcs_H_tilde = []
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            for observable, _ in (hamiltonian * (pauli_list[i][0] ** pauli_list[i][1]) * (pauli_list[j][0] ** pauli_list[j][1])).items():
                if observable == "I" * sigma_list[0][0].num_qubits:
                    qcs_H_tilde.append(make_qc_dsp_ancilla(qc_u=sigma_list[i][0],
                                                           qc_v_dual=sigma_list[j][0],
                                                           qc_v_inv=sigma_list_inv[j][0] if sigma_list_inv is not None else None,
                                                           observable=observable,
                                                           qc_name="denominator",
                                                           measurement=measurement,
                                                           z_index=z_index,
                                                           connection_graph=connection_graph,
                                                           barrier=barrier))  # constant power: 2
                else:
                    qcs_H_tilde.append(make_qc_dsp_ancilla(qc_u=sigma_list[i][0],
                                                           qc_v_dual=sigma_list[j][0],
                                                           qc_v_inv=sigma_list_inv[j][0] if sigma_list_inv is not None else None,
                                                           observable=observable,
                                                           qc_name="numerator",
                                                           measurement=measurement,
                                                           z_index=z_index,
                                                           connection_graph=connection_graph,
                                                           barrier=barrier))  # constant power: 2

    qcs_S_tilde = []
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            for observable, _ in ((pauli_list[i][0] ** pauli_list[i][1]) * (pauli_list[j][0] ** pauli_list[j][1])).items():
                if observable == "I" * sigma_list[0][0].num_qubits:
                    qcs_S_tilde.append(make_qc_dsp_ancilla(qc_u=sigma_list[i][0],
                                                           qc_v_dual=sigma_list[j][0],
                                                           qc_v_inv=sigma_list_inv[j][0] if sigma_list_inv is not None else None,
                                                           observable=observable,
                                                           qc_name="denominator",
                                                           measurement=measurement,
                                                           z_index=z_index,
                                                           connection_graph=connection_graph,
                                                           barrier=barrier))  # constant power: 2
                else:
                    qcs_S_tilde.append(make_qc_dsp_ancilla(qc_u=sigma_list[i][0],
                                                           qc_v_dual=sigma_list[j][0],
                                                           qc_v_inv=sigma_list_inv[j][0] if sigma_list_inv is not None else None,
                                                           observable=observable,
                                                           qc_name="numerator",
                                                           measurement=measurement,
                                                           z_index=z_index,
                                                           connection_graph=connection_graph,
                                                           barrier=barrier))  # constant power: 2

    return qcs_H_tilde, qcs_S_tilde


def compute_energy_gse_dsp_ancilla(hists_H_tilde: List[Counts],
                                   hists_S_tilde: List[Counts],
                                   hamiltonian: Hamiltonian,
                                   sigma_list: List[List[Tuple[QuantumCircuit, int]]],
                                   pauli_list: List[Tuple[Hamiltonian, int]],
                                   hist_type: str = "raw",
                                   return_all: bool = False) -> Tuple[complex, np.array]:
    num_qubits = sigma_list[0][0].num_qubits
    if hist_type == "raw":
        hists_H_tilde = [reshape_hist(hist) for hist in hists_H_tilde]
        hists_S_tilde = [reshape_hist(hist) for hist in hists_S_tilde]

    counter = 0
    H_tilde = np.zeros((len(sigma_list), len(sigma_list)), dtype="complex")
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            for observable, sign in (hamiltonian * (pauli_list[i][0] ** pauli_list[i][1]) * (pauli_list[j][0] ** pauli_list[j][1])).items():
                if observable == "I" * num_qubits:
                    sign *= compute_trace_dsp_ancilla_denominator(num_qubits, hists_H_tilde[counter])
                else:
                    sign *= compute_trace_dsp_ancilla_numerator(num_qubits, hists_H_tilde[counter])
                counter += 1
                H_tilde[i, j] += sign
    assert counter == len(hists_H_tilde)

    counter = 0
    S_tilde = np.zeros((len(sigma_list), len(sigma_list)), dtype="complex")
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            for observable, sign in ((pauli_list[i][0] ** pauli_list[i][1]) * (pauli_list[j][0] ** pauli_list[j][1])).items():
                if observable == "I" * num_qubits:
                    sign *= compute_trace_dsp_ancilla_denominator(num_qubits, hists_S_tilde[counter])
                else:
                    sign *= compute_trace_dsp_ancilla_numerator(num_qubits, hists_S_tilde[counter])
                counter += 1
                S_tilde[i, j] += sign
    assert counter == len(hists_S_tilde)

    eig_vals, eig_vecs = spy.linalg.eig(H_tilde, S_tilde)
    if return_all:
        return eig_vals, eig_vecs, H_tilde, S_tilde
    max_energy = len(hamiltonian)
    energy = np.sort(eig_vals.real[- max_energy < eig_vals.real][eig_vals.real[- max_energy < eig_vals.real] < max_energy])[0]
    eig_vecs = eig_vecs[:, np.argsort(eig_vals.real[- max_energy < eig_vals.real][eig_vals.real[- max_energy < eig_vals.real] < max_energy])]
    return energy, eig_vecs[:, 0]