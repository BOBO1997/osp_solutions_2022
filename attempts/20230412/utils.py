from typing import *
from collections.abc import Mapping
import copy
import numpy as np
import qiskit.quantum_info as qi

class Hamiltonian(Mapping):
    def __init__(self, input: Dict[str, complex]):
        self.num_qubits = None if len(
            input) == 0 else len(list(input.keys())[0])
        self.operators = input
        self.pindex = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        self.ptable = [[("I", 1.0), ("X", 1.0), ("Y", 1.0), ("Z", 1.0)],
                       [("X", 1.0), ("I", 1.0), ("Z", 1.j), ("Y", -1.j)],
                       [("Y", 1.0), ("Z", -1.j), ("I", 1.0), ("X", 1.j)],
                       [("Z", 1.0), ("Y", 1.j), ("X", -1.j), ("I", 1.0)]]

    def __add__(self, input):
        ret = copy.deepcopy(self.operators)
        for operator, coeff in input.items():
            if operator in ret:
                ret[operator] += coeff
            else:
                ret[operator] = coeff
            if ret[operator] == 0:
                del ret[operator]
        return Hamiltonian(ret)

    def __sub__(self, input):
        ret = copy.deepcopy(self.operators)
        for operator, coeff in input.items():
            if operator in ret:
                ret[operator] -= coeff
            else:
                ret[operator] = - coeff
            if ret[operator] == 0:
                del ret[operator]
        return Hamiltonian(ret)

    def __mul__(self, input):
        ret = dict()
        for left_operator, left_coeff in self.operators.items():
            for right_operator, right_coeff in input.items():
                temp_operator = ""
                temp_coeff = left_coeff * right_coeff # compute the multiplication of coefficients between left and right pauli strings
                for left_pauli, right_pauli in zip(left_operator, right_operator):
                    new_pauli, new_coeff = self.ptable[self.pindex[left_pauli]][self.pindex[right_pauli]]
                    temp_operator += new_pauli
                    temp_coeff *= new_coeff
                if temp_operator in ret: # add multiplied coefficient
                    ret[temp_operator] += temp_coeff
                else: # create a new pauli string and add multiplied coefficient
                    ret[temp_operator] = temp_coeff
                if ret[temp_operator] == 0: # if the updated pauli str becomes zero, then we empty it to save memory
                    del ret[temp_operator]
        return Hamiltonian(ret)

    def __pow__(self, input):
        if input == 0:
            return Hamiltonian({"I"*len(next(iter(self.operators))): 1.0})
        if input == 1:
            return Hamiltonian(self)
        else:
            ret = self ** (input - 1)
            return self * ret

    def __setitem__(self, key, value):
        self.operators[key] = value

    def __getitem__(self, key):
        return self.operators.get(key, 0.0)

    def __iter__(self):
        return iter(self.operators)

    def __len__(self):
        return len(self.operators)

    def __repr__(self):
        return repr(self.operators)

    def __str__(self):
        return str(self.operators)

    def reduce_identity(self, inplace=False):
        if inplace:
            del self.operators["I" * self.num_qubits]
        else:
            ret = Hamiltonian({})
            for pauli_str, coeff in self.items():
                if not pauli_str == "I" * len(pauli_str):
                    ret[pauli_str] = coeff
            return ret

from typing import *
from qiskit.quantum_info import DensityMatrix
from qiskit.result import Counts, Result


class ExtendedDM(DensityMatrix):
    def __init__(self, matrix, dims=None):
        super().__init__(matrix, dims=dims)

    def __matmul__(self, other):
        if isinstance(other, DensityMatrix):
            return ExtendedDM(self._data @ other._data, dims=self.dims())
        else:
            return ExtendedDM(self._data @ other, dims=self.dims())
        
    def __pow__(self, k):
        if k == 0:
            return ExtendedDM(np.eye(self.dim), dims=self.dims())
        if k == 1:
            return ExtendedDM(self._data, dims=self.dims())
        else:
            ret = self ** (k - 1)
            return self @ ret
    
    def dagger(self):
        return ExtendedDM(self._data.T.conjugate(), dims=self.dims())
    
    def tensor_pow(self, k):
        if k <= 0:
            return 0
        elif k == 1:
            return ExtendedDM(self._data, dims=self.dims())
        else:
            return ExtendedDM(self._data, dims=self.dims()) ^ ExtendedDM(self._data, dims=self.dims()).tensor_pow(k - 1)

    def normalize(self):
        return ExtendedDM(self._data / self.trace(), dims=self.dims())
    
    def partial_trace(self, qubits: List[int], normalize: bool = True):
        """
        qubtis: the subsystem to be traced over
        [with normalization]
        """
        if normalize:
            return ExtendedDM(qi.partial_trace(self._data, qubits)).normalize()
        else:
            return ExtendedDM(qi.partial_trace(self._data, qubits))
    
    def apply_unitary(self, U: DensityMatrix, normalize: bool = True):
        """
        [with normalization]
        """
        ret_raw = U @ self @ U.dagger()
        if normalize:
            return ExtendedDM(U @ self @ U.dagger()).normalize()
        else:
            return ExtendedDM(U @ self @ U.dagger())

def pauli_str_to_dm(pauli_str) -> ExtendedDM:
    ret = 1
    for pauli in pauli_str:
        if pauli == "I":
            dm = np.array([[1, 0], [0, 1]], dtype="complex")
        elif pauli == "X":
            dm = np.array([[0, 1], [1, 0]], dtype="complex")
        elif pauli == "Y":
            dm = np.array([[0, -1j], [1j, 0]], dtype="complex")
        elif pauli == "Z":
            dm = np.array([[1, 0], [0, -1]], dtype="complex")
        else:
            Exception
        ret = np.kron(ret, dm)
    return ExtendedDM(ret)

def hamiltonian_to_dm(hamiltonian):
    ret = None
    for pauli_str, coeff in hamiltonian.items():
        if ret is None:
            ret = pauli_str_to_dm(pauli_str) * coeff
        else:
            ret += pauli_str_to_dm(pauli_str) * coeff
    return ret

def add_observable_to_dm(density_matrix: DensityMatrix, basis: str, dm_endian: str = "big_endian", basis_endian: str = "little_endian") -> ExtendedDM:
    """
    density_matrix, basis -> both little-endian
    """
    flag_reverse_endian = 1 if basis_endian == dm_endian else -1
    observable = 1
    for pauli_char in basis[::flag_reverse_endian]:
        if pauli_char == "I":
            observable = np.kron(observable, np.array([[1, 0],
                                                       [0, 1]], dtype="complex"))
        elif pauli_char == "X":
            observable = np.kron(observable, np.array([[0, 1],
                                                       [1, 0]], dtype="complex"))
        elif pauli_char == "Y":
            observable = np.kron(observable, np.array([[0, -1j],
                                                       [1j, 0]], dtype="complex"))
        elif pauli_char == "Z":
            observable = np.kron(observable, np.array([[1, 0],
                                                       [0, -1]], dtype="complex"))
        else:
            print("unexpected pauli character is detected:", pauli_char)
            raise Exception
    return ExtendedDM(observable) @ density_matrix # big_endian by default (return is based on the endian of density matrix)

def dm_to_hist(density_matrix: DensityMatrix, dm_endian: str = "big_endian", hist_endian: str = "big_endian") -> Counts:
    """
    density_matrix: big_endian (given default)
    hist: big_endian (default)
    """
    num_qubits = int(np.log2(density_matrix.dim))
    hist = dict()
    flag_reverse_endian = 1 if hist_endian == dm_endian else -1
    # iteration start from 00...0 to 11...1
    for dm_index in range(0, density_matrix.dim):
        if np.abs(density_matrix.data[dm_index, dm_index]) > 0:
            hist[format(dm_index, "0"+str(num_qubits)+"b")[::flag_reverse_endian]] = np.abs(density_matrix.data[dm_index, dm_index])
    return Counts(hist)

def dms_to_hists(dm_list: List[DensityMatrix],  dm_endian: str = "big_endian", hist_endian: str = "big_endian") -> List[Counts]:
    return [dm_to_hist(dm, dm_endian, hist_endian) for dm in dm_list]

def get_density_matrices(result: Result) -> List[DensityMatrix]:
    """
    dm_list: list of density matrix with big-endian
    """
    dm_list = []
    for experimental_result in result.results:
        dm_list.append(experimental_result.data.density_matrix)
    return dm_list


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def add_observable(qc: QuantumCircuit, 
                   qr: QuantumRegister, 
                   cr: ClassicalRegister, 
                   pauli_str: str, 
                   inplace=False) -> Union[None, QuantumCircuit]:
    """
    density_matrix, basis -> both little-endian
    """
    if not inplace:
        print("do not use this option")
        qc = copy.deepcopy(qc)
        raise Exception
        
    idx_cr = 0
    for idx_qr, pauli_char in enumerate(pauli_str):
        if pauli_char == "I":
            continue
        else:
            if pauli_char == "Y": ###  ###
                qc.sdg(qr[idx_qr]) # HSdg Y SH = Z
            if pauli_char == "Y" or pauli_char == "X":
                qc.h(qr[idx_qr])
            if pauli_char == "Y" or pauli_char == "X" or pauli_char == "Z":
                qc.measure(qr[idx_qr], cr[idx_cr])
                idx_cr += 1
            else:
                print("unexpected pauli character is detected:", pauli_char, "in pauli_str:", pauli_str, "(idx_qr:", idx_qr, ", idx_cr:", idx_cr, ")")
                raise Exception
    if not inplace:
        return qc
    
    
from qiskit.result import Result, Counts

def connect_cregs(result: Result):
    result_new = copy.deepcopy(result)
    for experimentresult in result_new.results:
        num_clbits = len(experimentresult.header.clbit_labels)
        experimentresult.header.clbit_labels = [["cr", i] for i in range(num_clbits)]
        experimentresult.header.creg_sizes = [["cr", num_clbits]]
    return result_new
