import time
import numpy as np


from qiskit import *
from qiskit.circuit.library import *
from qiskit_aer import *

from qiskit.quantum_info.random import random_unitary
from qiskit.circuit import CircuitInstruction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('depth',type=int)
args = parser.parse_args()
print(args.depth)

def makeQuantumVolume(num_qubits,depth,seed):
        if seed is None:
            rng_set = np.random.default_rng()
            seed = rng_set.integers(low=1, high=1000)
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Parameters
        depth = depth or num_qubits  # how many layers of SU(4)
        width = int(np.floor(num_qubits / 2))  # how many SU(4)s fit in each layer
        name = "quantum_volume_" + str([num_qubits, depth, seed]).replace(" ", "")

        # Generator random unitary seeds in advance.
        # Note that this means we are constructing multiple new generator
        # objects from low-entropy integer seeds rather than pass the shared
        # generator object to the random_unitary function. This is done so
        # that we can use the integer seed as a label for the generated gates.
        unitary_seeds = rng.integers(low=0, high=99, size=[depth, width])

        # For each layer, generate a permutation of qubits
        # Then generate and apply a Haar-random SU(4) to each pair
        circuit = QuantumCircuit(num_qubits, name=name)
        qubits = circuit.qubits
        perm_0 = list(range(num_qubits))
        gates = [random_unitary(4, seed=0).to_instruction() for i in range(100)]
        for d in range(depth):
            perm = rng.permutation(perm_0)
            for w in range(width):
                seed_u = unitary_seeds[d][w]
                su4 = gates[seed_u]
                su4.label = "su4_" + str(seed_u)
                physical_qubits = int(perm[2 * w]), int(perm[2 * w + 1])
                #circuit.compose(su4, [physical_qubits[0], physical_qubits[1]], inplace=True)
                instruction = CircuitInstruction(su4,[qubits[physical_qubits[0]],qubits[physical_qubits[1]]] , [])
                circuit._append(instruction)

        
        return circuit


GpuCount=1
def doRun(shots,depth,qubits,sim,device,cuStateVec_enable,blocking_qubits,blocking_enable):
    t2 = time.time()
#    circuit = transpile(QuantumVolume(qubits,depth,seed=0),
#                        backend=sim,
#                        optimization_level=0)
#    circuit = QuantumVolume(qubits,depth,seed=0)
#    circuit = circuit.decompose()
#    print("transpile Complete")
    circuit = makeQuantumVolume(qubits,depth,0)
    circuit.measure_all()
    print("circuit Complete")
    circuits = [circuit for i in range(1)]

    t0 = time.time()
    sim.set_options(cuStateVec_enable=cuStateVec_enable)
    sim.set_options(max_parallel_experiments=4)
    sim.set_options(max_parallel_threads=4)
#    sim.set_options(batched_shots_gpu=True)
    sim.set_options(fusion_enable=False)
    sim.set_options(blocking_enable=blocking_enable)
    sim.set_options(blocking_qubits=blocking_qubits)
    results = sim.run(circuits,shots=shots,seed_simulator=12345).result()
    t1 = time.time()
    print(results._metadata["metadata"])
    print(results.results[0].to_dict()['metadata'])
    print(results.results[0].status)

    print(f"Config:{(shots,depth,qubits,device)}, Time:{t1-t0}")
    with open("Results.txt",'a') as f:
        if cuStateVec_enable:
            device = device + "_cuStateVec"
        f.write(f"{shots},{depth},{qubits},{GpuCount}_{device},{blocking_qubits},{blocking_enable},{t1-t0},{t0-t2}\n")
#    with open("Log.txt","a") as f:
#        f.write(str(results.results[0].to_dict()['metadata'])+"\n")
    return t1-t0

devices = ("GPU",)
blocking_enable = True 
for device in devices:
    sim = AerSimulator(method='statevector',device=device)
#    for blocking_qubits in np.arange(20,28,2):
    blocking_qubits=20
    for qubits in [34,36,38]:
            shots = 1
            depth = args.depth
            timeTaken = doRun(shots,depth,qubits,sim,device,False,blocking_qubits,blocking_enable)
            time2 = doRun(shots,depth,qubits,sim,device,True,blocking_qubits,blocking_enable)
            if timeTaken > 1000 or time2 > 1000:
               break

