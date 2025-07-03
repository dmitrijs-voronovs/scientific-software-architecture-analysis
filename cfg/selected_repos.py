from typing import List

from model.Repo import Repo

'''Top 5 repos based on number of stars'''
selected_repos: List[Repo] = [
Repo.from_dict({'author': 'google', 'repo': 'deepvariant', 'version': 'v1.6.1', 'wiki': None}),
Repo.from_dict({'author': 'root-project', 'repo': 'root', 'version': 'v6-32-06', 'wiki': 'https://root.cern'}),
Repo.from_dict({'author': 'OpenGene', 'repo': 'fastp', 'version': 'v0.23.4', 'wiki': None}),
Repo.from_dict({'author': 'scverse', 'repo': 'scanpy', 'version': '1.10.2', 'wiki': 'https://scanpy.readthedocs.io'}),
Repo.from_dict({'author': 'allenai', 'repo': 'scispacy', 'version': 'v0.5.5', 'wiki': 'https://allenai.github.io/scispacy/'}),
]

all_repos: List[Repo] = [
Repo.from_dict({'author': 'google', 'repo': 'deepvariant', 'version': 'v1.6.1', 'wiki': None}),
Repo.from_dict({'author': 'OpenGene', 'repo': 'fastp', 'version': 'v0.23.4', 'wiki': None}),
Repo.from_dict({'author': 'scverse', 'repo': 'scanpy', 'version': '1.10.2', 'wiki': 'https://scanpy.readthedocs.io'}),
Repo.from_dict({'author': 'allenai', 'repo': 'scispacy', 'version': 'v0.5.5', 'wiki': 'https://allenai.github.io/scispacy/'}),
Repo.from_dict({'author': 'qutip', 'repo': 'qutip', 'version': 'v5.0.4', 'wiki': 'https://qutip.org'}),
Repo.from_dict({'author': 'soedinglab', 'repo': 'MMseqs2', 'version': '15-6f452', 'wiki': 'https://github.com/soedinglab/mmseqs2/wiki'}),
Repo.from_dict({'author': 'su2code', 'repo': 'SU2', 'version': 'v8.1.0', 'wiki': 'https://su2code.github.io'}),
Repo.from_dict({'author': 'tum-pbs', 'repo': 'PhiFlow', 'version': '3.1.0', 'wiki': None}),
Repo.from_dict({'author': 'scipipe', 'repo': 'scipipe', 'version': 'v0.12.0', 'wiki': 'https://scipipe.org'}),
Repo.from_dict({'author': 'openbabel', 'repo': 'openbabel', 'version': 'openbabel-3-1-1', 'wiki': 'http://openbabel.org/'}),
Repo.from_dict({'author': 'qupath', 'repo': 'qupath', 'version': 'v0.5.1', 'wiki': 'https://qupath.github.io'}),
Repo.from_dict({'author': 'broadinstitute', 'repo': 'cromwell', 'version': '87', 'wiki': 'https://cromwell.readthedocs.io/en/latest/'}),
Repo.from_dict({'author': 'hail-is', 'repo': 'hail', 'version': '0.2.133', 'wiki': 'https://hail.is'}),
Repo.from_dict({'author': 'psi4', 'repo': 'psi4', 'version': 'v1.9.1', 'wiki': 'https://psicode.org'}),
Repo.from_dict({'author': 'CliMA', 'repo': 'Oceananigans.jl', 'version': 'v0.93.2', 'wiki': 'https://clima.github.io/OceananigansDocumentation/stable'}),
Repo.from_dict({'author': 'sofa-framework', 'repo': 'sofa', 'version': 'v24.06.00', 'wiki': 'https://www.sofa-framework.org'}),
Repo.from_dict({'author': 'stardist', 'repo': 'stardist', 'version': '0.9.1', 'wiki': None}),
Repo.from_dict({'author': 'COMBINE-lab', 'repo': 'salmon', 'version': 'v1.10.1', 'wiki': 'https://combine-lab.github.io/salmon'}),
Repo.from_dict({'author': 'broadinstitute', 'repo': 'gatk', 'version': '4.6.0.0', 'wiki': 'https://software.broadinstitute.org/gatk'}),
Repo.from_dict({'author': 'root-project', 'repo': 'root', 'version': 'v6-32-06', 'wiki': 'https://root.cern'}),
]