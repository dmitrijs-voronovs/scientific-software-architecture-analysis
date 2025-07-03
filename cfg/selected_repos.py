from typing import List

from models.Repo import Repo

'''Top 5 repos based on number of stars'''
selected_repos: List[Repo] = [
Repo.from_dict({'author': 'google', 'name': 'deepvariant', 'version': 'v1.6.1', 'wiki': None}),
Repo.from_dict({'author': 'root-project', 'name': 'root', 'version': 'v6-32-06', 'wiki': 'https://root.cern'}),
Repo.from_dict({'author': 'OpenGene', 'name': 'fastp', 'version': 'v0.23.4', 'wiki': None}),
Repo.from_dict({'author': 'scverse', 'name': 'scanpy', 'version': '1.10.2', 'wiki': 'https://scanpy.readthedocs.io'}),
Repo.from_dict({'author': 'allenai', 'name': 'scispacy', 'version': 'v0.5.5', 'wiki': 'https://allenai.github.io/scispacy/'}),
]

all_repos: List[Repo] = [
Repo.from_dict({'author': 'google', 'name': 'deepvariant', 'version': 'v1.6.1', 'wiki': None}),
Repo.from_dict({'author': 'OpenGene', 'name': 'fastp', 'version': 'v0.23.4', 'wiki': None}),
Repo.from_dict({'author': 'scverse', 'name': 'scanpy', 'version': '1.10.2', 'wiki': 'https://scanpy.readthedocs.io'}),
Repo.from_dict({'author': 'allenai', 'name': 'scispacy', 'version': 'v0.5.5', 'wiki': 'https://allenai.github.io/scispacy/'}),
Repo.from_dict({'author': 'qutip', 'name': 'qutip', 'version': 'v5.0.4', 'wiki': 'https://qutip.org'}),
Repo.from_dict({'author': 'soedinglab', 'name': 'MMseqs2', 'version': '15-6f452', 'wiki': 'https://github.com/soedinglab/mmseqs2/wiki'}),
Repo.from_dict({'author': 'su2code', 'name': 'SU2', 'version': 'v8.1.0', 'wiki': 'https://su2code.github.io'}),
Repo.from_dict({'author': 'tum-pbs', 'name': 'PhiFlow', 'version': '3.1.0', 'wiki': None}),
Repo.from_dict({'author': 'scipipe', 'name': 'scipipe', 'version': 'v0.12.0', 'wiki': 'https://scipipe.org'}),
Repo.from_dict({'author': 'openbabel', 'name': 'openbabel', 'version': 'openbabel-3-1-1', 'wiki': 'http://openbabel.org/'}),
Repo.from_dict({'author': 'qupath', 'name': 'qupath', 'version': 'v0.5.1', 'wiki': 'https://qupath.github.io'}),
Repo.from_dict({'author': 'broadinstitute', 'name': 'cromwell', 'version': '87', 'wiki': 'https://cromwell.readthedocs.io/en/latest/'}),
Repo.from_dict({'author': 'hail-is', 'name': 'hail', 'version': '0.2.133', 'wiki': 'https://hail.is'}),
Repo.from_dict({'author': 'psi4', 'name': 'psi4', 'version': 'v1.9.1', 'wiki': 'https://psicode.org'}),
Repo.from_dict({'author': 'CliMA', 'name': 'Oceananigans.jl', 'version': 'v0.93.2', 'wiki': 'https://clima.github.io/OceananigansDocumentation/stable'}),
Repo.from_dict({'author': 'sofa-framework', 'name': 'sofa', 'version': 'v24.06.00', 'wiki': 'https://www.sofa-framework.org'}),
Repo.from_dict({'author': 'stardist', 'name': 'stardist', 'version': '0.9.1', 'wiki': None}),
Repo.from_dict({'author': 'COMBINE-lab', 'name': 'salmon', 'version': 'v1.10.1', 'wiki': 'https://combine-lab.github.io/salmon'}),
Repo.from_dict({'author': 'broadinstitute', 'name': 'gatk', 'version': '4.6.0.0', 'wiki': 'https://software.broadinstitute.org/gatk'}),
Repo.from_dict({'author': 'root-project', 'name': 'root', 'version': 'v6-32-06', 'wiki': 'https://root.cern'}),
]