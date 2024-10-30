import os
from pathlib import Path

import dotenv
import pandas as pd
from tqdm import tqdm

from extract_quality_attribs_from_docs import Credentials
from extract_quality_attribs_from_github_metadata import GitHubDataFetcher

dotenv.load_dotenv()

repos = [
    "/google/deepvariant",
    "/root-project/root",
    "/OpenGene/fastp",
    "/scverse/scanpy",
    "/allenai/scispacy",
    "/broadinstitute/gatk",
    "/qutip/qutip",
    "/soedinglab/MMseqs2",
    "/su2code/SU2",
    "/tum-pbs/PhiFlow",
    "/scipipe/scipipe",
    "/openbabel/openbabel",
    "/qupath/qupath",
    "/broadinstitute/cromwell",
    "/hail-is/hail",
    "/psi4/psi4",
    "/CliMA/Oceananigans.jl",
    "/sofa-framework/sofa",
    "/stardist/stardist",
    "/COMBINE-lab/salmon",
]


def query_and_save_versions(package_versions_path):
    os.makedirs(package_versions_path.parent, exist_ok=True)
    data = []
    for repo_path in tqdm(repos, "Getting latest repository version"):
        _, author, repo = repo_path.split("/")
        git_repo = GitHubDataFetcher(os.getenv("GITHUB_TOKEN"), Credentials(
            author=author,
            repo=repo,
            version="latest"
        ))
        data.append(dict(author=author, repo=repo, version=git_repo.get_latest_version()))

    df = pd.DataFrame(data)
    df.to_csv(package_versions_path, index=False)


def print_as_credentials(package_versions_path):
    df = pd.read_csv(package_versions_path)
    for (author, repo, version) in df.itertuples(index=False):
        print(Credentials(author=author, repo=repo, version=version))



def main():
    package_versions_path = Path("metadata/versions/package_versions.csv")
    # query_and_save_versions(package_versions_path)

    print_as_credentials(package_versions_path)




if __name__ == "__main__":
    main()
