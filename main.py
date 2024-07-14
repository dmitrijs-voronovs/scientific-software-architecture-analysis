from tag_parser.tag_parser import get_tags
from utils.utils import get_golden_repos


def main():
    repos = get_golden_repos()
    tags = get_tags(repos)
    print(tags)


if __name__ == "__main__":
    main()
