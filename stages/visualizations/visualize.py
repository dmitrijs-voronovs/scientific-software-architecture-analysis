import dotenv

from actions.data_visualization import visualize
from constants.db import DBCollections
from servicess.MongoDBConnection import MongoDBConnection

dotenv.load_dotenv()


def main():
    visualize(DBCollections.Selected_projects)
    MongoDBConnection().close_connection()


if __name__ == "__main__":
    main()
