import dotenv

from actions.data_visualization import visualize
from services.MongoDBConnection import MongoDBConnection

dotenv.load_dotenv()


def main():
    visualize()
    MongoDBConnection().close_connection()


if __name__ == "__main__":
    main()
