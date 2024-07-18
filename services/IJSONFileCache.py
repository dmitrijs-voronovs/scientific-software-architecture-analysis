import json
from abc import ABC


class IJSONFileCache[T](ABC):
    _filename: str = None

    @classmethod
    def _verify_file_exists(cls) -> None:
        with open(cls._filename, "a+") as f:
            pass

    @classmethod
    def get(cls, key: str, fallback: T) -> T:
        cls._verify_file_exists()
        try:
            with open(cls._filename, "r") as f:
                data = cls._load_json_from_file(f)
                return data.get(key, fallback)
        except Exception as e:
            print(f"{cls.__name__}: Error in get", e)
            return fallback

    @staticmethod
    def _load_json_from_file(f) -> dict[str, T]:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

    @classmethod
    def set(cls, key: str, value: T) -> None:
        cls._verify_file_exists()
        try:
            with open(cls._filename, "r+") as f:
                data = cls._load_json_from_file(f)
                data[key] = value
                json.dump(data, f)
        except Exception as e:
            print(f"{cls.__name__}: Error in set", e)

    @classmethod
    def delete(cls, key: str) -> None:
        cls._verify_file_exists()
        try:
            with open(cls._filename, "r+") as f:
                data = cls._load_json_from_file(f)
                del data[key]
                json.dump(data, f)
        except Exception as e:
            print(f"{cls.__name__}: Error in delete", e)
