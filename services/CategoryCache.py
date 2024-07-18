from typing import Optional

from services.IJSONFileCache import IJSONFileCache
from utils.paths import Paths

type CategoryCacheItem = tuple[Optional[str], bool]


class CategoryCache(IJSONFileCache[CategoryCacheItem]):
    _filename = Paths.CACHED_CURSORS
