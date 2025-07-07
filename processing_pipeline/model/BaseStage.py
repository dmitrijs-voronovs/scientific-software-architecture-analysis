import math
import os
import shelve
import signal
import sys
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
from langchain_ollama import ChatOllama
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_incrementing
from tqdm import tqdm

from constants.abs_paths import AbsDirPath
from utilities.utils import create_logger_path


class BaseStage(metaclass=ABCMeta):
    @property
    @abstractmethod
    def data_model(self) -> BaseModel:
        pass

    @property
    @abstractmethod
    def temperature(self) -> float:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def cache_dir(self) -> Path:
        pass

    @property
    @abstractmethod
    def out_dir(self) -> Path:
        pass

    @property
    @abstractmethod
    def in_dir(self) -> Path:
        pass

    @property
    @abstractmethod
    def stage_name(self) -> str:
        pass

    error_texts_for_termination = ["HTTPConnectionPool",
                              "No connection could be made because the target machine actively refused it"]

    def __init__(self, hostname: str, batch_size: int = 10, n_threads: int = 5):
        self.model_fields = list(self.data_model.model_fields.keys())
        self.batch_size = batch_size
        self.hostname = hostname
        self.model = ChatOllama(model=self.model_name, temperature=self.temperature, base_url=self.hostname,
                                format=self.data_model.model_json_schema())
        self.n_threads = n_threads

    @staticmethod
    def _cleanup_and_exit(signal_num, frame):
        print("Caught interrupt, cleaning up...")
        sys.exit(0)  # Triggers the context manager's cleanup

    def _init(self):
        AbsDirPath.LOGS.mkdir(exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.add(create_logger_path(self.stage_name), mode="w")

        # Register the signal handler
        signal.signal(signal.SIGINT, self._cleanup_and_exit)

    @classmethod
    def is_retriable_error(cls, error) -> bool:
        error_s = str(error)
        return not any(error_text in error_s for error_text in cls.error_texts_for_termination)

    @retry(stop=stop_after_attempt(3), wait=wait_incrementing(5, 5), after=lambda retry_state: logger.warning(retry_state),
           reraise=True, retry=is_retriable_error)
    def request_ollama_chain(self, prompts: List[str]) -> List[BaseModel]:
        batch_answers = self.model.batch(prompts)
        return [self.data_model.model_validate_json(answer.content) for answer in batch_answers]

    @staticmethod
    @abstractmethod
    def to_prompt(x):
        pass

    @classmethod
    def filter_and_transform_df_before_processing(cls, df) -> pd.DataFrame:
        """Filter to apply to the data before processing"""
        return df

    @classmethod
    def transform_df_before_saving(cls, df) -> pd.DataFrame:
        """Filter to apply to the data before processing"""
        return df

    @classmethod
    def get_stage_labeled_field(cls, field_name):
        return f"{cls.stage_name}_{field_name}"

    def verify_file_batched_llm(self, file_path: Path, res_filepath: Path):
        with shelve.open(self.cache_dir / file_path.stem) as db:
            if db.get("processed", False):
                logger.info(f"File {file_path.stem} already processed")
                return
            logger.info(f"Processing {file_path.stem}")

            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                logger.error(e)
                return

            last_idx = db.get("idx", 0)
            if last_idx > 0:
                logger.info(f"Continuing from {last_idx}")
                res_filepath = res_filepath.with_suffix(f".from_{last_idx}.parquet")

            df = self.filter_and_transform_df_before_processing(df)
            df = df.iloc[last_idx:].copy()

            prompt_field = self.get_stage_labeled_field("prompt")
            df[prompt_field] = df.apply(self.to_prompt, axis=1)

            for batch_n in tqdm(range(0, len(df), self.batch_size), total=math.ceil(len(df) / self.batch_size),
                                desc=f"Processing {file_path.stem} in batches of {self.batch_size}"):
                batch_end = batch_n + self.batch_size
                batch_df = df.iloc[batch_n:batch_end]
                prompts = batch_df[prompt_field].tolist()
                batch_index = batch_df.index

                llm_responses = self.process_batch(prompts)
                if llm_responses is None:
                    logger.error(f"Error processing batch {last_idx + batch_n}")
                    continue

                df_with_responses = pd.DataFrame(llm_responses, columns=[self.get_stage_labeled_field(field) for field in
                                                                    self.model_fields], index=batch_index)
                df.update(df_with_responses)

                resulting_df = df.iloc[:batch_end].copy()
                self.transform_df_before_saving(resulting_df).to_parquet(res_filepath, engine='pyarrow', compression='snappy', index=False)
                db["idx"] = last_idx + batch_end

            db['processed'] = True
            logger.info(f"Processed {file_path.stem}")

    def process_batch(self, prompts):
        try:
            responses = self.request_ollama_chain(prompts)  # New batch query
            return [self.extract_response_data(r) for r in responses]
        except Exception as e:
            logger.error(e)
            if not self.is_retriable_error(e):
                logger.error("HTTPConnectionPool error, exiting")
                exit(1)
            return None

    def extract_response_data(self, response: BaseModel):
        return tuple([getattr(response, field) for field in self.model_fields])

    def execute(self, only_files_containing_text: List[str] | None = None, reverse: bool = False):
        logger.info(f"Executing {self.stage_name} stage")
        only_files_containing_text = only_files_containing_text or []

        try:
            for file_path in self.in_dir.glob("*.parquet"):
                keep_processing = len(only_files_containing_text) == 0 or any(
                    text_to_test in file_path.stem for text_to_test in only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = self.out_dir / f"{file_path.stem}.parquet"
                self.verify_file_batched_llm(file_path, res_filepath)
        except Exception as e:
            logger.error(e)
            raise e
        logger.info(f"Finished {self.stage_name} stage")
