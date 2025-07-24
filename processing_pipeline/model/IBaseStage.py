import concurrent.futures.thread
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
from processing_pipeline.model.IDFHandler import IDfHandler
from processing_pipeline.model.ParquetDFHandler import ParquetDFHandler
from processing_pipeline.processing_parameter_tuning.optimal_params import optimal_processing_parameters
from utilities.utils import create_logger_path


class IBaseStage(metaclass=ABCMeta):
    @property
    @abstractmethod
    def data_model(self) -> BaseModel:
        pass

    @property
    @abstractmethod
    def temperature(self) -> float:
        pass

    # noinspection PyPep8Naming
    @property
    def DFHandler(self) -> IDfHandler:
        return ParquetDFHandler()

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def cache_dir(self) -> Path:
        return AbsDirPath.CACHE / "uncategorized"

    @property
    @abstractmethod
    def in_dir(self) -> Path:
        pass

    @in_dir.setter
    def in_dir(self, value):
        self._in_dir = value

    @property
    @abstractmethod
    def out_dir(self) -> Path:
        pass

    @out_dir.setter
    def out_dir(self, value):
        self._out_dir = value

    @property
    @abstractmethod
    def stage_name(self) -> str:
        pass

    @property
    def file_ext(self) -> str:
        return self.DFHandler.file_ext

    error_texts_for_termination = ["HTTPConnectionPool",
                                   "No connection could be made because the target machine actively refused it",
                                   "An existing connection was forcibly closed",
                                   "RuntimeError cannot schedule new futures after interpreter shutdown"]

    def __init__(self, hostname: str, *, batch_size_override: int = None, n_threads_override: int = None,
                 disable_cache=False, model_name_override: str = None, in_dir_override: Path = None,
                 out_dir_override: Path = None):
        self.model_fields = list(self.data_model.model_fields.keys())
        self.hostname = hostname
        if model_name_override:
            self.model_name = model_name_override
        self.model = ChatOllama(model=self.model_name, temperature=self.temperature, base_url=self.hostname,
                                format=self.data_model.model_json_schema())

        optimal_params = optimal_processing_parameters[self.model_name]
        self.batch_size = batch_size_override or optimal_params.batch_size
        self.n_threads = n_threads_override or optimal_params.n_threads

        self.disable_cache = disable_cache
        if in_dir_override:
            self.in_dir = in_dir_override
        if out_dir_override:
            self.out_dir = out_dir_override

        self._print_all_params()

        self._init()

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

    @retry(stop=stop_after_attempt(5), wait=wait_incrementing(5, 5),
           after=lambda retry_state: logger.warning(retry_state), reraise=True)
    def request_ollama_chain(self, prompts: List[str]) -> List[BaseModel]:
        batch_answers = self.model.batch(prompts)
        return [self.data_model.model_validate_json(answer.content) for answer in batch_answers]

    @classmethod
    @abstractmethod
    def to_prompt(cls, x: pd.Series) -> str:
        pass

    @classmethod
    def filter_and_transform_df_before_processing(cls, df) -> pd.DataFrame:
        """Filter to apply to the data before processing"""
        return df

    @classmethod
    def transform_df_before_saving(cls, df) -> pd.DataFrame:
        """Filter to apply to the data before processing"""
        return df

    def get_stage_column_name(self, field_name):
        return f"{self.stage_name}_{field_name}"

    def _process_in_batches(self, file_path: Path, res_filepath: Path):
        # Fix for shelve not working in multithreading environment
        shelf_path = self._prepare_shelf_with_path(file_path)
        with shelve.open(shelf_path) as db:
            if not self.disable_cache and db.get("processed", False):
                logger.info(f"File {file_path.stem} already processed")
                return
            logger.info(f"Processing {file_path.stem}")

            try:
                df = self.DFHandler.read_df(file_path)
            except Exception as e:
                logger.error(e)
                return

            last_idx = 0 if self.disable_cache else db.get("idx", 0)
            if last_idx > 0:
                logger.info(f"Continuing from {last_idx}")
                res_filepath = res_filepath.with_suffix(f".from_{last_idx}{self.file_ext}")

            df = self.filter_and_transform_df_before_processing(df)
            df = df.iloc[last_idx:].copy()

            prompt_field = self.get_stage_column_name("prompt")
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

                df_with_responses = pd.DataFrame(llm_responses,
                                                 columns=self.get_columns(), index=batch_index)
                df.loc[batch_index, df_with_responses.columns] = df_with_responses.values

                resulting_df = df.iloc[:batch_end]
                df1 = self.transform_df_before_saving(resulting_df)
                self.DFHandler.write_df(df1, res_filepath)
                if not self.disable_cache: db["idx"] = last_idx + batch_end

            if not self.disable_cache: db['processed'] = True
            logger.info(f"Processed {file_path.stem}")

    def get_columns(self):
        # noinspection PyUnresolvedReferences
        return [self.get_stage_column_name(field) for field in
                # self.model_fields]
                self.data_model.model_fields.keys()]

    def _prepare_shelf_with_path(self, file_path) -> Path:
        if not self.disable_cache:
            (self.cache_dir / f"{file_path.stem}.dat").touch()
            (self.cache_dir / f"{file_path.stem}.bak").touch()
            (self.cache_dir / f"{file_path.stem}.dir").touch()

        return self.cache_dir / file_path.stem

    def process_batch(self, prompts):
        try:
            responses = self.request_ollama_chain(prompts)
            return [self.extract_response_data(r) for r in responses]
        except Exception as e:
            logger.error(e)
            if not self.is_retriable_error(e):
                logger.error("HTTPConnectionPool error, exiting")
                exit(1)
            return None

    def extract_response_data(self, response: BaseModel):
        return tuple([getattr(response, field) for field in self.model_fields])

    def execute_single_threaded(self, only_files_containing_text: List[str] | None = None, reverse: bool = False):
        logger.info(f"Executing {self.stage_name} stage")
        only_files_containing_text = only_files_containing_text or []

        try:
            for file_path in self.in_dir.glob(f"*{self.file_ext}"):
                keep_processing = self._keep_processing(file_path, only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = self.out_dir / f"{file_path.stem}{self.file_ext}"
                self._process_in_batches(file_path, res_filepath)
        except Exception as e:
            logger.error(e)
            raise e
        logger.info(f"Finished {self.stage_name} stage")

    def execute(self, only_files_containing_text: List[str] | None = None, reverse: bool = False, dry_run: bool = False):
        logger.info(f"Executing {self.stage_name} stage")
        only_files_containing_text = only_files_containing_text or []

        try:
            with concurrent.futures.thread.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures_to_filenames = {}
                for file_path in self.in_dir.glob(f"*{self.file_ext}"):
                    keep_processing = self._keep_processing(file_path, only_files_containing_text)
                    if keep_processing == reverse:
                        if dry_run: logger.info(f"File {file_path.stem} would be skipped")
                        continue

                    if dry_run:
                        logger.info(f"File {file_path.stem} would be processed")
                        continue

                    res_filepath = self.out_dir / f"{file_path.stem}{self.file_ext}"
                    futures_to_filenames[
                        executor.submit(self._process_in_batches, file_path, res_filepath)] = file_path

            for future in concurrent.futures.as_completed(futures_to_filenames):
                future.result()
                logger.info(f"File {futures_to_filenames[future]} processed")
        except Exception as e:
            logger.error(e)
            raise e
        logger.info(f"Finished {self.stage_name} stage")

    @staticmethod
    def _keep_processing(file_path, only_files_containing_text):
        keep_processing = len(only_files_containing_text) == 0 or any(
            text_to_test in file_path.name for text_to_test in only_files_containing_text)
        return keep_processing

    def _print_all_params(self):
        print(f"Stage {self.stage_name} params:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("\n")
