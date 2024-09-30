import argparse
import datetime
import pathlib
import re
from typing import Any, Collection, Literal

import tqdm
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'val'], default='train')
    parser.add_argument('--skip', nargs='+', default=[])
    return parser.parse_args()


class Converter:

    @staticmethod
    def build_pattern() -> re.Pattern:
        asctime_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
        asctime_group = rf'(?P<asctime>{asctime_pattern})'
        process_thread_pattern = r'\d+:\d+'
        process_thread_group = rf'(?P<process_thread>{process_thread_pattern})'
        filename_lineno_pattern = r'\S*?:\d+'
        filename_lineno_group = (
            rf'(?P<filename_lineno>{filename_lineno_pattern})'
        )
        name_pattern = r'\S+?'
        name_group = rf'(?P<name>{name_pattern})'
        func_name_pattern = r'\S+?'
        func_name_group = rf'(?P<func_name>{func_name_pattern})'
        iter_pattern = r'Iter \[\d+/\d+\]'
        iter_group = rf'(?P<iter_>{iter_pattern})'
        eta_pattern = r'ETA .*?'
        eta_group = rf'(?P<eta>{eta_pattern})'
        memory_pattern = r'Memory \d+\.\d{2}M'
        memory_group = rf'(?P<memory>{memory_pattern})'
        info_pattern = r'\S+=\S+'
        infos_pattern = rf'(?: {info_pattern})*'
        infos_group = rf'(?P<infos>{infos_pattern})'
        pattern = re.compile(
            rf'\[{asctime_group} {process_thread_group}\]'
            rf'\[{filename_lineno_group} {name_group} {func_name_group}\]'
            rf' INFO: {iter_group}(?: {eta_group})?(?: {memory_group})?'
            rf'{infos_group}',
        )
        return pattern

    def __init__(
        self,
        mode: Literal['train', 'val'],
        skip: Collection[str],
    ) -> None:
        self._mode = mode
        self._skip = list(skip)
        self._pattern = self.build_pattern()

    def __call__(self, log: str, summary_writer: SummaryWriter) -> Any:
        match = self._pattern.fullmatch(log)
        if match is None:
            return

        asctime = datetime.datetime.strptime(
            match.group('asctime'),
            "%Y-%m-%d %H:%M:%S,%f",
        )
        timestamp = asctime.timestamp()

        global_step_match = re.search(r'\d+', match.group('iter_'))
        assert global_step_match is not None
        global_step = int(global_step_match.group(0))

        infos = re.finditer(r'(\S+)=(\S+)', match.group('infos'))
        for info in infos:
            k = info.group(1)
            if k in self._skip:
                continue
            v = info.group(2)
            summary_writer.add_scalar(
                f'{self._mode}/{k}',
                float(v),
                global_step,
                timestamp,
            )


def main() -> None:
    args = parse_args()
    log_dir: pathlib.Path = args.log_dir
    tb_dir = log_dir.parent / 'tensorboard'

    converter = Converter(args.mode, args.skip)

    with (
        log_dir.open() as f,
        SummaryWriter(tb_dir) as summary_writer,
    ):
        for log in tqdm.tqdm(f):
            converter(log.strip(), summary_writer)


if __name__ == '__main__':
    main()
