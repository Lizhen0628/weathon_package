# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:38
# @Author  : LiZhen
# @FileName: file_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:


import os
import bz2
import yaml
import json
import gzip
import tarfile
from pathlib import Path
from typing import Union, List, Generator, Dict
from zipfile import ZipFile
from collections import OrderedDict


class FileUtils:
    """
    文件工具类：
    """

    @staticmethod
    def read_json(infile, encoding='utf8'):
        """
            读取json文件
        Args:
            infile: json 文件路径
            encoding: 文件编码格式
        Returns:加载json文件后的dict

        """
        infile = Path(infile)
        with infile.open('rt', encoding=encoding) as handle:
            return json.load(handle, object_hook=OrderedDict)

    @staticmethod
    def read_jsonl_list(file_path: Union[str, Path], encoding: str = 'utf-8', fields: List[str] = None,
                        dropna: bool = True) -> List[Dict]:
        """
        读取jsonl文件，并以列表的形式返回
        Args:
            file_path: jsonl 文件路径
            encoding: jsonl文件编码格式
            fields: 需要从json文件取出的键
            dropna: 如果键值不存在是否丢弃该数据

        Returns:以列表的形式返回读取内容

        """
        datas = []
        file_path = Path(file_path)
        if fields:
            fields = set(fields)

        with file_path.open("r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                if fields is None:
                    datas.append(data)
                    continue
                _res = {}
                for k, v in data.items():
                    if k in fields:
                        _res[k] = v
                if len(_res) < len(fields):
                    if dropna:
                        continue
                    else:
                        raise ValueError(f'invalid instance at line number: {idx}')
                datas.append(_res)
        return datas

    @staticmethod
    def read_jsonl_generator(file_path: Union[str, Path], encoding: str = 'utf-8', fields: List[str] = None,
                             dropna: bool = True) -> Generator:
        """

        读取jsonl文件，并以列表的形式返回
        Args:
            file_path: jsonl 文件路径
            encoding: jsonl文件编码格式
            fields: 需要从json文件取出的键
            dropna: 如果键值不存在是否丢弃该数据

        Returns:以生成器的形式返回读取内容
        """

        file_path = Path(file_path)
        if fields:
            fields = set(fields)
        with file_path.open("r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                if fields is None:
                    yield idx, data
                    continue
                _res = {}
                for k, v in data.items():
                    if k in fields:
                        _res[k] = v
                if len(_res) < len(fields):
                    if dropna:
                        continue
                    else:
                        raise ValueError(f'invalid instance at line number: {idx}')
                yield idx, _res

    @staticmethod
    def read_yaml(infile, encoding='utf8'):
        """
        读取yaml格式的文件
        Args:
            infile: 配置文件路径
            encoding: 默认utf8

        Returns:返回字典格式的配置

        """
        infile = Path(infile)
        with infile.open('r', encoding=encoding) as handle:
            return yaml.load(handle, Loader=yaml.Loader)

    @staticmethod
    def write_yaml( content, infile):
        """
        将文件内容写入yaml文件
        Args:
            content: 文件内容
            infile: yaml文件名称
        Returns:

        """
        infile = Path(infile)
        with infile.open('w', encoding='utf8') as handle:
            yaml.dump(content, handle, )

    @staticmethod
    def write_json( content, infile):
        infile = Path(infile)
        with infile.open('wt') as handle:
            json.dump(content, handle, indent=4, sort_keys=False)

    @staticmethod
    def copy_dir(source: Union[Path, str] = None, target: Union[Path, str] = None):
        """
        复制文件夹
        Args:
            source: 原文件夹
            target: 目标文件夹
        Returns: 返回目标文件夹路径
        """
        source, target = Path(source), Path(target)
        if not target.exists():
            target.mkdir()

        files = list(source.glob("*"))
        for source_file in files:
            target_file = target / source_file.name
            if source_file.is_file():
                target_file.write_bytes(source_file.read_bytes())
            else:
                FileUtils.copy_dir(source_file, target_file)

    @staticmethod
    def ensure_dir(dirname: Union[str, Path]) -> Path:
        """
        ensure dir path is exist,if not exist,make it
        Args:
            dirname: 文件夹 路径

        Returns:
        """
        dirname = Path(dirname)
        if not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=False)
        return dirname

    @staticmethod
    def ensure_file(file_name: Union[str, Path]) -> Path:
        """
        ensure file is exist,if not exist,make it.
        Args:
            file_name:

        Returns:

        """
        file_name = Path(file_name)
        FileUtils.ensure_dir(file_name.parent)
        if not file_name.exists():
            file_name.touch(exist_ok=False)
        return file_name

    @staticmethod
    def get_filepath(filepath):
        r"""
        如果filepath为文件夹，
            如果内含多个文件, 返回filepath
            如果只有一个文件, 返回filepath + filename
        如果filepath为文件
            返回filepath
        :param str filepath: 路径
        :return:
        """
        if os.path.isdir(filepath):
            files = os.listdir(filepath)
            if len(files) == 1:
                return os.path.join(filepath, files[0])
            else:
                return filepath
        elif os.path.isfile(filepath):
            return filepath
        else:
            raise FileNotFoundError(f"{filepath} is not a valid file or directory.")


class FileDecomposeUtils:
    """
    文件解压缩工具类
    """

    @staticmethod
    def unzip_file(source: Union[str, Path], target: Union[str, Path] = None) -> None:
        """
        解压缩zip文件
        Args:
            source: 压缩文件路径
            target: 解压缩路径，如果为空，解压到当前路径
        """
        source = Path(source)
        target = FileUtils.ensure_dir(target) if target else source.parent
        with ZipFile(source, "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(target)

    @staticmethod
    def untar_gz_file(source: Union[str, Path], target: Union[str, Path] = None) -> None:
        """
        解压缩tar.gz 文件
        Args:
            source: 压缩文件路径
            target: 解压缩路径，如果为空，解压到当前路径
        """
        source = Path(source)
        target = FileUtils.ensure_dir(target) if target else source.parent
        with tarfile.open(source, 'r:gz') as tar:
            tar.extractall(target)

    @staticmethod
    def ungzip_file(source: Union[str, Path], target: Union[str, Path] = None,
                    target_filename: Union[str] = None) -> None:
        """
        解压缩gzip文件
        Args:
            source: 压缩文件路径
            target: 解压缩文件夹，如果为空，解压到当前路径
            target_filename: 解压缩文件名称
        """
        source = Path(source)
        target_dir = FileUtils.ensure_dir(target) if target else source.parent
        target_filename = target_filename if target_filename else source.stem
        target_file = target_dir / target_filename
        with gzip.GzipFile(source, "rb") as source_reader, target_file.open("wb") as target_writer:
            for data in iter(lambda: source_reader.read(100 * 1024), b""):
                target_writer.write(data)

    @staticmethod
    def unbz2_file(source: Union[str, Path], target: Union[str, Path] = None,
                   target_filename: Union[str] = None) -> None:
        """
        解压缩gzip文件
        Args:
            source: 压缩文件路径
            target: 解压缩文件夹，如果为空，解压到当前路径
            target_filename: 解压缩文件名称
        """
        source = Path(source)
        target_dir = FileUtils.ensure_dir(target) if target else source.parent
        target_filename = target_filename if target_filename else source.stem
        target_file = target_dir / target_filename

        with bz2.BZ2File(source, "rb") as source_reader, target_file.open("wb") as target_writer:
            for data in iter(lambda: source_reader.read(100 * 1024), b""):
                target_writer.write(data)

    @staticmethod
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
            return fn

        return docstring_decorator
