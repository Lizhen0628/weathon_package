# -*- coding: utf-8 -*-
# @Time    : 2022/12/22 22:45
# @Author  : LiZhen
# @FileName: pdf_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import pdfplumber
from typing import List
from tqdm import tqdm
from pdf2docx import Converter


class PDFUtils:

    @staticmethod
    def extract_text(pdf_file: str, pdf_pwd: str = None):
        page_texts = []
        with pdfplumber.open(pdf_file, password=pdf_pwd) as pdf_reader:
            for page_id, page in enumerate(tqdm(pdf_reader.pages)):
                page_text = page.extract_text()
                page_texts.append((page_id, page_text))
        return page_texts

    @staticmethod
    def extract_tables(pdf_file:str,pdf_pwd:str = None):
        page_tables = []
        with pdfplumber.open(pdf_file,password=pdf_pwd) as pdf_reader:
            for page_id, page in enumerate(tqdm(pdf_reader.pages)):
                page_tables.append((page_id,page.extract_tables()))
        return page_tables

    @staticmethod
    def pdf2docx(pdf_file:str,docx_file:str,pdf_pwd:str=None, start:int=0,end:int = None,pages:List=None):
        converter = Converter(pdf_file,pdf_pwd)
        converter.convert(docx_file,start,end,pages)


if __name__ == '__main__':
    PDFUtils.extract_tables("")