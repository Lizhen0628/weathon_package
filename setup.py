# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 08:22
# @Author  : LiZhen
# @FileName: setup.py.py
# @github  : https://github.com/Lizhen0628
# @Description:


import codecs
import os
import sys

from distutils.util import convert_path
from fnmatch import fnmatchcase
from setuptools import setup, find_packages


def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()


# Provided as an attribute, so you can append to these instead of replicating them:
standard_exclude = ["*.py", "*.pyc", "*$py.class", "*~", ".*", "*.bak"]
standard_exclude_directories = [".*", "CVS", "_darcs", "./build", "./dist", "EGG-INFO", "*.egg-info"]


# (c) 2005 Ian Bicking and contributors; written for Paste (http://pythonpaste.org)
# Licensed under the MIT license: http://www.opensource.org/licenses/mit-license.php
# Note: you may want to copy this into your setup.py file verbatim, as
# you can't import this from another package, when you don't know if
# that package is installed yet.
def find_package_data(
        where=".",
        package="",
        exclude=standard_exclude,
        exclude_directories=standard_exclude_directories,
        only_in_packages=True,
        show_ignored=False):
    """
    Return a dictionary suitable for use in ``package_data``
    in a distutils ``setup.py`` file.
    The dictionary looks like::

        {"package": [files]}

    Where ``files`` is a list of all the files in that package that
    don"t match anything in ``exclude``.

    If ``only_in_packages`` is true, then top-level directories that
    are not packages won"t be included (but directories under packages
    will).

    Directories matching any pattern in ``exclude_directories`` will
    be ignored; by default directories with leading ``.``, ``CVS``,
    and ``_darcs`` will be ignored.

    If ``show_ignored`` is true, then all the files that aren"t
    included in package data are shown on stderr (for debugging
    purposes).

    Note patterns use wildcards, or can be exact paths (including
    leading ``./``), and all searching is case-insensitive.
    """
    out = {}
    stack = [(convert_path(where), "", package, only_in_packages)]
    while stack:
        where, prefix, package, only_in_packages = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if os.path.isdir(fn):
                bad_name = False
                for pattern in exclude_directories:
                    if (fnmatchcase(name, pattern)
                            or fn.lower() == pattern.lower()):
                        bad_name = True
                        if show_ignored:
                            print("Directory %s ignored by pattern %s" % (fn, pattern), file=sys.stderr)
                        break
                if bad_name:
                    continue
                if (os.path.isfile(os.path.join(fn, "__init__.py"))
                        and not prefix):
                    if not package:
                        new_package = name
                    else:
                        new_package = package + "." + name
                    stack.append((fn, "", new_package, False))
                else:
                    stack.append((fn, prefix + name + "/", package, only_in_packages))
            elif package or not only_in_packages:
                # is a file
                bad_name = False
                for pattern in exclude:
                    if fnmatchcase(name, pattern) or fn.lower() == pattern.lower():
                        bad_name = True
                        if show_ignored:
                            print("File %s ignored by pattern %s" % (fn, pattern), file=sys.stderr)
                        break
                if bad_name:
                    continue
                out.setdefault(package, []).append(prefix + name)
    return out


PACKAGE = "weathon"
VERSION = __import__(PACKAGE).__version__
AUTHOR = __import__(PACKAGE).__author__
EMAIL = __import__(PACKAGE).__email__

setup(
    name="weathon",
    version=VERSION,
    description="A private coding package",
    author=AUTHOR,
    author_email=EMAIL,
    url="https://github.com/Lizhen0628",
    packages=find_packages(),
    package_data=find_package_data(PACKAGE, only_in_packages=False),
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python",
    ],
    zip_safe=False,
    python_requires=">=3.5, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3*, !=3.4*",  # Python版本依赖
    install_requires=[
        "torch >= 1.0.0",
        "tqdm >= 4.27.0",
        "jieba >= 0.42.1",
        "zhon >= 1.1.5",
        "seqeval==1.2.2",
        "paramiko==2.11.0",
        "Levenshtein==0.20.5",
        "mmh3==3.0.0",
        "pandas==1.3.3",
        "scipy >= 1.2.0",
        "requests==2.28.1",
        "transformers >= 4.22.2",
        "scikit-learn >= 0.17.0",
        "pypinyin >= 0.47.1"
    ],
)
