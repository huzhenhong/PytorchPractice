# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 添加包含路径专用
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-09-17 17:37:24
 LastEditors  : huzhenhong
 LastEditTime : 2020-12-01 17:39:24
 FilePath     : \\PytorchPractice\\01-LineRegress\\path_magic.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import os
import sys

sys.path.append(os.path.abspath("./Util"))


def print_cwd():
    print(os.getcwd())


def print_path():
    print(sys.path)
