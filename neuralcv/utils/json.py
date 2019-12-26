import os
import json


def __print_curdir():
    """
    输出当前的工作目录
    :return:
    """
    print(f"Current dir: {os.path.abspath(os.curdir)}")


def parse_json_file(json_path, *arg, **kwargs):
    """
    解析json文件，并自动关闭
    :param json_path: json文件路径
    :param arg: 其他用于读取json文件的参数
    :param kwargs: 其他用于读取json文件的参数（kw形式）
    :return: 
    """
    try:
        with open(json_path) as f:
            return json.load(f, *arg, **kwargs)
    except Exception as err:
        __print_curdir()
        raise err


def write_json_file(obj: object, json_path, indent=4, *arg, **kwargs):
    """
    保存对象到json文件，并自动关闭
    :param obj: 带保存对象
    :param json_path: json文件路径
    :param indent: 自动对齐的空格数
    :param arg: 其他用于保存json文件的参数
    :param kwargs: 其他用于保存json文件的参数（kw形式）
    :return:
    """
    try:
        with open(json_path, "w") as f:
            json.dump(obj, f, indent=indent, *arg, **kwargs)
            print(f"Object saved in {json_path}.")
    except Exception as err:
        __print_curdir()
        raise err
