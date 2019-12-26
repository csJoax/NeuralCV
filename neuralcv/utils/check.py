import os


def check_sep(name: str, sep='/'):
    """
    在必要时添加分隔符。
    如果 name 为空值，返回 空串；
    如果 name 不为空值，在 name 前加上一个分隔符后返回；
    :param name:  输入的字符串
    :param sep: 待添加的分隔符
    :return: 处理后的字符串
    """
    return f"{sep}{name}" if name else ''


add_sep = check_sep


def check_dir(dir_name, show_func=None):
    """
    确保某个文件夹存在，如果不存在，则创建该文件夹
    :param dir_name: 文件夹名称
    :param show_func: 在创建文件夹时调用该函数以显示提示信息，默认为None
    :return:
    :raise OSError
    """

    try:
        os.mkdir(dir_name)
        if show_func:
            show_func()
    except OSError:
        if not os.path.isdir(dir_name):
            raise OSError(f'The path existed, but it is not a dir:{dir_name}')
