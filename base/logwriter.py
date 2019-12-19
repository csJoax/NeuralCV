import logging as Logging
from torch.utils.tensorboard import SummaryWriter

__all__ = ["Logging", "LogWriter"]

Logging.basicConfig(level=Logging.INFO, format='%(levelname)s: %(message)s')


class LogWriter(SummaryWriter):
    def __init__(self, lr, batch_size, other_info=None, *args, **kwargs):
        if other_info:
            comment = f'LR_{lr}_BS_{batch_size}_{other_info}'
        else:
            comment = f'LR_{lr}_BS_{batch_size}'
        super(LogWriter, self).__init__(comment=comment, *args, **kwargs)

    def net_info(self, net_config):
        """
        记录部分网络参数
        """
        Logging.info(f'''Starting training:
            Epochs:          {net_config.epochs}
            Batch size:      {net_config.batch_size}
            Learning rate:   {net_config.lr}
            Training size:   {net_config.n_train}
            Validation size: {net_config.n_val}
            Device:          {net_config.device.type}
        ''')

    def add_loss(self, loss, global_step, name=None, to_log=False):
        name = f"/{name}" if name else None  # 添加 name 与其他字符间的分隔符
        if to_log:
            Logging.info(f'Train loss{name}: {loss}')
        self.add_scalar(f'Loss/train/{name}' if name else 'Loss/train', loss, global_step)

    def add_score(self, score, global_step, name=None, to_log=True):
        name = f"/{name}" if name else None  # 添加 name 与其他字符间的分隔符
        if to_log:
            Logging.info(f'Validation score{name}: {score}')
        self.add_scalar(f'Score/Val{name}', score, global_step)
