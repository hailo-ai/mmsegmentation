from mmengine.hooks import CheckpointHook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class ExtCheckpointHook(CheckpointHook):

    def after_val_epoch(self, runner, metrics):
        if runner.iter == self.save_begin:
            runner.logger.info('Resetting best_score to 0.0')
            runner.message_hub.update_info('best_score', 0.0)
            runner.message_hub.pop_info('best_ckpt', None)
        if (runner.iter + 1 >= self.save_begin):
            runner.logger.info(
                f'Saving checkpoint at iter {runner.iter}')
            super().after_val_epoch(runner, metrics)
