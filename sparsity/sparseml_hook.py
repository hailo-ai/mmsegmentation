from mmseg.registry import HOOKS
from mmseg.utils.misc import calc_sparsity
from mmengine.hooks import Hook
from sparseml.pytorch.optim import ScheduledModifierManager


@HOOKS.register_module()
class SparseMLHook(Hook):
    def __init__(self, steps_per_epoch=1488, start_epoch=50, prune_interval_epoch=2):
        self.steps_per_epoch = steps_per_epoch
        self.start_epoch = start_epoch
        self.prune_interval_epoch = prune_interval_epoch

    def before_train(self, runner) -> None:
        self.manager = ScheduledModifierManager.from_yaml(runner.cfg.recipe)

        optimizer = runner.optim_wrapper.optimizer
        optimizer = self.manager.modify(runner.model.module,
                                        optimizer,
                                        steps_per_epoch=self.steps_per_epoch,
                                        epoch=self.start_epoch)
        runner.optim_wrapper.optimizer = optimizer

    def after_train(self, runner) -> None:
        self.manager.finalize(runner.model.module)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx % (self.steps_per_epoch * self.prune_interval_epoch) == 0:  # 2 Epochs
            calc_sparsity(runner.model.state_dict(), runner.logger)
            runner.logger.info(f"Epoch #{batch_idx // self.steps_per_epoch} End")

    def after_test_epoch(self, runner, metrics):
        runner.logger.info("Switching to deployment model")
        # if repvgg style -> deploy
        for module in runner.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        calc_sparsity(runner.model.state_dict(), runner.logger, True)
