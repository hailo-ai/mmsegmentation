from mmseg.registry import RUNNERS, HOOKS
from mmengine.hooks import Hook
from sparseml.pytorch.optim import ScheduledModifierManager

@HOOKS.register_module()
class SparseMLHook(Hook):
    def __init__(self, interval=10):
        self.interval = interval

    def before_train(self, runner) -> None:
        self.manager = ScheduledModifierManager.from_yaml(runner.cfg.recipe)

        optimizer = runner.optim_wrapper.optimizer
        optimizer = self.manager.modify(runner.model.module, optimizer, steps_per_epoch=1488, epoch=40)
        runner.optim_wrapper.optimizer = optimizer

    def after_train(self, runner) -> None:
        self.manager.finalize(runner.model.module)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx % (1488 * 2) == 0:  # 2 Epochs
            runner.logger.info(f"Epoch #{batch_idx // 1488} End")
            self._calc_sparsity(runner.model.state_dict(), runner.logger)

    def after_test_epoch(self, runner, metrics):
        runner.logger.info("Switching to deployment model")
        # if repvgg style -> deploy
        for module in runner.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self._calc_sparsity(runner.model.state_dict(), runner.logger)

    def _calc_sparsity(self, model_dict, logger):
        weights_layers_num, total_weights, total_zeros = 0, 0, 0
        prefix = next(iter(model_dict)).split('backbone.stage0')[0]
        for k, v in model_dict.items():
            if k.startswith(prefix) and k.endswith('weight'):
                weights_layers_num += 1
                total_weights += v.numel()
                total_zeros += (v.numel() - v.count_nonzero())
        logger.info(f"Model has {weights_layers_num} weight layers")
        logger.info(f"Overall Sparsity is roughly: {100 * total_zeros / total_weights:.1f}%")

