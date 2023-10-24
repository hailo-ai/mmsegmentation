from mmseg.registry import RUNNERS, HOOKS
from mmengine.hooks import Hook
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter
from yolov5.utils.neuralmagic import maybe_create_sparsification_manager
from yolov5.utils.torch_utils import de_parallel

@HOOKS.register_module()
class SparseMLHook(Hook):
    def __init__(self, interval=10):
        self.interval = interval

    def before_train(self, runner) -> None:
        print("before train\n before train\n  before train\n   before train")
        # ckpt = runner.model.state_dict()
        # ckpt["epoch"] = 0
        # ckpt["ema"] = ckpt.get("ema", None)
        # self.sparsification_manager = maybe_create_sparsification_manager(runner.model,
        #                                                                   ckpt=ckpt,
        #                                                                   train_recipe=runner.cfg.recipe,
        #                                                                   recipe_args=runner.cfg.recipe_args,
        #                                                                   device=runner.model.device, resumed=runner._resume)


        # # if self.args.recipe is not None:  # SPARSEML
        # start_epoch = 40  # self.start_epoch  # 295
        # self.scaler, scheduler, self.ema_model, epochs = self.sparsification_manager.initialize(
        #             loggers=None,  # None / self.tblogger / logger
        #             scaler=self.scaler,
        #             optimizer=runner.optim_wrapper.optimizer,  # self.optimizer,
        #             scheduler=runner.param_schedulers[-1],  # self.lr_scheduler,
        #             ema=None  # self.ema_model,
        #             start_epoch=start_epoch,
        #             steps_per_epoch=len(self.train_dataloader),
        #             epochs=50  # self.max_epoch,
        #             compute_loss=None,  # None / some loss function
        #             distillation_teacher=None,
        #             resumed=True,
        #         )
        self.manager = ScheduledModifierManager.from_yaml(runner.cfg.recipe)

        optimizer = runner.optim_wrapper.optimizer
        # optimizer = self.manager.modify(pl_module, optimizer, steps_per_epoch=trainer.estimated_stepping_batches, epoch=0)
        optimizer = self.manager.modify(runner.model.module, optimizer, steps_per_epoch=1488, epoch=40)
        runner.optim_wrapper.optimizer = optimizer

    def after_train(self, runner) -> None:
        self.manager.finalize(runner.model.module)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):  #, batch_idx=0, data_batch=None, outputs=None):
        # print(f"after_train_iter:: {batch_idx}")
        if batch_idx % (1488*2) == 0:  # 2 Epochs
            print(f"Epoch #{batch_idx // 1488} End")
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
                zeros_ratio = (v.numel() - v.count_nonzero()) / v.numel() * 100.0
                logger.info(f"[{weights_layers_num:>2}] {k:<58}:: {v.numel() - v.count_nonzero():<5} / {v.numel():<7} ({zeros_ratio:<4.1f}%) are zeros")
        logger.info(f"Model has {weights_layers_num} weight layers")
        logger.info(f"Overall Sparsity is roughly: {100 * total_zeros / total_weights:.1f}%")

