
---

version: 1.1.0

# General Hyperparams
start_epoch: 50
num_epochs: 120
init_lr: 0.00001
final_lr: 0.00001
weights_warmup_lr: 0
biases_warmup_lr: 0

# Pruning Hyperparams
init_sparsity: 0.01
final_sparsity: 0.60
pruning_start_epoch: 60
pruning_end_epoch: 110
pruning_update_frequency: 2.0

#Modifiers
training_modifiers:
  - !LearningRateFunctionModifier
    start_epoch: eval(start_epoch)
    end_epoch: eval(num_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(init_lr)
    
pruning_modifiers:
  - !GMPruningModifier
    params:
      - re:backbone.backbone.*.*.rbr_dense.conv.weight
      - re:backbone.neck.*.*.rbr_dense.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(final_sparsity)
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
---
