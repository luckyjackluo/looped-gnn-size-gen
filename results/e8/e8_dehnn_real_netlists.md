# Real MLCAD netlists (DEHNN features): adapt from synthetic pretrain

| variant | rmse_norm | rmse_physical | mae_physical |
|---|---|---|---|
| zeroshot_pretrain | 0.6124 | 7.175 | 6.167 |
| full_finetune_dehnn | 0.5815 | 6.783 | 5.806 |
| loop_triple_peft_k6_dehnn | 0.5465 | 6.333 | 5.258 |
| full_finetune_dehnn_seed1 | 0.5817 | 6.785 | 5.802 |
| full_finetune_dehnn_seed2 | 0.5814 | 6.782 | 5.806 |
| full_finetune_dehnn_seed3 | 0.5819 | 6.786 | 5.802 |
| loop_triple_peft_k6_dehnn_seed1 | 0.5446 | 6.308 | 5.211 |
| loop_triple_peft_k6_dehnn_seed2 | 0.5465 | 6.333 | 5.281 |
| loop_triple_peft_k6_dehnn_seed3 | 0.5469 | 6.343 | 5.293 |
