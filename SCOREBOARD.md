| Model Configuration       | Held-Out Perplexity | Accuracy | Refusal Accuracy | Avg. Latency | Training Details |

| **Baseline (AraGPT2)**    | 3,937.11            | 0%       | 0%               | 0.5s         | `aubmindlab/aragpt2-base` (135M parameters) |
| **Continued Pretraining** | 55.56               | 0%       | 0%               | 0.5s         | 1 epoch, LR: `5e-5`                         |
| **SFT**                   | val loss 0.152      | 0%       | 0%               | 0.5s         | 300 Q&A pairs, validation loss: `0.152`     |
