> Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., ... & Gadepally, V. (2023, September). From words to watts: Benchmarking the energy costs of large language model inference. In 2023 IEEE High Performance Extreme Computing Conference (HPEC) (pp. 1-9). IEEE. – https://arxiv.org/pdf/2310.03003

- inference has been less optimized than training
- but inference calls happen more frequently
- paper benchmarks inference energy costs of

*setup*

- model: meta llama (decoder only, transformer based)
	- 7b, 13b, 65b (largest model)
	- batchsize=64
	- maxtokens=128
- datasets: alpaca, gsm8k
	- 4,096 samples
- multi gpu model sharding, <32 gpus
	- pytorch fairscale
	- $\tau$=0.8, top-p=0.95 (common values, no tuning)
	- no quantization
- MIT supercloud hpc system:
	- 448 compute nodes
	- xeon cpu
	- 2x v100 35gb gpu (250w) → for 8, 16, 32 shards
	- 4x a100 80gb gpu (300w) → for smaller shards
	- maximum power draw capped at 250w
	- omnipath, 25gb ethernet
- metrics:
	- perf, latency, energy cost (but not correctness/quality)
	- total energy consumption divided by num of nodes (not fine granular)

*A1 - fig 2: tokens/s vs. gpu vs. model size vs. dataset*

- a100 >> v100 (by 1x-2x)
- gsm8k dataset seems easier
- llama7b >> llama65b (by 3x-5x)

*A2 - fig 3: energy/s vs. gpu vs. model size vs. dataset*

- joul per second
- a100 >> v100
- both datasets use the same energy
- llama7b >> llama65b (exponentially more, disproportionate to increase in performance)

*B - fig 4, 5: shards vs. batch size vs. model size vs. dataset*

- more batches don't need more energy per token
- more shards need more energy (circa proportional to batch size increase)
- llama65b ranges 300w-1000w

*C - fig 6, 7: shards vs. token size vs. energy vs. dataset*

- llama65b only
- max generation length doesn't matter
- there is a sweet spot where energy per token generated drops with increasing batch size

*D - fig 8, 9: shards vs. token size vs. energy vs. dataset*

- gsm8k, 512 generation length, 16 shards, max batch size → best energy efficiency

*E - power cap*

- capping from 250w to 175w: 6.7% slower inference
- capping from 175w to 150w: 19.49% slower inference
- static power cap not recommended

*F - gpu utilization in distributed inference*

- 94-98% $\pm$ 23-27% utilization
- higher with longer token size

*conclusion*

- power capping can be an effective tool for reducing inference energy

*outlook*

- hyperparam search
- a single gpu could be shared by multiple models, with minimal degradation
- model quantization, distillation, sparsification
- custom, energy-efficient hardware
