# performance engineering

*goal*

- memory and compute efficiency, at training and inference
- improved performance, cost, sustainability, …
- lower resource consumption = power, carbon footprint, co2 emission, water usage, …

*metrics*

- datacenter power usage efficiency (pue)
	- = total energy use (also cooling) / computing energy use
	- datacenter efficiency
	- 1 means no overhead, 1.67 on average
- datacenter carbon intensity
	- = tCO₂e/MWh
	- = tons of carbon dioxide and all equivalent greenhouse gasses / one million watt used for 1h
	- cleanliness of datacenter energy
- efficiency
	- = flops / joule
	- = floating point ops per second / watt
- training energy consumption
	- = $\text{pue} \cdot t \cdot e$
	- where $t$ is the train time, $e$ is the compute energy consumption
	- for federated learning: sum across all rounds and all devices used in that round
	- for network routers: consumption depends on mb up/down

*overview*

- using renewable energy:
	- moving datacenters closer to water
- reducing power consumption:
	- sparse models (instead of dense models)
	- zero shot models
- improving efficiency:
	- model optimization
	- infrastructure optimization (accelerator hardware, cloud, edge → more optimized than on-prem)

*inference optimization*

- just as important as training: less compute but many invocations
- quantization
	- lower precision of weights
	- pre-trainin vs. post-training
- pruning
	- drops unnecessary nodes and edges (structured) vs. weights (unstructured)
- distillation
	- student learns logits from teacher model, instead of labels
- cascading
	- chaining models, in increasing complexity
- context-aware model selection
	- choosing specialized model at runtime
	- self-adaptive system
	- automatically selecting model and infrastructure, based on traffic

*training optimization*

- federated learning
	- utilizing otherwise idle devices
	- inefficient: high synchronization overhead, has multiple rounds, …
- edge computing
- batches
- setting gpu power limit for gpus
	- trades off with training-time
	- energy bloat = finishing training faster than necessary, at higher energy cost
	- intrinsic bloat = gpus with less work are finishing too fast
	- extrinsic bloat = ie. hardware failure (straggler) limits throughput, other gpus are finishing too fast 

# ml4climate

- using models to help with climate change
- sustainability development goals (sdg) by united nations
- decision making, forecasting, physics simulations, analytical models → can outperform traditional models both in accuracy and efficiency (ie. FourCastNet)
- monitoring using edge devices, tinyml → strong privacy guarantees
- geospacial ai

*challenges*

- lack of data, inaccurate data
- insufficient data, not enough historical data
- lack of interpretability
