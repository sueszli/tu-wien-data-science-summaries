# Compare local and cloud object detection timings


A Data-intensive Computing project (194.048, summer 2024) comparing local Flask/OpenCV YOLO inference with an S3-triggered AWS Lambda pipeline.

[Aggregate timings](results.csv) summarize the original recorded samples in seconds. Each environment gets separate transfer and inference statistics. The pipelines used different hardware and different transfer measurement paths. These values describe those runs, not an end-to-end latency comparison or a general cloud performance result.

No cloud resources are needed to read the results. Object names, resource identifiers, raw logs, images, and deployment scripts are not included.
