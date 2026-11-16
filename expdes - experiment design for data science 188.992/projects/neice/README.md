# NEiCE reproducibility study

A group project on reproducing the evaluation in [Topic Modeling on Podcast Short-Text Metadata](https://arxiv.org/abs/2201.04419). NEiCE uses named entities to give a topic model more context when it has only short text, such as podcast titles and descriptions.

The group adapted the [original implementation](https://github.com/deezer/podcast-topic-modeling) around outdated dependencies and container assumptions, then compared the reproduced topic scores with the published results. The Deezer scores were broadly comparable, while the iTunes scores differed. The Spotify dataset was unavailable to the group.

The main lesson: shared source code helps, but reproducing results also depends on access to the same data and an exact record of the software environment.
