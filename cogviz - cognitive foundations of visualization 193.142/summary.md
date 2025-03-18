goal: making data easier to comprehend

*applications*

- data science
- data journalism
- design
- research

*evaluation*

- there is no absolute scale, we can only compare visualizations with one another
- visual query paradigm:
	- task = implicit/explicit goal to be accomplished using a visualization
	- visual queries = translation of task into visual information retrieval
	- effectiveness = how well you can find an answer to visual queries

*perceptual process*

understanding visual/cognitive system / how visual stimuli are processed allows us to objectively evaluate visualizations

- https://en.wikipedia.org/wiki/Perception
- https://en.wikipedia.org/wiki/Visual_cortex
- light reflected by lens
- projected to photoreceptors on retina:
	- rods = low light conditions, low resolution
	- cones = normal conditions, high resolution
	- highest resolution in centre (fovea), lowest in periphery
	- blind spot = region where optic nerve attaches to the eye
- transmitted to brain via optical nerve
- v1 stage: low-level feature perception
	- preattentive features / tunable features = orientation, color, texture, movement, depth
	- parallel processing, unconscious
	- processed in 200-250ms, faster than eye movement initiation
	- buffered in impermanent 'iconic memory'
- v2 stage: pattern perception
	- low-level features aggregated together to create patterns
	- sequential, influenced by attention
- v3 stage: object perception
	- stored in 'working memory'
	- features are kept
	- conscious
	- limited to ~7 chunks

*attention*

when designing, guide the attention of the viewer such that they can achieve their goal

- the process is lossy:
	- information is lost, gaps are inferred, memory is faulty
	- process strongly influenced by attention
- saccadic eye movement = fast movement of eyes, fast sampling, because scope of vision is limited
- inattentional change blindness = missing changes, because of lack of attention

*3 main guidelines*

- 1) eyes beat memory = moving eyes is faster than interaction with visualization
- 2) express tasks as explicit visual queries
- 3) use tunable features

*visual channels*

- https://ora.ox.ac.uk/objects/uuid:b98ccce1-038f-4c0a-a259-7f53dfe06ac7/files/m6893848f5ea3d126608a2e537b32ac77
- visualization = mapping between data properties and visual properties
- visual properties = symbols, graphical properties
- channel expressiveness = type of information that can be expressed by a channel (otherwise there is an expressiveness missmatch)
- channel effectiveness = how well it can be expressed

*information types*

- information types: nominal, ordinal, interval, ratio
- attribute types: quantitative, sequential, categorical

*channel expressiveness*

- quantity → position (axes), size (bubbles, bars, angle), slope/direction, color intensity
- order → position (list, table), texture/density, color intensity
- categories → position (bar position), color hue, shape

*channel effectiveness*

- https://www.statcan.gc.ca/en/data-science/network/data-visualizations
- accuracy (single)
	- = accuracy in expressing magnitudes, quantity
	- psychophysics = perceived intensity vs. actual intensity, can be overestimated/underestimated, each stimuli follows power law but with slight variations → also applies to different visual channels
	- sometimes you want to trade off accurarcy for scalability (information density in graph)
- discriminability (single)
	- = number of distinguishable values
	- depends on: channel type, num of values, position, size
	- too many values clutter the visualization, resolve by: grouping, filtering, faceting (putting them side by side)
- salience / pop-out (multiple)
	- = attracting attention, standing out from redundancy
	- preattentive-processing
	- some channels are stronger than others
	- some channels are asymmetric (they work differently based on the data distribution)
	- gray scale colors help guide attention
- seperability (multiple)
	- = tuning attention, especially during conjunctive search (multiple channels used to encode at once)
	- use integral dimensions for conjunctive search, use seperable dimensions for isolated search
- grouping (multiple)
	- = grouping and pattern information
	- gestalt laws = graphical objects grouped to create a pattern through: proximity, similarity, connection, enclosure, closure, continuity (some are stronger than others)

*color perception*

- https://tristen.ca/hcl-picker/#/hlc/6/1/15534C/E2E062
- most important channel
- cultural associations with colors aren't universal
- allows us to:
	- encode quantities
	- visualize patterns
	- encode categories / label data
	- highlight elements
- color perception:
	- trichromacy theory = there are 3 types of cones, sensitive to rgb colors
	- opponent process theory = color in visual cortex as 3 channel system (red-green, blue-yellow, black-white)
- color specification:
	- colorspaces / gamut = range of available colors
	- discriminability = we can at most recognize 5-10 colors as distinct
	- uniformity = value difference corresponds to perceived difference
	- rgb = red, green, blue
	- hsv/hsl = value, saturation, hue → intuitive
	- cie xyz/lab/luv = based on opponent process theory → perceptually uniform
	- cie lch/hcl = hue (distinguishable color), chroma (colorfulness), luminance (lightness) – maps cartesian coordinates of cie-lab to cylindrical coordinates → intuitive and perceptually uniform

*color maps*

- https://github.com/axismaps/colorbrewer/
- multi-hue sequential color scale:
	- chroma and luminance are fixed, hue varies
	- enables us to choose an equidistant color palette for visualizations
	- improves discriminability because we span a larger volume of the color space
- diverging color scales:
	- hue based on whether values are above/below a threshold (ie. positive, negative)
	- chroma and luminance can still vary to encode quantity

*accessibility*

- color blindness:
	- some cones / photoreceptors are damaged or abscent
	- mostly red-green color blindness
- use blue/orange, blue/red heavy color palettes
- use high saturation for small areas, low saturation for large areas
- use larger symbols and graphs
- use high contrast (by changing luminance) to create seperation
- color constancy = color perception changes based on background
