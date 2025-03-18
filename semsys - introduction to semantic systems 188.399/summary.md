*knowledge*

- = connected data

*knowledge graph*

- = directed labeled graph, for entities and relationships

*semantic systems*

- = systems that make use of knowledge representations (ontologies, taxonomies, knowledge graphs)
- inference: logic-reasoning, machine learning
- datasets: open data (wikidata), semantic web (hyperlinks)
- technologies: progerty graphs (pg), resource description frameworks (rdf)

*taxonomy*

- based on terms in a vocabulary
- lightweight taxonomy = clustered terms
- heavyweight taxonomy = hierarchical terms (inheritance), logic

*ontology*

- = taxonomy + relations and constraints
	- classes/concepts in domain/terminological space $\Delta^I$
	- objects/instances/entities/individuals → mapped to classes with interpretation function $I$ 
	- inheritance/hierarchy → ie. `is_a`
	- relations/roles → ie. `eats_food`, first order logic operations / description logic
	- assertions/restrictions → ie. limit on cardinality, ranges
- you can create/design, compare/map, combine/merge, retrieve/learn, populate ontologies

*ontology design*

- i. determine scope
	- what questions should it answer? (competency questions)
- ii. consider reuse
	- reuse existing ontology datasets
	- libraries: protege, bioontology, linked open vocab
- iii. enumerate terms
	- define nouns/terms/vocabulary and verbs/properties
- iv. define classes and taxonomy
	- define inheritance hierarchy → you can list them like directories on a filesystem
- v. define properties
	- properties should be defined for the highest possible class in the inheritance chain
	- properties can also have hierarchies/subproperties
- vi. define constraints
	- cardinality, value ranges, property constraints (summetry, transiviy, inverse properties, functional values)
- vii. create instances
	- populate ontology
- viii. check for anomalies
	- validate (consistency, satisfiability), infer additional knowledge
	- use reasoners/reference-engines to:
		- infer new knowledge (new relations between classes, new instances)
		- check correctness/consistency

*semantic web*

- web of documents with hyperlinks
- uniquie resource identifier URI → for each protocol (mail, phone, https)
	- uniform resource name URN
	- uniform resource locator URL
	- internationalized resource identifier IRI

# syntax: triples

RDF $\in$ RDFS $\in$ OWL

*rdf - resource description framework*

- https://www.w3.org/TR/rdf11-primer/
- https://github.com/VisualDataWeb/WebVOWL (visualizer)
- w3c graph based model
- syntax:
	- subject - resource with uri
	- predicate - property with uri
		- from xmlns/w3 namespace
	- object - resource, blank node, literal/plaintext
		- plaintext can have lanugage tags
		- blank nodes used for n-ary relations
		- can be a nested triple for reification (citing something)
- serialization formats:
	- xml → for xml compatibility, very verbose, not popular
	- turtle (subset of notation3/N3) → for manual editing, succinct, most common
	- n-triples → for high performance parsing, verbose, no prefixes
	- json-dl → for json compatibility

*rdfs - resource description framework schema*

- https://www.w3.org/TR/rdf-schema/
- assumptions:
	- non-unique nameing assumption → "aaa" slogan: anyone can say anything about anything on the web
	- open world assumption → missing information is not negative information
- purpose: nomination/assertions, inference
- poor expressivity, very simple assertions, very basic inference

*owl - web ontology language*

- https://www.w3.org/TR/owl2-primer/ (choose preferred syntax first)
- https://www.w3.org/TR/owl-features/
- extends rdfs, based on description/first order logic
- constructs: classes, properties, individuals

# syntax: toolchain

*rml - rdf mapping language*

- https://rml.io/docs/
- https://rml.io/implementation-report/
- used to map other file formats into triples
- superset of r2rml (used for databases)

*sparql - sparql protocol and rdf query language*

- https://www.w3.org/TR/sparql12-query/
- https://www.w3.org/2009/sparql/wiki/Main_Page
- https://www.w3.org/wiki/SparqlImplementations
- playground: https://query.wikidata.org/, https://dbpedia.org/sparql, https://data.europa.eu/data/sparql?locale=en
- used in triples databases for querying

*shacl - shapes constraint language*

- https://www.w3.org/TR/shacl/#dfn-shape
- https://graphdb.ontotext.com/documentation/10.8/shacl-validation.html
- playground: https://shacl.org/playground/
- used for data validation

# useful tools

apps:

- graphdb: https://www.ontotext.com/products/graphdb/ (alternatively: apache jena + fuseki ui)

cli tools:

- `rmlmapper` for rml: https://github.com/RMLio/rmlmapper-java/ (as jar binary)
- `pyshacl` for shacl: https://github.com/RDFLib/pySHACL
- `turtlefmt` for formatting turtle files: https://github.com/helsing-ai/turtlefmt
- `ttl-merge` for merging turtle files: https://github.com/julianrojas87/ttl-merge
