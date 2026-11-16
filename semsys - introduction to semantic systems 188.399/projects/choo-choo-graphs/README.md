# Vienna U-Bahn knowledge graph

A small project for Introduction to Semantic Systems (188.399). It models parts of Vienna's U1 and U4 lines in Turtle, validates the graph with SHACL, and finds interchange stations with SPARQL. The ontology's symmetric connection property lets an OWL reasoner infer the reverse of each recorded connection.

The example covers only a few stations. It is not live transit data or a full representation of the network. See [Wiener Linien's network maps](https://www.wienerlinien.at/fahrplaene) for passenger information.

Run from this folder with Python 3:

```sh
python -m pip install pyshacl
python demo.py
```

The expected transfer stations in this sample are Karlsplatz and Schwedenplatz. The program also verifies that reasoning infers a reverse connection.
