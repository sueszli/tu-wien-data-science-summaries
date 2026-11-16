from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Namespace

ROOT = Path(__file__).resolve().parent
UB = Namespace("https://example.org/vienna-ubahn#")


def main():
    graph = Graph().parse(ROOT / "network.ttl")
    reverse = (UB.Stephansplatz, UB.connectsTo, UB.Karlsplatz)
    if reverse in graph:
        raise SystemExit("The reverse connection should not be explicit.")
    conforms, _, report = validate(
        graph,
        shacl_graph=str(ROOT / "shapes.ttl"),
        inference="owlrl",
        inplace=True,
    )
    if not conforms:
        raise SystemExit(report)
    if reverse not in graph:
        raise SystemExit("The reverse connection was not inferred.")
    print("SHACL validation passed; reverse connection inferred.")
    for station, line_count in graph.query((ROOT / "transfers.rq").read_text()):
        print(f"{station}: {line_count} lines")


if __name__ == "__main__":
    main()
