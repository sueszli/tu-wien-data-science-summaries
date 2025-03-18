---
title: "Assignment 1"
subtitle: "188.399 Introduction to Semantic Systems 2024W"
author: "11912007 - Yahya Jabary"
output: pdf_document
documentclass: article
papersize: a4
toc: true
toc_depth: 3
number_sections: true
geometry:
    - top=10mm
    - bottom=15mm
    - left=10mm
    - right=10mm
header-includes:
    # smaller font
    # - \usepackage{lmodern}
    # - \usepackage[T1]{fontenc}
    # - \usepackage{sectsty}
    # - \usepackage{anyfontsize}
    # - \AtBeginDocument{\fontsize{6}{7.2}\selectfont}
    # - \allsectionsfont{\fontsize{8}{9.6}\selectfont}
    # - \let\oldtoc\tableofcontents
    # - \renewcommand{\tableofcontents}{\fontsize{6}{7.2}\selectfont\oldtoc}
    # smaller code
    - \usepackage{fancyvrb}
    - \fvset{fontsize=\scriptsize}
    - \usepackage{listings}
    - \lstset{basicstyle=\scriptsize\ttfamily}
    - \RecustomVerbatimEnvironment{verbatim}{Verbatim}{fontsize=\scriptsize}
---

<!--

required files:

- A1_JABARY_11912007_Ontology.owls
- A1_JABARY_11912007_Report.pdf

-->

![Ontology Visualization via `WebVOWL`](./assets/viz.png){ width=80% }

\newpage

# Task 1: Application Proposal

<!-- 2,3 paragraphs: suggest a semantic application that could be built on top of knowledge graphs. -->

In this project, we conceptualize and design an ontology for an application tailored to audiophiles. The ontology is structured to support a music discovery and event-planning platform that assists users in exploring music, locating concerts and tracking their favorite artists.

The application serves several key functions. First, it provides event discovery capabilities, allowing users to find upcoming concerts and festivals. These events can be filtered by venue, artist or genre, offering a personalized experience based on user preferences. Another core feature is artist exploration, which enables users to trace artist collaborations and band membership history. Additionally, users can discover related artists by exploring connections through shared genres or venues. To support music library management, the ontology enables users to organize and categorize their music collections, create playlists and share them with others. Moreover, it facilitates historical analysis, allowing users to track artist evolution across genres and analyze production patterns and collaborations over time.

The ontology should provide a robust foundation for future extensions: Geographic information could be integrated to add location data for venues, enabling proximity-based searches to enhance user experience when searching for events nearby. Social features could also be incorporated, such as user profiles, preferences and event attendance tracking, creating a more interactive and community-oriented experience. Additionally, temporal relations could be introduced, such as defining band membership periods and artist era classifications, adding a historical dimension to the application's data.

<!-- 5-10 competency questions for knowledge graph application -->

Some core **competency questions** that cover the mentioned use cases might include:

1. What concerts are happening in a specific venue in the next month?
2. Which artists are performing at upcoming festivals?
3. What songs has an artist recorded and on which albums do they appear?
4. What other bands have members of a specific band played in?
5. Which producers have worked with a particular artist?
6. What genres are most represented in a user's playlists?
7. Which venues regularly host events for a specific genre?
8. What instruments does each band member play?

# Task 2: Ontology Design

### The Final Ontology

In the following ASCII diagram we have displayed the 15 classes and their hierarchy in the ontology through the use of indentation. A nested class is a subclass of the class above it. Additionally the 20 properties and their schema definitions are listed beside each class, seperated by em-dashes. The goal is to provide the most concise and readable representation of the ontology structure. We did not instantiate any classes in the ontology, as the focus of this exercise was on the schema design.

```
Artist          –– hasName (xsd:string) 
    SoloArtist  –– hasName (xsd:string) 
    Band        –– hasName (xsd:string) 
Album           –– hasName (xsd:string), hasReleaseDate (xsd:date)
Song            –– hasName (xsd:string), hasDuration (xsd:duration)
Genre           –– hasName (xsd:string) 
MusicEvent      –– hasName (xsd:string), MusicEvent: hasEventDate (xsd:date)
    Concert     –– hasName (xsd:string) 
    Festival    –– hasName (xsd:string) 
Venue           –– hasName (xsd:string), hasLocation (xsd:string)
Award           –– hasName (xsd:string) 
RecordLabel     –– hasName (xsd:string) 
Instrument      –– hasName (xsd:string) 
Playlist        –– hasName (xsd:string) 
Producer        –– hasName (xsd:string) 
```

Additionally the following mermaid-flowchart-syntax inspired notation[^mermaid] displays the 10 relations and their directionality in the ontology. The arrow points from the subject to the object of the relation, split by the predicate.

[^mermaid]: See: https://mermaid-js.github.io/mermaid/#/flowchart

```
Song - hasGenre -> Genre
Song - performedBy -> Artist
Album - producedBy -> Producer
Song - isPartOf -> Album
Artist - performsAt -> MusicEvent
MusicEvent - hasVenue -> Venue
SoloArtist - memberOf -> Band
Artist - signedTo -> RecordLabel
Artist - plays -> Instrument
Playlist - contains -> Song
```

Finally, our knowledge graph ontology contains the following OWL assertions to enrich the initial RDFS ontology:

- Disjoint Classes:

    These assertions state that no individual can simultaneously be an instance of both classes in each pair. We declare `SoloArtist` and `Band`, as well as `Concert` and `Festival`, as disjoint.

    This is because we're modelling an n-to-m relationship between artists and bands and between concerts and festivals. An artist can be a solo artist or a band member, but not both. Similarly, an event can be a concert or a festival, but not both. There are many ways to model these relationships, but we chose to use disjoint classes to enforce clear categorization.

- Property Characteristics:

    The `memberOf` property is declared as asymmetric, modeling membership relationships between artists and bands. This ensures that while artists can be members of bands, bands cannot be members of artists.
    
    The `performedBy` property is declared as inverse functional, ensuring that a `Song` can only be performed by one `Artist`. This disallows multiple artists performing the same song – which might not make sense on first glance, but it serves to mark the main performer of a song in order to simplify the ontology. A more fine-grained model could include multiple performers for a song that are weighted by their contribution.

- Cardinality Constraints:

    The `Album` class is defined as a subclass of a restriction that specifies a minimum cardinality of 1 for the `producedBy` property. This ensures that every album must have at least one producer. The integer value is specified as a non-negative integer.

    This constraint enforces data completeness and maintains data quality in music databases. It ensures that albums cannot exist without producer information, which is crucial for crediting and royalty tracking.

- Property Chain:

    The `hasHeadliner` property is inferred through the chain of `hasVenue` and `performsAt` properties. If a venue hosts an event that an artist performs at, the artist is considered the headliner of the event. The noun "headliner" here is a slight misnomer, as we assume that any artist performing at an event is a headliner. This chain property simplifies the ontology by creating a direct relationship between venues and headlining artists.

- Value Restrictions:

    The `Festival` class is defined as a subclass of a restriction that specifies a minimum capacity of 1000 for the `hasCapacity` property. This ensures that any venue hosting a festival must have a minimum capacity of 1000 people. This constraint enforces realistic constraints on festival venues, helps with venue selection and event planning, and maintains data consistency for large-scale events.

These OWL constraints together ensure data consistency, enable automated reasoning, support data validation, maintain logical integrity and make the ontology more useful for practical applications.

### The Development Process & Challenges

We started the ontology development process by identifying the core entities and relationships that would be essential for our music discovery and event-planning application. We brainstormed the key features and user interactions that the application would support, such as event discovery, artist exploration, music library management and historical analysis. Based on these requirements, we defined the initial classes and properties that would form the foundation of our ontology.

Next, having designed an initial schema, very similar to a traditional graph database schema, we iteratively refined the ontology by adding more specific classes, properties and relationships. We considered the cardinal relationships between classes, such as the n-to-m relationship between artists and bands, and the hierarchical relationships between genres and subgenres. We also defined the properties that would link classes together, such as the `performedBy` property that connects songs to artists. We iteratively refined the ontology by adding more specific classes, properties and relationships as well as the inference rules and constraints that would enrich the ontology.

To implement the schema in the required Turtle OWL format, we faced the very unintuitive user interface of the Protege ontology editor. The tool's interface is not very user-friendly, and the process of defining classes, properties and restrictions is cumbersome. Fortunately, reading the official W3C primer on OWL from 2004 was more than sufficient to understand the syntax and semantics of the declarative language and efficiently work in a plaintext editor while iteratively validating the syntax in various online tools that parse OWL. This workflow allowed us to quickly iterate on the ontology design and also structure the file in a very maintainable way, similar to how we would structure a codebase or a YAML configuration file. The final ontology file is well-organized, with clear comments, sections and annotations that explain the purpose and structure of each class, property and restriction.
