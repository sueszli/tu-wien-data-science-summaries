# Introduction to Semantic Systems Assignment 2 (2024W)
# Task 1 Basic SPARQL on a didactic ontology (10 points)

> Q1: __SELECT__ Return all actors with their names

This query selects bindings for `?actor` and `?name` by retrieving resources of type `:Actor` and their associated `:fullName` values.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?actor ?name
WHERE {
    ?actor rdf:type :Actor .
    ?actor :fullName ?name .
}
```
Output:

| actor                                                 | name                   |
|:------------------------------------------------------|:-----------------------|
| http://semantics.id/ns/example#isabelle_huppert       | Isabelle Huppert       |
| http://semantics.id/ns/example#annie_girardot         | Annie Girardot         |
| http://semantics.id/ns/example#juliette_binoche       | Juliette Binoche       |
| http://semantics.id/ns/example#jean-louis_trintignant | Jean-Louis Trintignant |
| http://semantics.id/ns/example#ralph_fiennes          | Ralph Fiennes          |
| http://semantics.id/ns/example#william_dafoe          | William Dafoe          |
| http://semantics.id/ns/example/film#harrison_ford     | Harrison Ford          |
| http://semantics.id/ns/example/film#ryan_gosling      | Ryan Gosling           |
| http://semantics.id/ns/example/film#kathleen_quinlan  | Kathleen Quinlan       |
| http://semantics.id/ns/example/film#david_keith       | David Keith            |
| http://semantics.id/ns/example/film#will_smith        | Will Smith             |
| http://semantics.id/ns/example/film#jeff_goldblum     | Jeff Goldblum          |

> Q2: __ASK__ Is there a film directed by Michael Haneke after 2020?

This `ASK` query checks if any resource of type `:Film` exists in the dataset with `:hasDirector` as Michael Haneke and a `:releaseYear` greater than 2020 by filtering the integer value of the year.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

ASK {
    # find any film in the dataset
    ?film rdf:type :Film .
    # that has Michael Haneke as director
    ?film :hasDirector <http://semantics.id/ns/example#michael_haneke> .
    # and get its release year
    ?film :releaseYear ?year .
    # check if the release year is (strictly) greater than 2020
    FILTER (?year > xsd:integer("2020"))
}
```
Output: No
> Q3: __DESCRIBE__ Give me all information about the film 'Independence Day' released in 1996

The query uses a `DESCRIBE` clause to retrieve a concise RDF graph about the variable `?film`, filtered in the `WHERE` clause by its RDF type as `:Film`, its `rdfs:label` as the string "Independence Day", and its `:releaseYear` as the integer "1996".

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

DESCRIBE ?film 
WHERE {
    ?film rdf:type :Film .
    ?film rdfs:label "Independence Day"^^xsd:string .
    ?film :releaseYear "1996"^^xsd:integer .
}
```
Output:

```sparql
@prefix : <http://semantics.id/ns/example/film#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdf4j: <http://rdf4j.org/schema/rdf4j#> .
@prefix sesame: <http://www.openrdf.org/schema/sesame#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix fn: <http://www.w3.org/2005/xpath-functions#> .

<http://semantics.id/ns/example#film_10> a :Film, :Artwork, owl:NamedIndividual;
  rdfs:label "Independence Day";
  :hasActor :will_smith, :jeff_goldblum;
  :hasPerformer :will_smith, :jeff_goldblum;
  :hasCrew <http://semantics.id/ns/example#roland_emmerich>, <http://semantics.id/ns/example#dean_devlin>;
  :hasDirector <http://semantics.id/ns/example#roland_emmerich>;
  :hasFilmStudio <http://semantics.id/ns/example#twentieth_century_fox>;
  :hasGenre :genre_science_fiction, :genre_action;
  :hasScriptWriter <http://semantics.id/ns/example#roland_emmerich>, <http://semantics.id/ns/example#dean_devlin>;
  :releaseYear 1996 .
```
> Q4: __CONSTRUCT__ Return the directors and script writers who have worked together. You may use :collaboratedWith as the newly constructed property

This query uses the `CONSTRUCT` form to generate RDF triples that link directors to scriptwriters with the predicate `:collaboratedWith`, based on shared involvement in films, while ensuring type constraints and excluding cases where the director and writer are the same individual.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

CONSTRUCT {
    ?director :collaboratedWith ?writer
}
WHERE {
    # for each film find its director and script writer
    ?film :hasDirector ?director .
    ?film :hasScriptWriter ?writer .
    # make sure they have the correct types
    ?director rdf:type :Director .
    ?writer rdf:type :ScriptWriter .
    # and they are not the same person
    FILTER (?director != ?writer)
}
```
Output:

| subject                                         | predicate                                            | object                                         |
|:------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------|
| http://semantics.id/ns/example#paul_verhoeven   | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#david_birke     |
| http://semantics.id/ns/example#ridley_scott     | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#hampton_fancher |
| http://semantics.id/ns/example#denis_villeneuve | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#hampton_fancher |
| http://semantics.id/ns/example#ridley_scott     | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#david_peoples   |
| http://semantics.id/ns/example#denis_villeneuve | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#michael_green   |
| http://semantics.id/ns/example#robert_mandel    | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#alice_hoffman   |
| http://semantics.id/ns/example#roland_emmerich  | http://semantics.id/ns/example/film#collaboratedWith | http://semantics.id/ns/example#dean_devlin     |

> Q5: __CONSTRUCT__ Return the directors and films where the director is both director and script writer. You may use :directorandwriterof as the newly constructed property

The query constructs triples where the same individual, identified as both the director and scriptwriter of a film, is linked to the film using the predicate `:directorandwriterof`, with the condition that the individual must have the RDF types `:Director` and `:ScriptWriter`.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

CONSTRUCT {
    ?director :directorandwriterof ?film
}
WHERE {
    ?film :hasDirector ?director .
    ?film :hasScriptWriter ?writer .
    ?director rdf:type :Director .
    ?writer rdf:type :ScriptWriter .
    # same as before but now they have to be the same person
    FILTER (?director = ?writer)
}
```
Output:

| subject                                          | predicate                                               | object                                 |
|:-------------------------------------------------|:--------------------------------------------------------|:---------------------------------------|
| http://semantics.id/ns/example#michael_haneke    | http://semantics.id/ns/example/film#directorandwriterof | http://semantics.id/ns/example#film_1  |
| http://semantics.id/ns/example#michael_haneke    | http://semantics.id/ns/example/film#directorandwriterof | http://semantics.id/ns/example#film_2  |
| http://semantics.id/ns/example#michael_haneke    | http://semantics.id/ns/example/film#directorandwriterof | http://semantics.id/ns/example#film_3  |
| http://semantics.id/ns/example#michael_haneke    | http://semantics.id/ns/example/film#directorandwriterof | http://semantics.id/ns/example#film_4  |
| http://semantics.id/ns/example#anthony_minghella | http://semantics.id/ns/example/film#directorandwriterof | http://semantics.id/ns/example#film_5  |
| http://semantics.id/ns/example#roland_emmerich   | http://semantics.id/ns/example/film#directorandwriterof | http://semantics.id/ns/example#film_10 |
> Q6: __FILTER__ Return all films with 'Blade Runner' in their titles

The query retrieves all resources typed as `:Film` with a `rdfs:label` containing the string "Blade Runner".

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?film ?label
WHERE {
    ?film rdf:type :Film .
    ?film rdfs:label ?label .
    FILTER(CONTAINS(?label, "Blade Runner"))
}
```
Output:

| film                                  | label             |
|:--------------------------------------|:------------------|
| http://semantics.id/ns/example#film_7 | Blade Runner      |
| http://semantics.id/ns/example#film_8 | Blade Runner 2049 |
> Q7: __FILTER__ Return all the names of directors who made any films in 1990 or earlier

This query retrieves distinct director names by matching films with directors and release years in the RDF graph, filtering for films released on or before 1990, and ensuring type compatibility using an explicit cast to `xsd:integer`.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?directorName # drop duplicates
WHERE {
    # get director and year from film
    ?film :hasDirector ?director .
    ?film :releaseYear ?year .
    # get director's name to return
    ?director :fullName ?directorName .
    # filter movies released before 1990
    FILTER (?year <= xsd:integer("1990"))
}
```
Output:

| directorName   |
|:---------------|
| Ridley Scott   |
| Robert Mandel  |
> Q8: __ORDER and GROUP__ Return the actor with number of films they starred in, in descending order

This query retrieves actors alongside the count of distinct films they appear in, grouped by actor and ordered in descending order of the film count.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>

SELECT ?actor (COUNT(DISTINCT ?film) as ?filmCount) # return number of distinct films per actor
WHERE {
    ?film :hasActor ?actor .
}
GROUP BY ?actor # group movies by actor
ORDER BY DESC(?filmCount) # sort by number of distinct films
```
Output:


| actor                                                 |   filmCount |
|:------------------------------------------------------|------------:|
| http://semantics.id/ns/example#isabelle_huppert       |           4 |
| http://semantics.id/ns/example#annie_girardot         |           2 |
| http://semantics.id/ns/example#juliette_binoche       |           2 |
| http://semantics.id/ns/example#jean-louis_trintignant |           2 |
| http://semantics.id/ns/example/film#harrison_ford     |           2 |
| http://semantics.id/ns/example#ralph_fiennes          |           1 |
| http://semantics.id/ns/example#william_dafoe          |           1 |
| http://semantics.id/ns/example/film#ryan_gosling      |           1 |
| http://semantics.id/ns/example/film#kathleen_quinlan  |           1 |
| http://semantics.id/ns/example/film#david_keith       |           1 |
| http://semantics.id/ns/example/film#will_smith        |           1 |
| http://semantics.id/ns/example/film#jeff_goldblum     |           1 |
> Q9: __ORDER and GROUP__ Return the number of actors in each film, in ascending order of their release year

The query retrieves distinct films along with their labels, release years, and the count of unique associated actors by filtering and grouping data based on film entities, with results sorted chronologically by release year.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?filmLabel (COUNT(DISTINCT ?actor) as ?actorCount) ?year # count distinct actors per film
WHERE {
    # for each film get all: label, year, actor
    ?film rdf:type :Film .
    ?film rdfs:label ?filmLabel .
    ?film :releaseYear ?year .
    ?film :hasActor ?actor .
}
GROUP BY ?filmLabel ?year # group by film and year
ORDER BY ?year # sort chronologically
```
Output:

| filmLabel           |   actorCount | year   |
|:--------------------|-------------:|:-------|
| Blade Runner        |            1 | 1,982  |
| Independence Day    |            2 | 1,983  |
| The English Patient |            3 | 1,996  |
| Independence Day    |            2 | 1,996  |
| The Piano Teacher   |            2 | 2,001  |
| Cache               |            2 | 2,005  |
| Amour               |            2 | 2,012  |
| Elle                |            1 | 2,016  |
| Happy End           |            2 | 2,017  |
| Blade Runner 2049   |            2 | 2,017  |
> Q10: __UNION__ Return a combined list of films and their directors, and films and their film studios

This query uses a `SELECT` statement with a `WHERE` clause containing a `UNION` block to retrieve either `(film, director)` pairs or `(film, studio)` pairs by matching RDF triples with specific patterns and applying property paths for director and studio labels.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?film ?director ?studio
WHERE {
    {
        # only get (film, director) pairs
        ?film rdf:type :Film .
        ?film :hasDirector ?directorUri .
        ?directorUri rdfs:label|:fullName ?director .
    }
    UNION
    {
        # only get (film, studio) pairs
        ?film rdf:type :Film .
        ?film :hasFilmStudio ?studioUri .
        ?studioUri rdfs:label ?studio .
    }
}
```
Output:

| film                                   | director          | studio               |
|:---------------------------------------|:------------------|:---------------------|
| http://semantics.id/ns/example#film_1  | Michael Haneke    |                      |
| http://semantics.id/ns/example#film_2  | Michael Haneke    |                      |
| http://semantics.id/ns/example#film_3  | Michael Haneke    |                      |
| http://semantics.id/ns/example#film_4  | Michael Haneke    |                      |
| http://semantics.id/ns/example#film_5  | Anthony Minghella |                      |
| http://semantics.id/ns/example#film_6  | Paul Verhoeven    |                      |
| http://semantics.id/ns/example#film_7  | Ridley Scott      |                      |
| http://semantics.id/ns/example#film_8  | Denis Villeneuve  |                      |
| http://semantics.id/ns/example#film_9  | Robert Mandel     |                      |
| http://semantics.id/ns/example#film_10 | Roland Emmerich   |                      |
| http://semantics.id/ns/example#film_1  |                   | MK2                  |
| http://semantics.id/ns/example#film_3  |                   | Les Films du Losange |
| http://semantics.id/ns/example#film_4  |                   | Les Films du Losange |
| http://semantics.id/ns/example#film_5  |                   | Miramax Films        |
| http://semantics.id/ns/example#film_6  |                   | SBS Productions      |
| http://semantics.id/ns/example#film_7  |                   | Warner Bros.         |
| http://semantics.id/ns/example#film_8  |                   | Warner Bros.         |
| http://semantics.id/ns/example#film_9  |                   | Warner Bros.         |
| http://semantics.id/ns/example#film_10 |                   | 20th Century Fox     |
# Task 2: Querying knowledge graphs on the web (7.5 points)
> Q11: List the names of all Actors who starred in the movie Star Wars IV: A New Hope and order by their age

The query retrieves a distinct list of actors who starred in the film "Star Wars (film)", along with their ages, by binding the difference between the current year and their birth year (adjusted for whether their birthday has occurred this year) to the variable `?age`, ensuring that only English-language labels for actor names are included, and orders the results in descending order of age.

```sparql
# [endpoint=https://dbpedia.org/sparql]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>


SELECT DISTINCT ?actorName ?age # drop duplicates
WHERE {
    # get all actors in nodes named "Star Wars (film)"
    dbr:Star_Wars_\(film\) dbo:starring ?actor .
    ?actor rdfs:label ?actorName .
    ?actor dbo:birthDate ?birthDate .
    # calculate age
    BIND(year(now()) - year(?birthDate) - if(month(now()) < month(?birthDate) || (month(now()) = month(?birthDate) && day(now()) < day(?birthDate)), 1, 0) as ?age)
    # only use English names
    FILTER(LANG(?actorName) = "en")
}
ORDER BY DESC(?age) # sort descending by age
```
Output:

| actorName | age |
|---|---|
| Peter Cushing | 111 |
| Alec Guinness | 110 |
| Harrison Ford | 82 |
| Mark Hamill | 73 |
| Carrie Fisher | 68 |

> Q12: ASK Is there a movie that Steven Spielberg and Tom Hanks both directed?

The query uses a boolean form to determine whether there exists at least one resource, bound to the variable `?movie`, that is explicitly typed as `dbo:Film` and simultaneously has both `dbr:Steven_Spielberg` and `dbr:Tom_Hanks` as its directors, leveraging RDF triples defined within the specified ontology and resource namespaces.

```sparql
# [endpoint=https://dbpedia.org/sparql]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

ASK {
    ?movie rdf:type dbo:Film .
    ?movie dbo:director dbr:Steven_Spielberg .
    ?movie dbo:director dbr:Tom_Hanks .
}
```
Output: No
> Q13: Count the number of movies released after 1970 with at least one writer with the first name "Alex" and the number of movies starring an actor with the first name "Leo" released before or in 1970. The result of the query should be the sum of the two amounts.

The query calculates the total count of distinct films that either (1) were written by individuals with a given name starting with "Alex" and released after 1970 or (2) starred actors with a given name starting with "Leo" and were released in or before 1970, combining the results of two subqueries using a `UNION`, where each subquery retrieves distinct films filtered based on their relationship with relevant properties and applies conditions on associated literals using `FILTER` expressions.

```sparql
# [endpoint=https://dbpedia.org/sparql]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT (COUNT(DISTINCT ?movie1) + COUNT(DISTINCT ?movie2) AS ?total)
WHERE {
    {
        SELECT DISTINCT ?movie1 WHERE {
            ?movie1 rdf:type dbo:Film .
            ?movie1 dbo:writer ?writer .
            ?movie1 dbo:releaseDate ?date1 .
            ?writer foaf:givenName ?writerName .
            FILTER(REGEX(?writerName, "^Alex", "i"))
            FILTER(YEAR(?date1) > 1970)
        }
    }
    UNION
    {
        SELECT DISTINCT ?movie2 WHERE {
            ?movie2 rdf:type dbo:Film .
            ?movie2 dbo:starring ?actor .
            ?movie2 dbo:releaseDate ?date2 .
            ?actor foaf:givenName ?actorName .
            FILTER(REGEX(?actorName, "^Leo", "i")) # `i` flag for case-insensitivity, `^` for start of string
            FILTER(YEAR(?date2) <= 1970)
        }
    }
}
```
Output:

| total |
|---|
| 1 |

Note: There are 3 movies where the names are not at the beginning of the string. But the strings should be structured as `<first name> <last name>`.
# Task 3: Querying with/without inference (7.5 points)
> Query Q31: Subproperty Pattern

This query leverages the subproperty relationship between `hasActor` and `hasPerformer`.

With inference disabled it does not return results since no triples directly use the `hasPerformer` property.

With inference enabled it returns all films and their actors because `hasActor` is defined as a subproperty of `hasPerformer`, so the reasoner infers `hasPerformer` relationships from existing hasActor triples.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2?infer=False]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?film ?performer
WHERE {
    ?film :hasPerformer ?performer .
}
```
Output:

| film | performer |
|---|---|


```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2?infer=True]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?film ?performer
WHERE {
    ?film :hasPerformer ?performer .
}
```
Output:

| film | performer |
|---|---|
| http://semantics.id/ns/example#film_1 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_1 | http://semantics.id/ns/example#annie_girardot |
| http://semantics.id/ns/example#film_2 | http://semantics.id/ns/example#annie_girardot |
| http://semantics.id/ns/example#film_2 | http://semantics.id/ns/example#juliette_binoche |
| http://semantics.id/ns/example#film_3 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_3 | http://semantics.id/ns/example#jean-louis_trintignant |
| http://semantics.id/ns/example#film_4 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_4 | http://semantics.id/ns/example#jean-louis_trintignant |
| http://semantics.id/ns/example#film_5 | http://semantics.id/ns/example#juliette_binoche |
| http://semantics.id/ns/example#film_5 | http://semantics.id/ns/example#ralph_fiennes |
| http://semantics.id/ns/example#film_5 | http://semantics.id/ns/example#william_dafoe |
| http://semantics.id/ns/example#film_6 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_7 | http://semantics.id/ns/example/film#harrison_ford |
| http://semantics.id/ns/example#film_8 | http://semantics.id/ns/example/film#harrison_ford |
| http://semantics.id/ns/example#film_8 | http://semantics.id/ns/example/film#ryan_gosling |
| http://semantics.id/ns/example#film_9 | http://semantics.id/ns/example/film#kathleen_quinlan |
| http://semantics.id/ns/example#film_9 | http://semantics.id/ns/example/film#david_keith |
| http://semantics.id/ns/example#film_10 | http://semantics.id/ns/example/film#will_smith |
| http://semantics.id/ns/example#film_10 | http://semantics.id/ns/example/film#jeff_goldblum |

> Query Q32: Subclass Pattern

With inference disabled it only returns entities explicitly declared as Artwork (none).

With inference enabled it returns both explicit Artwork instances and Film instances, since Film is a subclass of Artwork.

Additionally, any resource that has the duration property will be inferred as Artwork since duration has Artwork as its domain.

The ontology defines these relationships through statements like: `:hasActor rdfs:subPropertyOf :hasPerformer`, `:Film rdfs:subClassOf :Artwork`, `:duration rdfs:domain :Artwork`.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2?infer=False]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?artwork
WHERE {
    ?artwork rdf:type :Artwork .
}
```
Output:

| artwork |
|---|


```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2?infer=True]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?film ?performer
WHERE {
    ?film :hasPerformer ?performer .
}
```
Output:

| film | performer |
|---|---|
| http://semantics.id/ns/example#film_1 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_1 | http://semantics.id/ns/example#annie_girardot |
| http://semantics.id/ns/example#film_2 | http://semantics.id/ns/example#annie_girardot |
| http://semantics.id/ns/example#film_2 | http://semantics.id/ns/example#juliette_binoche |
| http://semantics.id/ns/example#film_3 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_3 | http://semantics.id/ns/example#jean-louis_trintignant |
| http://semantics.id/ns/example#film_4 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_4 | http://semantics.id/ns/example#jean-louis_trintignant |
| http://semantics.id/ns/example#film_5 | http://semantics.id/ns/example#juliette_binoche |
| http://semantics.id/ns/example#film_5 | http://semantics.id/ns/example#ralph_fiennes |
| http://semantics.id/ns/example#film_5 | http://semantics.id/ns/example#william_dafoe |
| http://semantics.id/ns/example#film_6 | http://semantics.id/ns/example#isabelle_huppert |
| http://semantics.id/ns/example#film_7 | http://semantics.id/ns/example/film#harrison_ford |
| http://semantics.id/ns/example#film_8 | http://semantics.id/ns/example/film#harrison_ford |
| http://semantics.id/ns/example#film_8 | http://semantics.id/ns/example/film#ryan_gosling |
| http://semantics.id/ns/example#film_9 | http://semantics.id/ns/example/film#kathleen_quinlan |
| http://semantics.id/ns/example#film_9 | http://semantics.id/ns/example/film#david_keith |
| http://semantics.id/ns/example#film_10 | http://semantics.id/ns/example/film#will_smith |
| http://semantics.id/ns/example#film_10 | http://semantics.id/ns/example/film#jeff_goldblum |

> Query Q33: Property Symmetry Pattern

With inference disabled, the query will return only two results showing the explicit `friendOf` relationships:

- alice → bob
- charlie → david

With inference enabled, the query will return four results due to the `owl:SymmetricProperty` declaration on `:friendOf`:

- alice → bob
- alice ← bob
- charlie → david
- charlie ← david

This demonstrates how the symmetric property pattern automatically infers bidirectional relationships when inference is enabled.
This means that when inference is enabled, for every triple (`A :friendOf B`), the triple (`B :friendOf A`) is automatically inferred, effectively making friendship relationships bidirectional in the graph - yay.

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2]
PREFIX : <http://semantics.id/ns/example/film#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

INSERT DATA {
    :alice :friendOf :bob .
    :charlie :friendOf :david .
}

```

```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2?infer=False]
PREFIX : <http://semantics.id/ns/example/film#>

SELECT ?person1 ?person2
WHERE {
    ?person1 :friendOf ?person2 .
}
```
Output:

| person1 | person2 |
|---|---|
| http://semantics.id/ns/example/film#alice | http://semantics.id/ns/example/film#bob |
| http://semantics.id/ns/example/film#charlie | http://semantics.id/ns/example/film#david |


```sparql
# [endpoint=http://localhost:7200/repositories/ISS_AS2?infer=True]
PREFIX : <http://semantics.id/ns/example/film#>

SELECT ?person1 ?person2
WHERE {
    ?person1 :friendOf ?person2 .
}
```
Output:

| person1 | person2 |
|---|---|
| http://semantics.id/ns/example/film#alice | http://semantics.id/ns/example/film#bob |
| http://semantics.id/ns/example/film#bob | http://semantics.id/ns/example/film#alice |
| http://semantics.id/ns/example/film#charlie | http://semantics.id/ns/example/film#david |
| http://semantics.id/ns/example/film#david | http://semantics.id/ns/example/film#charlie |

# Addendum
Utility code snippet to parse CSV files from GraphDB:

```python
csvdata = """
film,director,studio
http://semantics.id/ns/example#film_1,Michael Haneke,
"""

import pandas as pd
from io import StringIO
df = pd.read_csv(StringIO(csvdata))
markdown_table = df.to_markdown(index=False)
print(markdown_table)
```

Utility code snippet to parse JSON output from sparqlbook results:

```python
import json

sparql_json = """
{"head":{"vars":["person1","person2"]},"results":{"bindings":[{"person1":{"type":"uri","value":"http://semantics.id/ns/example/film#alice"},"person2":{"type":"uri","value":"http://semantics.id/ns/example/film#bob"}},{"person1":{"type":"uri","value":"http://semantics.id/ns/example/film#bob"},"person2":{"type":"uri","value":"http://semantics.id/ns/example/film#alice"}},{"person1":{"type":"uri","value":"http://semantics.id/ns/example/film#charlie"},"person2":{"type":"uri","value":"http://semantics.id/ns/example/film#david"}},{"person1":{"type":"uri","value":"http://semantics.id/ns/example/film#david"},"person2":{"type":"uri","value":"http://semantics.id/ns/example/film#charlie"}}]}}
"""

sparql_json = json.loads(sparql_json) if isinstance(sparql_json, str) else sparql_json

headers = sparql_json["head"]["vars"]
rows = []
for binding in sparql_json["results"]["bindings"]:
    row = []
    for header in headers:
        row.append(binding[header]["value"])
    rows.append(row)

markdown_table = "| " + " | ".join(headers) + " |\n"
markdown_table += "|" + "|".join(["---" for _ in headers]) + "|\n"
for row in rows:
    markdown_table += "| " + " | ".join(row) + " |\n"

print(markdown_table)
```
