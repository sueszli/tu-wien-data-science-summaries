# 1. Construct Queries

## 1.1. Query

```sparql
PREFIX film: <http://semantics.id/ns/example/film#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# insert frequently collaborating actors and director preferences
INSERT {
    ?actor1 film:frequentCostarWith ?actor2 .

    ?director film:hasPreferredActor ?actor .

    ?director film:specializesInGenre ?genre .

    ?film1 film:fromSameEra ?film2 .
}
WHERE {
    {
        # find actors who worked together in at least 2 films
        SELECT ?actor1 ?actor2 (COUNT(DISTINCT ?film) as ?collaborations)
        WHERE {
            ?film film:hasActor ?actor1 .
            ?film film:hasActor ?actor2 .
            FILTER(?actor1 != ?actor2)
        }
        GROUP BY ?actor1 ?actor2
        HAVING (?collaborations >= 2)
    }
    UNION
    {
        # find directors who worked with same actor in multiple films
        SELECT ?director ?actor (COUNT(DISTINCT ?film) as ?collaborations)
        WHERE {
            ?film film:hasDirector ?director .
            ?film film:hasActor ?actor .
        }
        GROUP BY ?director ?actor
        HAVING (?collaborations >= 2)
    }
    UNION
    {
        # find directors who frequently work in specific genres
        SELECT ?director ?genre (COUNT(DISTINCT ?film) as ?genreFilms)
        WHERE {
            ?film film:hasDirector ?director .
            ?film film:hasGenre ?genre .
        }
        GROUP BY ?director ?genre
        HAVING (?genreFilms >= 2)
    }
    UNION
    {
        # find films from the same era, with a maximum difference of 2 years
        SELECT ?film1 ?film2
        WHERE {
            ?film1 film:releaseYear ?year1 .
            ?film2 film:releaseYear ?year2 .
            FILTER(?film1 != ?film2 && abs(?year1 - ?year2) <= 2)
            FILTER NOT EXISTS { ?film1 film:fromSameEra ?film2 }
        }
    }
}
```

## 1.2. Results

```sparql
PREFIX film: <http://semantics.id/ns/example/film#>
SELECT DISTINCT ?actor1 ?actor2 ?collaborations
WHERE {
    ?actor1 film:frequentCostarWith ?actor2 .
    {
        SELECT ?actor1 ?actor2 (COUNT(DISTINCT ?film) as ?collaborations)
        WHERE {
            ?film film:hasActor ?actor1 .
            ?film film:hasActor ?actor2 .
            FILTER(?actor1 != ?actor2)
        }
        GROUP BY ?actor1 ?actor2
    }
}
```

| actor1 | actor2 | collaborations |
|--------|--------|----------------|
| http://semantics.id/ns/example#isabelle_huppert | http://semantics.id/ns/example#jean-louis_trintignant | 2 |
| http://semantics.id/ns/example#jean-louis_trintignant | http://semantics.id/ns/example#isabelle_huppert | 2 |

```sparql
PREFIX film: <http://semantics.id/ns/example/film#>
SELECT DISTINCT ?director ?actor ?collaborations
WHERE {
    ?director film:hasPreferredActor ?actor .
    {
        SELECT ?director ?actor (COUNT(DISTINCT ?film) as ?collaborations)
        WHERE {
            ?film film:hasDirector ?director .
            ?film film:hasActor ?actor .
        }
        GROUP BY ?director ?actor
    }
}
```

| director | actor | collaborations |
|----------|-------|----------------|
| http://semantics.id/ns/example#michael_haneke | http://semantics.id/ns/example#isabelle_huppert | 3 |
| http://semantics.id/ns/example#michael_haneke | http://semantics.id/ns/example#annie_girardot | 2 |
| http://semantics.id/ns/example#michael_haneke | http://semantics.id/ns/example#jean-louis_trintignant | 2 |

```sparql
PREFIX film: <http://semantics.id/ns/example/film#>
SELECT DISTINCT ?director ?genre ?genreFilms
WHERE {
    ?director film:specializesInGenre ?genre .
    {
        SELECT ?director ?genre (COUNT(DISTINCT ?film) as ?genreFilms)
        WHERE {
            ?film film:hasDirector ?director .
            ?film film:hasGenre ?genre .
        }
        GROUP BY ?director ?genre
    }
}
```

| director | genre | genreFilms |
|----------|-------|------------|
| http://semantics.id/ns/example#michael_haneke | film:genre_drama | 4 |

```sparql
PREFIX film: <http://semantics.id/ns/example/film#>
SELECT DISTINCT ?film1 ?film2
WHERE {
    ?film1 film:fromSameEra ?film2 .
}
```

| film1 | film2 |
|-------|-------|
| http://semantics.id/ns/example#film_1 | http://semantics.id/ns/example#film_806 |
| ...   | ...   |

too many to display

# 2. Mandatory Update Query (with personal data)

## 2.1. Query

```sparql
PREFIX : <http://semantics.id/ns/example/film#>

DELETE {
    ?d :fullName "Christopher Nolan"
}
INSERT {
    ?d :fullName "YahyaJabary_11912007"
}
WHERE {
    ?d a :Director ;
       :fullName "Christopher Nolan"
}
```

## 2.2. Results

```sparql
PREFIX : <http://semantics.id/ns/example/film#>
SELECT DISTINCT ?director
WHERE {
    ?director a :Director ;
              :fullName "YahyaJabary_11912007" .
}
```
