exercise sheet 1, advanced database systems, 2024s

author: yahya jabary, 11912007

# execise 1: disk access in databases

*disk specifications:*

- **magenetic disk:**
  - block size: 1 kB
  - rotational speed: 10,000 rpm
  - seek time: 1 ms
  - transfer rate: 500 MB/s
  - track-to-track seek time: 1 ms
  - track size: 1000 kB
- **ssd:**
  - block size: 10 kB
  - transfer rate: 3000 MB/s

_database files:_

- $Item$:
  - $n_i$ = 50,000 records (number of records)
  - $R_i$ = 10 kB (record size)
- $Supplier$:
  - $n_s$ = 200 records
  - $R_s$ = 50 kB

*dbms:*

- block size: 1000 kB → larger than disk block sizes so the dbms will need to perform more i/o operations to interact with a full dbms block.
- unspanned → no indirection
- contiguous block allocation → blocks of the same file are adjacent
- main memory is empty when starting off
- intermediate results and hash values stay in memory

_assignment:_

- query: $Item \bowtie_{Item.supplier=Supplier.id}Supplier$
- calculate access time for the given query execution plans by postgres for each external storage

## a) hash join

```
Hash Join
    Hash Cond: (i.supplier = s.id)
    ->  Seq Scan on item i
    ->  Hash
        ->  Seq Scan on supplier s
```

*hash join*

assume we want to equi join: $R \bowtie_{\text{A}=\text{B}} S$

- i. partition phase:
    - find a hash function that can map values in the join columns to a buffer frame index between $[1;B\text{-}1]$. → the buffer frames we map the rows to are called "buckets" and the 1 remaining buffer frame is used to read new pages in.
    - read each page $p_R$ of $R$ to memory. then hash the join value of each row to find the right bucket to store a pointer in. → if buffer frames overflow, write them back to disk.
    - repeat for $p_S$ of $S$.
    - total cost: $2 \cdot (b_R + b_S)$ → factor of 2 because of initial reading and writing back the potentially full buckets to disk.
- ii. probing phase:
    - assuming $R_i$ and $S_i$ are all rows in the $i$th-bucket (and $R_i$ is the smaller one of them): read $R_i$ to $B\text-2$ buffer frames. → if not possible, either hash recursively or try another algorithm. the 2 remaining buffer frames are used to read new $S_i$ pages in and store the final result.
    - read each page of $S_i$ into memory. then check each row for matches with $R_i$.
    - if a matching row is found, write it into the buffer frame dedicated to results.
    - total cost: $b_i + b_s$
- **total cost of both phases**: $3 \cdot (b_R + b_S)$

*access time: magnetic disk*

- access time for one block:
  - $t_s$ - seek time: 1ms
  - $t_r$ - rotational delay: 0.5 \* (1 / 10,000) \* 60 = 3ms → we assume that it takes 0.5 rotations for a hit on average.
  - $t_{tr}$ - transfer time: 500MB/s = 1kB/0.002ms
  - total: 4.002ms
- access time for $n$ blocks:
  - $t_{t2t}$ - track to track seek time = 1ms
  - num of allocated tracks: 1000kB track size / 1kB block = 1000 blocks per track → for $n$ blocks we need $n$/1000 tracks → we would change tracks ($n$/1000-1) times
  - **random access**: $n$ * 4.002ms
  - **sequential access**: $t_s$ + $t_r$ + $n \cdot t_{tr}$ + track changes \* $t_{t2t}$ = 1ms + 3ms + $n$ \* 0.002ms + ($n$/1000-1) \* 1ms
- i. $Item$
    - total num of blocks: (50,000 records * 10 kB record size) / 1kB block size = 500,000 blocks
    - sequential access of 500,000 blocks: **1503ms**
- ii. $Supplier$
    - total num of blocks: (200 records * 50 kB record size) / 1kB block size = 10,000 blocks
    - sequential access of 10,000 blocks: **33ms**
- iii. total access time for hash join
    - $3 \cdot (b_{Item} + b_{Supplier})$ = 3 * (1503ms + 33ms) = **4608ms**

*access time: ssd*

- access time for one block:
  - $t_{tr}$ - transfer time: 3000MB/s = 10kB/3333.3 ns
- access time for $n$ blocks:
  - **sequential / random access**: $n$ * 3333.3ns
- i. $Item$
    - total num of blocks: (50,000 records * 10 kB record size) / 10kB block size = 50,000 blocks
    - sequential access of 50,000 blocks: **166.665ms**
- ii. $Supplier$
    - total num of blocks: (200 records * 50 kB record size) / 10kB block size = 1000 blocks
    - sequential access of 1000 blocks: **3.3333ms**
- iii. total access time for hash join
    - $3 \cdot (b_{Item} + b_{Supplier})$ = 3 * (166.665ms + 3.3333ms) = **509.9949ms**

## b) index nested loops join

```
Nested Loop
->  Seq Scan on supplier s
->  Index Scan using record_by_idx on item i
Index Cond: (supplier = s.id)
```

pseudo algorithm for naive nested loops join:

```
foreach page p_item of item:
    foreach page p_supplier of supplier:
    foreach tuple i ∈ p_item and s ∈ p_supplier:
        if i.supplier = s.id then Res := Res ∪ {(r,s)}
```

pseudo algorithm for index nested loops join:

```
itemIndex := generateIndex(i.supplier)

foreach page p_supplier of supplier:
    foreach tuple s ∈ p_supplier:
    Res := Res ∪ itemIndex.getMatches(s.id)
```

details:

- every `supplier.id` is looked up in an index of `item`, with the column `item.supplier` as the index key.
- if there is a match, the record from the index pointer gets read from disk.
  - the result contains 20 records.
  - the disk access isn’t sequential. we don’t know anything about the read order.
- we don’t know which kind index was used. we do not include the disk access costs for index creation.
- **total cost**: $b_{Supplier} + 20 \cdot r_{Item}$

*access time: magnetic disk*

- i. $Item$
    - total num of blocks: (20 records * 10 kB record size) / 1kB block size = 200 blocks
    - random access of 200 blocks: **800.4ms**
- ii. $Supplier$
    - sequential access of 10,000 blocks: **33ms** (same as previous example)
- iii. total: **833.4 ms**

*access time: ssd*

- i. $Item$
    - total num of blocks: (20 records * 10 kB record size) / 10kB block size = 20 blocks
    - random access of 20 blocks: **0.066666ms**
- ii. $Supplier$
    - sequential access of 1000 blocks: **3.3333ms** (same as previous example)
- iii. total: **3.399966ms**

# execise 2: selectivity

## a)

estimate the selectivity:

- `repository.contributors` has 100,000 rows
- equi-depth histogram: 7 buckets of equal size using the 6 dividers {1, 2, 4, 7, 12, 20}
- max value: 255
  - assumption: boundary values are included in the following bucket
  - buckets: {\[-∞;0], \[1;1], \[2;3], \[4;6], \[7;11], \[12;19], \[20;255]}
- assume uniform distribution

*i) predicate: `contributors ≥ 4`*

- because the histogram is equi-depth, we can use the bucket count to calculate selectivity
- 4 buckets satisfy the predicate: {\[4;6], \[7;11], \[12;19], \[20;255]}
- selectivity ≈ 4/7 buckets = 0.5714285714

*ii) predicate: `contributors > 12`*

- if the 2 buckets in {\[12;19], \[20;255]} wouldn’t contain the value 12, then they would satisfy the predicate.
- since the values are evenly spread, $\approx(1-\frac{1}{19-12}) = \frac{6}{7}$ of values in the \[12;19] bucket satisfy the predicate.
- selectivity ≈ $(1+\frac{6}{7})/7$ buckets = 0.2653061224

## b)

estimate the selectivity:

- avoid histograms for this part: histograms focus on ranges of values. they aren’t useful for selectivity estimation of equalities.
- `repository.contributors` has 400 distinct values.
  - (note: combined with the prior specification, this means that values can also be negative)

*i) predicate: `contributors == 5`*

- we assume uniform distribution (in the absence of other information)
- selectivity ≈ $\frac{1}{400}$ = 0.0025

*ii) predicate: `contributors != 5`*

- we assume uniform distribution (in the absence of other information)
- selectivity ≈ $1-\frac{1}{400}$ = 0.9975

*limitations of the method*

- **uniform distribution assumption:** this method assumes that all distinct values in the 'contributors' column are equally likely. this is very unlikely in reality. (ie. a negative exponential distribution might be a better approximation - but we can only find out through sampling).
- **lack of correlation:** it doesn't consider potential correlations between the 'contributors' column and other data in the table. these correlations could impact the actual selectivity of the predicate.

*possible solutions*

- **more detailed histograms:** instead of equi-depth histograms, you could use equi-width histograms or more sophisticated histograms that better capture the distribution of frequent values.
- **sampling:** take a representative sample of the data and examine the distribution of values in the 'contributors' column within the sample. this can give you a more realistic estimate of selectivity.
- **collect statistics:** gather more detailed statistics about the frequency of different values in the 'contributors' column. this will improve selectivity estimates for equality predicates.

## c)

estimate the selectivity:

- `user` has 50,000 rows
- `repository` has 100,000 rows (read from the table in the assignment, not part of this exercise)
- the `user.id` key can be joined on the foreign key `repository.owner`
  - assumption: no null values in `repository.owner` because it’s a foreign key.

*i) `repository ⋈_{owner=id} user`*

- since `user.id` is a key attribute, the result of the join can’t have more rows than `user` has.
- selectivity of join ≈ selectivity of `user.id` = 1/50,000 = 0.00002

*ii) `π_owner(repository)`*

- `repository.owner` has \[1;50,000] distinct values
  - 1 distinct value at least → one user owning everything
  - 50,000 distinct values at most → because there are no more keys to match with from the other table
- we assume uniform distribution (in the absence of other information) of repository ownership: therefore `repository.owner` = $(\frac{1+50,000}{2})$ = 25,001
- selectivity ≈ 25,001/100,000 = 0.250005

## d, e)

before optimization:

```sql
SELECT * FROM repository rep, user u, release rel
WHERE rep.owner = u.id AND rel.repo = rep.id
    AND (rel.name = 'v2' OR rel.version = 2)
    AND rep.commits > 105
    AND rep.contributors > 11;
```

after optimization:

```sql
SELECT *
FROM release rel
INNER JOIN repository rep ON rel.repo = rep.id
    AND rep.commits > 105
    AND rep.contributors > 11
INNER JOIN user u ON u.id = rep.owner 
    WHERE  rel.version = 2 OR rel.name = 'v2'; 
```

*rule-based, logical, heuristic optimization*

- simplify relational algebra, reduce i/o access:
  - replace $\times$ and $\sigma$ with $\bowtie$
  - apply $\sigma, \pi$ as early as possible - and apply the stronger filters first
  - remove unnecessary attributes early on
- huge search space:
  - there are $\frac{(2n)!}{n!}$ possible combinations of joining $n\text{+}1$ tables
  - dbms usually execute the "left-deep trees" because it allows pipelining and index nested loop joins.

*optimizations*

- replacing $\times$ and $\sigma$ with $\bowtie$
    - i replaced the selection `WHERE rep.owner = u.id AND rel.repo = rep.id` after the cartesian product with 2 joins.
    - this will eliminate rows without a foreign key.
- ordering joins
    - see: https://www.postgresql.org/docs/current/explicit-joins.html
    - `rep.owner` has 20,000 distinct values while `rel.repo` has 50,000 distinct values. therefore we have to join `repository` and `release` first.
    - this will eliminate even more rows without a foreign key.
    - but modern database query optimizers are so sophisticated that they often analyze table statistics and indexes to determine the best join order regardless of the way you write the query.
- applying $\sigma, \pi$ as early as possible
    - we filter as many rows as possible before joining with the third table.
- applying the stronger filters first
    - i reversed the order of predicates `(rel.name = 'v2' OR rel.version = 2)` so an integer based short circuit can happen before potential string comparisions.
    - i filtered by `rep.commits` before filtering by `rep.contributors` because `.commits` have more distinct values to filter and also the predicate `> 105` is already in the last equi-depth histogram bucket, while the predicate `>11` isn’t.

# execise 3, 4: query planning and optimization

the last 2 exercises turned into their own independent github project because i wanted to use jupyter notebook and write some code for better benchmarking.

see: https://github.com/sueszli/query-queen
