# Show how a PostgreSQL index changes a query plan


A small companion to the Advanced Database Systems notes (184.780). The original project explored how to read PostgreSQL execution plans and optimize queries. This example isolates one of those ideas on generated data.

Run on a scratch PostgreSQL database:

    psql -X -v ON_ERROR_STOP=1 -f explain.sql

The script shows the plan for a selective lookup before and after creating an index. It checks that both lookups return the same eight values. All writes use temporary objects in a transaction that rolls back. Compare plan nodes and buffer use, not absolute timings. Planner decisions depend on the server and its configuration.

The original database dump, notebooks, and review data are not included.
