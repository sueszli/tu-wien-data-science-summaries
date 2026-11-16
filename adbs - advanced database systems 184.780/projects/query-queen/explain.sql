BEGIN;

CREATE TEMP TABLE demo_events AS
SELECT i AS event_id, i % 10000 AS account_id
FROM generate_series(1, 80000) AS g(i);

ANALYZE demo_events;

EXPLAIN (ANALYZE, BUFFERS)
SELECT event_id FROM demo_events WHERE account_id = 42 ORDER BY event_id;

CREATE TEMP TABLE demo_before AS
SELECT event_id FROM demo_events WHERE account_id = 42;

CREATE INDEX demo_events_account_id_idx ON demo_events (account_id);
ANALYZE demo_events;

EXPLAIN (ANALYZE, BUFFERS)
SELECT event_id FROM demo_events WHERE account_id = 42 ORDER BY event_id;

DO $$
BEGIN
    IF (SELECT array_agg(event_id ORDER BY event_id) FROM demo_before)
       IS DISTINCT FROM ARRAY[42,10042,20042,30042,40042,50042,60042,70042]
       OR (SELECT array_agg(event_id ORDER BY event_id)
           FROM demo_events WHERE account_id = 42)
       IS DISTINCT FROM (SELECT array_agg(event_id ORDER BY event_id) FROM demo_before)
    THEN
        RAISE EXCEPTION 'lookup returned unexpected rows';
    END IF;
END;
$$;

ROLLBACK;
