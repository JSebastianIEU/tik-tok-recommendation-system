-- Template query for author-level EDA (1 row per author)
SELECT *
FROM authors
ORDER BY author_id DESC
LIMIT 1000;
