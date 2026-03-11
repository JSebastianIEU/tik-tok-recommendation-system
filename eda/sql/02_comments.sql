-- Template query for comment-level EDA (1 row per comment)
SELECT *
FROM comments
ORDER BY comment_id DESC
LIMIT 1000;
