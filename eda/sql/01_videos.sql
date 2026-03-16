-- Template query for video-level EDA (1 row per video)
SELECT *
FROM videos
ORDER BY created_at DESC NULLS LAST, video_id DESC
LIMIT 1000;
