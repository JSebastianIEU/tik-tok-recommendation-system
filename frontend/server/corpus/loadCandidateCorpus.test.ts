import assert from "node:assert/strict";
import test from "node:test";

import { buildArtifactBundleRecords } from "./loadCandidateCorpus";

test("buildArtifactBundleRecords maps Supabase bundle rows into report-ready records", () => {
  const records = buildArtifactBundleRecords({
    authors: [
      {
        author_id: "author-1",
        username: "creator.one",
        followers_count: 1200
      }
    ],
    videos: [
      {
        video_id: "video-1",
        author_id: "author-1",
        caption: "How I market before launch",
        hashtags: ["#marketing", "#launch"],
        keywords: ["marketing", "launch"],
        search_query: "launch strategy",
        posted_at: "2026-04-10T10:00:00Z",
        video_url: "https://www.tiktok.com/@creator.one/video/1",
        duration_seconds: 42,
        language: "en"
      },
      {
        video_id: "video-2",
        author_id: "author-1",
        caption: "Older video",
        hashtags: [],
        keywords: [],
        posted_at: "2026-04-08T10:00:00Z",
        video_url: "https://www.tiktok.com/@creator.one/video/2",
        duration_seconds: 18,
        language: "en"
      }
    ],
    video_snapshots: [
      {
        video_id: "video-1",
        event_time: "2026-04-10T11:00:00Z",
        views: 120,
        likes: 12,
        comments_count: 3,
        shares: 4
      },
      {
        video_id: "video-1",
        event_time: "2026-04-10T12:00:00Z",
        views: 240,
        likes: 30,
        comments_count: 5,
        shares: 7
      }
    ],
    comments: [
      {
        video_id: "video-1",
        text: "Where do you find users first?",
        created_at: "2026-04-10T12:01:00Z"
      },
      {
        video_id: "video-1",
        text: "This is helpful.",
        created_at: "2026-04-10T12:03:00Z"
      }
    ]
  });

  assert.equal(records.length, 2);
  assert.equal(records[0]?.video_id, "video-1");
  assert.deepEqual(records[0]?.hashtags, ["#marketing", "#launch"]);
  assert.deepEqual(records[0]?.comments, ["This is helpful.", "Where do you find users first?"]);
  assert.equal(records[0]?.metrics.views, 240);
  assert.equal(records[0]?.metrics.likes, 30);
  assert.equal(records[0]?.author.author_id, "author-1");
  assert.equal(
    typeof records[0]?.author === "object" && records[0]?.author
      ? records[0].author.username
      : "",
    "creator.one"
  );
  assert.equal(records[1]?.metrics.views, 0);
});
