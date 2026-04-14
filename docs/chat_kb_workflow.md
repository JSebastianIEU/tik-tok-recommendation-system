# Chat Knowledge Base Workflow (Manual v1)

This project uses a curated TikTok knowledge base for global guidance in chat coaching.

## Source of Truth

- File: `frontend/server/knowledgeBase/data/tiktok_kb.v1.json`
- Scope: global guidance (algorithm, hooks, retention, CTA, creators, hashtags, content formats)
- Refresh mode: manual only

## Entry Contract

Each entry must include:

- `id`
- `title`
- `category` (`algorithm|hooks|retention|cta|creators|hashtags|content_formats`)
- `content` (1-3 concise facts)
- `action_hint`
- `impact_area`
- `keywords`
- `updated_at` (ISO timestamp)
- `confidence` (`high|medium`)
- `active` (boolean)

Optional:

- `objective_tags` (`reach|engagement|conversion|community`)

## Edit / Review / Publish

1. Update `tiktok_kb.v1.json`.
2. Validate locally:
   - `cd frontend && npm run kb:validate`
3. Run modeling checks:
   - `cd frontend && npm run test:modeling`
4. Commit the KB update with a short note about what changed.

## Runtime Behavior

- Server startup performs strict KB validation.
- If validation fails, server startup fails closed with a clear error.
- Chat blends KB insights into normal coaching responses.
- KB is not shown as a separate section unless user explicitly asks for details/examples.
