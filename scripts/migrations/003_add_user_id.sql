ALTER TABLE rec_ui_feedback_events ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_rec_ui_feedback_events_user_id ON rec_ui_feedback_events(user_id);
CREATE INDEX IF NOT EXISTS idx_rec_request_events_user_id ON rec_request_events(user_id);
