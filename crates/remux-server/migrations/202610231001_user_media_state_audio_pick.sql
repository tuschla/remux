-- `stream_id` names the stream a remembered `audio_idx` points into only while
-- both were written together: older builds update `audio_idx` alone, so the
-- index the stream was recorded with is kept next to it.
ALTER TABLE user_media_state ADD COLUMN stream_audio_idx INTEGER;
-- The user picked that track over the one PlaybackInfo had the stream play,
-- so other releases of the item follow its language. A track that merely
-- played by default is remembered for its own stream only.
ALTER TABLE user_media_state ADD COLUMN audio_chosen INTEGER NOT NULL DEFAULT 0;
