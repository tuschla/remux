-- The user picked the remembered `audio_idx` over the track PlaybackInfo had
-- `stream_id` play, so other releases of the item follow its language. A track
-- that merely played by default is remembered for its own stream only.
ALTER TABLE user_media_state ADD COLUMN audio_chosen INTEGER NOT NULL DEFAULT 0;
