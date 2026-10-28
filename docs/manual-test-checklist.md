# Manual test checklist

Run through this after bigger changes (about 10 minutes). Start with `streamlit run app.py`.

## Start-up
- [ ] The page loads without a red error box; the sidebar shows the demo video.
- [ ] With a weight file temporarily renamed, the app shows a clear "Model weights not found"
      message instead of a traceback (rename it back afterwards).

## Sidebar
- [ ] Switching Demo 1 / Demo 2 changes the video, team names and the four kit colours.
- [ ] Uploading your own video replaces the demo; "Input video" is shown.
- [ ] Emptying a team name disables "Start Detection".

## Team Colors tab
- [ ] The frame slider changes the frame on the right and the player thumbnails on the left.
- [ ] Moving the slider to the very last frame does not crash.
- [ ] Choose "P color" for a team, click a player of that team: the matching colour box updates.
- [ ] Editing a colour box by hand keeps the edit (it is not overwritten by the earlier click).
- [ ] On a frame without players, an info message appears instead of an error.

## Detection tab
- [ ] Start Detection with defaults: progress bar moves, annotated frames and map appear.
- [ ] Players have boxes labelled with the right team name; the dots on the map use team colours.
- [ ] The ball is drawn on the map and leaves a short track.
- [ ] Toggling each annotation option (keypoints, players, palettes, ball track) changes the output.
- [ ] "Process every Nth frame" = 3 is clearly faster and still looks right.
- [ ] Stop Detection ends the run without an error.
- [ ] "Save output" with a name writes `outputs/<name>.mp4`; running again does not overwrite it.
- [ ] The saved video plays and shows the frame next to the tactical map.

## Edge cases
- [ ] A very short clip (a few frames) runs to the end.
- [ ] A video with no visible pitch lines does not crash (the map stays empty).
