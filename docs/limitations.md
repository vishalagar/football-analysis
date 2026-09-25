# Limitations

What the numbers can and can't tell you, and why. Most of this improves
with a football-specific model (see [model.md](model.md)).

## The ball

- **The stock model often misses it.** COCO's ball class was learned from
  close-ups, and a broadcast ball can be 5–10 px wide. The zoomed ball-focus
  pass and gap filling help, but only when the ball was seen recently.
- **Balls in the air are guessed.** The tracker works in image space, so a
  lofted pass looks like a ball flying over players' heads. Possession only
  changes when the ball comes close to someone's feet, which usually happens
  on landing.
- **Look-alikes.** A white boot, a head, or a line marking can score as a
  ball. The jump filter rejects most of them, but not all.

## Players and teams

- **Crowded boxes merge.** At a corner, overlapping players can become one
  box, and ids may swap when they separate. This affects the pass network
  more than the totals.
- **Ids are not squad numbers.** A player who leaves the frame and comes
  back gets a new id. Shirt numbers aren't read.
- **Similar kits confuse the clustering.** Two dark kits, or white against
  light grey, can merge. Goalkeepers usually get flagged as outliers because
  their kit differs, so they drop out of team stats. Referees in a colour
  close to one team's can be counted as that team.
- **Lighting.** Half-shade stadiums split one kit into two tones. The
  clustering copes with shading of the same colour but not with extreme
  contrast.

## Possession and passes

- **"Closest feet to the ball" is a proxy for control.** It works well in
  open play and less well in tight challenges, where two players are equally
  close.
- **A pass is inferred, not seen.** Possession moving between team-mates
  within 6 s counts as a pass, so a deflection that lands with a team-mate
  also counts. Passes the tracker never sees start (because the ball wasn't
  detected) are missed.
- **Turnovers include everything.** Tackles, interceptions and loose balls
  all count as "ball won back".
- **Frame rate matters.** At fewer than about 5 analysed frames per second,
  quick one-touch exchanges fall between samples. Use the Speed control on
  slow machines.

## Camera and footage

- **Cuts reset tracking** by design, so replays and close-ups don't produce
  nonsense movement. The side effect is that ids restart after every cut.
- **Replays count twice.** A replay played after a goal is analysed like
  live play. Trim replays out if the totals matter.
- **Pitch mapping holds for one camera angle.** Broadcast cameras pan and
  zoom, and positions drift as they do. Re-map after big moves. (Automatic
  pitch-keypoint detection is on the list below.)
- **Close-ups and tactical cams.** The model expects a wide or mid shot.
  Close-ups make players huge and the ball rare.

## Distance covered

Without mapping, distance uses player height as a ruler. That works on
average, but it is biased by players partly off-screen or bent over. After
mapping, the numbers are in real metres. They only cover the time a player
was tracked, which is far less than a full match on broadcast footage.

## Performance

- WebAssembly on an older laptop may manage only 2–4 analysed frames per
  second. Everything still works, but event detection gets coarse.
- The first load is about 40 MB (model plus runtime). After that it's
  cached.

## Streams

DRM-protected services cannot be analysed (see [web-app.md](web-app.md)).
That is a platform restriction, not a bug.

## Possible next steps

1. Fine-tuned football model with ball, goalkeeper and referee classes (the
   training script exists and needs a dataset and a GPU).
2. Pitch-keypoint model for automatic calibration that follows the camera.
3. Shirt-number reading for player identity across cuts.
4. Replay detection (broadcast replays have a logo wipe) so they can be
   skipped.
5. Webcam gesture control: swipe to seek, pinch to zoom, point to draw on the
   pitch.
