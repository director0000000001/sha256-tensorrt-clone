# sha256 — Roadmap to a deliverable neural version

**Written 2026-09-04, after the day-22 audit and the LivePortrait re-measurement.**
Owner's brief: highest quality, as fast as possible, the app will be used by a
team of five people other than the owner.

## What "highest level" is made of

| part | today | target | how |
|---|---|---|---|
| the picture | photographs chosen and cut 12-13 times a second; reads as a game character | a face generated every frame, no cuts | LivePortrait engine. Measured 2026-09-04 on the RTX 5070 Ti: **25.5 ms a frame** with fp16 weights and one CUDA graph (B005's 38.4 was autocast without graphs). Output identical to eager to 0.0000 |
| reaction | ~260 ms glass to glass, 227 of it before the engine starts (ZV-E10 over USB) | under 150 ms | phone-camera test first (free); if the camera is the cause, an HDMI capture card at 1080p60. Shutter 1/125-1/250 (free): the fastest mouth frames are 2.3x blurrier than still ones. MediaPipe's own VIDEO-mode lag disappears because LivePortrait reads motion itself |
| stability | fixed 2026-09-04: preview blackouts, the capture button on a tracking loss, load 47 s → 18 s, the "face not found" line | the same on five machines | GPU and camera check at startup with a plain message; a replay that starts from the engine's live state |
| delivery | nothing started: no shippable virtual camera (pyvirtualcam is GPL), no licence server, no kill switch, no site, no encryption, no `.exe` | installable by a non-programmer on five machines | see phase 3 and 4 |

## The three facts that decide the whole plan

1. **The neural engine needs an NVIDIA GPU on every user's machine.** 25.5 ms is a
   5070 Ti. A 3060-class card is roughly 2 to 2.5x slower, which is 15 fps. A
   laptop without a discrete NVIDIA GPU cannot run it at all; the photograph
   engine (CPU) is the fallback there. **The users' GPUs must be known before
   phase 2 ends.**
2. **Each user needs one good portrait, not a library.** LivePortrait animates a
   single photograph. The 250-photograph ladder, the clip generation and the
   builders are the photograph engine's need only. Five users means five
   portraits, generated the way the owner's was, plus one enrolment step each.
3. **Delivery is a separate project of about a week** and it cannot start before
   the engine is chosen, because the package now has to carry torch and the CUDA
   runtime (about 2.5 GB).

## Phases

Days are working days for one engineer at full time, starting 2026-09-05.

### Phase 0 — decisions, day 0 (the owner, today)

- Which GPU and which camera do each of the five users have? Model names.
- One shared avatar or one per user?
- The deadline conversation with the client, with the numbers below in hand.
- Phone-camera latency test (`scripts/measure_latency.py` with DroidCam or
  Phone Link), and the shutter set to 1/125 for a new `--script speech` recording.

### Phase 1 -- the neural engine, offline, days 1-5

**Day 1, done.** `LivePortraitEngine` behind the existing `AvatarEngine`
protocol (`src/sha256/avatar/engine_liveportrait.py`), registered as backend
`liveportrait`. Portrait prepared once at load (appearance feature, source
keypoints, crop). Per frame: MediaPipe's full landmarker for the crop (the
detector-only cut is still open, see Phase 2), motion -> warp -> decode in one
CUDA graph, paste back with LivePortrait's own crop transform. The four
CPU-side allocations inside LivePortrait's forward were moved to the GPU for
graph capture, vendored under `src/sha256/avatar/liveportrait/` (MIT,
provenance in that package's own `__init__.py`).

First render (`scripts/render_liveportrait.py`, frames 300-700 of
`driving-speech.mkv`): the picture was recognisably the avatar, correctly
placed and sized -- and pasted under LivePortrait's own template mask, which
covers the whole 512 crop square. On screen that is the portrait's studio
background and shoulders laid over the room, edged in black outside the
square. Not a defect in the network; a mask built for a different framing
convention.

**Day 2, done, three fixes:**

1. **The paste mask.** `_oval_mask` builds the portrait's own face oval from
   its landmarks placed in the crop -- same technique as the warp engine's own
   composite (`engine_warp._trim_forehead`/`_destination_mask`), same
   fractions (`_OVAL_SHRINK`, `_OVAL_FEATHER`, `_OVAL_FOREHEAD_REACH`),
   written again rather than imported so the two engines keep sharing no
   internals. Computed once per portrait at load, not per frame -- see the
   function's own docstring for why that is sound rather than a shortcut.
2. **Colour match.** `_match_face_colour` / `_fit_lighting_plane`: the same
   low-frequency relighting plane the warp engine fits between its portrait
   and the room, fit here between the generated face and the scene under the
   mask, with the same frame-to-frame inertia (`_CORRECTION_INERTIA = 0.8`).
3. **The paste, bounded to the mask's own box.** `_mask_bbox` finds where the
   oval actually is inside the 512 square -- about 13% of its area on this
   avatar -- and every frame warps only into that box, not the whole square.

   Measured 2026-09-04, this machine, `driving-speech.mkv` frames 300-700,
   before -> after:

   | | before (whole square) | after (oval's own box + colour match) |
   |---|---|---|
   | paste, p50 | 22.32 ms | **7.57 ms** |
   | paste, p90 | 22.98 ms | **8.10 ms** |
   | neural total, p50 | 50.79 ms | **34.85 ms** |
   | model (unchanged) | 27.69 ms | 26.74 ms |

   16 tests in `tests/unit/test_engine_liveportrait.py` (was 11): the mask's
   box is asserted smaller than the crop, feathered not binary; `_paste` is
   tested on the CPU alone (no GPU, no loaded networks) for byte-identical
   pixels outside its box, correct colour-matching on a synthetic scene, and
   inertia carrying between frames. The GPU test (portrait driving itself,
   graph against eager) still passes. mypy: one pre-existing-class error
   (a `cv2.subtract` stub gap `engine_warp.py` already carries at its own
   matching line), everything else clean; the vendored network code is
   excluded from mypy by its own override in `pyproject.toml`, the six calls
   into it from `engine_liveportrait.py` each carry their own
   `# type: ignore[no-untyped-call]`.

   **Looked at again, frames 120 and 300**: the oval is right and the room
   shows through around it now, which is the fix. There is still a visible
   seam at the mask's own edge -- most noticeable at the hairline, a band
   where the portrait's forehead tone meets the room's rather than fading
   into it -- and the mask stops at the jaw with nothing below it, so a
   turned or lowered head will show a hard edge above the neck. Neither is
   surprising: the warp engine's own `_MASK_FEATHER`/`_JAW_REACH` were tuned
   against the owner's eye over several days, not derived once and correct.
   This is the next thing to look at, not yet fixed.

**Still open, not blocking the owner's first look:**

- **The budget.** `tracker+neural` is 42.35 ms p50 against the 33.2 ms camera
  interval, 100% of frames over -- down from 63.34 ms before today, but still
  over. The full landmarker (7.4 ms today) is doing more than the crop needs;
  a face detector alone is the next lever, in Phase 2.
- **The seam and the jaw**, above -- Day 3 intersected the portrait's oval
  with the driver's, which measurably helps the hairline in aggregate but not
  on the left-turn frames specifically; see "Day 3, done" below for the
  number and the still that shows it did not close.
- Load time: 13.1 s this run, 21.8 s the run before -- this machine's mood
  (see PROJECT_STATUS.md trap 5), not yet measured as a back-to-back pair.

**Day 3, done, one fix, and it is only half of one:**

1. **The mask follows the driver, not only the portrait.** `_frame_mask`
   intersects the portrait's own oval (computed once at load, as before) with
   a second oval built the same way (`_oval_mask`, now used for two different
   faces) from the *driver's* landmarks, every frame. Neither oval alone was
   enough: the portrait's, warped onto a driver with different
   forehead-to-hairline or jaw-to-chin proportions, could reach into the
   driver's own hair or stop short of the driver's own chin -- exactly the
   hairline band and the floating chin the owner's Day 2 pictures showed. The
   intersection can only be smaller than each half, so it can never show
   either. Same rule the warp engine already follows in
   `engine_warp._destination_mask` ("the join happens over skin on both
   sides"), written again rather than shared -- the two engines still share no
   internals, only the registry contract.

   `animate` now builds this mask every frame (`mask_ms`, new) from the
   crop-space points already computed for the model; `_Portrait.last` carries
   it alongside the face and the transform so a held frame (tracking lost)
   keeps the mask that belongs with it rather than the portrait's own.
   `_paste` takes the mask as a parameter instead of reading `loaded.mask`
   directly. 8 new tests (`TestFrameMask`, `TestSharedLoadTracker`, tracker
   `TestRunningMode`) plus the 4 existing `TestPaste` tests updated for the
   new parameter -- 726 total (was 716). mypy: still the one pre-existing
   `cv2.subtract` error, now also present at its own line in this file (same
   stub gap, same shape, not new). ruff: the vendored `liveportrait/` package
   is now excluded from linting the same way it already was from mypy --
   `extend-exclude` in `pyproject.toml` -- since none of its 346 findings were
   ever this project's to fix.

   **Measured**, this machine, before -> after, mask-edge Sobel gradient
   (unitless, only comparable before to after -- see
   `scripts/render_liveportrait.py`'s own docstring):

   | | `driving-speech.mkv` 300-700 | `driving.mkv` 1300-1650 (both turns) |
   |---|---|---|
   | hairline, p50 | 48.97 -> **39.62** | 64.57 -> **50.72** |
   | jaw, p50 | 46.25 -> 50.21 | 50.40 -> 49.95 |
   | whole band, p50 | 48.47 -> 46.89 | 58.80 -> 51.46 |
   | mask cost, p50 | -- | 1.23-1.26 ms |
   | tracker+neural, p50 | 42.93 -> 44.74 | 40.90 -> 42.36 |

   The hairline number improves on both clips, clearly. The jaw number does
   not move -- expected: no jaw extension exists on either oval to disagree
   about (the owner rejected one on the warp engine, 2026-08-15, and none was
   added here), so the jaw edge is set by the portrait's own oval either way.
   The cost is real: +1.2-1.3 ms of mask work a frame, and `tracker+neural`
   rose 1.5-1.8 ms -- on a budget already 100% over at 33.2 ms, so it changes
   nothing about Phase 2's plan to close that gap, only the size of it.

   **Looked at, eight stills (four per clip, `recordings/day23-mask-before-
   after*.png`), and the picture is not as clean as the number.** On four --
   both near-frontal speech frames and the right-turn frames -- the fix is
   invisible because there was nothing visibly wrong to begin with. On the
   two left-turn frames (`driving.mkv` 1380, 1420) the hairline band the
   owner's Day 2 pictures showed is **still there, unchanged in appearance,
   before and after.** The aggregate p50 improved because most frames in that
   window were already fine; it does not mean this specific defect closed.
   **Working hypothesis, not yet checked:** at this yaw the portrait's and
   the driver's ovals may be wrong in the same direction rather than
   different directions -- both ceilings computed from a 2D `brow_top`/
   `oval_top` reading that a turned head foreshortens -- in which case the
   intersection of two wrong masks is still wrong, and the real fix is in how
   the ceiling is computed under yaw, not in whose oval is used. The
   right-turn frames not showing it argues against a simple yaw-magnitude
   explanation and for something sign- or pose-specific; not run down yet.

**Day 4, done -- the owner ran it live for the first time, and named two
things: "the AI face is slightly to the side of my face" and "reactions
don't sit on the face."** The left-turn hairline investigation this section
used to end on is still open -- his own report took priority, and finding it
turned up a mechanism the hairline question never got to.

**The tool: `scripts/measure_follow.py`.** Three trackers on one clip -- the
driver frame through the app's own filter (what actually drives the engine),
the driver frame unfiltered (the truth of where his face and expression are),
and the engine's own output tracked the same way, so the avatar's eyes/nose/
mouth can be read and compared to his on the same frame. Position is reported
as avatar-minus-him in pixels; expression following as a gain (1.0 = the
avatar's own tracked reading moves exactly as far as his did) and a
correlation. Numbers only compare before a change to after, same clip.

**"To the side" was stitching's own 2-D shift, and nothing else.** Four
candidates were measured on a fresh 62-second clip (`recordings/driving.mkv`,
re-recorded 2026-09-04 for this): pinning the driver's translation onto the
portrait, pinning scale, removing the mean of the expression delta, and
dropping stitching's shift. The first three changed nothing. The shift was the
whole effect:

| | with the shift | without |
|---|---|---|
| nose, p10..p90 | -14.8 .. +15.2 px | **-6.1 .. +4.0 px** |
| "Close your eyes", nose dx | -13.4 px | -1.0 px |
| "TALK", nose dx | +10.2 px | -1.7 px |
| correlated with | avatar's apparent size, r=0.88 | -- |
| 40° turn (frame 1590) | a grey wedge above the jaw | gone |

Stitching was trained to paste a re-animated face back onto its *still*
source photograph without a seam at the neck; here the driver's own crop
already re-centres the face every frame, so the shift fights the crop instead
of correcting a paste. `stitching` defaults to `false` now (the correction
half of the network still runs when explicitly enabled -- it is the shift,
not the whole module, that was the problem, but nobody has needed the two
apart). Cost: nothing, this is *less* work per frame, not more.

**"Reactions don't sit on the face" was real, and partly fixed.** The
driver's relative expression alone reproduced only 36% of a blink and 73% of
a mouth opening, against his own tracked reading. LivePortrait ships eye and
lip retargeting networks in the same checkpoint stitching's is in
(`retarget_eye`, `retarget_mouth`) and neither was loaded before today.
Wired in, on his own close ratio -- computed on MediaPipe's points, since
that is what this engine tracks with, not LivePortrait's own landmark scheme
the networks were trained against. That mismatch turned out to matter:

| | eye | lip |
|---|---|---|
| gain, off | 0.42 | 0.75 |
| gain, calibrated | **0.67** (full clip) / 0.70 (isolated) | **0.99** |
| r | 0.79 -> 0.92 | 0.99 (unchanged) |
| calibrated constant | `eye_gain = 0.06` | `lip_gain = 0.20` |

Lip reaches the target cleanly: gain 1.0 at `lip_gain = 0.2`, correlation
0.99 throughout the sweep, a clean monotonic response (0, 0.05, ... 0.3 ->
0.75, 0.81, ... 1.10). **Eye does not, and this is a real finding, not an
under-tuned one.** Its response rises smoothly from gain 0 to 0.06-0.07 and
then plateaus -- 0.08 gave *less* than 0.07, not more -- while a first probe
at gain 1.0 (both networks' original guess) was actively worse than off,
overshooting into territory that read as *more* shut regardless of the
driving target. Both networks take the same shape of input (the portrait's
own kp, plus close ratios) at gain 1.0 in LivePortrait's reference pipeline;
the plausible read is that a MediaPipe-based ratio and a LivePortrait-based
one agree well enough in shape for the lip network's training to tolerate,
and not well enough for the eye network's -- not measured further, since the
gain sweep already shows the ceiling rather than an unfinished tuning curve.
**Blinking on the app will be visibly closer, not identical.**

Position held through the calibration: nose dx still -5.8..+4.0 px (p10..p90)
with both networks on. 728 -> **737 tests**. mypy: the same one pre-existing
error, at its own line, none new. ruff: the vendored `liveportrait/` package
is now excluded the same way it already was from mypy (`extend-exclude` in
`pyproject.toml` -- 346 findings there were never this project's to fix).

**What was not run down:** the left-turn hairline band this section used to
be about. **At |yaw| > 15° the avatar's whole face lags the real turn by
18-20 px, both directions** -- measured today, not stitching, not yet
explained. **The "That looked wrong" capture button cannot replay this
engine.** It stores landmarks and one rendered frame by design (privacy: "not
a photograph of anyone" beyond that one frame) -- exactly right for the
photograph engine, which only ever needed landmarks to redrive itself. This
engine's motion extractor reads raw pixels from the driver crop, not
landmarks, so a captured moment with no saved frames for it cannot be
reconstructed offline. Fixing that means deciding to store more of the
owner's face on disk, which is his call, not a default to change quietly.

**A footgun closed:** `render_liveportrait.py --clip` had no default that
matched the clip most days actually needed -- it silently pointed at the
speech clip and cost one wasted run today. `--clip` is required now.

**Files from today under `recordings/` and `captures/`, all his face, all
gitignored, none looked at yet:** `driving.mkv` (the new 62 s clip) and its
`driving-follow*.csv`; `day24-overlay-f*.png` (8 stills, before any of
today's fixes) and `day25-nostitch-f1590.png` (after -- the wedge, gone);
`captures/moment-20260904-233810/` (the "That looked wrong" press that
started today, plus `overlay.png` built to check it).

**Day 5, done -- his six live complaints, each run down to one mechanism and
fixed: the anchor, the crop, and the eye/lip targets.**

**1. "Face to the side, not symmetric" — the anchor.** It learned the
driver's rest from the first 15 frames after load, whatever they held. His
own capture (`captures/moment-20260905-010827`) showed why that is not
enough: nose +27 px, eye +18 px off his, an asymmetry of +9.5 -- reproduced
exactly offline by anchoring on frames turned -12°: nose +29, eye +17, asym
+12, plus the avatar's mouth sitting ajar and its eyes 26% wider than rest.
`_anchor_frame_ok` now gates which frames the anchor is allowed to learn
from (frontal within 4°/4°/3°, eyes open, mouth shut); a **"Re-centre"
button** calls the already-existing `reset_anchor` (was wired to nothing);
the status line says when the anchor is still being learned.

**2. "Zooms in and out, enlarges the eyes" — the crop.** It sized itself from
all 468 mesh points, and an open mouth or a smile moves its own corners
outward, growing the box the network reads its own size from. Avatar size
against his own varied **1.01..1.25** over the clip, worst exactly at the
widest mouth opening. `crop_matrix` now sizes and centres from
`_RIGID_POINTS` -- points expression cannot move (eye corners, face sides,
forehead, nose) -- times a fixed `_RIGID_EXTENT_FACTOR` so it still matches
LivePortrait's own framing convention. Size variance fell to **1.00..1.08**,
mouth-wide scale from 1.25 to 1.03. **Not free:** at |yaw| > 15° the nose
offset got *worse* (19 px → 34 px) and the eyebrow-raise correlation fell
0.95 → 0.30 -- reported, not chased further this session.

**3 and 4. "Half-shut drunk eyes" and "thin, angry lips" — the retargeting
targets.** Both networks were driven by his *raw* close ratio. His eyes read
narrower than the portrait's own at rest (0.30 against 0.334), so a raw
target asked the network to narrow the avatar for the mere fact of him
sitting still -- measured as the avatar's eyes 27% narrower than his over a
whole clip. His raw lip ratio, at rest, was *tighter* than the portrait's own
already-closed lips, and the network had nowhere to put that demand but the
lips themselves -- thickness fell to 38-44% of his own. `_eye_target` now
asks relative to his own rest (his rest → the portrait's own ratio, capped
1.2x above it); `_lip_target` is floored at the portrait's own closed ratio,
never tighter.

**Before -> after, measured on `driving.mkv` (1844 frames), same clip
throughout:**

| | before today (day 4's end) | after A (anchor) | after B (crop) | after C (targets) |
|---|---|---|---|---|
| nose dx, p10..p90 (px) | -5.8..+4.0 | -5.8..+4.0 | -4.3..+5.7 | -4.5..+6.0 |
| scale, p10..p90 | 1.011..1.159 | 1.011..1.160 | 1.002..1.075 | 1.002..1.077 |
| mouth-wide scale, p90 | 1.254 | -- | 1.027 | 1.03 |
| ear gain / r | 0.67 / 0.92 | 0.67 / 0.92 | 0.61 / 0.79 | 0.62 / 0.80 |
| mar gain / r | 0.99 / 0.99 | 0.99 / 0.99 | 1.09 / 0.99 | 0.94 / 0.99 |
| lip thickness (upper/lower, avatar/his) | 0.38 / 0.44 (live report) | -- | -- | 0.60-0.78 / 0.73-0.78 |
| \|yaw\|>15°, nose dx | 11.3 | -- | -34.0 | -34.1 |

**The lip finding is worth stating plainly: retargeting turned out not to
earn its place.** Re-swept with the floor in place and Task B's crop
already fixed: the mouth-following gain from relative motion *alone*, no
lip network at all, had already reached 0.94 (r 0.98) -- Task B's crop
stability turns out to have fixed most of the mouth-tracking gap too, not
only the zoom. Every `lip_gain` above 0 bought a little more mar gain at a
real, measured thickness cost (0.20 → 1.09 gain but only 68/70% thickness).
`lip_retargeting` is off by default now; the option and `_lip_target` stay
wired in for an avatar or a driver that needs it more than this one does.
Eye retargeting keeps earning its keep (`eye_gain 0.06`, ear gain 0.36 → 0.62
off a baseline that never had this problem to begin with) and stays on.

**Lip thickness itself is not fully explained.** It recovers well at the
very start of a clip (0.77/0.78, clearing the informal 0.75 bar) and sags
later in it (0.60/0.73 by "SIT STILL again") even with the lip network fully
off -- something else, not yet identified, thins the mouth over time. Not
run down this session.

737 -> **767 tests**. mypy: the same one pre-existing error. ruff clean.

**What was not run down.** The **forehead mask line** from Day 3 -- visible
again in `captures/moment-20260905-010827/seen.png`, frontal and close --
his live report moved to the position/zoom/target problems instead and it
was never revisited. The **|yaw| > 15° lag**, now 34 px rather than 19,
made worse by Task B rather than fixed. The **capture button's own limit**:
it stores landmarks and one rendered frame, enough for the photograph engine
to replay itself but not this one, whose motion extractor reads raw pixels
it never saved. Fixing that means storing more of the owner's face on disk
by design, which is his call.

**Files from today under `recordings/` and `captures/`, all his face, all
gitignored, none looked at yet:** `driving-follow-taskA.csv`,
`-taskB.csv`, `-taskC-final.csv` (one per task, this measurement run each
built on the last); `day25-final.mp4` and its five stills (the finished
state, for his own side-by-side); `captures/moment-20260905-010827/` (the
capture that started today's list) and its `overlay.png`.

**Day 6, done (2026-09-05, Fable 5.1 implementing directly on the owner's
instruction) -- his third live verdict, and the crop rebuilt around what
actually stays still on a face.**

His four notes after the day-5 build: (1) "my jaw shows, the face got
shorter"; (2) "random reactions -- raises the brows, pulls the eyes wide";
(3) "teeth on a smile are terrible"; (4) "the reactions have to match mine,
that is a hard prerequisite".

**(1) and (2) were the same mistake: the day-5 rigid crop.** Its box was
sized from the rigid points' *larger side* -- cheek to cheek on any face --
times one constant taken from the portrait. Two things followed. His face is
proportionally longer than the portrait's (forehead-to-chin over
forehead-to-nose: 1.91 against 1.815), so the constant under-sized his crop
by 5% and the avatar's chin landed 14-20 px above his own -- the jaw he saw.
And a cheek-to-cheek width shrinks with cos(yaw), so every turn zoomed the
crop in: the turn lag went 19 → 34 px on day 5, and the brow-raise
correlation fell 0.95 → 0.30, because the motion extractor reads its own
framing, not ours -- it wants the driver at the driver's *own* whole-face
framing (day 4's), and every experiment below that gave it the portrait's
framing instead killed the brows (r 0.27-0.30) no matter what else changed.

**Three tries, measured on `driving.mkv` each time, before the one that
holds:**

| | day 4 | day 5 | A: his ratio, running mean | B: "own" width-gov. | C: "paste" | D: "own" height-gov. | E: "paste" height-gov. |
|---|---|---|---|---|---|---|---|
| still: chin dy (px) | -2.6 | -14.4 | -15.5 | -11.9 | -4.7 | -4.4 | -7.1 |
| still: scale | 1.083 | 1.020 | 1.144 | 1.018 | 1.064 | 1.063 | 1.059 |
| still: avatar EAR (his 0.293) | 0.318 | 0.286 | **0.419** | 0.286 | 0.286 | 0.281 | 0.299 |
| brows gain / r | 1.14 / 0.95 | 0.23 / 0.30 | 0.75 / 0.75 | 0.35 / 0.84 | 0.24 / 0.27 | 0.88 / 0.90 | -0.36 / -0.50 |
| mar gain / r | 1.05 / 0.99 | 0.97 / 0.99 | 0.90 / 0.98 | **0.17 / 0.29** | 0.92 / 0.99 | **0.29 / 0.50** | 0.62 / 0.86 |
| eyes shut: avatar EAR | 0.233 | 0.290 | 0.337 | 0.112 | 0.323 | 0.215 | **0.411** |
| \|yaw\|>15°: nose dx | -20 | -34 | -37 | -49 | -30 | -49 | -32 |

- **A** learned his whole-over-rigid ratio as a *running mean during the
  anchor window and applied it while averaging* -- so the anchor's frames
  were cut at fifteen different sizes and every frame after at a sixteenth.
  The avatar sat 12% too big with its eyes 0.419 wide at rest and its brows
  up: exactly note (2), reproduced offline. Rule learned: **the anchor and
  the frames it serves must be cut at one framing.**
- **B/C** fixed that (median, applied once) but kept the width-governed
  box; C put the driver at the portrait's framing for the network and only
  *pasted* larger. C fixed the jaw and lost the brows; B had a paste bug and
  is not evidence of anything.
- **D/E** switched the box to the rigid points' *height* (forehead top to
  nose tip) to stop the yaw zoom -- and found the nose tip's own protrusion
  under pitch: -13.5% at +14° of pitch, which is exactly what his head does
  when his mouth opens wide. D got the brows back (0.90) and lost the mouth
  (0.29).

**What every 2-D measure of the face does under his own motion** (p10..p90
per segment, relative to its own rest value; the filtered tracker, as the
engine sees it):

| segment | pitch / yaw | rigid height | rigid width | whole height | 1 / pose distance |
|---|---|---|---|---|---|
| look up/down | pitch to +14° | -11.7% | -7.4% | -10.0% | -4.7% |
| mouth wide open | pitch to +13° | -13.5% | -5.5% | -6.3..**+6.7%** | -3.6..+0.2% |
| talk | pitch -5..+10° | -9.5..+4.0% | -4.2% | -2.5..+2.6% | -2.6..-0.6% |
| turn right | yaw -48..+31° | -14.1% | **-20.5%** | -10.3% | -4.3..-2.1% |
| smile, closed | -- | +1.2% | -2.8% | -3.4% | -2.6..-2.0% |
| still again | pitch -5° | **+8.2%** | -3.5% | +0.7% | -2.8..-2.2% |

The last column is MediaPipe's own head pose: the translation of a rigid
whole-face model fitted to the mesh, which neither an expression nor a turn
can move -- it reads only how near he is (the -2..-5% in the table is him
actually leaning back; every other column agrees he did). **That is the crop
size now:** `crop_matrix(face_height=...)` takes the face height in pixels;
the engine measures the driver's resting height (forehead top to chin, the
day-4 measure LivePortrait's framing was tuned against) and resting distance
over the anchor's accepted frames -- medians, because the tracker's first
frame reads a face 10% shorter -- and afterwards passes
`height_rest x z_rest / z` every frame (`_driver_face_height`). Before the
freeze the crop is sized from each frame itself, which at rest is the same
height, so the anchor's frames and the frames after are cut alike (the rule
above, kept for free). The rigid points only *centre* the box now. The
`extent_factor` / `driver_framing` machinery from tries A-E is gone.

**And then the mouth died anyway (try F: the distance-sized crop at the
old scale 2.3): mar gain 0.30, and on the wide-open-mouth frames the whole
face collapsed** -- eyes shut, nose dragged down into a shut mouth (frame
1067: his mouth ratio 0.55, the avatar's 0.005). Run down in order: the crop
box at those frames was steady (835-855 px); the network's input crop looked
textbook; the motion extractor's own numbers were moderate (scale ratio
0.97, |exp - exp_d0| 0.12); the eye network off changed nothing; the
network's raw 512 output was already the collapsed face. One frame, the same
portrait and driver, the driver's box swept in size:

| driver box, x resting face height | 1.96 | 2.07 | 2.18 | **2.30** | 2.41 | 2.53 |
|---|---|---|---|---|---|---|
| output mouth ratio (his 0.549) | 0.53 | 0.51 | 0.44 | **0.01** | 0.16 | 0.08 |
| output eye ratio (his 0.260) | 0.19 | 0.17 | 0.26 | **0.14** | 0.15 | 0.12 |

A cliff between 2.18 and 2.30, and 2.30 is exactly LivePortrait's own
`CropConfig` scale -- which we had applied to the wrong height since day 1.
Their "face" box comes from a landmark set that starts at the eyebrows; ours
is the MediaPipe mesh's whole height, forehead top to chin, about a seventh
taller. 2.3 of *our* heights shows the network a face ~10% smaller than it
was trained on, and on a hard expression it falls off. No earlier crop had
ever held that frame at 2.3 with an anchor cut at 2.3: day 5's
width-governed box shrank 5% on that frame (a turn or a wide mouth narrows
the cheek-to-cheek width) and rendered it; day 4's whole-mesh box *grew* 7%
on it and rendered it too -- with lip retargeting still on, and with an
anchor cut smaller than the frame. Why a mismatched anchor/frame pair
survives where the matched pair of the sweep collapses is **not
understood**; the sweep is one frame, one portrait, and is reported as
that.

**2.1 fixed that frame and broke another.** The whole clip at 2.1 measured
well (mar gain 0.84, brows r 0.78, chin -6.6) -- and the owner's own
proposal, *look at the side-by-side, not the numbers*, showed frames
1760-1808 (he frowns, chin down 5°, eyes open) with the avatar's eyes
squeezed shut and lips pursed: a grimace day 5 did not have (avatar eye
ratio 0.30 there; at 2.1, 0.10) -- the "random reactions" he reported, made
worse. The same one-frame sweep on that frame and on a brow raise:

| driver box, x resting face height | 2.07 (2.1) | 2.18 (2.2) | 2.30 (2.3) |
|---|---|---|---|
| frown, chin down 5° -- eye ratio (his 0.283) | **0.10** | **0.30** | 0.12 |
| ... mouth ratio (his 0.008, shut) | 0.04 | 0.01 | 0.12 |
| brows raised -- eye ratio (his 0.376) | 0.28 | 0.32 | 0.35 |
| mouth wide + pitch 10° -- mouth ratio (his 0.549) | 0.51 | 0.44 | 0.01 |

Two cliffs, not one: the wide mouth collapses at 2.3, the frown collapses at
2.1 *and* at 2.3, and 2.2 renders all three. **`CROP_SCALE` is 2.2, for the
portrait and the driver alike.** Why the network is this sensitive to a 5%
framing change is not understood and is the single most important open
question about this engine -- a stable crop was what exposed it, because
every earlier crop wandered across these cliffs frame by frame and the
failures read as noise. The whole-clip numbers below are the test of
whether 2.2 holds beyond three frames; the owner's eye on the side-by-side
is the test of whether the numbers mean anything.

**(2), the eyes pulled wide, had a second cause:** `_eye_target`'s 1.2x
headroom -- a driver whose eyes read wider than his rest asked the eye
network for eyes 20% wider than the portrait's own, and that network's 63
outputs are not confined to the lids. Headroom is 1.0 now: the portrait's
open eyes are as wide as the avatar goes; narrower still follows him.

**(3), the teeth.** Two things in one picture. The teeth themselves are
hallucinated -- the portrait's lips are closed, the network has never seen
this mouth's interior and paints a generic one; the lever is a portrait with
a slightly open mouth, not code. But the *grimace* he saw is measurable: on
"smile showing teeth" the avatar's eyes squeezed to EAR 0.152 (day 5) while
his stayed at 0.267 -- the network entangles a wide smile with a squint.
Final numbers below say how much of that this day recovered.

**Whole clip, the distance-sized crop at each scale (`driving-follow-
day6-final.csv` = 2.3, `-s21.csv` = 2.1, `-s22.csv` = 2.2, eye headroom 1.0
in all three) against day 4 and day 5 (what he tested):**

| | day 4 | day 5 (tested) | 2.3 | 2.1 | **2.2 (this build)** |
|---|---|---|---|---|---|
| still: chin dy (px; his note 1) | -2.6 | -14.4 | -8.8 | -6.6 | **-8.0** |
| still end: chin dy | -2.0 | -20.5 | +1.6 | +0.9 | -8.1 |
| still: scale (1.08 is this pair's identity) | 1.083 | 1.020 | 1.099 | 1.073 | **1.066** |
| mouth wide: scale p90 (day 5's zoom fix) | 1.254 | 1.033 | 1.067 | 1.105 | 1.156 |
| still: avatar EAR (his 0.293; note 2) | 0.318 | 0.286 | 0.325 | 0.308 | **0.301** |
| talk: avatar EAR p90 (his 0.332) | 0.410 | 0.335 | 0.291 | 0.347 | 0.371 |
| frown, chin down (fr. 1760-1808): avatar EAR (his 0.28-0.34) | 0.16-0.20 | 0.30-0.36 | -- | **0.10-0.12** | **0.31** |
| brows gain / r (note 2) | 1.14 / 0.95 | 0.23 / 0.30 | 0.67 / 0.83 | 0.31 / 0.78 | 0.18 / 0.24 |
| eyes shut: avatar EAR | 0.233 | 0.290 | 0.192 | 0.196 | **0.351** |
| ear gain / r | 0.85 / 0.79 | 0.52 / 0.63 | 0.71 / 0.65 | 0.67 / 0.64 | 0.51 / 0.72 |
| smile with teeth: avatar EAR (his 0.267; note 3) | 0.213 | **0.152** | 0.218 | 0.234 | **0.265** |
| smile with teeth: mouth-width gain | 0.79 | 0.86 | 0.91 | 0.96 | 0.88 |
| mar gain / r | 1.05 / 0.99 | 0.97 / 0.99 | **0.30 / 0.50** | 0.84 / 0.98 | 0.77 / 0.98 |
| \|yaw\| > 15°: nose dx (+ / -) | -20 / +19 | -34 / +25 | -51 / +24 | -27 / +20 | -33 / +21 |
| lip thickness, start (upper/lower) | -- | 0.77 / 0.78 | 0.71 / 0.70 | 0.77 / 0.81 | 0.75 / 0.75 |

**What the side-by-side (`day27-side-by-side.mp4`, 12 stills) shows at
2.2, frame by frame:** rest, talk, the wide-open mouth (which collapsed at
2.3), the toothy smile (eyes 0.265 against his 0.267 -- the day-5 grimace
gone), the 40° turn, and the frown frames (which grimaced at 2.1) all render
as a plausible face doing roughly what he does. **What it also shows: the
avatar is flatter than he is.** His brow raise (1724) and his frown (1784)
come out as a neutral face; his closed-mouth smile (800) barely registers;
in the eyes-shut segment the avatar's eyes stay half open (0.351). Nothing
breaks at 2.2 -- and the expression gain that 2.1 had (eyes shut 0.196,
brows r 0.78) is gone with it. Read against his four notes: (1) the chin is
8 px above his, was 14-20 -- better, not day 4's 2.6; (2) the *random*
reactions are gone in the clip (no wide eyes, no grimace) but the *real*
reactions are under-driven; (3) the toothy-smile squint is gone, the teeth
are the portrait's problem; (4) not met: the avatar follows him, weakly.

**Where the trade sits.** Three crop scales, three different failures: 2.3
collapses a wide mouth, 2.1 grimaces a frown, 2.2 does neither and
under-reacts. The next lever is not the crop: it is a multiplier on the
relative expression itself (`exp_s + k * (exp - exp_d0)`, k > 1 -- what
LivePortrait's own demos call a driving multiplier), tried at 2.2 on the
same frames first (the risk is amplifying exactly the two failures the
scale choice just avoided), and the portrait-side lever for the teeth.
Both are his call.

**(4) is his exit criterion, not mine to declare met.** The numbers above
are the clip; his eye on the live app decides. What is still known to be
open: the |yaw| > 15° lag (never better than day 4's 19-20 px); teeth on a
closed-lip portrait; the forehead mask line from day 3.

775 tests (767 → 775; the day-6 tests cover the height/distance freeze,
medians, reset, and the crop's indifference to width and mouth). mypy: the
same 12 pre-existing errors. ruff clean.

**Files from today under `recordings/`, gitignored:** the measurement CSVs
(his tracked geometry, no pixels) `driving-follow-day6.csv`, `-day6-own.csv`,
`-day6-paste.csv`, `-day6-own2.csv`, `-day6-paste2.csv`, `-day6-final.csv`
(tries A-F), `-day6-s21.csv` (2.1) and `-day6-s22.csv` (2.2, this build);
**`day27-side-by-side.mp4` and its 12 stills (`-f60 … -f1808.png`), his
face beside the avatar's at 2.2 -- made at his request, for him to look at,
to be deleted once he has.** Every other image and clip of his face made
for the diagnosis (`day26-*`, the 2.1 side-by-side) was looked at and
deleted the same day; the temporary avatar copies `avatars/exp_*` are
deleted too.

**Later the same day: a second clip, and his two live observations
reproduced in it.** He re-ran the recorder (which silently replaced
`driving.mkv` -- the recorder now refuses to overwrite without
`--overwrite`, and has a `natural` script for a call posture) and reported
two things from the live app: *"when I tilt my head just slightly down it
makes a sulky, offended face"* and *"when my face is simply still it tries
different expressions on its own"*. Both are in the new clip, at the 2.2
build (`driving2-follow-s22.csv`, `day27b-side-by-side.mp4` + 14 stills):

| | him | avatar |
|---|---|---|
| still, frames 0-119: eye ratio p10..p90 | 0.314..0.320 | **0.291..0.365** |
| still, frames 0-119: brow p10..p90 | 0.237..0.247 | **0.161..0.213** |
| eyes up, frames 470-530: eye ratio | 0.43-0.44 | 0.27-0.33 (headroom 1.0 caps it) |
| eyes up, frames 470-530: brow | 0.16 | **0.12** (brows pulled down) |
| eyes up, frames 470-530: mouth ratio | 0.005 | **0.02-0.05** (ajar, pouting) |

He does not move and the avatar wanders ±12% in the eyes and ±14% in the
brows -- the crop is steady (distance and height within ±0.5% on those
frames), so this is the motion extractor's own frame-to-frame noise passed
straight through. Looking up reads as brows down + mouth ajar on the
portrait: the sulky face (frames 500/520). The frown segment renders flat
and calm; the toothy smile is fine.

**Portrait 2** (`avatars/my_avatar_neural2`, his own upload: smiling, teeth
showing, 642x673): real teeth appear -- and the avatar grins at rest and
while he talks neutrally (frames 100, 1200), because the source face's
smile *is* its rest. Unusable as the main portrait; proof that the teeth
lever is real. What is needed is neutral, lips slightly parted.

**Day 7, next -- two engine levers, in this order, each measured on the new
clip's still and eyes-up frames before anything is kept:**
1. **Temporal smoothing of the extracted motion** (an EMA on exp/t/scale/
   pose inside the captured graph, state reset with the anchor) -- for the
   wandering. Low risk; the cost is a fraction of a frame of lag.
2. **The pitch/gaze leak** -- the extractor under-reads pitch (MediaPipe
   -5° read as +3°, +10° as -2°), so a tilt lands in the expression as a
   pout. Candidate fix: re-decompose the driver's motion around MediaPipe's
   rotation (`exp' = exp + kp @ (R_ext - R_mp)`, then relative motion with
   R_mp) -- principled, untested, conventions between the two pose
   parametrisations must be checked first.
Plus his side: a `natural` clip (`--script natural --out recordings/
natural.mkv`) and a neutral portrait with parted lips. The expression
multiplier is deferred behind these two (amplifying noise before smoothing
it would be backwards). Exit unchanged.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day27b-side-by-side.mp4` + 14 stills (2.2, new clip),
`recordings/day27b-portrait2.mp4` + 6 stills, and the older
`day27-side-by-side*` (2.2, the old clip).

**Day 7, done (2026-09-05, Sonnet 5, lever 1 only) -- temporal smoothing
built and measured; the plain-EMA version does not clear its own bar.**
Task A: an EMA over the extractor's own exp/t/scale/pose, in state tensors
on `_Portrait` that persist between graph replays (`_ema_`, in-place
`mul_`/`add_` so it captures into the graph), gated by a new
`motion_smoothing` option (`[0, 1)`, default `0.0`, validated at
`load_avatar`'s own `_capture` call). At `0.0` every line is a no-op --
`state*0.0` is exactly `0.0`, `0.0 + new*1.0` is exactly `new` -- checked
bit-for-bit against the unsmoothed path, not just approximately. Primed at
load from the portrait's own reading (not zero); re-primed to the anchor's
own running mean on every accepted anchor frame while it is still being
learned, so a smoothed graph's first replays after the freeze start from
the clean mean, not from noise collected during the anchor window;
`reset_anchor` deliberately leaves the state alone (no jump the instant it
is called, matching `exp_d0`'s own behaviour) until the next accepted frame
re-primes it. 789 tests (775 → 789), mypy the same 12 pre-existing errors,
ruff clean.

Task B: measured on the same clip (`recordings/driving.mkv`, `driving.json`
segments) at `motion_smoothing` 0.5 and 0.65 against the no-smoothing
baseline (`driving2-follow-s22.csv`):

| | baseline (off) | 0.5 | 0.65 | bar |
|---|---|---|---|---|
| still start: avatar eye p10..p90 (his 0.314..0.320), width | 0.291..0.365 (**0.074**) | 0.294..0.348 (**0.054**) | 0.294..0.345 (**0.051**) | width ≤ 0.037 |
| still start: avatar brow p10..p90 (his 0.237..0.247), width | 0.161..0.213 (**0.052**) | 0.160..0.206 (**0.046**) | 0.159..0.205 (**0.046**) | width ≤ 0.026 |
| blink: ear gain / r (% of baseline) | 0.807 / 0.953 | 0.741 / 0.897 (**92%**) | 0.660 / 0.839 (**82%**) | ≥ 90% |
| mouth wide: mar gain / r (% of baseline) | 0.562 / 0.971 | 0.560 / 0.976 (100%) | 0.554 / 0.977 (99%) | ≥ 90% |
| talk: mar gain / r | 0.764 / 0.941 | 0.702 / 0.898 | 0.652 / 0.845 | -- |
| turn: nose dx p50, yaw>+15 / yaw<-15 (px) | -35.5 / 35.0 | -35.5 / 33.2 | -35.9 / 32.0 | -- |

**Neither value clears the spread bar.** A single-pole EMA on white noise
would be expected to leave `sqrt((1-a)/(1+a))` of the raw std: 58% at 0.5,
46% at 0.65. The *measured* eye spread only fell to 73% (0.5) / 69% (0.65)
of the baseline's -- the extractor's frame-to-frame wander is not simple
white noise, so a plain EMA buys less than the textbook number and costs
more lag for it than expected: 0.65 also drops the blink ear gain to 82% of
baseline, under the 90% floor.
Per the rule set for this measurement (**if neither value clears the
spread bar, report both and stop -- do not go above 0.65, do not substitute
a different filter without saying so**): stopped here. `motion_smoothing`
is left at `0.0` in `my_avatar_neural`'s own `engine_options` -- the option
and the whole EMA mechanism stay wired in, correct and tested, waiting for
either a higher ceiling to be accepted deliberately (more lag) or a
different filter shape. No pictures were rendered for this build: nothing
here is a "before/after" worth keeping, only a measurement that the
first thing tried was not enough.

**Day 7, still open:** lever 1 (smoothing) needs a shape that does not cost
this much lag for this little gain -- a speed-adaptive (One-Euro) filter is
the obvious next candidate, not decided or built here. Lever 2 (the
pitch/gaze leak behind the sulky face) is untouched. His side (the
`natural` clip, the parted-lips portrait) is still outstanding. Exit
unchanged.

**Day 7, later (Fable 5.1 analysis on the `natural` clip he then recorded,
`recordings/natural.mkv`, 1690 frames, `natural.json`).** The clip that
finally holds his call posture, and it is worse than the scripted one:

| segment | his eye ratio | avatar eye ratio | his brow | avatar brow |
|---|---|---|---|---|
| look at camera (0-118) | 0.288..0.316 | 0.218..0.344 | 0.252..0.269 | 0.158..0.192 |
| look at the screen, still (119-416) | 0.292..0.381 | **0.180..0.292** | 0.208..0.263 | 0.135..0.189 |
| talk (713-1306) | 0.249..0.329 | **0.131..0.299** | 0.245..0.307 | 0.140..0.181 |
| still again (1602-1689) | 0.262..0.338 | **0.085..0.243** | 0.197..0.352 | 0.144..0.189 |

His eyes are open all clip; the avatar's are half shut while he talks
(frames 1000, 1380: eyes squeezed, lips pursed) and squinting-frowning when
his eyes open *wider* than rest (frames 300-390: his ratio 0.37-0.39, the
avatar squints -- the "sulky face"). The head-down segment (417-712) goes to
pitch -32°, far past anything the engine can hold from a frontal portrait,
and the face smears with the forehead band showing (frames 560/620).

Run down, in order, each cheaply:
- **Not the crop.** The 1/z box tracks his inter-ocular width within 1-2% on
  every sitting segment (box/io 0.99-1.01) -- the day-6 crop is doing its job.
- **Not the anchor posture.** Re-rendered with the anchor learned while he
  looks at the screen (`--first 119`): frames 300/400/1000/1380 identical.
  The problem frames are deviations from *either* anchor.
- **Not pitch, as such.** The extractor's own pitch runs *opposite in sign*
  to MediaPipe's (slope -1.7 on the up/down segment, -0.9 on the natural
  clip) -- a convention, consistent internally, not a leak. What moves the
  expression on neutral-face frames is **his eyes opening wider than his
  rest**: |exp - exp_d0| 0.09-0.10 at frames 314-329 (eye ratio 0.38-0.39,
  pitch flat at -1.3°) against 0.01-0.02 at rest -- and the network renders
  "wider than the portrait's eyes" as a *squint with lowered brows*. Talking
  closes the avatar's eyes the same way (frame 1000: his 0.28, avatar
  squeezed): mouth motion leaking into the eye keypoints.
- **Not the portrait.** `portrait3.jpg`/`portrait4.jpg` (his uploads:
  neutral, lips parted, 2048x2048 -- exactly what was asked) rendered on the
  same clip (`my_avatar_neural3`/`4`): the same squint at 300, the same
  squeezed eyes at 1000/1380. Plus a new cost: a parted-lips *rest* means the
  avatar's mouth sits open whenever his is shut (frames 60/200/400), because
  `_lip_target` is floored at the portrait's own ratio -- for these portraits
  that floor must go and lip retargeting must come back on, or the mouth
  never closes. Teeth at frame 1380: no better than the closed-lip portrait.

**The finding that opens a door: LivePortrait's 21 implicit keypoints are
semantically separable, and the same on both portraits.** Nudging each
keypoint's expression by ±0.04 on x/y/z with the portrait driving itself and
reading the result back with MediaPipe:

| region | keypoints (axis) | effect on that region | leak into the other |
|---|---|---|---|
| eyes | **11, 13, 15, 16 (y)** | eye ratio ±0.16..0.20 each | mouth ≤ 0.004 (15/16 move brows 0.06-0.12) |
| mouth | **19, 20 (y, z)**, 17 (y), 14, 7 | mouth ratio ±0.30..0.37 | eyes ≤ 0.03 |
| brows | 2 (y: 0.196), 3 (y), 1 (y) | brow ±0.07..0.20 | -- |
| the rest (0, 4-6, 8-10, 12, 18) | -- | ≤ 0.015 anywhere | pose/shape |

So the eye channel can be cut out of the relative expression and driven by
the *measured* eye ratio alone (the eye retargeting network already wired
in, its delta masked to the same four keypoints) -- which is exactly the
thing the extractor gets wrong. This is **lever 2, redefined**: not a pitch
correction, a region-masked expression. Proposed as the next Sonnet task,
with the natural clip as the measure and frames 300/1000/1380 as the eye.
Probe scripts (scratch, not in the repo): `gaze_probe.py`,
`pitch_leak_probe.py`, `keypoint_sensitivity.py`.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day28-natural-side-by-side.mp4` + 16 stills (2.2, natural
clip), `day28-natural-anchor119.mp4` + 6 stills, `day28-natural-p3.mp4` +
7, `day28-natural-p4.mp4` + 7; CSVs `natural-follow-s22.csv`,
`natural-gaze.csv` (geometry only). Avatars `my_avatar_neural3`/`4` kept --
selectable in the app, but see the parted-lips caveat above.

**Day 8, done (2026-09-06, Sonnet 5, on Fable 5.1's written instruction) --
the eye channel cut from the extractor, kept; an extreme-pose hold, kept.**

**Task A.** `EYE_KEYPOINTS = (11, 13, 15, 16)`. Two new options:
`eye_expression_gain` (`[0, 1]`, default **0.0**) weights the eye keypoints
of the *relative expression* -- at 0.0 the extractor's own reading of the
eyes never reaches the portrait, only the eye retargeting network does;
`eye_mask_output` (default **true**) confines that network's own delta to
the same four keypoints, so a driven eye target cannot itself leak into the
mouth or brows. Both built once at load into static masks
(`rel_mask`/`eye_mask`, `_region_masks`) the graph multiplies every frame;
at `1.0`/`false` the whole thing is bit-identical to day 7. Validated once,
at load (not per frame like `motion_smoothing` -- the value only ever feeds
a precomputed tensor, so there is nothing to re-check after that) -- a
deliberate difference from the letter of "validated where `motion_smoothing`
is", noted here rather than silently.

**Task B1 -- `driving.mkv`, `eye_expression_gain=0.0`, sweeping `eye_gain`:**

| | baseline (day 7) | 0.06 | 0.12 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| blink: ear gain / r | 0.807 / 0.953 | 0.095 / 0.594 | **0.482 / 0.833** | 0.222 / 0.522 | -0.087 / -0.561 |
| eyes shut: avatar ear p10 (≤0.08) | 0.039 | 0.245 | **0.059** | 0.054 | 0.031 |
| still: avatar ear p10..p90 | 0.291..0.365 | 0.279..0.300 | 0.228..0.268 | 0.100..0.199 | 0.025..0.042 |
| still: avatar brow p10..p90 | 0.161..0.213 | 0.162..0.243 | 0.180..0.259 | 0.208..0.293 | 0.247..0.333 |
| smile teeth: avatar ear p50 | 0.147 | 0.261 | 0.209 | 0.078 | 0.077 |
| mouth-wide mar gain (% of baseline) | 0.562 | 0.537 (96%) | 0.525 (93%) | 0.499 (**89%**) | 0.457 (**81%**) |
| talk mar gain (% of baseline) | 0.764 | 0.743 (97%) | 0.711 (93%) | 0.708 (93%) | 0.690 (90%) |

**No value reaches the 0.70 blink-gain target -- not monotonic, either: 0.12
is the peak, 0.20 and 0.30 are worse than 0.12 on every row, not just
blink.** With the extractor cut out, the eye network is now the *only*
driver of eye motion instead of a small correction on top of one, and it
overshoots the way the old *combined* channel did in day 4's calibration --
past 0.12 the avatar's eyes narrow even at rest (`still` collapsing toward
shut) and the toothy smile squints again (0.078, worse than day 5's 0.152
that started this whole investigation). 0.20 and 0.30 also breach the
mar-gain integrity check (>10% off baseline) that was meant to catch
exactly this kind of cross-talk -- not simple keypoint leakage (the mouth
keypoints are untouched by the masks) but the warping network's own dense
motion field responding to an extreme deformation elsewhere in the image.
**Per the stated rule ("if none reaches 0.70, pick the best and say so"):
`eye_gain = 0.12`**, the only value that also stays inside the mar-gain
tolerance.

**Task B2 -- `natural.mkv`, `eye_gain=0.12` throughout, isolating which
change matters:**

| | baseline (day 7) | chosen (gain=0.0) | control (gain=1.0, masked) |
|---|---|---|---|
| screen-still: avatar ear p10 (≥0.24) | 0.180 | **0.236** | 0.092 |
| screen-still: avatar ear width (≤0.06) | 0.112 | **0.024** | 0.148 |
| talk: avatar ear p10 (≥0.20) | 0.131 | **0.218** | 0.072 |
| still-again: avatar ear p10 (≥0.20) | 0.085 | **0.212** | 0.068 |
| talk mar gain (% of baseline) | 0.799 | 0.816 (**102%**) | 0.754 (94%) |
| frame 300: his 0.376, avatar ear (≥0.22) | 0.267 | **0.265** | 0.221 |
| frame 1000: his 0.308, avatar ear (≥0.22) | 0.184 | **0.230** | 0.080 |
| frame 1380: his 0.325, avatar ear (≥0.22) | 0.141 | **0.256** | 0.085 |

**The control -- masking alone, extractor still at full strength -- is not
the fix; it is sometimes worse than the day-7 baseline** (frame 1000: 0.080;
screen-still p10: 0.092, both below baseline). Cutting the extractor's own
reading of the eyes (`eye_expression_gain=0.0`) is what matters; the mask
on the eye network's own output is a second, smaller safeguard. The chosen
build clears every acceptance line except one, and that one by 0.004
(screen-still p10 0.236 against 0.24) -- everything else, including all
three single-frame checks (the direct evidence for the day-7 squint/closure
complaint), passes with room. **Kept**: `eye_expression_gain: 0.0`,
`eye_mask_output: true`, `eye_gain: 0.12` in `my_avatar_neural`'s own
`engine_options`.

**Task C.** `hold_pitch_degrees`/`hold_yaw_degrees` (default 20.0/35.0,
must be positive), hysteresis at 0.75x to enter/leave without flicker
(`_should_hold`, `loaded.holding`). Past the threshold, `animate` reuses the
last rendered face but still computes this frame's own crop and mask, so
the held face follows him at his real position; `reset_anchor` clears it.
Rendered on `natural.mkv` with Task B's kept options
(`day29-natural-eyech.mp4` + 10 stills): frames 500/560/620 (pitch to -32°,
the forehead-band smear in day 7's own stills) now hold a calm, coherent
face that tracks his position and comes back the instant he does --
compared side by side with `day28-natural-anchor119-f500/560/620.png`, no
smear, no forehead band. Frames 60/300/1000/1380 (the eye-channel
evidence) read as a genuinely following face for the first time: eyes open
at rest, open and alert while he talks, open when his are wide -- none of
day 7's squeezed-shut or squint-and-frown.

**What did not change, on purpose.** Brows still leak via their own
keypoints (2, 3, 1 -- not masked, no attempt made); the parted-lips
portraits (`my_avatar_neural3`/`4`) still need lip retargeting turned back
on before they are usable, not attempted; the |yaw| > 15° lag from day 4 is
untouched. 823 tests (789 -> 823; +34 from Task A's masks and Task C's
hold). mypy the same 12 pre-existing errors. ruff clean.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day29-natural-eyech.mp4` + 10 stills (frames 60, 200, 300, 400,
500, 560, 620, 1000, 1380, 1650). CSVs (geometry only, not deleted):
`driving2-follow-eyech-g0.06/0.12/0.20/0.30.csv`,
`natural-follow-eyech.csv`, `natural-follow-eyech-ctrl.csv`.

**Day 9 (2026-09-06, his live test, then Sonnet 5's quick measurement of it)
-- day 8's fix does not generalise: the mouth and brows invent reactions
the same way the eyes did, and day 8 never touched them.** His own words:
*"it takes on emotions/reactions automatically... when I do nothing but
lift my head up, it moves its mouth and half-shuts its eyes -- it invents
reactions, this is terrible, it must be fixed before anything else."* He
is running the app correctly; this is real, not operator error.

Measured on `driving.mkv`'s "Look UP" segment (421-538) at today's kept
build (`driving2-follow-eyech-g0.12.csv`, already in hand -- no new render
needed):

| | him | avatar |
|---|---|---|
| mar, whole segment | **0.003..0.006** (mouth shut, no change) | **0.003..0.101** (opens on its own) |
| frames 469-525 specifically: mar | 0.004..0.005 | **0.023..0.062** |
| brow, same frames | ~0.24, barely moving | **0.135..0.152**, dropping |
| ear, same frames (his eyes open wide) | 0.43..0.44 | 0.227..0.254 -- does not follow, moves the wrong way |

Exactly frames 469-525 are where his own eye ratio rises to 0.43-0.44 (he
is looking up, eyes open wide) -- and that is when the avatar's mouth opens
and its brow drops, though his mouth never moved. **This is the same
mechanism day 8 fixed for the eyes, showing up in a place day 8 did not
touch.** Day 8's `EYE_KEYPOINTS` table was built from a *synthetic* test
(nudge one keypoint at a time on a still portrait, read the isolated
effect) -- it answers "does keypoint 19 moving open the mouth" (yes), not
"does the extractor's own *natural* reading of keypoint 19, across real
frames, correlate with something the driver is not actually doing with his
mouth" (apparently yes, when his gaze/eyes change). The mouth and brow
keypoints (19/20/17; 1/2/3) still take the raw extractor's relative
expression unmasked, exactly as before day 8.

**Proposed next step, not measured or built yet:** extend day 8's own
pattern -- `rel_mask` zeroed on the mouth keypoints too, mouth driven only
by the lip retargeting network (`lip_retargeting`, off since day 5) against
his own measured `mar`, the same way the eyes are now driven only by the
eye network against his measured `ear`; the brow keypoints likely need the
analogous treatment, though nothing currently retargets brows from a
measured ratio, so that may need its own new mechanism rather than
reusing one that exists. Day 5's finding that "the lip network only presses,
never opens" was measured *before* this leak was known to exist, on a
driver clip that never isolated a pure gaze-up gesture from a mouth
gesture -- it needs re-checking under this specific case, not assumed
still true. His priority, stated plainly: this closes before Phase 2 or
Phase 3 get any time. Handoff to Fable 5.1 for the precise instruction --
this entry is written for that.

**Day 9, done (2026-09-06, Fable 5.1 implementing directly on his
instruction -- "no instruction, start on the defect yourself"): the mouth
and brows no longer invent reactions, and the eyes are the portrait's own
at rest.** Three changes, each measured on both clips against the day-8
build (`driving2-follow-eyech-g0.12.csv`, `natural-follow-eyech.csv`):

1. **Expression gates** (`expression_gates`, default on; the GATES section
   of `engine_liveportrait.py`). The extractor's relative expression on the
   mouth keypoints (7, 14, 17, 19, 20) and brow keypoints (1, 2, 3) is
   multiplied, every frame, by a scalar in [0, 1] computed from *his own
   tracked geometry* against the anchor's rest: mouth ratio above rest
   (start 0.010, full 0.030), mouth width above rest (+3% .. +8%, over the
   *jaw* width at mouth height -- over inter-ocular it grows 0.4%/degree of
   pitch, a closed smile's worth at +10 degrees; faded past |yaw| 12-25
   degrees), brow height above the eye corners over the eye-to-chin height
   (+-8% .. 16%; blink-proof, 0.3%/degree of pitch against 1.2-1.9% for
   the lid-based measure; faded past |pitch| 10-20 and |yaw| 15-30). Opens
   at once, closes at 0.7 per frame. Written into a static `(1, 21, 3)`
   tensor before each replay, like `ratios`. Not a cut: a smile, a word or
   a raise passes whole. Rest values `rest_mouth_width`/`rest_brow` freeze
   with the anchor; `reset_anchor` clears them.
2. **Zero at rest for the retargeting networks** (`retarget_zero_at_rest`,
   default on). Measured 2026-09-06 with the real weights: asked for *no
   change* (target = the portrait's own ratio), the eye network answered
   +0.089/+0.111 on the upper-lid keypoints -- **more than the whole
   difference between its rest and full-blink outputs** (0.072/0.066). At
   `eye_gain` 0.12 that alone closed the avatar's eyes from the portrait's
   0.334 to 0.23-0.27 at rest, and is why no gain ever reached the 0.70
   blink target on day 8: the "closing" that was measured was mostly this
   constant. The network's output at the portrait's own ratio is now read
   once at load (`_retarget_rest`) and subtracted (`eye_rest`/`lip_rest`).
3. **`eye_headroom`** option (was the constant `_EYE_TARGET_HEADROOM`),
   1.3 kept: with the eye net's delta confined to the eye keypoints (day 8)
   and zero at rest, "wider than the portrait" no longer lifts the brows --
   on the wide-eyed frames (his 0.43-0.44) the avatar's eyes go 0.31 ->
   0.375-0.38, brow measure unchanged to the third decimal, every other
   line of the table identical. Day 6's "pulls the eyes and raises the
   brows" verdict was taken with the unmasked network and the rest offset
   in it.

`eye_gain` re-swept under 2 (0.12 / 0.25 / 0.5, `driving2-follow-gate-*.csv`):
blink ear gain 0.26 / **0.77** / 0.83, eyes-shut avatar ear p50 0.26 /
**0.13** / 0.09, still-again avatar ear p10 0.30 / 0.26 / **0.15 (over)**,
blink r 0.86 / **0.89** / 0.81. **0.25 kept** -- the first value ever to
clear the 0.70 blink bar, with the eyes at rest still the portrait's.

**The complaint's own frames, `driving.mkv` 421-538 (his gaze up, mouth
shut), day 8 -> day 9:**

| | him | day 8 | day 9 |
|---|---|---|---|
| mouth ratio, p50 / p90 / max | 0.005 (p90) | 0.023 / 0.055 / **0.101** | **0.003 / 0.005 / 0.006** |
| frames 469-525, mouth ratio | 0.004..0.005 | 0.023..0.067 | **0.003..0.004** |
| eye ratio p10..p90 | 0.307..0.441 | 0.225..0.254 (half shut) | **0.316..0.372** (0.309..0.330 at headroom 1.0) |
| brow measure (brow_c) p10..p90 | -- | -- | 0.173..0.175 against 0.174..0.180 at rest: flat |
| still 0-119, avatar eye ratio | his 0.314..0.320 | 0.228..0.268 | **0.316..0.331** (portrait 0.334) |
| still, avatar mouth ratio p90 | 0.005 | 0.007 | 0.004 |

`natural.mkv` (call posture): screen-still avatar mouth ratio p90 **0.064 ->
0.003** (his 0.005), still-again **0.151 -> 0.003**, screen-still avatar eyes
0.236..0.260 -> **0.301..0.333**, talk mar gain 0.82 -> **0.96** (r 0.85 ->
0.92), smile mouth-width gain 1.35 -> 1.26, smile mar gain 0.92 -> 1.05.
Frames 300/1000/1380 (his eye -> avatar eye | his mouth -> avatar mouth):
0.376 -> **0.357** | 0.003 -> 0.002; 0.308 -> **0.318** | 0.009 -> **0.011**
(was 0.137); 0.325 -> **0.335** | 0.293 -> 0.400 (was 0.297). (Kept build,
`natural-follow-day9.csv`; the gate-only sweep at headroom 1.0 read 0.346 /
0.318 / 0.332 on the eyes.)

**What must still come through, `driving.mkv`, day 8 -> day 9 (`eye_gain`
0.25):** closed-smile width gain 1.17 -> 1.22, toothy-smile mar gain 0.93
-> 0.90, mouth-wide mar gain 0.53 -> 0.56, talk mar gain 0.71 -> 0.72 (r
0.92), eyes-shut avatar ear p10 0.059 -> 0.052. **Costs, stated:** the
toothy smile's *width* gain 0.41 -> 0.23 -- per frame the avatar's width
saturates at 0.67-0.69 in both builds (his 0.75; the gate is fully open,
+27..35%), so this is the network's ceiling read through a slope, minus
~0.02; and a slow closed smile arrives late: at his +2% width (frames
781-797) the day-8 avatar was already +10% wide (the extractor amplifying
a beginning), the gated one waits until his +3-6% (frame 805), about half
a second.

**Brows: measured, and held by default (`brow_gain` 0.0).** With the gate
open on his real brow raise (his measure +34..38% on both clips, frontal,
mouth shut): `driving.mkv` frames 1678-1708, avatar brow +4..8% (right way,
a fifth of his); `natural.mkv` frames 1570-1600, the same gesture, avatar
brow **-12..-16% -- a frown for a raise**, in the day-8 build too (lid-based
brow 0.139-0.149 against 0.16-0.20 before it). The extractor's brow
reading is not reliably the right way up from one clip to the next, and a
gate cannot turn a wrong-way reading right. A brow that frowns when he
raises it is exactly this day's complaint, so until brows are driven from
his own measured reading the way the eyes are (no such network exists in
LivePortrait -- it would be a new mechanism), they do not move. `brow_gain`
1.0 gives the gated brows back for measuring. His frown: driving frames
1720-1756, his -12%, avatar -2..-11% (right way) -- also held now.

**Also:** `scripts/measure_follow.py`/`render_liveportrait.py` take
`--option KEY=VALUE` (repeatable, JSON values) to override an avatar's
`engine_options` for one run -- sweeps no longer copy avatars or edit
`metadata.json`; `measure_follow.py` reports the two gate measures as
`mouth_wj`/`brow_c`. Kept options (`my_avatar_neural/metadata.json`):
`eye_gain 0.25, eye_headroom 1.3, retarget_zero_at_rest true,
expression_gates true, brow_gain 0.0` (the rest as day 8). Kept build
re-measured whole (`driving2-follow-day9.csv`): blink ear gain 0.78 (r
0.88), eyes-shut avatar ear p10 0.051, closed-smile width gain 1.33,
mouth-wide mar gain 0.58, talk mar gain 0.73; brows flat everywhere
(brow_c gain -0.02). Tests 823 -> 855 (+32: gate measures, gates, the
frame path, `_drive`'s gate and rest terms, headroom). mypy the same
pre-existing errors, ruff clean. (Note for whoever runs the suite from
Git Bash: `tests/unit/test_ui_background.py` aborts the process at exit
127 there -- a Qt platform matter, not a failure; `QT_QPA_PLATFORM=offscreen`
runs all 53 UI/virtual-camera tests green.)

**Not done, on purpose:** brows (above); the lip network stays off (its
`lip_rest` is wired and tested, unmeasured with real weights since it does
not run); the parted-lips portraits; |yaw| > 15 degree lag; the |yaw| 12-25
fade means a smile *while turned well away* arrives only through the mouth
ratio. **Next: his live verdict on this build, then Phase 2.**

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day30-driving-day9.mp4` + stills `day30-driving-day9-f{60,469,
500,520,700,830,940,1250,1700}.png`, `recordings/day30-natural-day9.mp4` +
stills `day30-natural-day9-f{300,560,1000,1380}.png`. CSVs (geometry only,
not deleted): `driving2-follow-gate-g0.12/0.25/0.5.csv`,
`driving2-follow-gate-g0.25-h1.3.csv`, `natural-follow-gate-g0.25.csv`,
`driving2-follow-day9.csv`, `natural-follow-day9.csv` (the kept build).

**Day 9, later (2026-09-06, Fable 5.1, on his live verdict of the build
above): "it no longer follows my mouth well when I talk, the face is not as
clear; eyes and brows have gone inert; it still makes its own emotions at
moments" -- mouth first, then brows, on his order.**

*"Not as clear" is not blur.* Laplacian sharpness of the rendered face on
the same natural.mkv frames, day 8 -> day 9: eyes region 101 -> 159, mouth
92 -> 99, whole face 160 -> 173 (higher is crisper). What he saw is the
articulation, below.

**Mouth: speech is motion, not distance from rest.** The distance-only gate,
replayed on his own tracked readings over natural.mkv's 594 talking frames:
shut on **20%** of them, half-open on 11%, **49 open/shut flips in 20 s**
(closed runs of ~3 frames) -- every closed-lip consonant and every pursed
vowel shut it and the release dragged the lips back. Fix: a second way to
open, *the mouth is moving* -- per frame |dMAR|/0.004 + |dwidth|/1.5%
(either direction), soft-gated 0.75..1.25, and a frame that reads as
moving holds the gate fully open for 12 frames (0.4 s)
(`_mouth_activity`, `_MOUTH_HOLD_FRAMES`). Units from the filtered
tracker's own frame-to-frame reading (his clips): still or gazing up,
|dMAR| p99 0.0004-0.0015 and |dW| p99 0.4-0.6%; talking, |dMAR| p50 0.004
(call) to 0.010 (scripted). Lip thickness was tried and dropped (moves 2% a
frame on a tilt, mouth shut). Result, the gate as actually applied (now
logged by `measure_follow.py` as `gate_mouth`/`gate_brow`): natural talk
**shut 8% / open 92% / 1 flip** (from 20% / 49); driving talk open 100%;
still and gaze-up **shut 100%**; screen-still shut 98%. Nothing invented
came back: gaze-up avatar mouth max 0.006, still 0.004, screen-still
0.003, still-again 0.003 (all as before). Following: natural talk mar gain
0.96 (same), driving talk 0.73 -> 0.74, mouth-wide 0.58 -> 0.59, closed
smile 1.33, natural smile width 1.27 -> 1.43; his pursed "o" at frame 1000
(mar 0.009): avatar 0.011 -> 0.019-0.026 (day 8: 0.137) -- lip motion
inside speech passes again. Known: a head turn's own width jitter opens
the gate on 69% of turning frames (he also talked while turning, so the
harm there is unmeasured).

**Brows: driven from his own measured brow height, raises only.**
LivePortrait has no brow network, so one was built by hand in keypoint
space (`_brow_lift`, `brow_follow`, `_BROW_LIFT_Y`, the `brow_delta` slot
added to the driving keypoints in `_drive`). Calibrated on the portrait
itself by nudging each brow keypoint in y and reading the avatar's brow
measure back (table in the engine): keypoint 2 is the brow (-0.025 ->
+29%, -0.05 -> +69%, both sides even, eyes and mouth untouched); 3 widens
the mouth 6.6% for the same raise; 1 is not monotone and moves the eyes.
So keypoint 2 alone, -0.12 of y per unit of his relative raise.

*Why raises only.* Nine brow measures were tried (brows to the eye
corners, the nose bridge, the glabella; over the chin, the nose length,
inter-ocular): **on every one, his gaze turned up with the head still
(driving 469-525) reads -9..-17% of rest, and his real frown -8..-12%.**
The two cannot be told apart, and lowering the avatar's brows on that
number would frown it every time he looks up -- the day-9 complaint
itself. A frown is therefore not carried. On the raise side the confounds
are all under 10%: eyes shut +5..+10% (the corner points ride with the
lids), a blink +4..+9%, a closed smile +5%, a turn +8%, a tilt +6%, talking
+6% (scripted) to +13% (call posture); a real raise +33..+38%. Dead band
10%. Faded past |pitch| 10-20 and |yaw| 15-30 as the gate was.

*Measured, `brow_follow` 1.0 (`driving2-follow-day9d.csv`,
`natural-follow-day9d.csv`), against the brows-held build:*

| | his brow (rel. to rest) | brows held | driven, raises only |
|---|---|---|---|
| driving raise 1678-1708, p50 | +37.7% | -3.0% | **+34.1%** |
| natural raise 1570-1600, p50 | +33.8% | -3.1% | **+17.0%** |
| driving brows segment, gain / r | | -0.02 / -0.21 | **0.57 / 0.89** |
| natural brows segment, gain / r | | -0.05 / -0.54 | **0.54 / 0.96** |
| driving frown 1720-1756, p50 | -12.3% | -1.8% | -1.8% (not carried) |
| gaze up 469-525, avatar brow p10 | (his -11.6%) | -2.8% | **-2.9%** (a symmetric drive: -7.8%) |
| still / screen-still, avatar brow spread | | 0.174..0.180 / -5..+2% | 0.174..0.180 / -6..+1% (no added jitter) |

Every other line unchanged (gaze-up mouth max 0.005, still mouth 0.004,
blink ear gain 0.79, talk mar gain 0.73 / 0.97, closed smile 1.33). The
same gesture drives the avatar 90% of the way on the scripted clip and 50%
on the call-posture one -- the keypoint's effect inside a real frame is
weaker than on the still portrait and varies with posture; 1.0 kept rather
than pushed, so the scripted clip does not overshoot. An earlier pass with
a symmetric 3% band (`*-day9c.csv`) carried frowns at -8% but also dipped
the brows -7.8% on the gaze-up -- the table's last column is why it is
raises only.

Kept options added: `brow_follow 1.0` (`brow_gain` stays 0.0 -- the
extractor's own brow reading is still not used). Tests 855 -> 866 (the
whole suite runs green in one go from Git Bash with
`QT_QPA_PLATFORM=offscreen`). ruff clean, mypy the same pre-existing error.

**Not done:** the "own emotions at moments" line of his verdict -- he has
not yet said *which* moments; candidates listed in the chat answer (the
1.3 headroom's wide-eyed look on a gaze up; the nine ungated keypoints;
the extractor's overshoot on a genuine small opening, frame 1380 0.29 ->
0.40). Eyes: what is missing since day 8 is gaze (the eyeballs always look
straight) -- an experiment was proposed, not built: let the extractor's eye
reading back in at 0.3-0.5 only while his lids are at rest. Frowns (above).
Turns opening the mouth gate (above).

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day31-driving.mp4` + stills `day31-driving-f{60,469,520,700,
830,940,1250,1690,1740}.png` (1690 = his brow raise, 1740 = his frown),
`recordings/day31-natural.mp4` + `day31-natural-f{300,560,1000,1380,1585}.png`
(1585 = his brow raise). The `day30-*` renders of the morning's build were
deleted, superseded. CSVs (geometry only): `*-day9b.csv` (activity gate),
`*-day9c.csv` (symmetric brow drive), `*-day9d.csv` (kept build).

**Day 9, evening (2026-09-06, Fable 5.1, on his third live verdict of the
day: "better, but: (1) the face *bounces* on the slightest sideways shift
of my head, or while I talk; (2) a slight smile spreads the mouth a lot;
the eyes are slightly too big; it smiles to one side, like a cynical
smile").**

*Bounce, measured before touching anything* (`pose_jitter_probe.py`,
driving.mkv, frame-to-frame |change| p50/p90): the extractor's angles are
*not* it (pitch 0.04/0.14 degrees sitting still). The crop's **size**
jumps 1.5/6.3 px (of ~700) while he talks and 1.2/3.3 on a turn -- it is
sized by MediaPipe's head-pose distance z, which reads 0.02..0.08 units
differently each frame as his mouth moves; his own inter-ocular distance
moves 0.18/0.79 px. On top, the extractor's own *scale* reads 0.1%/1.1%
differently each frame even sitting still, and the avatar's inter-ocular
distance jittered 0.18/1.20 px against his 0.10/0.32 on the still segment
-- a face breathing in size for no reason. Two fixes: (a) the crop's size
through a One Euro filter (`crop_smoothing` 1.0 Hz, `crop_smoothing_beta`
0.02; `_OneEuro`, `_crop_geometry`/`_similarity` split out of
`crop_matrix`, `_smooth_crop`); (b) the extractor's scale and translation
smoothed on their own (`size_smoothing` 0.85, same `_ema_`, expression and
pose untouched). The crop's *centre* was filtered too at first and that
was withdrawn (`crop_centre_smoothing` 0.0): the size flicker went but the
position against his own jittered more on the still segments (natural.mkv
camera-still dx p90 0.63 -> 0.98 px) -- a One Euro's speed estimate hunts.
Also the mouth gate: a turn's own width jitter no longer counts as motion
(`_WIDTH_PER_DEGREE_YAW`) and the gate opens over two frames instead of
one (`_GATE_ATTACK`), so the mouth region does not jump into place.

*Smile / side smile, measured* (`mouth_tilt` added to `measure_follow.py`:
the mouth corners' height difference over inter-ocular; `mouth_axis_probe.py`):
his closed smile tilts his mouth +0.018 of inter-ocular at the median (his
rest +0.014); the avatar's moved **twice as far as his (slope 1.96)** and,
talking on natural.mkv, wandered -0.028..+0.025 with r 0.43 against his --
not his asymmetry, the extractor's. Nudging each mouth keypoint on each
axis on the portrait: the **x of keypoints 14, 17, 20 and the y of 7 tilt
the mouth** (+-0.05 on 17's x: tilt 0.06-0.12, width 2-6%) and carry
nothing else of use; the **z of 17, 19, 20 is the width** (20's z: +32% /
-58%); the y of 19/20 the opening; 14's y the corners' height, symmetric.
So `mouth_lateral_gain` (default 0.0) cuts the four tilt components and
`mouth_width_gain` (0.7 kept) weights the width -- his +15% at the top of
a closed smile had read as +26%. With the tilt components cut the avatar's
tilt *spread* while talking equals his (35 vs 35, 37 vs 37 on the smile;
was 53 and 40) but a constant offset remains -- the photograph's own mouth
sits +0.020 higher on one side and this face's generator adds +0.014 on a
symmetric smile -- so `mouth_tilt_trim` (0.005 on keypoint 17's x, about
-0.012 of tilt) takes most of it out.

*Eyes*: `eye_headroom` 1.3 -> 1.1 and a new `eye_scale` 0.95 (the target at
his rest is 95% of the portrait's open eyes -- the portrait's 0.334
against his 0.314..0.320). Still-segment avatar eyes 0.316..0.331 ->
0.306..0.324; his 0.314..0.320. Blink gain 0.79 -> 0.79.

*Measured, kept build (`driving2-follow-day9g.csv`, `natural-follow-day9g.csv`
plus the `-day9h` smile-segment runs) against the morning's
(`*-day9d.csv`); frame-to-frame |change|, p50/p90:*

| bounce | morning | evening | him |
|---|---|---|---|
| driving still: avatar inter-ocular, px | 0.18 / 1.20 | **0.18 / 0.50** | 0.10 / 0.32 |
| driving still: avatar-minus-him dy nose, px | 0.25 / 1.58 | **0.20 / 0.89** | |
| driving still-again: inter-ocular / dy nose | 0.15 / 1.75, 0.16 / 2.77 | **0.17 / 0.54, 0.27 / 0.79** | 0.08 / 0.30 |
| driving talk: avatar inter-ocular | 0.25 / 0.93 | **0.23 / 0.77** | 0.18 / 0.79 |
| natural camera-still: inter-ocular / dy nose | 0.19 / 1.36, 0.20 / 2.73 | **0.14 / 0.77, 0.21 / 1.18** | 0.14 / 0.63 |
| natural talk / smile: avatar inter-ocular | 0.44 / 1.70, 0.54 / 1.91 | **0.29 / 1.03, 0.44 / 1.23** | 0.31 / 1.25, 0.38 / 1.29 |
| scale (avatar/him) still, % | 0.10 / 0.67 | **0.07 / 0.29** | |

The avatar's size now moves frame to frame about as much as his own face
does; sitting still its position jitters ~40% less in y. A turn's dx is
unchanged (0.76 / 3.2-3.6 px -- the unfiltered centre following a moving
head, not a bounce).

| side smile: mouth tilt, x1000 of inter-ocular, p10 / p50 / p90 | him | morning | evening |
|---|---|---|---|
| driving closed smile | +12 / +18 / +27 | +5 / +17 / +36 | **+8 / +13 / +24** |
| driving talk | +21 / +31 / +45 | -- | **+1 / +24 / +34** |
| natural talk | -14 / +3 / +21 | -28 / +6 / +25 | **-9 / +5 / +24** |
| natural smile | -10 / +7 / +28 | -22 / -1 / +18 | **-4 / +8 / +30** |
| at rest | +14 (driving), +10 (natural) | +20, +18 | **+7, +6** |

The avatar's mouth now tilts as much and as often as his, no more.

*Eyes*: still-segment avatar 0.316..0.331 -> **0.306..0.322** (his
0.314..0.320); gaze-up 0.316..0.372 -> 0.308..0.334; blink gain 0.79 kept.
Natural frame 300 (his eyes wide, 0.376): 0.358 -> 0.333.

*Smile width -- not fixed, and two things tried are on record.* His closed
smile widens his mouth +8% at the median and +15% at its widest; the
avatar's +8% and **+26%** -- the median matches, the peak over-spreads --
and that number did not move: weighting the z of 17/19/20 at 0.7 made it
+29% (those components *narrow* during a smile), weighting keypoint 14's y
at 0.7 left it at +26% and cost the toothy smile (+27% -> +21%, his +33%).
`mouth_smile_gain` stays 1.0. On natural.mkv his smile is an open-mouth
laugh whose width hardly changes (-1..+2%), and the avatar's narrows (-10%)
-- there the complaint cannot be about width. What his eye called
"spreads the mouth a lot" was, on the numbers, most likely the tilt, which
is gone; if the peak spread still reads wrong live, the next lever is a
soft cap on the *rendered* width (measure the avatar's own mouth width and
pull keypoint 14 back), not a gain.

*Unchanged, checked*: gaze-up avatar mouth max 0.005, still mouth 0.003,
screen-still 0.003; talk mar gain driving 0.73 -> 0.77, natural 0.97 ->
0.87 (r 0.94); mouth-wide 0.58 -> 0.66; brow raise +34% / +19%; the mouth
gate on turns 69% -> 64% open, 6 -> 4 flips.

Kept options added: `eye_scale 0.95`, `eye_headroom 1.1`, `mouth_lateral_gain
0.0`, `mouth_tilt_trim 0.005`, `mouth_smile_gain 1.0`, `size_smoothing
0.85`, `crop_smoothing 1.0`, `crop_centre_smoothing 0.0`,
`crop_smoothing_beta 0.02`. Tests 866 -> 883. ruff clean, mypy the same one
pre-existing error.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day32-driving.mp4` + stills `day32-driving-f{60,469,700,830,
940,1250,1690}.png`, `recordings/day32-natural.mp4` + `day32-natural-f{300,
1000,1380,1585}.png`. The `day31-*` renders were deleted, superseded. CSVs
(geometry only): `*-day9e-smile.csv` (tilt measured, before), `*-day9f.csv`
(centre filtered too), `*-day9g.csv` (kept build, full), `*-day9h.csv`
(smile gain 0.7 on 14's y, smile segments).

**Day 9, night (2026-09-06, 23:21 -- he pressed "That looked wrong" twice
on the evening build: "a catastrophe").** The two captures
(`captures/moment-20260905-232130`, `-232151`: the frame on screen and the
last 300 tracked frames, no camera imagery) read as follows. 23:21:30 --
he is in a **closed-lip grin** (MediaPipe smile blendshapes 0.58-0.79,
mouth ratio 0.004) and the avatar shows an **open, toothy grin**, eyes
wide. 23:21:51 -- he is **at rest** (no smile, ratio 0.005, eyes 0.34) and
the avatar's **lips are parted**, staring. Two causes, both measured from
the tracking he saved:

1. **The width gate opened the whole mouth.** A smile's width opened the
   gate on every mouth component, and the extractor's reading of a strong
   smile carries parted lips with it (the clips' smiles were milder). Now
   the lips' opening -- the y of keypoints 17, 19, 20 (`_MOUTH_OPENING`,
   from the axis probe: +-0.05 on 19/20's y is a mouth ratio of 0.24-0.27)
   -- has **its own gate that knows only the ratio**: how far his ratio is
   from rest, and whether *it* is moving. A smile widens the avatar's
   mouth; only his own lips parting parts the avatar's. `_gates` returns
   `(mouth, opening, brow)`; `measure_follow.py` logs `gate_open`.
2. **His live tracking is 3-7x noisier than the clips.** With his mouth
   shut, |dMAR| per frame read 0.0027 (p90) / 0.011 (p99) and |dW| 0.7% /
   2.0% at 22 fps -- the clips' worst was 0.0015 / 0.6%. At the motion
   units tuned on the clips (0.004 / 1.5%), **17% of his shut-mouth frames
   read as "moving" and 9% set the hold** -- the parted lips at 23:21:51.
   Units now 0.010 / 2.5% (his live p99 is 1.1 / 0.8 units: a one-frame
   partial open at worst, no hold; a talking frame at the call clip's p90
   |dMAR| 0.019 is still 1.9 units) and the motion is measured per 30-fps
   frame (`last_dt` from the capture timestamps -- at 22 fps a frame sees
   1.4x the change).

*A wrong turn, on record.* His face is 425-475 px tall in the captures, so
the 2.2-height crop (~950 px, shifted up an eighth) reaches ~260 px above
the 720 px frame: **28% of the extractor's input is a black band.** That
looked like the live/offline difference -- until the clips were measured:
he sits at the same distance in all of them (inter-ocular 196-244 px), so
every number in this file was taken *with* that band, and the extractor
copes. Tried anyway on a 1.9x zoomed copy of natural.mkv (`natural-zoom.mkv`,
inter-ocular 408 px, 69% of the crop outside the frame): replicated edges
were far *worse* than black (talk mar gain 1.12 -> 0.31; frame 1380 mouth
0.185 -> 0.017) -- smeared edge pixels read as a face, black reads as
nothing. `crop_border` exists ("black" default, "replicate", "reflect")
for measuring, not for use. The honest limit stands: at 720p the network
sees its whole training context only for a face under ~330 px (inter-
ocular under ~160 px); at his distance it sees ~72% of it. A wider camera
framing, or sitting a little further back, would give it the rest.

*Measured (`*-day9i.csv` against the evening's `*-day9g.csv`):*

| | evening | night |
|---|---|---|
| driving closed smile: **opening gate** (lips) | (one gate) | **shut 100%** -- the lips never part |
| driving closed smile: avatar mouth ratio p90 / his | 0.006 / 0.005 | 0.010 / 0.005 (shut) |
| driving closed smile: width gain, avatar p90 (his 0.634) | 1.63, 0.708 | **1.25, 0.670** |
| driving toothy smile: width gain | 0.22 | 0.35 |
| driving talk: mar gain / opening gate | 0.77 / -- | 0.75 / open 88%, shut 5% |
| natural talk: mar gain (r) | 0.87 (0.94) | 0.89 (0.93) |
| natural smile (a laugh): mar gain / opening gate | 0.94 | 1.00 / open 100% |
| natural screen-still, tilt: mouth gate shut | 99%, 93% (2 flips each) | **100%, 100%** (0 flips) |
| gaze-up avatar mouth max, still mouth p90 | 0.005, 0.003 | 0.005, 0.003 |
| driving turn: mouth gate open | 64% | 40% |
| blink ear gain | 0.79 | 0.80 |

The closed smile's over-spread came down with it (the opening components
add width too: 17's y at +0.05 is +13% width) -- +29% -> +24% at the peak
of his +15%. Tests 883 -> 887. ruff clean, mypy the same one pre-existing
error. **What cannot be checked here:** his live jitter itself -- the
clips are quieter than his live tracking, so the thresholds were set from
the two captures, and only a live test says whether 0.010 / 2.5% is
enough. If the lips still part at rest, the next capture's `tracking.npz`
gives the number directly (the |dMAR| p99 of shut-mouth frames).

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day33-driving.mp4` + stills `day33-driving-f{60,469,700,830,
940,1250,1690}.png`, `recordings/day33-natural.mp4` + `day33-natural-f{300,
1000,1380,1585}.png`. The `day32-*` renders were deleted, superseded. Also
his own `captures/moment-20260905-232130`, `-232151` (tracking + the frame
on screen) -- his to delete. `recordings/natural-zoom.mkv` (his face,
zoomed) kept for the border question only; CSVs `natural-zoom-follow-black
/-replicate.csv`, `*-day9i.csv`.

**Day 10 opens with his fifth live verdict (2026-09-06, on the night
build), verbatim, and a request to continue in a fresh chat.** His words:
*"The eyes are now more shut. When I look to the side, the eyes don't
follow. Blinking isn't as natural as it was yesterday. Yesterday's version
had more flexible expressions and I wanted a balance of that -- it isn't
working yet. When I look down it takes on a terrible, foreign face; I tilt
my head slightly down and it is already a completely different face. This
is a big problem."* Nothing was changed in response; the next session
starts here. Hypotheses, each with the measurement that settles it:

1. *Eyes more shut* -- the evening's `eye_scale 0.95` and `eye_headroom
   1.3 -> 1.1`. "Yesterday" is the morning build (scale 1.0, headroom 1.3),
   which he found "slightly too big" -- the middle is `eye_scale 1.0,
   eye_headroom 1.15-1.2`. Measure: still-segment avatar eye ratio against
   his 0.314..0.320 (morning 0.316..0.331, evening 0.306..0.322).
2. *Eyes don't follow a sideways glance* -- gaze. Since day 8 the eye
   keypoints take nothing from the extractor (`eye_expression_gain 0.0`),
   and the eye network only opens and shuts; the eyeballs always look
   straight. Two routes: (a) a hand-built gaze drive like the brows' --
   MediaPipe's iris points (468-477) give his gaze offset directly; the
   nudge probe (`mouth_axis_probe.py` pattern, on EYE_KEYPOINTS' x/z) says
   which keypoint component moves the irises; (b) the extractor's eye
   reading let back in at 0.3-0.5 *only while his lids are at rest*
   (|ear/rest - 1| < 0.1), so the day-8 half-shut leak on a gaze-up cannot
   return. Measure on driving.mkv 120-420 (gaze L/R): avatar iris offset
   against his; and 421-538 must stay at eyes >= 0.31.
3. *Blinks less natural* -- `eye_gain` 0.25 is unchanged since the morning;
   blink ear gain measured 0.79-0.80 all day. Suspects: the smaller open
   state (1) making the same blink look smaller; the size/crop smoothing
   (`size_smoothing`, `crop_smoothing`) changing the eye region's timing.
   Measure: blink segment 539-658, avatar ear per frame, morning vs night
   CSVs (`driving2-follow-day9d.csv` vs `-day9i.csv`) -- the shape of the
   blink, not just its gain.
4. *Looking down / a slight tilt down = a foreign face* -- the largest.
   The hold only engages at |pitch| > 20 degrees; between -5 and -20 the
   frontal portrait is re-drawn pitched down, and a downward gaze lowers
   his lids, so the eye network shuts the avatar's eyes on top -- brows
   held, lips gated: a dead, tilted face. Candidates: (a) a *pitch gain*
   < 1 on the relative rotation (the avatar pitches, say, 60% of his --
   the portrait cannot render steep pitch well anyway); (b) hold earlier
   (`hold_pitch_degrees` 20 -> 10-12) with a blend instead of a freeze;
   (c) the crop's vy shift under pitch (day-6 cliff -- check the framing
   at pitch -10..-20). Measure on natural.mkv 417-712 (tilt down to -32):
   render stills at pitch -5 / -10 / -15 / -20 and read the avatar's eye
   and mouth ratios against his; and ask him to press "That looked wrong"
   on the next such moment -- `tracking.npz` gives his exact pitch/gaze.
5. *"More flexible yesterday; I want a balance"* -- the gates and the
   brow/lateral cuts each removed invented motion and some real motion
   with it. Worth a single `expression_gain`-style knob he can feel live
   (mouth gate floor, brow follow, eye headroom together), and a live A/B
   with him rather than more clip measurement -- the clips are quieter
   than his live tracking (night entry) and have not predicted his
   verdicts well twice now.

The kept options at the close of day 9 are exactly
`avatars/my_avatar_neural/metadata.json`; the measurement tools are
`scripts/measure_follow.py` (`--option`, `--last`, gate columns) and
`scripts/render_liveportrait.py` (`--option`, `--stills`); his own captures
are the ground truth for live behaviour.

**Day 10, analysis of (4) -- the pitch-down "foreign face" -- from the
night build's own CSV, no new render (2026-09-06, Fable 5.1; the
implementation is Sonnet 5's, instruction below).** `natural-follow-day9i.csv`,
segment 417-712 (the "tilt slightly down" instruction: in fact ONE tilt,
492 -> 684, a 4 s plateau at -32 degrees in the hold; his eyes are shut
0.05-0.10 on the descent 477-495 -- a blink as he starts -- and open
0.29-0.39 on the return 669-693, so the RETURN frames are the clean ones
for stills). Avatar-minus-him pixels, binned by his MediaPipe pitch:

| his pitch | n | his EAR | avatar EAR | dy forehead | dy eye | dy mouth | dy chin | scale (io) |
|-----------|---|---------|------------|-------------|--------|----------|---------|------------|
| -5..0     | 84 | 0.295 | 0.300 | +4  | -5  | -5  | -5 | 1.05 |
| -10..-5   | 10 | 0.289 | 0.283 | +28 | +15 | +7  | 0  | 1.02 |
| -15..-10  | 13 | 0.116 | 0.272 | +43 | +29 | +24 | +3 | 0.98 |
| -20..-15  | 21 | 0.134 | 0.305 | +55 | +38 | +39 | -1 | 0.95 |
| < -20 (held) | 158 | 0.146 | 0.305 | +57 | +50 | +52 | +20 | 0.98 |

Per frame on the return: -17.9 deg dy_eye +46 / dy_chin -1; -12.6: +28 / +3;
-6.9: +16 / +2; -3.4: +2 / -4. Read: at rest every feature sits within 5 px
of his; from -5 degrees on, the avatar's forehead, eyes and mouth slide
DOWN 2-3 px per degree faster than his while the chin stays -- a face
compressed from the top, the mouth pushed toward the chin, and 7-10%
narrower (scale 0.95-0.98 against 1.05-1.08 at rest; his own io GROWS
220 -> 237 px because he leans in as he tilts, the avatar's does not
follow, 231 -> 228). That is his "completely different face" at a slight
tilt. Two things the table rules out: the eyes are NOT shut by the eye net
on a look-down (avatar 0.27-0.31 while his read 0.10-0.15; hypothesis (4)
in the block above was wrong about that), and the gates are all 0 here, so
no expression is being invented -- this is a rigid-motion error.

Three candidate mechanisms, not yet separated (each is a different lever):
- **A. over-rotation**: the rendered pitch is larger than his (the
  relative rotation `rot_d @ R_d0^T` applied to `x_c_s`); would move the
  forehead down AND the chin up, the chin then hidden under his own oval
  mask (dy_chin ~0 could be that artefact).
- **B. a translation/scale leak**: the extractor's `t` reads a pitched face
  as "lower" and its `scale` as "smaller" (its relative t/scale are added
  to the portrait's); would slide the whole face down and shrink it.
- **C. the crop**: `_driver_face_height` (rest height x z_rest/z) not
  following his lean-in under pitch; would leave the avatar too small.

**Day 10, instruction for Sonnet 5 -- (4) pitch-down, in this order; stop
and report at any step whose numbers contradict the step before.**

*Step 1 -- separate A/B/C on natural.mkv 480-700, no engine change.*
1a. `scripts/measure_follow.py` (the output tracker block, ~line 295-320,
    `got` is the tracker on the avatar's own frame): add columns
    `o_yaw`/`o_pitch`/`o_roll` from `got.head_pose` (same instrument on
    both faces -- the only apples-to-apples pitch comparison), and
    `crop_px` = the crop's size in frame pixels this frame, read from the
    engine's `portrait.last[1]` (`to_frame`, 2x3: size = 512 x the norm of
    its first column). Keep every existing column. Run:
    `uv run python scripts/measure_follow.py --clip recordings/natural.mkv
    --last 712 --csv recordings/natural-follow-day10-base.csv` (~3 min).
    Report over frames 480-700 with his pitch in [-20, -3] and not held:
    slope of `o_pitch` on `pitch` (least squares, and the ratio at -7 and
    -12.6); `crop_px / d_io` at rest (417-470) vs at -12.6 and -17.9 --
    if it drops more than 4%, that is C.
1b. Copy the day-7 probe
    `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-desktop-sha256/afec2b6f-5311-4591-bd3c-05eba767cb36/scratchpad/pitch_leak_probe.py`
    into this session's scratchpad (it already runs the motion extractor
    eagerly on `loaded.static_in` after the anchor and prints MediaPipe
    pitch, extractor pitch, scale/scale_d0 per frame). Add the extractor's
    relative `t` (`t - t_d0`, y component, in crop units x 512 for pixels)
    and run it on `recordings/natural.mkv 480 700 3`. Report, against his
    MediaPipe pitch: slope of the extractor's relative pitch (day 7 read
    -0.9 on this clip -- confirm), of relative t.y (pixels per degree), of
    relative scale (% per degree). t.y or scale moving with pitch = B.
1c. Stills of the night build for the before picture:
    `uv run python scripts/render_liveportrait.py --clip recordings/natural.mkv
    --first 400 --last 712 --no-warp --out recordings/day34-tilt-base.mp4
    --stills 450,687,681,675,506,500` (450 rest; return frames 687/681/675
    = -6.9/-12.6/-17.9 with his eyes open; 500/506 descent, lids lowered).
    His face -- gitignored, announce, delete after he looks.

*Step 2 -- the levers, all in `_drive` (engine_liveportrait.py ~2498-2521),
each a float option baked at capture like `eye_gain`, each defaulting to
today's behaviour exactly.* Parse them with `_eye_scale`-style helpers that
raise `ValueError` out of range; document them in the module's option list
and add them to `avatars/my_avatar_neural/metadata.json` at their defaults
until the sweep picks values.
- `pitch_gain` (default 1.0, allowed [0, 2]) and `pitch_cap_degrees`
  (default 0.0 = no cap, allowed >= 0): on the extractor's RELATIVE pitch.
  Needs a new static slot `pose_d0` (3, 1) beside `R_d0` (slot list ~1547,
  init ~1692): primed in `load_avatar` from the same raw-pose stack that
  primes `pose_prev` (~1914) -- so `R_d0 == R(pose_d0)` holds from the
  first frame -- and copied in `_learn_anchor` where `pose_prev` is
  (~2438, `sums["pose"] / n` -- exactly what `R_d0` is built from at
  ~2424). `reset_anchor` leaves it alone, as it leaves `R_d0`. Then,
  replacing the `rot_d = get_rotation_matrix(pitch_s, yaw_s, roll_s)` line:
  ```
  rel = pose_now - loaded.pose_d0                      # degrees, (3, 1)
  rel_pitch = rel[0]
  if cap > 0:  rel_pitch = cap * torch.tanh(gain * rel_pitch / cap)
  else:        rel_pitch = gain * rel_pitch
  used = loaded.pose_d0 + torch.stack([rel_pitch, rel[1], rel[2]])
  rot_d = get_rotation_matrix(*used.unbind(0))
  ```
  Keep the old line on the Python-level branch `gain == 1.0 and cap == 0.0`
  so the default stays bit-exact (a float round trip through pose_d0 is
  not). The gain is sign-agnostic, so the extractor's inverted pitch
  convention (day 7) does not matter; the cap is in the extractor's own
  degrees. Yaw and roll untouched (roll is taken out by the crop; yaw is
  the separate turn-lag item).
- `translation_gain` (default 1.0, [0, 1]): `t_new = t_s + g * (t_now -
  t_d0)`. At 0 the avatar sits exactly where the crop (his rigid points)
  puts it.
- `scale_gain` (default 1.0, [0, 1]): `scale_new = scale_s *
  (scale_now / scale_d0) ** g`. At 0 the size is the crop's alone (which
  already follows his distance through z and is One-Euro smoothed).
Which of the three to sweep is decided by Step 1: A -> pitch_gain/cap;
B -> translation_gain and/or scale_gain; C -> stop and report (a crop
change; the day-6 cliff is 5% away and needs its own instruction).

*Step 3 -- sweep, natural.mkv only until a candidate exists.* Stills first
(`--option KEY=VALUE`, same six frames): for A, pitch_gain 0.7 / 0.5 /
0.35 with cap 0, then the best gain with `pitch_cap_degrees` 10; for B,
translation_gain 0.5 / 0 and scale_gain 0.5 / 0. Then measure_follow
`--last 712` with the candidate. Numbers that decide, on the same bins as
the table above (his pitch -10..-5 and -15..-10, return frames):

| measure | now | target |
|---|---|---|
| dy_eye - dy_chin at -7 / -12.6 deg | +15 / +25 px | <= 8 / <= 12 px |
| scale (io) at -15..-10 | 0.98 (rest 1.05) | >= 1.00 |
| o_pitch / pitch slope over [-20, -3] | (Step 1a) | 0.5-0.8 -- follows, gently |
| still segment 119-416 avatar EAR | 0.30-0.32 | unchanged +-0.005 |
| talk 713-1306 mar gain | 0.87 | >= 0.85 |

Then ONE full run of `driving.mkv` on the final candidate: blink ear gain
>= 0.75, turn lag no worse than -33/+21 px, toothy-smile width unchanged.
A candidate that clears the pitch lines and breaks a must-not-break line
is reported, not kept.

*Step 4 -- keep, prove, record.* Write the kept values into
`my_avatar_neural/metadata.json`; unit tests (the file's own style, fake
nets on CPU, see `_engine_and_loaded` ~492): the relative-pitch helper
(gain 1/cap 0 identity; gain 0.5 halves; cap 10 with rel 40 stays under
10 and keeps the sign), the three option validators (refused at first
drive, as `test_an_invalid_value_is_refused_at_first_drive_call`),
`pose_d0` primed at load and equal to `pose_prev` after `_learn_anchor`,
and `reset_anchor` leaving it. `uv run pytest tests/unit -q` (823 pass
today), `uv run mypy src` (12 pre-existing errors, no new), `uv run ruff
check`. Write "Day 10, done" here with the before/after table, the stills
kept for him (`recordings/day34-tilt-*`, his face, delete after), and
what was NOT fixed. Nothing is committed unless he asks.

*Not in this instruction:* items (1) eyes, (2) gaze, (3) blink shape and
(5) the balance knob from the block above -- (4) first, alone, so its
before/after is clean. One observation for (2): while he looks down his
lids read 0.10-0.15 and the avatar's stay 0.27-0.31 -- on a call the
avatar keeps staring ahead when he looks at the keyboard; whether that is
wanted is his call, not a bug to fix here.

**Day 10, Step 1 done (2026-09-06, Sonnet 5) -- stopped at the instruction's
own C-branch; Steps 2-4 (the pitch_gain/translation_gain/scale_gain levers)
NOT built.** Measured on `natural.mkv`, no engine code changed --
`scripts/measure_follow.py` gained three diagnostic columns used below
(`o_yaw`/`o_pitch`/`o_roll`, the avatar's own pose read by the same
MediaPipe instrument as his; `crop_px`, the crop's edge length in source-
frame pixels; `held`, whether the frame is the extreme-pose hold's frozen
reuse) -- kept, they cost nothing at `motion_smoothing`-style defaults and
the next crop instruction will want them too.

**Correction to Step 1's own request first: "not held" needs the engine's
own flag, not a pitch cutoff.** The hold (`_should_hold`) has hysteresis --
enters above 20 degrees, releases only below 15 -- so on the RETURN
(669-684) frames at -15 to -20 degrees are still held from the descent,
while the same pitch on the way DOWN (504-513) is not yet held. A pitch-
range filter alone pools six genuinely held frames into the "-20..-3,
should be live" bucket; `held` (now a column) is exact. All numbers below
exclude held frames.

*1a/1b together -- A (over-rotation) does not hold up; C (the crop) does.*

| measure | at rest | -6.9 deg | -12.6 deg | -17.9 deg\* |
|---|---|---|---|---|
| avatar's own pitch (`o_pitch`, MediaPipe on the output) | -3.7 (his ~0) | -10.5 (his -6.9) | -16.5 (his -12.6) | -23.7 (his -17.9, held) |
| excess (`o_pitch` - his pitch) | -3.7 | -3.6 | -3.9 | -5.8 (held, not comparable) |
| crop_px / his own inter-ocular distance | 4.480 | 4.429 (-1.1%) | 4.331 (-3.3%) | 4.288 (-4.3%, held) |
| avatar io / his io (`scale`) | 1.049 | 1.019 (-2.9%) | 1.014 (-3.4%) | 0.999 (-4.8%, held) |

\* -17.9 deg is inside the hold on this pass of the clip (see the box
above) -- the crop and mask are still fresh every frame even while held
(`animate`'s own comment), so `crop_px`/`scale` there are real; `o_pitch`
is not (it is the frozen face's own pose, not this frame's).

Read: the "excess" pitch is **~3.6-3.9 degrees, present already near rest
and not growing with |pitch|** over the whole unheld range (confirmed on
segments 0-118 and 119-416 too: avatar reads -3.7/-5.2 degrees while he
reads -0.06/-1.50 -- essentially the same constant, nowhere near his own
motion). A true over-rotation (gain > 1 on the *relative* pitch, which is
zero at rest by construction) would be **zero at rest and grow with
|pitch|** -- the opposite of what is measured. `pitch_gain`/
`pitch_cap_degrees`, as specified in Step 2, act only on the relative
term (`pose_now - pose_d0`); they cannot touch a bias that is already
present at zero relative pitch, so building them would not address this.

The day-7 probe, re-run on this clip with the held frames now correctly
excluded (they were re-reading one frozen crop and had flattened the
original day-7 numbers): the extractor's own pitch reads **-0.78x** his
MediaPipe pitch over the unheld descent (sign-flipped, as day 7 found;
0.78 here vs 0.9-1.7 reported on other segments/clips -- consistent
family, same convention, not identical -- expected, different clip and
range) -- **under 1:1, not over**. Its relative translation barely moves
with pitch (r -0.22, ~0.18 px/deg) and its relative scale, if anything,
grows slightly with pitch (r +0.24, +0.05%/deg) -- the wrong sign to
explain a shrink. Neither supports B as the mechanism either.

**C, by contrast, is clean and consistent both ways.** `crop_px` (how wide
a source-frame region `_driver_face_height` cuts) shrinks *relative to his
own true inter-ocular distance* as his pitch grows -- -3.3% at -12.6 deg,
crossing Step 1's own 4% line by -17.9 deg -- and the avatar's *rendered*
size (`scale`, io/io in frame pixels, unaffected by the held/unheld split)
drops by the same amount at the same points (-3.4%/-4.8%). Both numbers
move together because they are the same mechanism seen twice: the crop is
undersized, so what comes back through the same `to_frame` is undersized
too. Independently corroborated: his own tracked `d_io` *grows* 220 -> 247
px over this segment (he leans toward the camera as he tilts down) while
`face_height_rest * z_rest / z` is not keeping pace -- the lean-in, not
just the tilt, is not being fully absorbed by the z-based crop size.

**Stills confirm it visually and point at a second, connected symptom.**
`recordings/day34-tilt-base-f{450,687,681,675}.png` (rest, -6.9, -12.6,
-17.9 degrees; sent to him, his face, delete after he looks): at rest and
-6.9 the two faces track closely; at -12.6 and sharply by -17.9 a straight
horizontal line cuts across the upper forehead, above which the pasted
region stops looking like a face. This is the forehead mask line already
on the open list (day 3) -- not a new defect -- but an undersized crop is
a plausible reason it is far more visible here than at rest: the mask's
own vertical reach (`_OVAL_FOREHEAD_REACH`, `_frame_mask`) is built from
the same crop geometry that is now wrong by several percent, so the
oval's edge and the network's own rendered content stop agreeing on where
the forehead ends. Worth checking first when the crop fix lands, not a
claim -- not measured directly here.

**Decision, per the instruction's own branch:** stopping at Step 1. C
crosses the stated threshold and both A and B are contradicted by their
own measurements (a flat, rest-present offset for A; a weak and wrong-
signed correlation for B). Steps 2-4 as written build lures for A/B;
building them now would spend the day on a lever this data says will not
close the gap, and the crop fix the numbers actually point to is exactly
the one the instruction itself flagged as needing its own turn -- the
day-6 cliff (CROP_SCALE 2.1 grimaces, 2.3 grimaces, 2.2 is the only
frame-count-of-three that renders all three probes) sits close enough
that a `_driver_face_height` change should be swept the same cautious way,
not folded into this instruction's pitch levers.

**State: nothing in `src/` changed.** `scripts/measure_follow.py` gained
the three columns above (diagnostic only, no default-changing behaviour).
`uv run pytest tests/unit -q -m "not gui"`: 826 passed (32 gui-marked
deselected). `uv run pytest tests/unit -q` (everything, unfiltered)
crashes reproducibly partway through the gui-marked tests (exit 127, no
summary line) -- confirmed pre-existing and unrelated to today: the
crashing file passes cleanly in isolation (14/14), and today's only
touched files are this ROADMAP entry and `measure_follow.py`, neither on
pytest's collection path for `tests/`. Flagging for whoever next runs the
full suite with a display/Qt change nearby; not chased further here, out
of this instruction's scope. `ruff check scripts/measure_follow.py`:
clean. mypy does not cover `scripts/` (pyproject's own note).

**Next instruction, proposed:** a `_driver_face_height`/crop-size fix for
the lean-in-under-pitch shortfall, swept as cautiously as day 6's own
cliff (one-frame stills before any full measure run, the same six probe
frames), with the forehead-mask-line stills above as the visual check for
whether it also closes that. Items (1)(2)(3)(5) from "Day 10 opens with"
are still untouched.

**Day 10, Fable 5.1's review of Step 1 (2026-09-06), and the next
instruction.** Sonnet's Step 1 was done carefully and two of its catches
stand for good: (a) the extreme-pose hold freezes `static_in`, so any probe
that re-reads it on a held frame re-samples one stale crop -- the `held`
column is now the only correct "not held" filter, and day 7's own
extractor-pitch slope on this clip (-0.9) was contaminated the same way
(-0.78 unheld); (b) `crop_px`/`o_pitch`/`o_roll`/`o_yaw` are the right
columns and stay. Its conclusion does not stand, for four concrete reasons:

1. **The -3.7 degree "rest offset" is the photograph.** MediaPipe on
   `portrait.png` itself reads pitch **-3.56** (yaw 1.1, roll 0.8). The
   portrait is tilted down 3.6 degrees in the photo; the avatar carries it
   at every pitch. Cosmetic, not the engine, not this defect; if he wants
   the avatar level at rest that is a portrait choice.
2. **C cannot produce the measured shape.** A crop 3% too small is a
   uniform 3% shrink about the crop centre, which sits near the brows
   (`vy -0.125`): the chin would move UP ~10 px and the eyes ~0. The table
   shows eyes +29 and mouth +24 moving DOWN together at -12.6 degrees with
   "chin 0". That is a translation of the whole face -- and the chin's own
   motion is hidden: the paste mask is HIS oval, so a face slid down has
   its chin cut off and the composite's chin is his. `dy_chin ~ 0` is an
   artefact of measuring on the composite, on every day this table has
   been read. C is real and stays on the list (-1.1% at -7, -3.3% at
   -12.6 -- nothing at the "slight tilt" he complains of, a 2.13 effective
   crop scale by -13, i.e. toward day 6's 2.1 frown cliff) but it is the
   secondary item.
3. **Two unit slips.** The keypoint grid is [-1, 1] over the crop, y down
   (`make_coordinate_grid`): 1 t-unit = 256 crop px, not 512. The
   extractor's relative t.y is then +6 / +8 frame px at -7 / -12.6 degrees
   (of the 16 / 29 observed) and reverses sign at -16..-20 while the slide
   keeps growing -- a quarter of the effect, wrong direction at the end;
   B is a contributor, not the mechanism. And `o_pitch` on a face whose
   upper half has slid down is not a clean rotation reading -- the pose
   solver fits scale and translation too -- so "slope 0.99, no
   over-rotation" is weak evidence either way.
4. **The forehead line is the slide, not the crop.** A 3% crop error moves
   the mask edge ~4 px; the line on frame 681 sits where the avatar's
   forehead is ~29 px lower than his: his skin above the edge, the
   portrait's forehead below it, a tone step sharpened by the feather. Fix
   the slide and the line should fall back to its day-3 level -- a
   prediction to check on the same still.

**The candidate nobody has tested: the rotation's pivot.** In `_drive`,
`x_c_s @ rot_new` rotates the canonical keypoints about the keypoint-
space ORIGIN. In LivePortrait's own pipeline the driving video is cropped
once for the whole clip, so when the driver pitches about the neck his
face also slides down/forward in that fixed crop, `t_d` reads the slide
and `t_s + (t_d - t_d0)` re-applies it -- rotation and slide come from
the same head and agree. Our crop re-centres on his rigid points every
frame: his slide is removed on the driver side (`t_now - t_d0 ~ 0`), but
the portrait's keypoints still swing about their origin under the
relative rotation, by (origin-to-face distance) x sin(rel pitch), with
nothing left to cancel it. His face, in the crop, turns about the rigid-
point centroid (eyes/nose); the avatar's turns about a point that is not
on the face. Signature: eyes and mouth move together, forehead a little
more (the cos compression on top) -- exactly the table's shape. The
probe below settles it in one run and calibrates the fix.

**Next instruction (Sonnet 5), in order; stop and report where a number
says so.**

*Step 1 -- the swing probe, no engine change.* `pitch_swing_probe.py` in
the scratchpad: self-driving on the portrait. Rebuild the portrait crop
exactly as `load_avatar` does (`_read_portrait`, `_landmarks_of`,
`crop_matrix(points, scale, vy_ratio)`, `warpAffine`, `_to_tensor`) and
take `x_c_s`, `exp_s`, `scale_s`, `t_s`, `R_s`, `f_s`, `x_s` and
`pose_s = stack(_degrees(info[k]))` from a loaded avatar (all set in
`load_avatar`, ~1884-1915). Render eagerly, outside the graph:
```
def render(d_pitch, pivot):            # degrees, extractor's own convention
    R_rel = get_rotation_matrix(pose_s[0] + d_pitch, pose_s[1], pose_s[2]) \
            @ R_s.permute(0, 2, 1)
    base = x_c_s if pivot is None else x_c_s - pivot
    rotated = base @ R_rel if pivot is None else base @ R_rel + pivot
    x_d = scale_s[..., None] * (rotated @ R_s + exp_s) + t_xy[:, None, :]
    out = nets["decoder"](feature=nets["warping"](f_s, kp_source=x_s.half(),
                                                  kp_driving=x_d.half())["out"])
```
(`t_xy` as in load_avatar: `t_s` with z zeroed.) Assert first that at
`d_pitch = 0`, `x_d` equals `x_s` within 1e-5 -- the portrait must come
back unchanged. Then `d_pitch` in {-15, -10, -5, 0, +5, +10, +15} x
`pivot` in {None, mean of `x_c_s` over all 21 keypoints, mean over
`EYE_KEYPOINTS`}: write each 512 render as
`recordings/day35-swing-<pivot>-<deg>.png` (the portrait, not his face)
and run MediaPipe on it; report per render the eye-centre / mouth /
chin / forehead y (the `measure_follow` anchors) minus the `d_pitch = 0`
render's, and the render's own `head_pose.pitch`. Which sign of
`d_pitch` is "down" is whatever makes MediaPipe's pitch on the render go
negative -- say so in the table.

Decision: with `pivot = None`, at the `d_pitch` whose render reads
MediaPipe ~ -10 degrees: **|dy_eye| >= 8 px in the 512 render** (~16
frame px at crop ~1000; observed 29 at -12.6 with ~8 from t) confirms
the swing as the dominant mechanism. The pivot to build is the one that
leaves dy_eye within +-3 px while the chin still moves (the face turns
instead of sliding). If the eyes do NOT move with `pivot = None`, the
slide is on the driver side after all: build `translation_gain` (Step
2's lever B from the previous instruction, unchanged spec) and sweep
0.5 / 0 on natural 681/687 instead of the pivot; report.

*Step 2 -- `rotation_pivot` option in `_drive` (~2503-2521).* Values
`"none"` (default), `"centroid"`, `"eyes"`; a static `loaded.pivot`
(1, 1, 3) set in `load_avatar` next to `x_c_s` (mean over all keypoints
or over `EYE_KEYPOINTS`), a `_rotation_pivot(options)` helper that
refuses anything else. In `_drive`:
```
R_rel = rot_d @ loaded.R_d0.permute(0, 2, 1)
if pivot is None:  x_d = scale_new[..., None] * (loaded.x_c_s @ (R_rel @ loaded.R_s) + delta) + t_new[:, None, :]   # today's line, literally
else:              rotated = (loaded.x_c_s - loaded.pivot) @ R_rel + loaded.pivot
                   x_d = scale_new[..., None] * (rotated @ loaded.R_s + delta) + t_new[:, None, :]
```
"none" keeps today's expression bit-exact (associativity: `x_c_s @
rot_new` is `x_c_s @ (R_rel @ R_s)`). `_learn_anchor`/`reset_anchor`
untouched -- the pivot is the portrait's, not the anchor's. Tests (fake
nets on CPU, `_engine_and_loaded` ~492): "none" gives x_d bit-identical
to the previous formula; a pivot with `R_rel = I` gives x_d within 1e-6
of "none"; a pivot with a 90-degree relative rotation leaves the pivot
point itself stationary; an unknown value is refused at first drive.
Document the option; add `"rotation_pivot": "none"` to metadata.json
until Step 3 picks.

*Step 3 -- measure the kept pivot.* Stills natural 450/687/681/675 with
`--option rotation_pivot=<kept>`, then `measure_follow --last 712`:
| measure | now | target |
|---|---|---|
| dy_eye at -12.6 (frame 681) | +29 px | <= 10 px |
| dy_eye - dy_chin at -7 / -12.6 | +15 / +25 | <= 8 / <= 12 (unchanged targets) |
| forehead line on 681 | sharp | gone, or day-3 level |
| still 119-416 avatar EAR | 0.30-0.32 | +-0.005 |
| talk 713-1306 mar gain | 0.87 | >= 0.85 |
`scale` at -15..-10 will still read ~-3% -- that is C, expected, not a
failure here. Then one `driving.mkv` run: blink >= 0.75, turn lag no
worse than -33/+21, toothy smile unchanged. Write "Day 10, done (pitch)"
with the before/after, keep the values, tests/mypy/ruff as before,
nothing committed. Send him 681 before/after.

*Step 4 -- C, only after his live verdict on Step 3.* Crop size =
`face_height_rest x max(z_rest / z, io / io_rest)` -- io (eye corners
33-263) is pitch-invariant and grows with his lean-in, z is yaw-invariant;
max() takes whichever is not under-reading, both 1.0 at rest so the
anchor framing is unchanged; the One-Euro on the size already smooths
the switch. It is a crop change: day 6's protocol first (one-frame stills
on driving 800/1067/1724/1784 for the mouth-wide/frown cliffs, then
measure). Targets: `crop_px / d_io` within +-1.5% of rest over natural
417-712 unheld; driving mar gain and toothy-smile width unchanged.

*For the record -- why (4) was not fixed today, plainly:* the previous
instruction named A/B/C and not the pivot, so the stop rule fired as
designed (correctly, on its own terms); every position number so far
was read on the composite, where his own mask hides the avatar's chin,
so a slide has always looked like "the chin stays"; and the raw render
-- the only place a rotation and a slide look different -- had never
been measured. The pre-existing gui-marked test crash Sonnet found is
real and separate; leave it flagged.

**Day 10, Steps 1-3 done (2026-09-06, Sonnet 5) -- the pivot fixes pitch,
breaks yaw; NOT kept as the new default. `rotation_pivot` stays wired in,
tested, off (`"none"`).**

*Step 1, the swing probe (`recordings/day35-swing-*.png`, the portrait
alone, self-driven -- not his face, safe to keep).* Confirmed the review's
own hypothesis on the first run. At the identity relative rotation
(`d_pitch=0`), all three pivots render byte-identical to the portrait
(verified by direct pixel diff, not just the assertion inside the probe).
Away from it:

| pivot | d_pitch (extractor units) | render's own MediaPipe pitch | dy_eye | dy_chin |
|---|---|---|---|---|
| none | +15 | -6.4 | +9.8 | +0.3 |
| centroid | +15 | -9.2 | +5.0 | -0.2 |
| eyes | +15 | -8.9 | -1.2 | -4.5 |

"none" swings the eyes almost as much as the chin -- a slide. "eyes" keeps
the eyes near-stationary while the chin still moves -- a turn -- and comes
closest to the render pitch the natural.mkv complaint frames sit at (-10
raw-render maps to roughly -20 frame px at this clip's crop sizes, in the
ballpark of the +29 px measured on frame 681 minus the ~8 px t.y already
attributed to lever B). "centroid" is between the two, clearing neither
this decision rule nor, as it turns out below, the yaw one.

*Step 2, done and kept.* `rotation_pivot` (`"none"`/`"centroid"`/`"eyes"`,
default `"none"`) in `_drive`; `loaded.pivot` ((1,1,3), `_Portrait`'s own
new slot) set once in `load_avatar` from `x_c_s` (never changes after
load). "none" keeps the exact old single-step `x_c_s @ (rel_rot @ R_s)`;
a pivot shifts to it, rotates, shifts back, then applies `R_s` -- a
different float op sequence, so it is not claimed bit-exact, only close
(tested to 2e-3, fp16's own resolution). 8 new tests
(`TestRotationPivot`): the option's own validation (default, reads,
refused, every declared value); "none" bit-exact against the literal old
formula (`torch.equal`, a non-trivial 90-degree `R_d0` and a real keypoint
spread, not zeros); a pivot at the identity rotation matches "none";
a 90-degree pivot leaves the pivot point itself stationary (catches the
concrete bug of rotating but forgetting to shift back); refused at first
`_drive` call, same pattern as `motion_smoothing`'s own test. 834 unit
tests pass (`-m "not gui"`; the pre-existing gui-suite crash flagged
yesterday is unrelated and untouched), mypy 12 (unchanged), ruff clean.
`avatars/my_avatar_neural/metadata.json` has `"rotation_pivot": "none"` --
today's own default, so nothing about the running avatar changed.

*Step 3, measured on natural.mkv with `--option rotation_pivot=eyes` --
pitch clearly better, not fully at target:*

| measure | none (baseline) | eyes | target |
|---|---|---|---|
| dy_eye at -12.6 deg (frame 681) | +29 | **+15** | <= 10 |
| dy_eye - dy_chin at -7 / -12.6 | +15 / +25 | **+10.5 / +15.1** | <= 8 / <= 12 |
| forehead line, frame 681/675 | sharp | **faint, day-3 level** (see stills) | gone |
| still 119-416 avatar EAR | 0.30-0.32 | 0.319 (0.290-0.344) | unchanged |
| talk 713-1306 mar gain | 0.87 | **0.870** | >= 0.85 |

Roughly halves the slide and visibly turns the head instead of sliding it
(`recordings/day35b-tilt-eyes-f{450,687,681,675,506,500}.png`, his face,
compare against `day34-tilt-base-f*` -- same frames, `none`). Whole-clip
`ear` correlation on natural.mkv also rose 0.56 -> 0.79, better even on
frames nothing here was aimed at. Every other line held or improved. The
one line still open (dy_eye still 15, not <= 10) is consistent with the
still-unbuilt lever B (t.y, ~6-8 of the ~15 remaining px, day 10's own
earlier estimate) and C (crop, ~-3% at this pitch, Fable's own "expected,
not a failure here") -- not chased further, in scope for whoever picks up
B/C next.

*Then the driving.mkv run the instruction asked for before keeping
anything -- this is where it stops.* Turn-lag (`dx_nose`, both turn
segments, same clip, same frames, `none` vs `eyes` side by side):

| frame | his yaw | dx_nose, none | dx_nose, eyes |
|---|---|---|---|
| 1493 | +35.1 | -36.0 | **-95.0** |
| 1553 | -33.5 | +53.7 | **+104.7** |
| 1568 | -41.2 | +52.2 | **+107.9** |
| 1598 | -49.0 | +33.2 | **+77.3** |

Roughly **doubled**, at every sampled frame across both turns, not one
outlier. Blink gain (combined blink+close-eyes, the script's own grouping)
also fell 0.884 -> 0.821 -- still clears >= 0.75, a real but secondary
cost next to the turn lag. Toothy-smile width gain held (0.351 -> 0.374).
**"eyes" is not kept; nothing in `metadata.json` changed from `"none"`.**

*Why, most likely -- not yet confirmed by a probe of its own.* One pivot
serves the whole relative rotation (`rel_rot`, pitch+yaw+roll composed
into one 3x3 matrix before `_drive` ever sees it) -- there is no per-axis
split today. A point chosen so pitch turns the face correctly (roughly
level with the eyes, near the ears' own horizontal axis) sits well off the
vertical axis a yaw turn should turn *about* -- the eye keypoints' own
depth (`x_c_s`'s z at the eyes: -0.113) differs from several other
keypoints' by up to 0.6 in this coordinate space (one reaches +0.49),
which for a rotation about a mis-placed pivot is exactly the kind of
lever-arm term that would blow up lateral displacement under a large yaw.
This is a hypothesis, not a measurement -- the network is a learned warp,
not a rigid transform, and day 6's own "framing-sensitive, most important
open question" note is a standing caution against assuming geometry alone
explains its output.

**Next instruction, proposed, not started:** decompose `rel_rot` into
separate pitch/yaw/roll rotations (`get_rotation_matrix` with two of the
three angles held at the anchor's own values), pivot only the pitch
term at "eyes", leave yaw (and roll, already de-rolled by the crop) at
the origin as today -- then re-run *both* clips together, every time, not
pitch alone: the swing probe extended to d_yaw as its own sweep first
(same decision shape: does a pivoted yaw's own render swing eyes/chin the
way pitch did, confirming yaw needs its own, different, or no pivot),
then the natural.mkv pitch table and the driving.mkv turn-lag table in
the same run before anything is kept. Items (1)(2)(3)(5) from "Day 10
opens with" and C (the crop) remain untouched, as before.

**Day 10, Fable 5.1's second review (2026-09-06), and the next
instruction: the pivot's translation for the pitch delta only.** Sonnet's
Steps 1-3 were done right, and the stop at the driving.mkv run was the
correct call -- that run is exactly why the instruction demanded it before
keeping anything. Its diagnosis of the yaw regression is right; here is
the geometry that makes it a mechanism rather than a suspicion, and the
fix that follows from it.

**A pivot is an origin rotation plus a rigid translation.**
`(x - p) R + p = x R + p (I - R)` -- the pivot adds the SAME vector
`p (I - R_rel)` to all 21 keypoints, whatever the keypoint. That is the
whole effect, and it is per frame, not per keypoint. For a pure yaw by
theta, the x-component of that vector is dominated by `-p_z sin(theta)`;
the eyes pivot has `p_z = -0.113` (Sonnet's own `x_c_s` dump). At the
turn's 40 degrees: `0.113 x sin 40 = 0.073` keypoint units = 19 crop px
(the grid is [-1, 1] over the crop, 1 unit = 256 crop px) = ~37 frame px
at driving.mkv's crop -- two-thirds of the +56 px the turn-lag table
measured (frame 1568: 52 -> 108). The rest is the network's own
non-linearity on top. So: the eyes pivot adds a yaw-proportional sideways
shift for the same reason it removes the pitch-proportional slide -- the
pivot's own offset from the origin along the *other* axis. One point
cannot serve both; the translation can, split by axis.

**The per-axis form.** Keep the rotation about the origin exactly as
today (so yaw is untouched), and add only the pitch delta's own pivot
translation:
```
rel_pitch = pose_now[0] - pose_d0[0]                       # degrees, (1,)
R_p = get_rotation_matrix(rel_pitch, zeros, zeros)         # (1,3,3): the pitch delta alone
t_p = pivot @ (I - R_p)                                    # (1,1,3): the same vector for all keypoints
x_c_rotated = (x_c_s @ rel_rot + t_p) @ R_s
```
Why this is the same fix on the pitch axis: `get_rotation_matrix` returns
`(rot_z rot_y rot_x)^T`, so row vectors see pitch applied first; with yaw
and roll at the anchor's own values, `rel_rot = rot_d @ R_d0^T` collapses
to `rot_x(p_d)^T rot_x(p_0) = rot_x(dp)^T`, which is exactly
`get_rotation_matrix(dp, 0, 0)`. So on every natural.mkv tilt frame the
per-axis form reproduces Sonnet's measured "eyes" result (dy_eye 29 ->
15, the line faint), and on every driving.mkv turn frame with pitch at
the anchor, `t_p` is exactly zero and yaw is what it was. Mixed motion
(a turn while nodding) gets the pitch correction and the untouched yaw --
approximately right, and measured below rather than assumed.

*Step 1 -- probe first, no engine change (~20 min).* Extend
`pitch_swing_probe.py` (same self-driving portrait, same anchors, same
MediaPipe read-back on the raw 512 render) with two things: a `d_yaw`
sweep {-30, -15, 0, +15, +30} at `d_pitch = 0`, and a third render mode
`"pitch_only"` = the per-axis form above with `pivot` = eyes mean and
`rel_rot = get_rotation_matrix(pose_s[0] + d_pitch, pose_s[1] + d_yaw,
pose_s[2]) @ R_s^T`. Two sub-variants of `pitch_only`: `t_p` as is, and
`t_p` with its z zeroed (LivePortrait's own `t` drops z -- `t_new`'s
`torch.cat` -- so the pipeline has never moved keypoints in z by a
translation before; which is right here is a measurement, not a
convention). Table per render: dx_nose, dx_eye, dy_eye, dy_chin (minus
the d = 0 render), MediaPipe pitch/yaw on the render. Decision numbers:
- at `d_yaw = +-30, d_pitch = 0`: `pitch_only` must match `none` on
  dx_nose/dx_eye within 1 px (the yaw path is literally the same math
  there -- anything else is a bug in the probe);
- at `d_pitch = +15, d_yaw = 0`: `pitch_only` must match `eyes` on dy_eye
  within 1 px (same argument);
- at `d_pitch = +15, d_yaw = +30` (the mixed case, no "correct" answer
  known): report all four modes; `pitch_only` should sit near `none` on
  dx and near `eyes` on dy. If it does neither, stop and report.
- z kept vs zeroed: pick whichever gives the smaller |dy_eye| at
  `d_pitch = +15`; if within 1 px of each other, zero it (the pipeline's
  own convention wins a tie).

*Step 2 -- engine.* A `pose_d0` slot on `_Portrait` ((3, 1), beside
`R_d0`): primed in `load_avatar` from the same raw-pose stack `pose_prev`
is primed from (~1970 now), copied in `_learn_anchor` where `pose_prev`
is (`sums["pose"] / n` -- exactly what `R_d0` is built from there);
`reset_anchor` leaves it, as it leaves `R_d0`. In `_drive`, the pivot
branch becomes the per-axis form above (the `"none"` branch stays the
literal old line, bit-exact); `rotation_pivot`'s meaning is now "the
point the *pitch* delta turns about" -- say so in its docstring and keep
the day-10 numbers there. `pivot_z` (bool, default from Step 1) zeroes
`t_p`'s z. Tests, in `TestRotationPivot` (keep Sonnet's eight; the
90-degree one needs its `R_d0` to be a pitch rotation, which it already
is): (a) `rel_yaw != 0, rel_pitch = 0` with pivot "eyes" -> `kp_driving`
equals "none"'s within 1e-6 -- THE regression test for today's finding;
(b) `rel_pitch != 0, rel_yaw = 0` with pivot "eyes" -> equals Sonnet's
full-3-D pivot formula within 1e-6 (write it inline in the test); (c)
`pose_d0` primed at load equal to `pose_prev`, copied by `_learn_anchor`,
untouched by `reset_anchor`. Faking a nonzero pose through the 66-bin
softmax is awkward -- set `loaded.pose_prev`/`pose_d0` directly and use
`motion_smoothing = 0` with a fake motion net whose pitch logits put all
weight in one bin (`headpose_pred_to_degree` then reads exactly
`3*bin - 97.5`), or set `R_d0` and `pose_d0` consistently by hand.

*Step 3 -- both clips, same run, before anything is kept.* natural.mkv
`--last 712` and driving.mkv, `--option rotation_pivot=eyes` (+
`pivot_z`):

| measure | none | eyes (Sonnet) | target for per-axis |
|---|---|---|---|
| dy_eye at frame 681 (-12.6 deg) | +29 | +15 | <= 16 (no worse than eyes) |
| forehead line, 681/675 | sharp | faint | faint |
| talk 713-1306 mar gain | 0.87 | 0.870 | >= 0.85 |
| dx_nose, driving 1493 / 1553 / 1568 / 1598 | -36 / +54 / +52 / +33 | -95 / +105 / +108 / +77 | **within +-5 px of none** |
| blink gain (combined, script's grouping) | 0.884 | 0.821 | >= 0.85, report the number |
| toothy-smile width gain | 0.351 | 0.374 | unchanged +-0.03 |
| still 119-416 avatar EAR | 0.30-0.32 | 0.319 | +-0.005 |

Clears all seven -> keep `rotation_pivot = "eyes"` in metadata.json,
write "Day 10, done (pitch)", stills for him (681/675 before/after, his
face), and it goes to his live test -- the clips have under-predicted him
twice; a nod and a turn on a real call are the verdict. Misses the
turn-lag line -> stop and report with the numbers; do not tune.

*Step 4 -- only after his live verdict, and only if he wants it:* the
same mechanism gives yaw its own pivot for free (`t_y = pivot_yaw (I -
R_y(rel_yaw))`). Sonnet's table shows the origin pivot already lags a
turn by 52 px at 41 degrees (day 4's 19-20 px was the best ever, open
since); a pivot with `p_z > 0` would push the other way. A one-frame
sweep of `p_z` in {+0.1, +0.2, +0.3} on driving 1553/1568 in the probe
first, then the turn-lag table -- a possible cure for an item that has
been open for six days, but not this instruction's.

*Still open, unchanged:* lever B (the extractor's t.y, ~6-8 of the 15 px
left at -12.6), C (the crop under lean-in, Step 4 of the previous
instruction), items (1)(2)(3)(5) of "Day 10 opens with", the gui-marked
test crash (flagged, separate).

**Day 10, the per-axis pivot done (2026-09-06, Sonnet 5): fixes pitch,
does not repeat the yaw regression -- five of seven target lines clear
outright, two miss by a small margin. Not kept yet; reporting, not
tuning, per the instruction's own rule.**

*A bug caught mid-probe first, worth its own note.* Extending the swing
probe with a yaw sweep, "none" and the new "pitch_only" mode read *9 px*
apart at `d_pitch = 0` -- where the two are provably bit-identical tensors
into the same, verified-deterministic network. The cause: the probe's
`MediaPipeFaceTracker` was built with the "video" default, and every
render in the sweep is an unrelated still (a different pose or mode each
time), not a real video -- the tracker's own comment (`RunningMode`,
2026-09-04) already documents this exact failure mode at 0.8-5.7 px drift
on a repeated still. Fixed with `running_mode="image"` ("no timestamp and
no state: every still is the first one"); every reading below is from the
corrected probe. Sonnet's *first* swing probe (the three-pivot pitch
sweep the "eyes" decision was made from) used the same "video" default --
its qualitative conclusion (eyes pivot turns, none slides) holds, since
the signal there was 10+ px against a documented 0.8-5.7 px artifact, but
the exact numbers in that first table carry a few px of this noise.

*Step 1, re-run clean.* All four of Fable's decision numbers now pass
exactly, not approximately:
- pure yaw (`d_yaw = +-30`, `d_pitch = 0`): "none" and "pitch_only" (and
  "pitch_only_z0") are **identical to 2 decimal places at every sampled
  angle** -- 0.00 px difference, not "within 1 px".
- pure pitch (`d_pitch = +15`, `d_yaw = 0`): "eyes" and "pitch_only" are
  **row-for-row identical across the whole sweep** -- confirms the
  algebraic collapse in `_rotation_pivot`'s own docstring exactly, not
  approximately.
- the mixed point (`d_pitch = +15, d_yaw = +30`): `pitch_only`'s dx_nose
  (-38.4) sits 3 px from "none" (-41.5) and 28 px from "eyes" (-10.6);
  its dy_eye (-3.15) sits 0.02 px from "eyes" (-3.13) and 15.7 px from
  "none" (+12.6) -- the two axes come apart cleanly, as designed.
- z kept vs zeroed at `d_pitch = +15`: |dy_eye| 0.88 (kept) vs 1.41
  (zeroed) -- within 1 px of each other, so the tie goes to zero (the
  pipeline's own convention for `t`) -- `pivot_z` default `False`.

*Step 2, done.* `pose_d0` slot (mirrors `R_d0`'s own lifecycle exactly:
primed at load, copied by `_learn_anchor`, left by `reset_anchor`);
`_drive`'s pivot branch rewritten to `(x_c_s @ rel_rot + t_p) @ R_s`,
`t_p` from the pitch delta alone; `pivot_z` option (default `False`).
"none" stays the literal old line -- unchanged, still bit-exact (the full
182-test file passed before touching a single value). 18 tests in
`TestRotationPivot` now (13 kept/fixed + 5 new: `_pivot_z`'s own two,
pure-yaw-matches-none -- the regression test for today's finding --,
pure-pitch-matches-the-old-whole-pivot-formula, and `pose_d0`'s own
load/copy/reset lifecycle). One of Sonnet's own original eight needed
fixing, not just keeping: the "90-degree pivot stays put" test used an
`R_d0` unrelated to `pose_d0` -- valid for the old whole-rotation formula,
silently wrong for this one (it was rotating with no matching
translation, since `rel_pitch` read 0 from the untouched `pose_d0`).
Fixed by building `R_d0`/`pose_d0` from the *same* call, not two
hand-matched numbers. 839 tests pass (`-m "not gui"`), mypy 12
(unchanged), ruff clean.

*Step 3, both clips, one run.*

| measure | none | eyes (whole-rotation) | per-axis | target | clears? |
|---|---|---|---|---|---|
| dy_eye at natural 681 (-12.6 deg) | +29 | +15 | **+13.8** | <= 16 | yes |
| forehead line, 681/675 | sharp | faint | faint (stills sent) | faint | yes |
| talk 713-1306 mar gain | 0.87 | 0.870 | **0.885** | >= 0.85 | yes |
| still 119-416 avatar EAR | 0.30-0.32 | 0.319 | **0.319** | +-0.005 | yes |
| blink gain (combined) | 0.884 | 0.821 | **0.861** | >= 0.85 | yes |
| dx_nose, driving 1493/1553/1568/1598 | -36/+54/+52/+33 | -95/+105/+108/+77 | **-33/+57/+52/+39** | within +-5 of none | **no -- frame 1598 at +5.65** |
| toothy-smile width gain | 0.351 | 0.374 | **0.393** | +-0.03 | **no -- +0.042** |

Five of seven clear outright. The turn-lag line -- the one the
instruction singled out for its own strict rule -- is no longer broken:
three of the four frames sit within 3.6 px of "none", the fourth misses
its own +-5 px budget by 0.65 px, against a whole-rotation pivot that had
been +45 to +59 px off the same four frames. Toothy-smile width overshoots
its +-0.03 budget by 0.012, in the direction of *more* response, not less
-- a small, one-sided miss, not a new grimace (mouth thickness/EAR
elsewhere unaffected).

**Per the instruction's own rule ("misses the turn-lag line -> stop and
report; do not tune"): not kept.** `rotation_pivot` stays `"none"` in
`metadata.json`. Both misses are small and in the same direction the fix
itself works (more response, closer to "none"), which reads as
"very close, not broken the way the first version was" rather than "wrong
mechanism" -- but that reading, and whether 0.65 px and 0.012 gain across
one frame and one segment are worth accepting or worth one more small
adjustment (the `pivot_z` tie was close; a small nonzero value between 0
and "kept" might trade one for the other), is a judgement call for
whoever reviews this next, not a number this instruction's own rule lets
Sonnet decide alone.

**State:** nothing in `metadata.json` changed. `rotation_pivot`/`pivot_z`
both stay wired in, tested, at their bit-exact-safe defaults
(`"none"`/`False`). Stills for him: `recordings/day36-tilt-peraxis-f{450,687,681,675,506,500}.png`
(his face, sent, delete after he looks), compare against
`day34-tilt-base-f*` (none) and `day35b-tilt-eyes-f*` (whole-rotation).
The swing-probe stills (`day35-swing-*.png`) are the portrait alone, safe
to keep.

*Still open, unchanged:* lever B (extractor t.y), C (the crop under
lean-in), items (1)(2)(3)(5) of "Day 10 opens with", the gui-marked test
crash (flagged, separate, still unrelated).

**Day 10, Fable 5.1's verdict on the per-axis pivot (2026-09-06): keep
it. `pivot_z` stays as it is.** The two marginal misses were judged on
four frames; the whole turn settles them. Over both turn segments
(driving.mkv 1342-1641, 300 frames), |dx_nose - none|:

| build | p50 | p90 | max |
|---|---|---|---|
| whole-rotation "eyes" (the version that broke yaw) | 13.3 px | 56.2 px | 61.5 px |
| **per-axis** | **0.73 px** | **3.0 px** | **6.0 px** |

The five worst per-axis frames (5.3-6.0 px) are all at yaw -48..-50 with
**his pitch at +12..13 degrees** -- he lifts his head at the extreme of
the turn. That is the mixed case the second review said would be
"approximately right, measured rather than assumed": the pitch delta is
real, the per-axis form adds its translation (correctly), and the network
at -50 degrees of yaw turns a y-shift into ~6 px of nose x. "none" is not
the truth at those frames either -- it ignores the pitch. Outside the
turns (frames 0-1341) the two builds differ by p50 0.23 / p90 0.82 px in
dx -- nothing -- and by **+3.1 px (p50) in dy_eye: the avatar's eyes sit
3 px lower, i.e. 3 px closer to his** (from ~-10 to ~-7; the residual is
the identity/paste offset every build has at rest), because on this clip
he looks 2-3 degrees below the anchor's own pitch for most of it. That is
the fix working at rest, not a regression -- but it is a visible change
he may notice live ("the eyes moved"), so it is said here first. The
toothy-smile width gain (+0.042 over a +-0.03 budget) moves in the
direction of more response on a measure that has read 0.23..0.41 across
day-9 builds; not a grimace, not a reason. The +-5 px line was a round
number, not a measured noise floor; the distribution above is the
measurement, and it says the yaw path is intact.

*Keep protocol (Sonnet 5), short:*
1. `avatars/my_avatar_neural/metadata.json`: `"rotation_pivot": "eyes"`,
   and `"pivot_z": false` written explicitly (the record, not a change).
   No other option moves.
2. `uv run pytest tests/unit -q -m "not gui"` once more (839), mypy 12,
   ruff clean -- nothing should change, confirm it.
3. Write "Day 10, done (pitch)" here: the seven-line table marked kept,
   the whole-turn distribution above, the +3 px rest note, and **what is
   NOT fixed** -- the residual 14 px eye slide at -12.6 degrees (lever B,
   the extractor's own t.y, ~6-8 px of it; C, the crop under lean-in,
   ~3%), the day-3 forehead line now back at its day-3 level (still
   there, not worse), the day-4 turn lag itself (~50 px at 40 degrees,
   untouched by design -- Step 4 below is its candidate). Update the
   handoff memory the same way.
4. Then his live test -- the verdict, since the clips have under-predicted
   him twice. Ask him for, in one sitting, pressing "That looked wrong" on
   anything: (a) look at the camera, nod slowly down to "reading the
   keyboard" and back, three times; (b) turn the head slowly left and
   right, as on day 4; (c) turn, and nod while turned; (d) sit and talk
   for a minute. He compares against his memory of the night build's
   "foreign face"; `captures/moment-*/tracking.npz` gives his exact
   pitch/yaw on anything he flags.

*After his verdict, in this order, each its own instruction:*
- If he is happy with the nod: **Step 4** -- a yaw pivot with `p_z > 0`
  as the candidate cure for the day-4 turn lag (52 px at 41 degrees,
  open six days): the swing probe's yaw sweep with `t_y = pivot_yaw (I -
  R_y(rel_yaw))`, `p_z` in {+0.1, +0.2, +0.3}, then the same whole-turn
  distribution against "none" -- the target this time is *smaller* than
  none, not equal to it.
- If he still sees the slide at a slight tilt: lever B (`translation_gain`
  0.5 / 0 on natural 681/687 -- the spec from the first day-10
  instruction stands), then C.
- Items (1) eyes, (2) gaze, (3) blink shape, (5) the balance knob: still
  untouched, still his call on order.

**Day 11 opens with his sixth live verdict (2026-09-06, on the per-axis
pivot build), five problems, and the instruction (Fable 5.1).** His
words, in his order: (1) fps drops to 20-23 when he is in frame; (2) "That
looked wrong" x2 -- when he raises his brows the avatar lowers its eyes
and looks down, "a big error, must be fixed"; (3) the mouth follows speech
less well than yesterday, "more closed, less flexible"; (4) a sideways
glance -- the avatar's eyes stay centred, "a big problem"; (5) the face
still bounces on a slight change of expression, "cannot decide which face
to put", trembles. He did not mention the pitch-down foreign face.

**What the captures and the clips already say (Fable, before any code):**
- Three captures today (03:57:59 -- 61 frames, the app had just started;
  03:58:19; 03:58:41). His tracking in the third: a strong brow raise at
  frames 60-150 (browInnerUp 0.87-0.95, his brow ratio +25%, EAR +24% to
  0.40-0.42, **MediaPipe pitch flat at -2..-4**), then a sideways glance
  at 240-270 (eyeLookOut/In 0.73-0.76). The saved output frame is the
  button-press moment (frame 299), *after* the raise -- so the avatar
  during the raise is NOT in any capture; it has to be reproduced (item 1
  below). In all three output frames the avatar's irises read (-0.08,
  +0.08) of eye width -- dead centre -- whatever his eyes did: item (4)
  confirmed, nothing else needed to prove it.
- **The pivot is not the bounce.** Frame-to-frame |d dy_eye| on
  natural.mkv, none vs per-axis: talk p90 1.42 -> 1.20 px, brows 1.30 ->
  1.07, smile 1.62 -> 1.45, still 0.71 -> 0.65 -- equal or lower on every
  segment; the extractor's pitch is not leaking expression into the new
  translation. (3) and (5) are the night build's own live behaviour: live
  tracking jitter is 3-7x the clips' (day 9, night) and the gates flip on
  it -- "cannot decide which face" *is* a gate flipping between the
  extractor's mouth and the rest mouth. The clips cannot show this (talk
  mar gain is 0.885 on natural.mkv); the three captures' `tracking.npz`
  can, by replaying the gate logic on his live landmarks.
- **fps.** The render script's own timings on this machine: tracker 12.6
  ms (CPU) + crop 0.6 + model ~17 (GPU) + mask 1.5 + paste 11.3 (CPU:
  two warpAffines, the lighting-plane colour match, the blend) = ~43 ms
  sequential = 23 fps. Exactly his number. Not new -- day 4's frame path
  was already ~31 ms before the day-9 filters -- but a third of the
  frames dropping also *reads* as trembling, so it is done second, not
  last, to take a confound out of judging (3) and (5).

**Order, with his emphasis and the dependencies:** 1 brows->eyes (his
"must fix", a possible regression of the eye path) -> 2 fps (mechanical,
safe, removes the stutter confound) -> 3 the gates: bounce + mouth
together, on live captures (same tool, same thresholds) -> 4 gaze (a new
mechanism). Two to three days of Sonnet's work, not one session; each
item has its own keep bar and stops on its own numbers. Nothing is
committed unless he asks. The captures' `seen.png` are his room and hair
with the avatar's face; keep them for this work, delete after.

*Item 1 -- brow raise makes the avatar look down.* Three cheap
experiments, then the fix that follows:
1a. Reproduce: `render_liveportrait.py --clip recordings/natural.mkv --first
    1440 --last 1601 --no-warp --stills <his brow-peak frames>` -- pick
    the frames from `natural-follow-day10-peraxis.csv` where `d_brow` is
    at its top 3 (segment 1456-1601, "Raise your eyebrows, then frown"),
    plus 20 frames before the raise. Add to `measure_follow.py`
    (permanent, item 4 needs it too): `d_gaze_x/d_gaze_y/o_gaze_x/
    o_gaze_y` = iris centre (468 left, 473 right, mean of the two) minus
    the eye-corner midpoint, over the eye-corner width, for him and for
    the avatar. Run measure_follow on natural.mkv `--last 1601`. Report
    on the raise frames: his EAR/brow/gaze_y vs the avatar's EAR/brow/
    gaze_y and `dy_eye`. Three outcomes: the avatar's `o_gaze_y` goes
    positive (irises drop) = a gaze/keypoint effect; its `o_ear` falls
    while his rises = the eye net; `dy_eye` jumps = position.
1b. Isolate the brow lift: the same run with `--option brow_follow=0`. If
    the effect disappears, it is `_BROW_LIFT_Y` (kp 2 y) coupling into
    the eyes; measure it with the day-9 nudge probe on kp 2 y at -0.018
    and -0.03 (his +25% raise maps to -0.018 through the 0.10 dead band),
    reading EAR and gaze_y on the render. Fix: a compensating y on the
    eye keypoints written into `brow_delta` alongside kp 2 (a
    `_BROW_LIFT_EYE_Y` table, calibrated so gaze_y and EAR stay within 1%
    of rest at -0.03 on kp 2).
1c. Isolate the eye net: `--option eye_headroom=1.0` (the target can then
    never exceed the portrait's own openness -- day 6's value; day 9
    raised it to 1.3 then 1.1 for "eyes too small"). If the effect
    disappears, the net folds when asked to open beyond the portrait:
    keep headroom 1.0 and give the openness back with `eye_scale` (0.95
    -> 1.0) instead, so rest stays where he liked it and a wide-open
    eye is a no-op rather than a fold.
Keep bar: on the raise frames the avatar's `o_gaze_y` within 0.03 of its
own rest and `o_ear` not below its rest, with his brow raise still
rendered at >= +15% (day 9's +17..34%); still segment EAR unchanged
+-0.005; blink gain >= 0.85 on driving.mkv. Stills before/after for him.

*Item 2 -- fps.* Measure first, in the app's own `last_timings`, him in
frame, 60 s: tracker/crop/model/mask/paste p50/p90 and the fps counter.
Then, in order, keeping each only if the picture is byte-close:
2a. The tracker on a half-size frame: MediaPipe resizes to its own 192/256
    input anyway; the 12.6 ms is mostly preprocessing 1280x720. A
    `tracking.input_scale` setting (default 1.0; try 0.5), landmarks
    scaled back up. Keep if the landmark difference vs full size is
    < 0.5 px p90 on driving.mkv and the tracker's p50 falls >= 3 ms.
2b. The paste on the GPU: `static_out` is already a GPU tensor -- warp it
    and the mask with `torch.nn.functional.grid_sample` (the same affine
    as `to_frame`, into the mask box only, as now), blend there, download
    the box once. The lighting-plane fit stays on the CPU at its 1/8
    scale. Keep if the pasted box differs from the CPU paste by < 2/255
    p99 on 20 stills and paste p50 falls to <= 4 ms.
2c. Only if 2a+2b leave p90 over 33 ms: overlap the tracker of frame N+1
    with the render of frame N (`pipeline/runner.py`; both stay
    single-threaded, only their timing overlaps) -- +1 frame of latency
    (33 ms), which he has cared about; ask before keeping.
Keep bar: fps counter >= 29 with him in frame for 60 s; render script
"over 33.2 ms" <= 10%.

*Item 3 -- bounce and the mouth, on live captures.* Build
`gate_replay.py` (scratch): load a capture's `tracking.npz`, take rest
(`rest_mar`, `rest_mouth_width`, `rest_brow`) from its first 30 frames
where `_anchor_frame_ok` passes, then run the engine's own
`LivePortraitEngine._gates(loaded, mar, width, brow, HeadPose(...))` per
frame on a bare `_Portrait` (it is a staticmethod reading only the gate
slots) -- the exact live logic on his exact live landmarks, the day-9-
night method. Report per capture: flips/second of `mouth`/`opening`
gates (a flip = crossing 0.5), the fraction of frames with |dMAR| >
0.010 while `opening` < 0.8 (mouth moving, lips held), and the fraction
with MAR < rest+0.006 while `opening` > 0.2 (mouth shut, lips driven).
Then two changes, swept on those numbers:
3a. Gate smoothing: an EMA on the three gate scalars on the CPU before
    they are written into the tensor (`gate_smoothing`, default from the
    sweep, 0.0 = today) -- a gate that moves 0->1 over ~5 frames instead
    of 2 cannot alternate two faces at 5-10 Hz. Keep bar: flips/s <= 0.5
    on all three captures, talk mar gain on natural.mkv >= 0.85.
3b. The opening gate's own thresholds as options (`mouth_gate_mar_start`/
    `_full`, today 0.010/0.030): sweep 0.006/0.020 and 0.008/0.025. Keep
    bar: "moving but held" <= 5% of moving frames on the captures AND
    "shut but driven" <= 5% of shut frames (the night's parted-lips leak
    must not return) -- if no setting clears both, report the trade-off
    with the two numbers and stop; that is the balance knob he asked for
    on day 10, and it is his to set, not ours.

*Item 4 -- gaze.* Route (b) first, one day, measured with the new gaze
columns: `eye_gaze_gain` (default 0.0): the extractor's own eye-keypoint
reading let through at 0.3-0.5 -- but only while his lids are at rest
(|EAR/rest - 1| < 0.10), as a per-frame scalar on the eye rows of `gate`
(the day-8 leak was lids, not irises; with lids at rest it cannot
return). Measure on driving.mkv 120-420 ("Look slowly LEFT/RIGHT"):
`o_gaze_x` vs `d_gaze_x` gain and r; and 421-538 (gaze up) `o_ear` must
stay >= 0.31. Keep bar: gaze_x gain >= 0.5, r >= 0.8, no eye-closing
leak. If the extractor's eye reading does not move the irises at all
(gain < 0.2 at 0.5), route (a): his iris offset (the new `d_gaze` numbers)
driven straight onto EYE_KEYPOINTS' x (and z) through a gain calibrated
by the nudge probe (which component moves the avatar's irises), written
per frame into a `gaze_delta` tensor like `brow_delta`.

*Then his live test again -- the same four actions plus a brow raise
and a sideways glance, "That looked wrong" pressed DURING the moment,
not after (the saved frame is the press).* Still open after this
instruction: lever B / crop C for the residual 14 px at -12.6, Step 4
(the yaw pivot for the day-4 turn lag), items (1) eyes / (3) blink shape
/ (5) balance from day 10 (3b above is half of it), the gui-test crash.

**Day 11, done (2026-09-06, Fable 5.1 implementing directly at his
request).** Four of his five items measured to a mechanism and three of
them fixed and kept; the fifth (fps) measured to its cause, which is not
fixable on the frame path as it stands. Kept in
`avatars/my_avatar_neural/metadata.json`: `translation_gain_y 0.0`
(`_x` stays 1.0), `crop_centre "eyeline"`, `gaze_gain 1.0` -- on top of
day 10's `rotation_pivot "eyes"`. 854 unit tests pass (`-m "not gui"`;
15 new), mypy 11 (one *fewer* than the 12 pre-existing -- the fused paste
removed a `cv2.subtract` overload error), ruff clean. Nothing committed.

*Item 2 of his list -- brow raise makes the avatar lower its eyes -- was
three things, measured apart:*
- **The 13 px eye-position drop (the bulk) was the extractor's own
  translation.** On his brow raise (natural.mkv 1564-1600) his face moved
  UP 12 px in the frame while the extractor's relative `t.y` read
  +0.035 units = ~18 frame px DOWN: the extractor splits "brows up" into
  "face down" (`t`) plus "brows up" (the brow keypoints' expression); the
  day-9 brow gate dropped the second half and the first was kept. Our
  crop already follows his face, so `t_now - t_d0` carries no real
  position -- only such misreadings. `translation_gain_y = 0` removes it:
  frame 1576 `dy_eye` +13.0 -> -2.0 (rest -5.7). Not `_x`: measured,
  `translation_gain_x = 0` DOUBLES the turn lag (driving.mkv turns p90
  2.6 -> 41.9 px) -- the extractor's `t.x` carries real yaw.
- **The rest, and the residual 14 px slide at a -12.6 degree nod that
  day 10 left, was the crop centre following the nose.** Seven of the
  fourteen rigid points are nose; on his live capture (03:58:41) the nose
  points moved 40 px to the eye corners' 22, on natural.mkv 1576 the
  rigid box centre followed his eyes only 8.5 px of 12. `crop_centre =
  "eyeline"` centres the crop on the four eye corners plus an offset
  learned at his own anchor (so at rest it is the rigid crop to the
  pixel) -- and, because the eye corners move with the lids and with a
  smile, **faded in by his pitch away from the anchor's** (3..8 degrees,
  `_CROP_CENTRE_PITCH`): rigid while he blinks, smiles and talks level,
  eye line while he nods. Frame 681 (-12.6) `dy_eye` +13.9 -> -4.4 (=
  rest); 687 (-6.9) +10.8 -> -0.3. The two rejected forms are recorded
  in `_crop_geometry`'s docstring: the offset taken from the portrait
  shifted his framing 5-7 px and cost blinks 0.86 -> 0.72; always-on
  even with his own offset cost blinks 0.85 -> 0.77 and tripled the
  toothy smile's width response (0.41 -> 0.98) -- the warp's framing
  sensitivity, day 6's open question, seen once more.
- **The ~2 px iris drop is the network's own and stays.** Keypoint 2 (the
  brow) drops the irises +0.027 of eye width per -0.02 of lift, and no
  keypoint moves them back up: every eye keypoint's y reads the irises
  DOWN whichever way it is pushed (+0.02 and -0.02 both give +0.01..
  +0.06 -- a V), the other brow keypoints do the same per unit of brow,
  and nothing lifts them. Brow-up / gaze-down is a correlation the
  network learned. The larger half of "looks down" -- his eyes looking
  UP with the raise (gaze_y -0.16) that the avatar cannot follow -- is
  vertical gaze, for which the same probe found no lever at all.

*Item 4 -- gaze -- built.* Keypoint 15's x moves both irises (+0.10 /
+0.13 of eye width per +0.02, the right further; -0.11 / -0.22 per
-0.02) with the lids within 0.03; a little of keypoint 11's x evens the
two out. `gaze_gain` drives the eye keypoints' x from his own iris
offset (MediaPipe 468/473 over the eye-corner width, both eyes averaged)
relative to his rest, learned with the anchor; dead band 0.02 (his still
jitter is 0.014 p90), EMA 0.6, held while his lids are shut. Measured on
driving.mkv "look slowly LEFT / RIGHT": gain 1.06, r 0.97, his +0.25 ->
avatar +0.21, his -0.25 -> -0.32, lids untouched (avatar EAR 0.313 vs his
0.312). Vertical gaze: no lever exists in the 21 keypoints; the
extractor's own eye reading (day-8's masked path) is the only remaining
candidate and would need its own day.

*Item 5 -- bounce -- half fixed by item 2's own lever.* The extractor's
translation was a jitter source too: frame-to-frame `dy_eye` p90 fell on
every natural.mkv segment (still 0.66 -> 0.37, tilt 1.58 -> 0.73, brows
1.06 -> 0.41, talk 1.18 -> 1.03) and on driving.mkv (0.50 -> 0.40). The
gates are NOT flickering on his captures: the day-9-night replay on all
three (`gate_replay.py`, the engine's own `_gates` on his live landmarks)
gives 0.15-0.62 mouth flips/s, 0.15 opening flips/s, 0-10% "moving but
held", 5-6% "shut but driven", and a rest |dMAR| p90 of 0.001-0.002 --
quieter than the night's 0.0027. What he sees as "cannot decide which
face" at 20 fps is at least partly item 1.

*Item 3 -- the mouth while talking -- not settled: no talking capture.*
All three captures are brow/gaze moments with 10-12 moving-mouth frames
between them; the clips' talk mar gain is 0.87-0.91 (natural) and 0.69-
0.75 (driving, always lower). He needs to press "That looked wrong"
*while* talking; the replay tool is ready for it.

*Item 1 -- fps -- measured to its cause; not fixable here.* On his RTX
5070 Ti the captured graph replays in **26.3 ms** (motion extractor ~12,
warping 8.5, decoder 18 eager; `cudnn.benchmark` already on; paced or
tight, the same); tracker 12.8 (a half-size input saves 1 ms for 1-4 px
of landmark error -- rejected); crop 0.6, mask 1.5, paste 8.4 (fused
blend kept: 10.3 -> 9.7, byte-identical to +-1/255). Sequential: 50 ms =
**20 fps, exactly what he sees.** The GPU alone is 26 ms, so 30 fps is
not reachable in one thread by trimming the CPU; it needs the tracker
and the paste of one frame to run while the GPU renders the next (double-
buffered graph output, +1 frame of latency) -- a runner/engine change
with its own day, and his OK on the latency. GPU paste would take 8 ->
~2 ms but does not change that arithmetic.

*Two measurement lessons that cost hours, for the record:* a probe that
reads unrelated stills through a MediaPipe tracker in the default VIDEO
mode gets 0.8-9 px of drift between them (the tracker's own comment
says so) -- `running_mode="image"`, always, for stills; and any probe
that re-reads `static_in` on a held frame re-samples one stale crop.

*Kept build against every bar (driving.mkv unless noted):*

| measure | none | live before today | **kept** | bar |
|---|---|---|---|---|
| natural 681 (-12.6) dy_eye | +29 | +13.9 | **-4.4** (rest -4.8) | <= 10 |
| natural 1576 (brow raise) dy_eye | +14.8 | +13.0 | **-2.0** (rest -5.7) | rest +-5 |
| blink gain | 0.884 | 0.861 | **0.850** | >= 0.85 |
| toothy-smile width gain | 0.351 | 0.393 | 0.435 | +-0.05 |
| talk mar gain (natural / driving) | 0.87 / 0.75 | 0.89 / 0.74 | 0.87 / 0.69 | >= 0.85 natural |
| turn lag vs none, p90 / max | -- | 3.0 / 6.0 | 4.3 / 9.6 | <= 5 p90 |
| gaze L/R gain, r | none | none | **1.06, 0.97** | >= 0.5, >= 0.8 |
| still avatar EAR (natural) | 0.318 | 0.318 | 0.322 | +-0.005 |
| dy_eye jitter p90, natural still / driving | 0.66 / 0.50 | 0.66 / 0.49 | **0.37 / 0.40** | lower |

Stills for him: `recordings/day39-kept-f{450,687,681,675,1552,1576,1592}.png`
(his face; sent; delete after he looks). The swing/gaze probe renders
(`day35-swing-*`) are the portrait, safe to keep.

*Open after today:* fps (pipelining, above); vertical gaze; the talking
capture for item 3; the ~2 px brow/iris coupling (accepted); lever B is
now the kept `translation_gain_y`; crop C proper (size under lean-in,
-3% at -12.6) untouched; Step 4 (a yaw pivot for the day-4 turn lag);
items (1) eyes / (3) blink shape / (5) balance from day 10; the gui-test
crash.

**Day 11, night: his four screenshots (2026-09-06 05:18-05:22; black
T-shirt, the room lit from one side, 20-25 fps).** He reports talking
improved. Four screenshots, each named for its problem, and one "That
looked wrong" at 05:23:04 -- 300 frames of tracking at 25.1 fps while
talking; its `seen.png` is the *camera* frame (the capture does not store
the avatar's frame; proposal below). `_anchor_frame_ok` finds no rest in
that capture (mouth open throughout), so the gate replay's rest is its 30
shutest frames: mouth flips 0.42/s, opening 0.25/s, moving-but-held 5%,
shut-but-driven 18% -- talking behaves as the day-11 build measured.

*white_line_on_cheek and look_down_emotion_and_white_line -- a light band
along his left jaw (image left), ear to chin.* Not the portrait's white
background: the band's brightest pixel is 157 (first shot) and 193
(second) on its darkest channel, the background is 245; and at 35 degrees
of yaw and 8.7 of roll on driving.mkv (frames 1490, 1556, rendered today)
no near-white pixel appears on the avatar half that is not on the camera
half. Zoomed, the band is smooth pale skin between the avatar's beard line
and the outside: the portrait's own bare jawline -- its beard stops short
of the jaw contour, his runs along it -- pasted over his jaw, which at
night sits in shadow. The mask's join is the portrait oval's, 4% of face
width inside the portrait's contour with a 7% feather, i.e. on the
portrait's bare skin, and the colour correction is a plane
(`_fit_lighting_plane`): it cannot darken one band at the jaw. The
daytime clips hide it because his own jaw is lit like the portrait's.
Two fixes, both measurable with a band-mismatch metric (mean |output -
scene| in the 10 px inside the mask edge, jaw half): (a) the join moved
inside the beard at the jaw -- a jaw-only shrink of the driver oval's
lower points, ~8% of face width, so his own jaw shows (his 2026-08-15
preference); (b) the correction from a plane to the warp engine's blurred
low-resolution difference (`engine_warp._match_colour`, ~1 ms), which can
darken locally. (a) first -- geometry, costs nothing; (b) helps every
night-time mismatch. A portrait whose beard runs the jaw as his does
would remove the cause.

*look_down_emotion -- pitched down, irises at the bottom of wide eyes, a
pouted "o".* The "o" is not it: natural.mkv 1277/1290 (his "o"s, rendered
today, stills sent) give a level "o" with the irises centred, and on both
clips the avatar's gaze_y on open-and-narrow mouths equals its shut-mouth
value (-0.106 / -0.093 against -0.107 / -0.098). What the screenshot
shows is his head pitched down (eye pivot: forehead grows, chin tucks --
as designed) with the avatar's irises rigid in the lids where his either
counter-rotate toward the camera or drop to the screen -- vertical gaze,
no lever (day 11) -- and the eyes wide: his own EAR in the 05:23 capture
is 0.41 against 0.31-0.32 on the daytime clips, so either his eyes were
wider at night or the anchor learned a narrower rest and `eye_headroom
1.1` opened them. Needs the moment captured with the avatar's frame in it.

*smile_without_teeth -- a very wide closed-lip smile.* Not reproduced on
the clip: driving.mkv 925/968 (his toothy smile) show teeth, and still
show them with the eye-line crop forced fully on (`_CROP_CENTRE_PITCH`
patched to always; the always-on variant's o_mar 0.196 against his
0.188). The shape is exactly what the gates produce when `mouth` is open
and `opening` shut: width from keypoints 7/14, no parting of 17/19/20.
The opening gate reads `_lip_ratio` = inner-lip gap over MOUTH WIDTH,
0.010/0.030 above rest: the wider the smile, the larger the gap must be
to pass -- a 4-px gap reads 0.067 at a 60-px mouth and 0.033 at a 120-px
grin, his rest 0.006-0.02. The clip's smile has an 18-px-class gap
(0.188) and passes; a grin with the teeth just showing does not. Fix:
gate the opening on the gap over a width-independent size (the jaw width
`_JAW_L/_JAW_R` the width ratio already uses, or eye-corners-to-chin),
thresholds re-derived so the clips' talk / smile / shut segments keep
their gate values, then verified on a smile capture (he presses the
button while smiling).

*closed_eyes -- narrowed lids, level head, faint smile.* The avatar's
lids copy his (`_eye_target`, eye_gain 0.25): on natural.mkv TALK, 33 of
the 35 frames where his EAR falls under 60% of rest are frames where his
iris sits more than 0.05 of eye width below rest (gaze down: the
screen), 2 are not; corr(his EAR, his iris height) is -0.85 driving /
-0.91 natural. His lids drop when he looks down; the avatar's irises
cannot follow (vertical gaze), so it shows the narrowing alone -- eyes
"closed". In the driving "look DOWN" his EAR reaches 0.35 of rest, the
avatar's 0.55. Either accept, or a `gaze_down_relief`: while his iris is
below rest and the drop is slow (not a blink -- a blink hides the iris
and lands in 2-3 frames), hold the eye target at rest so the avatar keeps
its eyes open looking straight. His call: open-and-straight or narrowed.
(A blink caught by the screenshot is the other reading; a capture with
the avatar's frame would settle it.)

*Two tool corrections.* `tracking.npz`'s `pose` rows are (yaw, pitch,
roll) (runner.capture); `gate_replay.py` and today's `capture_stats.py`
read them as (pitch, yaw, roll) -- no gate result changes (both under 5
degrees on every capture) but the day-9-night and day-11 replays'
"pitch" was yaw; fixed in both. And the capture's `seen.png` is the
camera frame, not what he saw: proposal -- the runner also writes
`shown.png`, the composited output at the press, so a screenshot's
complaint can be replayed against the avatar's actual frame instead of
guessed at from the clips, as all four had to be tonight.

*Stills for him (his face; delete after he looks):*
`recordings/day40-turn-f{925,968,1486,1490,1551,1556}.png`,
`day40-o-f{1277,1290}.png`, `day40-smile-eyeline-f{925,968}.png`, and
the three `day40-*.mp4`. `day40-o-f1290` and `day40-smile-eyeline-f968`
sent.

*Next, in order:* (1) `shown.png` in the capture -- one line, makes every
later complaint reproducible; (2) the jaw shrink and the band metric
(white line); (3) the opening gate over a width-independent size
(smile); (4) his decision on gaze-down relief (closed eyes); the
look-down emotion waits for a capture with the avatar in it. fps
(pipelining) and vertical gaze stand as day 11 left them.

**Day 12 opens with (2026-09-07; his go, given at the end of day 11's
night: "let's start fixing tomorrow"; Fable 5.1 implements directly, as
on day 11).** In this order, each step measured before the next:

1. **`shown.png` in the capture** (`pipeline/runner.py capture`): beside
   `seen.png` (the camera frame) write the composited output frame at the
   press. Until this exists every complaint has to be guessed from the
   clips, as all four were on day 11's night. Then he records four
   moments with the button: a toothy smile; looking at the screen while
   talking; an "o" while looking down; the head tilted at night with the
   jaw band visible. Unit test: a capture holds both files.
2. **The jaw band ("white line")**: a jaw-only shrink of the driver oval's
   lower points (~8% of face width; `_oval_mask` gets a jaw shrink, the
   forehead reach untouched) so the join sits inside the beard and his own
   jaw shows. Metric first: mean |output - scene| in the 10 px inside the
   mask edge, jaw half, on his night capture (once step 1 exists) and on
   natural.mkv; bars: blink gain >= 0.85, talk mar gain >= 0.85 unchanged.
   If the band survives the shrink, the colour field from the plane to
   the warp engine's blurred low-resolution difference (~1 ms).
3. **The opening gate over a width-independent size** (smile without
   teeth): `_lip_ratio`'s denominator from the mouth corners' distance to
   the jaw width at mouth height (`_JAW_L/_JAW_R`, what
   `_mouth_width_ratio` already divides by); `_MOUTH_GATE_MAR` and
   `_MOUTH_ACTIVITY_MAR` re-derived so the clips' shut / talk / smile
   segments keep their gate values within 0.05 (gate_replay on all
   captures, the pose order fixed); verified on his smile capture.
4. **Closed eyes -- his decision first**: open-and-straight while he reads
   the screen (`gaze_down_relief`: hold the eye target at rest while his
   iris is below rest by > 0.03 of eye width and the drop is slow; a
   blink -- iris hidden, EAR under 0.5 of rest within 3 frames -- still
   closes), or narrowed as now. Bars if built: driving "look DOWN" avatar
   EAR >= 0.9 of rest, blink gain unchanged >= 0.85.
5. The look-down emotion waits for its capture with `shown.png`.

The day-11 probes are in `scripts/probes/` now (`gate_replay.py`,
`capture_stats.py`, `white_leak.py`, `render_eyeline_on.py`) -- the
session scratchpad they lived in does not survive a new chat.

Not on day 12 unless he asks: fps pipelining (+1 frame latency, his OK
pending), vertical gaze (no lever), crop C, the yaw pivot.

**Day 12 (2026-09-07, Fable 5.1, his "start from step 1, in order"; steps
1-3 built and measured, step 4 built behind a switch that is off until he
decides).** 927 tests pass (up from 913: the capture's two files, the jaw
shrink and its option, the opening measure, the relief); mypy the same 12
pre-existing errors; ruff clean. Nothing committed.

*Step 1 -- the capture, and a correction.* The night's premise was wrong:
`seen.png` was never the camera frame. `PipelineRunner.capture` writes
`snapshot.frame.data`, which is the renderer's canvas -- the engine's
composited output (`_paste` returns the scene with the face blended in) --
and the day-9 entry above already says so ("his room and hair with the
avatar's face"); the 05:23:04 capture's own `seen.png`, zoomed, carries the
pale cheek band, which only the avatar has. So what was missing was the
opposite file: the camera frame at the press, without which the neural
engine (which reads pixels, not landmarks) cannot be replayed and the jaw
metric below has no "scene" to compare against. The capture now writes
`camera.png` -- the source frame of the same snapshot, so the same tick as
`seen.png` -- beside it. It is a photograph of him, written knowingly; the
runner's docstring says so and the "That looked wrong" dialog names all
three files. `_as_bgr` converts either frame for `imwrite` (NV12 skipped
with a warning). Test: a capture holds both files and they are not the
same picture. His part unchanged: four moments with the button -- a toothy
smile; looking at the screen while talking; an "o" while looking down; the
head tilted at night with the band visible.

*Step 2 -- the jaw band.* Metric first, `scripts/probes/jaw_band.py`: the
mask the engine actually pasted with (`loaded.last`) warped into the frame,
the 10 px inside its 0.5 contour (distance transform), jaw half (rows below
the mask's middle); `abs` = mean |output - scene| there in grey levels,
`step` = mean(output) - mean(scene), signed (a pale band over a shadowed
jaw is positive). Clip mode renders as the app does (`--option` for the
engine option, `--stills` writes camera|output face zooms); capture mode
takes `seen.png` + `camera.png` under the driver's oval from the capture's
last frame. Then the shrink: `_oval_mask(..., jaw_shrink=)`, each oval
point below the eye line pulled toward the face centre by up to
`_OVAL_JAW_SHRINK` = 0.08 of face width -- nothing at the eye line, the
full amount from the nose tip down (the crop is rolled with the face, so
"down" holds under roll); applied to the driver's oval only, in
`_frame_mask`, so the portrait's mask and its box are untouched; the
forehead reach as before. Engine option `mask_jaw_shrink` (0 is the
day-11 mask; validated at load). On the synthetic face the chin rises
0.077 of face width and the mouth row narrows 0.18 (two sides), the eye
row 0.05 (the hull's taper), the top row not at all.

| jaw band, p50 / p90 (grey levels) | natural.mkv before | after | driving.mkv before | after |
|---|---|---|---|---|
| abs, jaw half | 24.8 / 26.5 | **15.8 / 18.4** | 27.5 / 29.6 | **15.6 / 18.1** |
| step, jaw half | 15.6 / 19.8 | **8.0 / 13.2** | 16.0 / 22.3 | **8.0 / 13.9** |
| abs, whole band | 21.4 / 23.5 | 16.5 / 18.4 | 22.1 / 23.8 | 15.1 / 17.2 |
| blink gain (driving) | 0.850 (day 11) | **0.90** | | |
| talk 713-1306 mar gain (natural) | 0.87 (day 11) | **0.859** | | |

The mismatch at the jaw drops by a third and the pale step halves on both
daytime clips; both bars clear 0.85. Stills, his face, delete after he
looks: `recordings/day12-jaw-{natural,driving}-s{0,default}-f*.png`
(natural 800 / 1277 / 1576, driving 925 / 1490 / 1556; `s0` is before).
At driving 1556 (the 35-degree turn the night measured) his own beard now
runs the jaw and the chin; the avatar's bare jawline is gone from the
join. The night verdict -- the band in his shadowed room -- needs his
capture with `camera.png` (capture mode above); if it survives there, the
colour field from the plane to a blurred low-resolution difference (b)
stays the next lever. Not done today.

*Step 3 -- the opening gate over the jaw width.* `_mouth_open_ratio`:
inner-lip gap over the jaw width at mouth height (58/288, what
`_mouth_width_ratio` divides by). `_lip_ratio` is not touched: it is
LivePortrait's lip-retargeting convention and the anchor's `_ANCHOR_MAX_MAR`
reads it; only what the gates and their rest read changed (`animate`
passes `c_d_open` to `_gates` and, as `_render`'s `mar`, to the anchor's
`rest_mar`; `c_d_lip` stays for `_anchor_frame_ok` and `lip_target`). The
factor: mouth width over jaw width at rest is 0.39-0.43 on both clips and
all nine captures (natural 0.420, driving 0.396), so `_MOUTH_GATE_MAR`
(0.010, 0.030) -> (0.004, 0.012) and `_MOUTH_ACTIVITY_MAR` 0.010 ->
0.004. Replayed with the engine's own `_gates` on both clips (tracked with
the app's settings) and the nine captures, old measure and thresholds
against new: the per-segment gate means moved 0.000 on every still, talk
and smile segment and on every capture (worst 0.004), 0.023 on driving's
"turn RIGHT" (the jaw foreshortens differently from the mouth) -- inside
the 0.05 bar. Live jitter on the new scale: shut-mouth |dMAR| p99
0.0023-0.0024 on the clips, 0.0043 on capture 232151 (day 9 night's 0.011
times 0.4); the test constants follow. Unit test: the same 4-px gap at a
60-px and a 120-px mouth reads the same opening ratio while `_lip_ratio`
halves -- the smile-without-teeth case, which the clips do not contain and
his smile capture will show. `gate_replay.py` and `capture_stats.py` read
the new measure (the shut-mouth threshold 0.006 -> 0.0024; the variants'
thresholds scaled). Bars with the gates in: blink gain 0.90, talk mar gain
0.859 (>= 0.85; the talk segment's opening gate reads 0.916 either way).

*Step 4 -- closed eyes, his call, built so the choice is measured.*
`gaze_down_relief` (engine option, default **off**: the narrowed look as
now). On: while his iris sits more than 0.03 of eye width below his rest
(`rest_gaze`, the day-11 anchor reading; +y down) and the lids came down
slowly, `_relieved_ear` builds the eye target from his rest ratio -- no
narrower than rest -- so the avatar keeps its eyes open looking straight.
A blink still closes: eyes under 0.5 of rest, or a fall of 0.25 of rest
within the last 3 frames (`ear_recent`, cleared with the anchor), pass
through untouched; so does anything wider than rest. The anchor and the
gaze blink hold still read his raw ratio. driving.mkv, "Look UP, then
DOWN" 421-538, frames with his iris > 0.03 below rest: the sidecar's segment holds only the
first five frames of the glance (a blink-speed drop to 0.35 of rest); the
look-down itself runs 534-575, his iris 0.06 below rest and his lids
settled at 0.67 of rest -- 36 frames with the iris > 0.03 below rest and
the eyes over 0.3 of rest (the measure).

| driving.mkv | relief off | **relief on** | bar |
|---|---|---|---|
| look-down avatar EAR, p50 of rest | 0.82 | **1.05** | >= 0.9 |
| look-down avatar EAR, p10 / min | 0.64 / 0.53 | 0.65 / 0.53 | (the three frames after the drop, held by the blink rule) |
| blink gain | 0.902 | **0.909** | >= 0.85 |

Both bars clear. What he must also know before saying yes: the rule fires
wherever the iris reads low against the eye corners, and on this clip that
is also the toothy smile (36 of 118 frames), the wide-open mouth (32 of
118) and the head turns (9-14 of ~150) -- his squint on a wide smile would
be undone while the iris reads down. A "no smile" exclusion (mouth width
under rest + 3%) removes those but also two-thirds of the real look-down
frames (his width reading moved with the pitched head), so it is not the
discriminator; if he wants the relief and the squint, that needs its own
measure. Off by default, so nothing he sees changed. CSVs:
`recordings/driving-follow-day12-relief.csv` (on), `-gates.csv` (off). To try it:
`"gaze_down_relief": true` in `avatars/my_avatar_neural/metadata.json`;
his verdict decides the default.

*Not done.* The look-down emotion (step 5) and the night jaw verdict wait
for his captures with `camera.png`. fps pipelining, vertical gaze, crop C,
the yaw pivot: as day 11 left them.

**Day 12, his first look (2026-09-06, 23:41 screenshot; he had shaved that
morning).** "Quite good, I am satisfied with this development" -- and one
picture: the avatar's face ends at the mask edge and his own bare cheeks,
jaw and chin show under it, a ledge. That is the jaw shrink doing exactly
what it was built to do, on a face that no longer has the beard it was
built for: the join was moved 8% inside the contour so it would sit
*inside his beard*; clean-shaven, it sits on bare skin and cuts the
avatar's stubbled chin short. So the shrink is a beard setting, not a
default: `_OVAL_JAW_SHRINK` is 0 again (the join at the contour, as
before day 12, for every user), and `mask_jaw_shrink` is written into his
metadata.json at 0.0 -- back to 0.08 when the beard is back, the day-12
table stands for that case. The night band was a beard-vs-bare-jaw
mismatch too (his beard in shadow under the portrait's bare jaw); shaved,
its cause is gone with it, and tonight's capture says whether anything
remains. The portrait's own moustache and chin stubble are now the
mismatch in the other direction; a clean-shaven portrait would remove it
(his call, a portrait lever). Tests: the default test reads 0.0.

**Day 12, night (2026-09-07, 00:15-01:40; his four captures were to come, one
did): "the face trembles when I smile -- it cannot settle how wide the smile
should be." Found, and it was not the smile.**

*The trail.* His capture at 00:15:55 (the night jaw: band gone, step +0.8
grey levels against +16 on the clips with the beard; the forehead line is
what shows now). Then the smile, on driving.mkv's closed-lip smile
(806-870, his width steady at 0.628-0.636, gates 1.00 throughout, crop
constant to 0.5 px, pose constant, no hold, no blink): the avatar's mouth
width fell 0.666 -> 0.561 in two frames and climbed back over twelve,
three times in sixty frames, the corners dropping 6-8 px with it. Not the
camera (the mouth box's sharpness, brightness and frame-to-frame change
flat at the drops), not the measuring tracker (stills of the render show
it: 815 smiling, 816-819 neutral, 824 half, 828 smiling). The motion
extractor's own reading, read eagerly on the same crops: keypoint 14's y
relative to the anchor flipping -0.010 / -0.007 / -0.004 between
consecutive frames -- three levels, no in-between.

*The cause.* The extractor ran in fp16 by a plain `.half()` on the whole
module (day 1). LivePortrait's own half-precision is `torch.autocast`,
which keeps the normalisations in fp32; ours put the ConvNeXt-V2's global
norms in fp16 too. That is not a precision loss, it is a different
function: on the portrait, fp16 read scale 1.86 (fp32: 1.01), t -0.073
(-0.010), the eye keypoints' y at -0.31..-0.14 (+0.23..+0.30) and the
anchor's eye expression at -0.07 (-0.002). Frame-to-frame |change| on a
held smile, fp16 against fp32: keypoint 14 y 0.0030 vs 0.0002, yaw 0.64
vs 0.07 degrees, scale 0.022 vs 0.003, t x 0.0034 vs 0.0003 (bf16 nearly
as good as fp32). The cost is nothing: 5.1 ms fp16, 5.0 ms fp32 eager
(cuDNN's convolutions run TF32 either way). The other three networks in
fp16 are fine: warping, decoder and the three MLPs re-run in fp32 give
the same picture (mean 0.12-0.16 grey levels, p99 2; the tracked EAR,
mouth ratio and width identical), so they stay half.

*What it explains.* Twelve days of gains were fitted to that function:
eye_gain 0.25 against LivePortrait's own 1.0 (the eye network was reading
an out-of-range x_s); the lip network "only ever pressed the lips
thinner" and stitching "sat the face to the side" (both read x_s); the
crop-scale cliff and the grimaces at 2.1/2.3; "the avatar wanders while
I sit still"; "the face bounces on the slightest sideways shift"; day
7's EMA that could not clear its bar. And the relative expression came
out about 1.8x too large, flickering -- the closed-lip smile he liked was
partly noise.

*The build now.* Motion extractor in fp32 (`_load_nets`), the image cast
on the way in. `eye_gain` 1.0 (0.25 barely closed the eyes on the
correct x_s: blink gain 0.16). A new `expression_gain` (default 1.0,
[0, 3]; the day-6 lever) on the whole relative expression except the
eyes' rows, so the amplitude is put back by measurement; his metadata
carries 1.8. `mouth_smile_gain`'s bound is [0, 3] now, for the closed
smile if he wants it wider. Measured on both clips (eye_gain 1.0):

| driving.mkv | fp16, day 11 | fp32, eg 1.0 | fp32, eg 1.5 | **fp32, eg 1.8** | fp32, eg 2.2 |
|---|---|---|---|---|---|
| blink gain (>= 0.85) | 0.902 | 0.971 | 0.972 | **0.971** | 0.971 |
| talk + open-mouth mar gain | 0.636 | 0.842 | 0.957 | **0.987** | 0.953 |
| natural talk 713-1306 mar gain (>= 0.85) | 0.859 | -- | -- | **0.941** | -- |
| smile width gain (both smiles / closed) | 0.728 / 1.383 | 0.182 / 0.078 | 0.320 / 0.202 | **0.441 / 0.288** | 0.588 / 0.383 |
| brow gain | 0.641 | 0.674 | 0.693 | 0.701 | 0.711 |
| closed smile: avatar width jitter p90 | 2.46% | 0.30% | 0.44% | **0.50%** | 0.54% |
| still: avatar EAR / brow / width spread | .038 / .013 / .022 | .027 / .015 / .016 | same | **.027 / .015 / .016** | same |
| turn L/R: dx_nose p90 | 47.7 px | 19.7 | 19.6 | **19.9** | 19.9 |
| natural 681 / 1576 dy_eye | -5.2 / -1.6 | -- | -- | -6.8 / -6.9 | -- |

1.8 kept: the talk gain reads 1.0 on driving and 0.94 on natural, the
blink 0.97, the turn lag less than half of day 11's, the smile's jitter a
fifth, and the still face steadier than it has ever been. What he will
see smaller: the closed-lip smile (0.29 against the old, inflated 1.38)
-- `mouth_smile_gain` above 1 is the knob, unmeasured. The natural
clip's still-segment EAR spread is 0.044 against 0.035 (eye_gain 1.0
passes more of the tracker's own jitter on his lids), and 1576's dy_eye
-6.9 sits at the edge of the rest +-5 bar. 213 engine tests; ruff and
mypy as before. Not committed until he has looked.

*His look, 03:00 (five captures 02:52-02:55: closed smile, toothy smile,
turn with a blink, eyes shut, a 42-degree turn).* "The only thing I do
not like is that my forehead shows -- that line is bad and noticeable.
Everything else is good. Push this version to main, it is the best we
have had." Pushed. The forehead line -- the mask's top edge at the
hairline, the portrait's paler forehead over his -- is the next thing.

*Re-opened by this, in order of likely payoff, each a measurement away:*
lip_retargeting and stitching (both read x_s, both were judged on the
broken one); the crop scale (2.2 was chosen against fp16's cliffs, and
2.2 in our box is about LivePortrait's 2.5 -- the expression reads small
partly for that); eye_headroom / eye_scale; the gate thresholds do not
move (they read his geometry, not the network).

**Day 13 opens with (2026-09-07; his two last captures of the night, 03:24,
an "o" and a wide-open mouth, and his words: "if we fix these too it will
be very good").** In this order, each measured before the next, on the
fp32 build (a5d1217):

1. **`scale_gain`** -- the eyes and brows grow when the mouth opens. On
   driving.mkv's wide-open frames the avatar's inter-ocular distance is
   1.087 of its rest (his 1.007), correlation 0.98 with his mouth ratio,
   and the extractor's relative scale reads 1.081 on the same frames; in
   fp16 the scale head did not respond (1.010), which is why this is new.
   The crop is already sized from his resting face height by head-pose
   distance, so the driver's relative scale carries only what the
   extractor mistakes for it -- the day-11 argument for
   `translation_gain_y`, applied to `scale_new`: `scale_gain` in
   `[0, 1]`, default 0 (the portrait's own scale, always), the old line
   bit-exact at 1. Bars: on the wide-open frames avatar inter-ocular
   within 2% of rest; talk mar gain and blink gain unchanged (>= 0.85);
   the head-turn lag not worse than 20 px p90.
2. **The jaw under an open mouth** -- his own jaw shows below the
   avatar's on an "o". The paste mask is the portrait's oval intersected
   with his; his oval follows his dropped jaw, the portrait's is the
   closed-mouth chin, and the intersection stops there. Fix: the
   portrait's oval re-rendered per frame with its points below the mouth
   line moved down by his chin's drop from rest (crop pixels, ramped
   from the mouth to the chin), so the join follows the opening; the
   mask box extended downward by the largest drop the anchor's face
   height allows. Metric: the jaw band (`jaw_band.py`) on driving.mkv
   986-1103 and on the two 03:24 captures' open frames (clip mode on
   the capture is not possible -- the last frame is after the "o" -- so
   the driving segment is the bar); stills at 1040 and 1060 for him.
   Bars: jaw abs on the open frames not worse than the shut ones'; blink
   and talk gains unchanged.
3. **The forehead line** -- his one complaint on the build, worse on a
   turn. Measured tonight on his frontal captures: the avatar's forehead
   is +25 grey levels brighter than his in the ten pixels inside the
   mask's top edge (the jaw: -6). First the number on driving.mkv's
   turns (1380-1600, the day-3 "left-turn hairline band"), then two
   levers in order: the top edge lower on the forehead
   (`_OVAL_FOREHEAD_REACH` 0.85 -> a sweep), and the colour correction
   from a plane to the warp engine's blurred low-resolution difference,
   which can darken the forehead band alone. Bars: forehead step within
   +-8 grey levels frontal and turned, the jaw's not worse.
4. Then Phase 3 as agreed on the night of day 12: GPU check with the
   photograph engine as the fallback, a shippable virtual camera,
   enrolment, the `.exe`, the kill switch.

Re-opened by the fp32 finding and not on this list unless a step above
needs them: lip_retargeting, stitching, the crop scale,
eye_headroom/eye_scale, `mouth_smile_gain` above 1 for the closed smile.
A clean-shaven portrait taken in his room with his camera is the largest
single lever and is his to take.

**Day 13 (2026-09-08): the scale, the jaw under an open mouth, the forehead
line -- the three in the order they were listed, each measured before the
next, on the fp32 build (a5d1217).**

*Step 1 -- `scale_gain`, the eyes that grew with an open mouth.* The
measurement first, on driving.mkv with `measure_follow.py` (a `rel_scale`
column now: the extractor's size-smoothed relative scale as `_drive` reads
it): on the wide-open frames (986-1103, his mouth ratio over half the
segment's peak, 55 frames) the extractor reads 1.100 of the anchor (p50;
1.105 max) and the avatar's inter-ocular distance 1.088 of its rest, his
own 1.005, correlation with his mouth ratio 0.98 -- the day-13 note's
numbers, reproduced. The same head also reads 1.065 (p90) while he talks,
1.099 while he raises his brows and 1.077 through the right turn: whatever
makes the face taller or wider in the crop is read as nearer, and the crop
is already sized from his resting face height by head-pose distance, so
none of it is distance. `scale_gain` in `[0, 1]`, default 0: `scale_new =
scale_s * (1 + (scale_now / scale_d0 - 1) * scale_gain)`, the old line
bit-exact at 1 (the run at 1.0 reproduces the baseline frame for frame:
avatar inter-ocular within 0.02 px at the median). A (1, 1) tensor baked
at load like `t_gain`.

| driving.mkv | scale_gain 1 (old) | **scale_gain 0** | bar |
|---|---|---|---|
| wide-open: avatar inter-ocular / rest, p50 / p90 | 1.088 / 1.093 | **1.001 / 1.004** | within 2% |
| talk: avatar inter-ocular / rest, p50 / p90 | 1.042 / 1.056 | **1.014 / 1.029** | |
| brow raise: avatar inter-ocular / rest, p50 / p90 | 1.039 / 1.116 | **1.016 / 1.033** | |
| blink gain | 0.919 | **0.931** | >= 0.85 |
| talk + open-mouth mar gain | 0.988 | **1.112** | >= 0.85 |
| smile width gain / brow gain | 0.442 / 0.701 | 0.445 / 0.702 | unchanged |
| turns 1342-1641: dx_nose p90 | 19.9 px | **22.5 px** | <= 20 px |

Three of the four bars clear; the turn lag does not, and the reason is
worth having: through the right turn the extractor reads the face 7.7%
larger (rel_scale 1.077), and at gain 1 that inflation happened to push
the avatar's nose the way his was turning -- the tracker read the avatar's
inter-ocular at 0.876 of rest on that turn against the foreshortened
0.823 it reads now. The scale was hiding two and a half pixels of the yaw
lag that is already on the open list ("yaw pivot"), by growing the whole
face 6% on every turn. Kept at 0: the eyes, brows and face that grew on
every open mouth, every word and every brow raise were what he saw; the
2.5 px on a turn is the lag's own, and the pivot is the fix for it. The
mar gain at 1.11 is the avatar's mouth width no longer inflated with the
face -- the opening itself is unchanged in pixels; `expression_gain`
below 1.8 would trim it and the smile with it (0.44), so it stays.

*Step 2 -- the jaw under an open mouth.* His nose-tip-to-chin reach in
crop pixels (`_nose_chin`; the chin's own y moves with the crop's
centring, this does not) is learned at the anchor like the brow rest
(`rest_chin`, `chin_sample` into `_learn_anchor`), and each rendered
frame's excess over it (`_chin_drop`, never negative, capped at
`_CHIN_DROP_MAX` 0.35 of the portrait's crop face height, ~80 px) lets
the portrait's oval down: `_oval_mask(chin_drop=...)` moves its points
below the mouth line down by the drop ramped from nothing at the mouth
corners to all of it at the chin, `_frame_mask` rebuilds the portrait's
half with it when the drop is over half a pixel, and the mask box is
extended below by the cap at load (`_mask_bbox(reach_below=...)`) so the
paste still warps into one fixed box. A held frame reads a drop of 0 --
its chin has not moved. `chin_follow` in `[0, 1]`, default 1, is the
measured-against knob (0 is the day-12 mask). Measured with
`jaw_band.py` on driving.mkv:

| jaw band abs, grey levels | chin_follow 0 | **chin_follow 1** |
|---|---|---|
| open 986-1103, p50 / p90 / max | 27.3 / 35.3 / 43.2 | **27.5 / 33.2 / 35.0** |
| open 986-1103, step p90 / max | +33.5 / +41.9 | **+28.4 / +34.9** |
| shut 0-119, p50 / p90 | 29.4 / 29.8 | 29.8 / 30.4 |

The bar (open not worse than shut) holds at the median as it already did;
where the fix lives is the tail, the wide-open frames: the worst frame
43 -> 35, and the stills at 1040 and 1060 (`recordings/day13-jaw-cf{0,1}-
driving-f*.png`) show it -- at 1060 the day-12 mask stops at the
avatar's closed-mouth chin with his own open jaw under it, the day-13
mask follows the jaw down. At rest the drop reads 3 px (p90) -- the
anchor's reach is a mean, the frame's a reading -- under a 16-px
feather; open, 24 px at the median and 51 at the widest.

*Step 3 -- the forehead line.* The number first. `jaw_band.py` reads the
forehead now (the same ten-pixel band over the mask's top fifth of rows,
`fore abs` / `fore step`) and has a `--render` capture mode: the capture's
300 frames of tracking fed to *this* build's engine with `camera.png` as
the picture, the last frame measured against `camera.png` under the mask
the engine actually pasted with -- so a mask or colour change is judged on
the frames he complained about, not only on a clip shot in daylight. His
five 02:52-02:55 captures re-rendered by the day-12 build: forehead step
+26.1..+27.1 grey levels, abs 46-51 (the two 03:24 captures, sat lower:
-1.8 / +11.3); the jaw band on the same captures +2.5 (his shave).
driving.mkv's turns 1380-1600: +1.8 p50, +17.6 p90, +27.3 max; frontal
0-119: -2.6 -- the clip barely has it, which is why it took his captures
to see it.

Why the plane cannot fix it: one brightness and one gradient per channel
is what a plane is, and a face too pale at its top edge and slightly dark
at its bottom while right in the middle is curvature at the edges -- the
thing the warp engine's own measurement kept the plane from representing,
because the same curvature *inside* the face is a brow shadow or a beard,
the user's own, which must never be printed onto the avatar. So the plane
stays the whole-face correction and what it leaves is put back **at the
mask's edge alone** (`_edge_residual`): the residual (alpha-weighted
difference minus the plane, at the same eighth-scale box), blurred over
`_EDGE_BLUR` cells with the blur weighted by the mask, faded from all of
it at the contour to nothing `_EDGE_REACH` (0.2) of the box's height
inside. The join is where the two pictures have to agree; the middle of
the face keeps the avatar's features. `edge_match` in `[0, 1]`, default 1
(0 is the plane alone). The second lever, `mask_forehead_reach` (the
ovals' reach up the forehead, 0.85 in code as before), was swept with it.
Over the seven captures (median / the worst, 025329, the toothy smile):

| captures, re-rendered | forehead step | forehead abs | jaw abs |
|---|---|---|---|
| plane alone, reach 0.85 (day 12) | +26.3 / +27.1 | 48.0 | 21.9 |
| plane alone, reach 0.65 | +13.0 / +15.2 | 43.7 | 21.1 |
| edge 0.5, blur 5, reach 0.85 | +17.8 / +22.1 | 31.3 | 14.7 |
| edge 1, blur 5, reach 0.85 | +9.3 / +17.6 | 15.6 | 9.2 |
| edge 1, blur 5, reach 0.65 | +5.4 / +12.5 | 14.4 | 9.2 |
| edge 1, blur 5, reach 0.55 | +3.6 / +7.9 | 14.8 | 9.2 |
| edge 1, blur 3, reach 0.85 | +6.0 / +15.0 | 13.0 | 8.4 |
| edge 1, blur 3, reach 0.75 | +4.3 / +12.9 | 13.0 | 8.4 |
| **edge 1, blur 3, reach 0.65** | **+3.0 / +11.2** | **12.5** | **8.4** |

The reach alone moves the join onto skin and leaves the two skins
different (abs 43.7); the edge match is what closes it, and the narrower
blur (3 cells, 24 px) keeps more of the edge's own reading. On the clip,
with the kept settings:

| driving.mkv | plane, reach 0.85 | **edge 1, blur 3, reach 0.65** | bar |
|---|---|---|---|
| turns 1380-1600: forehead step p50 / p90 / max | +1.8 / +17.6 / +27.3 | **-0.1 / +5.3 / +8.4** | within +-8 |
| turns: forehead abs p50 / p90 | 12.3 / 22.2 | **5.3 / 10.6** | |
| turns: jaw abs p50 / p90 | 24.4 / 29.5 | **11.7 / 14.0** | not worse |
| frontal 0-119: forehead step / abs | -2.6 / 11.1 | **-0.9 / 3.4** | within +-8 |
| frontal: jaw abs p50 | 29.8 | **11.6** | not worse |

Kept: `edge_match` 1 and `_EDGE_BLUR` 3 in the code, `mask_forehead_reach`
0.65 in his metadata.json -- the code default stays 0.85, the whole
forehead, for the reason the warp engine kept it (his 2026-08 ask); 0.65
is the measured setting for this portrait over this face and is his to
move. What he will see: the top third of his brow-to-hairline is his own
forehead, the avatar's below it; the pale line gone on the frontal
captures (025254, both stills), softened to a gradient on the toothy
smile (025329 at +11 -- his hair over the forehead's left, structure, not
tone); on the turn (1490) the hairline band gone and the jaw edge sunk
into his beard's tone. Stills: `recordings/day13-forehead-*.png`. The jaw
band falling from 22-30 to 8-12 everywhere is the same mechanism at the
bottom edge; the day-12 "white line" measure now reads a third of what it
did.

*His look, 2026-09-09 02:12 (one capture, `moment-20260909-021214`): "it
looks like the AI face came down a little and my eyebrows show."* The
face had not moved (the tracker on the composite against his frame: eyes
-0.4 px, nose -15, chin -9 -- the avatar's features sit where they did,
above his). The mask had: at reach 0.65 its 0.5 contour sat 18 px above
the top of MediaPipe's brow points at the face's centre and its feather
(alpha 0.05) 26 px above -- and his brow hair reaches higher than the
points that trace it, so the feather ran across his brows and they showed
through as a dark smear over the avatar's own (the still
`recordings/day13-brows-r0.65.png`). The same capture re-rendered at
0.75: 26 px; at 0.85: 37 px, the brows clean, and the forehead step on
this night's capture +0.8 (abs 6.1) -- the edge match carries it alone
here (`day13-brows-r0.85.png`). **Reverted: `mask_forehead_reach` 0.85 in
his metadata.json, the code default.** The lesson is the rule's shape,
not the number: a fraction of the brow-to-hairline distance has no floor
above the brows, and on a short forehead (his, tilted 6 degrees down
that night) two thirds of it lands the feather on the brow hair. If the
reach is ever lowered again it needs a clearance floor -- the erosion
(4% of face width) plus the feather (7%) plus the brows' own height above
their points -- measured, not assumed. The 09-07 captures at 0.85 with
the edge match: +6.0 median / +15.0 worst, the table above.

*His second look, the same night (a screenshot, brows raised): "not fixed
-- the eyebrows show; when I raise them, and fainter at rest. Look for
the dark colour above the AI face's brows."* Not the mask this time: on
the 02:12 capture (his brows are raised in it) the mask's alpha reaches
0.95 a full 29 px above his brow top. The edge match itself: re-rendered
with `edge_match` 0 the smear is absent and with 1 it is there (the
zooms `recordings/day13-brows2-plane.png` / `-edge-excluded.png`, left
half before the fix). The residual's fade reaches a fifth of the box
inside the contour -- about 100 px at his distance -- and his brows sit
30-40 px inside it at rest, 15-20 on a raise, so their darkness was
blurred into the correction and printed onto the avatar's forehead: the
warp engine's own "a brow shadow, 70% passed" failure, back at the edge.
Fix: the features that are his and not the join are cut out of the
residual. `_EDGE_FEATURES` (each brow with its eye's corners and lids,
the mouth) are kept from every tracked frame in frame pixels
(`loaded.features`), handed through `_paste` to
`_face_colour_correction`, drawn as convex hulls on the eighth-scale
grid and dilated 2 cells (`_feature_exclusion`, the brow hair above
MediaPipe's brow line), and `_edge_residual(exclude=)` zeroes those
cells in the residual *and* in the blur's weight, so they read as
unknown and fill from the skin around them. On his capture the band
6-30 px above his brow top: darkest tenth 173 -> 178 grey levels, the
smear gone in the zoom; forehead step -0.4. The 09-07 captures again,
reach 0.85 with the exclusion: forehead step **+3.9 median / +13.8
worst** (was +6.0 / +15.0 without it -- his brows had been pulling the
residual there too), jaw abs 8.4. Cost: three small polygons and a
dilation on a ~60x60 grid, nothing measurable. The rule this leaves: an
edge-matched colour field must exclude his features explicitly; a fade
alone is no protection once he raises his brows.

*His third look (02:52, a screenshot): "a small smudge, like low-quality
pixels, above the brows; and when I raise my brows the jaw shows too,
look at the jaw line; and when I bow my head and look down moderately,
my jaw shows more."* Three things.

(1) The pixels were the exclusion zone itself: a hard-edged zone on the
eighth-scale grid, resized up by eight, is eight-pixel blocks wherever
the residual steps at its edge. Softening the zone and fading the
residual to the plane alone over it took the blocks away and left a
dark patch instead (the band 6-30 px above his brow top, darkest tenth:
178 hard, 163 soft-to-plane; his forehead that night was brighter than
the avatar's, the residual around the zone was *brightening* it, and
the zone stayed dark). What the zone needs is the skin's own correction
carried across it: the same skin-only residual blurred wide
(`_EDGE_FILL_BLUR`, 15 cells) reaches over the brows from the forehead
above and the cheek below, and is blended in by the zone softened over
`_EDGE_EXCLUDE_SOFTEN` (5 cells). Darkest tenth 177, no step, no patch
(`recordings/day13-brows3-edge-filled.png`); forehead step on the
capture -0.2, the 09-07 captures unchanged.

(2) The jaw, on the clips first: driving.mkv's brow raise (1642-1760),
its look up-and-down (421-538) and natural.mkv's tilt (417-712), today's
build against yesterday's settings (`chin_follow 0, scale_gain 1,
edge_match 0`): jaw abs p50 11.2 / 11.0 / 10.5 against 28.5 / 28.5 /
23.4, the stills at 1700, 520 and 681 clean. The clips do not show what
he sees; his beard on them is a week fuller than tonight's stubble and
hides the join. The mechanism that fits all three of his words: under a
pitch the avatar's chin, turned about the eyes by the extractor's
under-read pitch, and the mask's bottom part company -- the mask's
portrait half sat at the portrait's *resting* chin because `_chin_drop`
was clamped at zero for a rise, so on a look down the portrait's neck
was pasted between the avatar's risen chin and the mask, and the edge
match now paints that band in his jaw's tone: "my jaw shows more". The
drop is signed now (`+-chin_drop_max`; `_oval_mask` moves the jaw
points up for a negative one), so the mask's bottom follows his chin up
as well as down. On the tilt clip: jaw abs p90 13.8 -> 11.7, max 25.3
-> 19.3, step p90 9.2 -> 6.5 (the deepest frames); the look segment
unchanged (11.0); stills 681/687 (`recordings/day13-jaw-tilt-f681-
{before,signed}.png`). **His two captures then said no, and the signed
drop is reverted** -- and behind them was a wall the whole day-13 jaw
work had been standing on. In order:

*The replay, fixed first.* `jaw_band.py --render` fed the capture's 300
frames of tracking over `camera.png`, and that picture is the *last*
frame's only: an anchor learned from the other 299 frames' landmarks was
cut from a picture they do not match, and the render came out smeared
(the first chin reading on 021214, which the signed drop had been judged
on, was that smear's). It now feeds the last frame's own tracking every
time -- a sane render at the moment's own geometry, the anchor learned
from that frame when it is frontal enough (03:28:24 was, 15 of 15;
03:28:34, yaw -5.5, was not). What a replay still cannot carry is his
live expression, since the frame is then its own anchor; the capture
writing the live anchor is the tool that would.

*What his captures showed (03:28:24 brows raised, 03:28:34 looking
down).* The **render is right**: MediaPipe on the raw 512 render, mapped
into the frame, puts the avatar's chin within 4 px of his own on both
(+3.4 and -0.3 px). The **mask is not**: its 0.5 contour sits 5 px
*above* his chin on both, and its feather reaches the chin line, so the
avatar fades out across his chin and the underside of his own shows
through -- "my jaw". A signed drop makes it worse, not better: on the
brow-raise capture his nose-to-chin reach read 9 px short of rest, so the
mask went *up* 9 px while the rendered chin had not moved, cutting the
avatar's chin 12 px short (contour at 518 against his chin at 534,
against 529 with the drop off). `_chin_drop` is clamped at 0 again and
its docstring says why. The tilt clip's apparent gain was the mask
cutting into a beard thick enough to hide it; a bearded clip is no
measure of a bare chin.

*So: `mask_chin_reach`* -- both ovals let down at the chin by a fraction
of the portrait's crop face height, ramped from the mouth line like the
drop, so the join sits under his chin on his neck where the portrait's
own under-chin skin is pasted and the edge match carries the tone. Swept
on the two captures (mask contour against his chin / jaw band abs):

| reach | contour vs his chin | jaw abs |
|---|---|---|
| 0 (the mask as it was) | -5 / -5 px | 9.6 / 9.7 |
| 0.04 (~9 px) | +10 / +10 | 9.8 / 9.1 |
| 0.08 (~19 px) | +26 / +24 | 11.9 / 11.3 |
| 0.12 (~28 px) | +41 / +38 | 11.9 / 11.9 |

0.04 is where the join leaves his chin line without becoming a chin of
its own (`recordings/day13-chin-reach-*.png`; the band's abs rises with
the reach because the band itself moves onto the neck's shadow).

**And then the wall.** Measured on his portrait's own crop: face height
233 px, the portrait's chin at row **489 of 512** -- twenty-two pixels of
crop below the chin, because `crop_vy_ratio` -0.125 spends the crop's
room *above* the face (256 px of hair) and leaves almost none under it.
The oval's own edge needs more than that: the erosion pulls its 0.5
contour in by `_OVAL_SHRINK` (4% of face width, 9 px) and the feather
fades over about half its kernel (8 px) past that. A hull let down past
the last row is not let down at all -- `fillConvexPoly` clips it at the
array's edge and `erode` does not shrink a shape at that border -- so the
mask ends in a **hard row instead of a feather**, and the paste draws a
straight dark line across his neck. That is what today's drop had been
doing: on driving.mkv's open-mouth segment the mask's alpha reached the
crop's last row on **118 of 118 frames** (the still at 1060 shows the
line). The same wall limits the *render*: on 18 of those frames the
avatar's own rendered chin sits within 10 px of the crop's bottom, row
510 at the widest.

So the drop and the reach now share whatever room the crop actually has,
less that edge margin (`load_avatar`, logged at load). On his portrait
that room is **zero**: both are clamped to 0, the mask is exactly the
one he had before today's jaw work, and the clipping is gone -- 0 of 118
frames at the last row, the mask's 0.5 contour at row 492 with its tail
at 497, the open-mouth band 11.7 p50 / 16.5 max and no line
(`recordings/day13-jaw-open-f1060-capped.png`). The features stay,
tested, for a crop that has room.

**The jaw join cannot move in this crop, and that is the finding.** The
lever is the framing itself -- `crop_vy_ratio`, which decides how the
crop's 512 rows are spent between his hair and his neck -- and it moves
what the motion extractor sees, which is the day-6 cliff's own axis. It
gets a sweep of its own, both clips, every gain, before anything is
kept. Until then the join sits where the mask's erosion and feather put
it: about 5 px above his chin line.

*One more thing in that capture.* 032824 has no face at all for its first
74 frames -- two and a half seconds in which the engine's passthrough put
his own face on the virtual camera. Exactly the case his "connection
dropped" blur is for (parked above).

**Who wrote which half of this day, and what the next session should
check (2026-09-09, ~03:45).** Fable 5.1 reached its usage limit partway
through the jaw work and he switched to Opus 5 (1M context), which
continued in the same session with the whole conversation in context.
The line falls exactly here: **Fable 5.1** did steps 1-3 (`scale_gain`,
the chin drop, the forehead line), his first look (the reach reverted to
0.85), his second (the feature exclusion), his third (the fill, the
signed drop, `mask_chin_reach` and its sweep on the two captures, up to
writing 0.08 into his metadata.json). **Opus 5** did everything after the
reach sweep: the clip A/B that showed the reach making the band worse,
the crop measurement behind it, `crop_room.py`, the headroom cap and its
edge margin, taking `mask_chin_reach` back out of his metadata, and this
entry from "And then the wall" onward.

Four judgements of that second half are worth a second opinion, in
order of how much rests on them:

1. **The edge margin's shape.** `_OVAL_SHRINK + _OVAL_FEATHER / 2` plus
   one pixel is an argument about how `erode` and `GaussianBlur` behave
   at an array's border, not a measured constant. It is verified only by
   its effect (118 of 118 clipped frames -> 0 of 118, the mask's tail at
   row 497 of 511), so it may be a pixel or two conservative. If the
   crop sweep gives the chin real room, this constant decides how much
   of it the mask may use, and it should be measured then rather than
   argued.
2. **Reverting the signed chin drop on two captures.** One of them
   (03:28:34, yaw -5.5) could not seed an anchor, so its numbers are the
   frame's own geometry, not his live state; the other (03:28:24) did,
   15 of 15. Both agree the rendered chin lands within 4 px of his, and
   the clips agree, but two captures and a beard-hiding clip is thin
   evidence for a mechanism that only shows on a bare chin.
3. **`mask_chin_reach` kept in the code at default 0 and out of his
   metadata.** It is inert on his portrait (the crop has no room), so it
   ships untested live -- a feature whose only measurements are a sweep
   on two still captures. The alternative was to delete it until the
   crop can use it.
4. **The clip A/B that started this.** Reach 0.08 read *worse* than 0 on
   both clip segments (open 13.2 vs 11.4, shut 14.2 vs 11.4) while the
   captures' stills read *better*. The reading taken was that the band
   metric moves onto the neck's own shadow and stops describing the
   join -- which is an argument for distrusting the metric exactly where
   it was asked a new question, and it deserves a look.

(3) A limit of the capture replay found on the way: `jaw_band.py
--render` learns its anchor from the capture's own 300 frames, and
`_anchor_frame_ok` takes only |pitch| <= 4 degrees -- on 021214 (pitch
-6.2 throughout) every frame is rejected, so the replay ran without an
anchor (`rest_*` None, chin drop 0, the expression relative to the
portrait's own). The mask and colour numbers on it stand; anything
anchor-relative does not. The fix is the capture writing the live
anchor (exp_d0, R_d0, t_d0, scale_d0, the rest values) so a replay is
the session itself -- not built tonight; until it is, a capture taken
while frontal earlier in the same moment is the workaround.

*Cost, and the whole build measured once more.* Per-stage timings over
driving.mkv 950-1650 (open mouth, talk, turns), this machine:

| ms, p50 / p90 | chin_follow 0, edge_match 0 | **the day-13 build** |
|---|---|---|
| crop | 0.64 / 0.69 | 0.64 / 0.70 |
| model (graph replay) | 29.6 / 30.0 | 29.4 / 29.7 |
| mask | 1.51 / 1.61 | **2.62 / 2.83** |
| paste | 7.91 / 9.23 | **8.36 / 9.73** |
| total | 39.6 / 40.7 | **40.9 / 42.0** |

+1.2 ms a frame: the portrait's oval rebuilt whenever the drop is over
half a pixel (it is on most frames -- at rest the reach jitters 3 px,
under a 16-px feather), and the edge residual with a box 80 px taller.
The day-11 arithmetic stands: 30 fps needs the tracker and the paste
off the GPU's frame, not these. If it ever matters, a 4-px dead band on
the drop skips the rebuild at rest for nothing visible. The chin drop
itself on the clip: p50 24 px on the open segment, 51 px at the widest,
3 px at rest -- well inside the 80-px cap.

`measure_follow.py` on the whole build (scale 0, chin follow, edge
match, reach 0.65), against the step-1 run: blink gain 0.964, talk +
open-mouth mar gain 1.22, brow gain 0.77, smile width 0.48, wide-open
avatar inter-ocular 0.981 / 1.005 of rest (p50 / p90), turns dx_nose p90
18.6 px. Read with care: the mask now hands the tracker HIS upper
forehead and a matched edge, and MediaPipe's whole-face fit on the
composite moves with that -- the brow, mouth and nose readings shift
without the rendered face changing. The step-1 table (same mask both
sides) is the honest A/B for the scale; this line is what the tracker
reads of what he will see. CSVs: `recordings/driving-follow-day13-*.csv`
(`sg0`/`sg1` the scale A/B, `final` the build).

*The build.* `scale_gain` 0, `chin_follow` 1, `edge_match` 1 in the
code's defaults, `mask_forehead_reach` 0.85 (the default; 0.65 tried and
reverted on his look, above) -- all four written into his metadata.json,
like every option before them. 231 engine tests
(five new classes: `TestScaleGain`, `TestChinDrop`, `TestForeheadLine`
and the mask tests' chin-drop cases); ruff and mypy as before. Not
committed until he has looked.

**Day 14 opens with:** the crop's own framing -- `crop_vy_ratio`, swept
the way day 6 swept `crop_scale` (both clips, blink / talk / smile /
brow gains, turn lag, and the grimace frames), for room below the chin
so the jaw join can sit under it and a wide-open jaw can be followed at
all. Then the capture carrying its live anchor, so a replay is the
session and his moments can be measured directly. Then, in order of what
today exposed: the yaw lag the scale was hiding
(turns dx_nose p90 22.5 px; the pivot for yaw, on the open list since day
10); the mouth gain at 1.11 (expression_gain 1.8 was fitted with the
face's inflation in -- a lower value trims the smile too, so it is a
measurement, not a number); the toothy smile's forehead (hair over skin,
structure -- a reach that follows his hairline per frame, if it is worth
it). Then Phase 3 as agreed. Still open from day 12: lip_retargeting,
stitching, the crop scale, eye_headroom/eye_scale, `mouth_smile_gain`
above 1, gaze_down_relief (his call), vertical gaze, crop C, fps
pipelining, the clean-shaven portrait.

**Parked on his ask (2026-09-09, 02:30): his two ideas, to be built in
Phase 3 -- "save these for sure".** The app is to be shown by his friends
to their boss; a failure in front of the boss must look like the
internet, not like the app.

1. **"The connection dropped" -- a full-frame blur of the camera output,
   by hand and automatically.** By hand: a button in the window and a
   global hotkey (the meeting app is in front during a call). Automatic,
   on the failures that today put *his own face* on the virtual camera:
   the face lost (the engine's `_passthrough` returns the raw camera
   frame, `is_placeholder=True`, and the controller publishes it as it
   is), a pipeline exception (`PipelineRunner._run` logs and *returns* --
   the loop stops, the camera keeps the last frame or goes black), the
   face partly out of frame (the paste box clipped), the extreme-pose
   hold (a frozen avatar). Shape: freeze the last good avatar frame for
   ~1.5 s (a real stall freezes first), then the live frame pixelated
   (downscale 1/8-1/16 and up, plus a heavy blur), recover over ~0.5 s
   once the engine has been stable for a second -- hysteresis, never a
   flicker. The preview shows the true output with a "BLUR" indicator so
   he knows what the other side sees. His own caution, agreed: convincing
   once or twice a call, not every minute -- the automatic mode fires on
   real failures only.
2. **An output quality selector -- Native / 720p soft / 480p / 360p.**
   Downscale and upscale back to the camera's own size inside the
   pipeline, so the virtual camera's format never changes and Zoom/Meet
   see the same stream; at 480p the mask edge, the stubble mismatch and
   small defects disappear into what looks like a cheap webcam. This is
   the base the blur sits on: one new stage in
   `PipelineController.process_once` between the renderer and the sink
   (works for both engines, under 1 ms a frame), a setting for the
   default quality, the UI control beside "Virtual camera".

Order proposed: the quality stage first, the button and hotkey on it,
the automatic triggers last. About a day for both, with tests.

**Day 14 (2026-09-10, Fable 5.1 implementing directly on his "review what
we did last, tell me the next step and start on it"): the crop's vertical
framing swept -- this axis has no cliff, and the jaw gets its room.**

*What was swept, and how.* `crop_vy_ratio` moves the crop's centre up the
face by that fraction of the crop; it decides how the 512 rows are spent
between hair and neck. Day 6 swept the crop's *size* and found two cliffs
five percent apart; this was the same protocol on the *placement*, cheap
first, whole-clip second. Cheap: one engine process (`vy_sweep.py`, in the
session scratchpad -- the fore_sweep pattern: reload the avatar per
value, anchor from the clip's first 45 frames as a full run would, then
each hard frame after a 20-frame run-up so the crop's One-Euro, the size
EMA, the gates and the tracker's own smoothing settle), the raw 512
render and the composite read back by MediaPipe in image mode. Twelve
frames of driving.mkv (rest 60, look up 500, eyes 700, closed smile 800,
toothy smile 925, open mouth 1060, the day-6 wide-mouth-plus-pitch 1067,
talk 1200, both turns 1420/1568, brows 1724, frown 1784) and seven of
natural.mkv (rest, screen 300, the tilt 675/681, talk 1000, the open
smile 1380, brows 1576), six values from -0.16 to 0.0. Then
`measure_follow.py` over both clips at -0.09 / -0.06 / -0.03 / 0.0
against a fresh -0.125 baseline of the same build, and `jaw_band.py` on
the open-mouth segment at four values. Portraits 2 and 3 got the eight
expression frames at three values.

*The room.* His portrait in the crop, rows 0..511:

| vy | face top | chin | rows under the chin | drop cap (day-13 margin) |
|---|---|---|---|---|
| -0.16 | 274 | 507 | 4 | 0 |
| **-0.125 (was)** | 256 | 489 | 22 | 7 |
| -0.09 | 238 | 471 | 40 | 25 |
| **-0.06 (kept)** | 223 | 455 | 56 | 40 (**52** with the margin below) |
| -0.03 | 207 | 440 | 71 | 56 |
| 0.0 | 192 | 425 | 86 | 71 |

*The hard frames: nothing breaks anywhere.* Every one of the nineteen
frames renders at every value; the raw render's own readings barely
move. The day-6 cliff frame (1067, his mouth ratio 0.85, eyes 0.11):
avatar mouth 0.79 / 0.92 / 0.90 / 0.94 / 0.89 / 0.83 from -0.16 to 0.0,
eyes 0.22-0.25 throughout. The frown (1784, his eyes 0.298): 0.29-0.31
everywhere -- no grimace. Brows (1724): 0.166-0.171 raw at every value.
Rest eyes 0.306-0.321. Position of the pasted face against his: eye
dx/dy within a pixel of each other across the whole sweep. The one frame
that moved: natural's open smile (1380, his mouth ratio 0.32) read 0.33 /
0.39 / 0.47 / 0.49 at -0.125 / -0.06 / -0.03 / 0.0 -- the mouth opens
more as the face rises in the crop, and the whole-clip numbers say the
same thing smoothly. Portraits 2 and 3 (one grins at rest, one rests with
parted lips): the same picture at -0.125 / -0.06 / 0.0, the wide mouth
rendered at all three, chin rows 490 / 457 / 426 -- the geometry is the
crop rule's, not this portrait's. Turn 1568 is held (yaw past 35) at
every value, so it says nothing.

*Whole clips.* driving.mkv, the day-13 bars plus what the day-6 sweep
watched:

| driving.mkv | -0.125 | -0.09 | **-0.06** | -0.03 | 0.0 | bar |
|---|---|---|---|---|---|---|
| blink gain | 0.92 | 0.89 | **0.90** | 0.87 | 0.85 | >= 0.85 |
| talk + open-mouth mar gain | 1.14 | 1.22 | **1.21** | 1.22 | 1.22 | >= 0.85 |
| talk-only mar gain | 1.07 | 1.07 | **1.10** | 1.21 | 1.25 | |
| smile width gain | 0.45 | 0.52 | **0.51** | 0.65 | 0.73 | |
| brow gain | 0.71 | 0.70 | **0.72** | 0.73 | 0.73 | |
| wide-open: avatar inter-ocular / rest | 0.984 | 0.983 | **0.981** | 0.980 | 0.981 | within 2% |
| turns: \|dx_nose\| p90, not held | 21.9 | 21.8 | **21.7** | 21.3 | 22.0 | not worse |
| frown 1760-1808: avatar EAR p50 (his 0.289) | 0.290 | 0.288 | **0.291** | 0.293 | 0.295 | ~ his |
| still: avatar EAR spread p90-p10 | 0.006 | 0.008 | **0.005** | 0.008 | 0.007 | |
| chin drop, open segment p50 / max | 7 / 7 | 24 / 25 | **24 / 40** | 24 / 51 | 24 / 51 | |
| jaw band, open segment: abs p90 / max | 14.1 / 16.5 | -- | **12.2 / 14.1** | 12.4 / 13.5 | 12.3 / 12.7 | not worse |
| jaw band: step max | 12.4 | -- | **10.4** | 9.6 | 8.6 | |

natural.mkv:

| natural.mkv | -0.125 | **-0.06** | -0.03 | 0.0 |
|---|---|---|---|---|
| talk mar gain | 0.93 | **1.03** | 1.15 | 1.19 |
| smile width gain | 0.88 | **1.04** | 1.15 | 1.20 |
| brow gain | 0.83 | **0.80** | 0.84 | 0.85 |
| tilt, deepest tenth: dy_eye / dy_chin | -8.7 / -15.7 | **-8.5 / -12.0** | -9.0 / -14.1 | -8.9 / -12.1 |
| talk: avatar EAR p50 (his 0.308) | 0.312 | **0.312** | 0.310 | 0.315 |
| still: avatar EAR spread | 0.009 | **0.010** | 0.008 | 0.008 |

Slopes, not cliffs: as the face rises in the crop the mouth and the smile
respond more (driving talk 1.07 -> 1.25, smile width 0.45 -> 0.73;
natural talk 0.93 -> 1.19) and the blink a little less (0.92 -> 0.85);
brows, turns, the frown, the still face and the pasted position do not
move. A reading of why, not a proof: LivePortrait's own crop centres on
its landmark box (eyebrows to chin) and our rigid box (forehead top to
nose) has its centre about a fifth of a face height higher, so at the
same -0.125 our face sits ~50 px *lower* in the crop than a face in
their training crop; -0.03 is roughly where theirs sits. The extractor
reads a face nearer its trained placement as more expressive in the
mouth -- consistent with day 6's finding that it reads its own framing.

*The stills say what the band cannot.* `recordings/day14-vy/zoom-jaw-
f1060.png` and `-f1067.png` (his face, the open mouth at six values, the
jaw zoomed): at -0.16 and -0.125 the avatar's chin ends and his own
darker beard-chin shows under it -- the day-13 complaint; from -0.09 on
the avatar's chin runs down to his jaw line with the join under it. The
band metric's "abs" barely moves because the band moves with the mask
onto his neck; its tail (max 16.5 -> 12.7-14.1) and the step max (12.4
-> 8.6-10.4) are what improve.

**Kept: `crop_vy_ratio` -0.06, in his metadata.json and as the code's
`CROP_VY_RATIO`** (three portraits share the geometry, and the axis is
benign on both clips; the default is what the five other users' avatars
will be cut with). Why not 0.0, which has the most room: the mouth at
1.25 / 1.19 and the smile at 0.73 are a change he would see -- and the
day-13 question of whether the mouth is already over-driven at 1.11 is
open, so more of it is his call, not a side effect of a crop. -0.06
keeps the mouth at today's gain (1.10 / 1.03), the blink at 0.90, gains
0.06 of smile, and leaves 56 rows under the chin: with the margin below,
a 52-px drop cap against the day-13 open-jaw maximum of 51.

*The edge margin, measured (the first of Opus 5's four day-13
judgements).* The day-13 margin -- `_OVAL_SHRINK + _OVAL_FEATHER / 2` of
the face width plus one, ~15 px -- was an argument about kernels. Built
the portrait's own oval with the hull's chin placed row by row toward
the bottom (`edge_margin.py`, scratchpad; CPU only):

| hull's chin row | 0.5 contour | 0.02 tail | last-row alpha |
|---|---|---|---|
| 505 | 502 | 507 | 0.000 |
| 507 | 504 | 509 | 0.000 |
| 508 | 505 | 510 | 0.008 |
| 509 | 506 | 511 | 0.043 |
| 510 | 507 | 511 | 0.118 |
| 511 | 511 | 511 | 0.57 (the hard line) |
| 513+ | 511 | 511 | 1.000 |

The erosion pulls the contour in by the kernel's *radius* (3 px here,
not the 9 of its diameter) and the feather's tail ends 1-2 px past the
hull; only a hull on the last row itself defeats the erosion. So the
margin is the mask's own measured tail plus two rows (`load_avatar`,
`_MASK_EDGE_CLEARANCE`): 4 px, four times less than argued -- at
-0.125 the drop cap would have been 18 px, not 7. The other three
judgements, reviewed: (2) reverting the signed drop -- agreed, the
render's chin does not rise with a brow raise and the day-14 stills show
the portrait's oval as the right guess for a rise; (3) `mask_chin_reach`
inert at default 0 -- agreed to keep, and it is no longer inert: at
-0.06 it has room, so it can be measured live if the join still shows;
(4) the clip A/B distrusted -- agreed, and the same thing happened
today: a band ten pixels inside a contour that moves onto the neck
describes the neck, not the join; the zoomed stills are the evidence.

*Costs and what did not change.* One frame of the day-13 build's timings
is unchanged by a crop shift (the same crop, the same box). The forehead
at -0.06: the crop has 223 rows above the face top instead of 256, all
hair; the forehead band on the open segment reads 4.4-4.8 abs at every
value. The driver's live framing: his face is 425-475 px tall, the 2.2
crop overflows the 720 frame, and at -0.06 about 60 of those overflow
rows move from the top of the crop to its bottom (black, below his
neck) -- the clips have the same framing and did not mind; his live look
is the test. The kept build re-measured once whole: `load_avatar` logs
the drop cap at 52 px, the open-mouth frames 1040-1067 use 38-42 of it
with the mask's alpha never on the last row, jaw band on 986-1103 abs
11.1 / 12.2 / 14.0 (p50 / p90 / max), step max 10.4, forehead abs 4.7.
239 engine tests (two new: the measured tail, the ratio's geometry), 892
unit tests with the gui ones deselected, ruff and mypy clean on the
engine file.

*Files for him, all under `recordings/` (his face; delete after he
looks):* `day14-vy/driving-sheet-out-f{60,500,700,800,925,1060,1067,
1200,1420,1568,1724,1784}-*.png` and `natural-sheet-out-*.png` (six /
four values side by side), `zoom-jaw-f1060.png` / `f1067.png` (the jaw,
zoomed), `jaw-kept-*.png` (the kept build's open-mouth stills). The
`*-sheet-raw-*` sheets are the portrait alone, safe. CSVs:
`driving-follow-day14-vy*.csv`, `natural-follow-day14-vy*.csv`.

*What he should look at live:* an "o" and a wide-open mouth -- the join
under the chin, no line on the neck; talking and a smile -- the mouth
should feel as it did; the forehead, unchanged. If he wants more mouth,
the pair (`crop_vy_ratio` toward 0, `expression_gain` down) is the next
two-axis sweep, not a guess.

**Day 15 opens with:** his look at this (day 13 and day 14 are still
uncommitted; his word commits them). Then the capture carrying its live
anchor (exp_d0 / R_d0 / t_d0 / scale_d0 / pose_d0 / the rest_* values /
face_height_rest / z_rest / centre_offset into `tracking.npz`), so a
replay is the session itself -- day 13's 021214 capture could not be
replayed for that. Then the yaw lag (the pivot for yaw, open since day
10; turns p90 21-22 px on every day-14 run), the mouth gain as a
measurement (now with the crop's placement as a second axis), the
toothy smile's forehead, then Phase 3 as agreed.

**Day 15 opens with his look at the day-14 build (2026-09-11, 08:10, one
capture `moment-20260911-081054`): "on a cheerful, slightly overdone
smile with teeth, the teeth grow a little in height. Everything else I
like very much."** The jaw, the forehead, the mouth while talking: no
complaint. Nothing changed yet; the analysis:

*The capture.* The button was pressed after the moment, as always: the
last frame has him talking with his lips just parted (his mouth ratio
0.099, opening over jaw 0.041) and the avatar's lips parted 0.174 /
0.072 -- **1.76x his** -- with the avatar's lips read 20% thinner than his
(upper 0.049 vs 0.062, lower 0.112 vs 0.135 of inter-ocular; the
portrait's own lips, day 6's "lip thickness 0.75-0.78"). The smile
itself sits at t = 11.6-12.5 s of the capture (his width 0.52-0.53 of
jaw, mouth ratio 0.23-0.26) and cannot be replayed with the live anchor
-- every one of the 300 frames is at pitch -5..-10 deg, `_anchor_frame_ok`
rejects them all, so a replay would run against the portrait's own rest
(the day-13 replay limit, again). That is the first thing to build.

*The clips, from the day-14 CSVs (avatar mouth ratio / his, frames with
his ratio >= 0.05; p50 / p90):*

| | -0.125 | **-0.06 (kept)** | -0.03 | 0.0 |
|---|---|---|---|---|
| driving, toothy smile 868-985 | 0.96 / 1.04 | **1.02 / 1.08** | 1.11 / 1.15 | 1.11 / 1.16 |
| driving, talk 1104-1341 | 0.90 / 1.09 | **0.97 / 1.15** | 1.10 / 1.32 | 1.14 / 1.34 |
| driving, wide open 986-1103 | 1.04 / 1.32 | **1.10 / 1.35** | 1.10 / 1.47 | 1.12 / 1.50 |
| natural, smile 1307-1455 | 0.92 / 1.04 | **1.05 / 1.23** | 1.23 / 1.53 | 1.30 / 1.51 |
| natural, talk 713-1306 | 0.80 / 1.07 | **0.96 / 1.26** | 1.12 / 1.47 | 1.18 / 1.51 |

The day-14 crop added 6-13% to the opening on the smiles (the "mouth
responds more" slope, seen from the other side) -- real, small, and not
the 1.76x of the capture's frame. The p90s above the p50s say where the
rest is: the opening over-shoots most at *small* openings (a parted-lip
smile, a syllable), where `expression_gain` 1.8 multiplies a small
relative reading and the network has not saturated; at a wide-open mouth
the ratio is 1.1. And the lips themselves: a thinner-lipped portrait
shows more tooth for the same gap -- identity, not code, and the
clean-shaven / parted-lips portrait lever is still open.

*The plan, in order (nothing built until his go):*

1. **The capture carries its live anchor** (`exp_d0`, `R_d0`, `t_d0`,
   `scale_d0`, `pose_d0`, `rest_ear` / `rest_mar` / `rest_mouth_width` /
   `rest_brow` / `rest_chin` / `rest_gaze`, `face_height_rest`, `z_rest`,
   `centre_offset`, `eye_rest` / `lip_rest`) into `tracking.npz`, and
   `jaw_band.py --render` / a new `capture_replay.py` seeds the engine
   from it instead of learning one -- so this capture's smile at t = 12 s
   can be rendered as he saw it and measured. Half a day with tests.
2. **`mouth_opening_gain`**: the same knob `mouth_smile_gain` is, on the
   opening components (`_MOUTH_OPENING`, the y of keypoints 17 / 19 / 20)
   -- default 1.0, bit-exact. Swept on driving talk (bar: talk mar gain
   >= 0.85, today 1.10) and the toothy smile, then replayed on his
   capture: the value where the smile's opening matches his without the
   talk going under the bar. If no single value does both, the opening
   gain becomes a curve of the opening itself (less gain for small
   openings) -- measured, not assumed.
3. His look.

**His two Phase-3 features, restated by him today, to be built after
the teeth (his order: quality switches, then the battery screen):**

1. **Output quality switch -- 360p / 480p / 720p / 1080p.** One stage in
   `PipelineController.process_once` between `renderer.render` and
   `virtual_camera.send`: downscale to the chosen height (INTER_AREA) and
   back up to the camera's own size (INTER_LINEAR), so the virtual
   camera's format never changes and Zoom / Meet see one stream. "Native"
   is the camera's own size -- 720p with the ZV-E10 today, so the 1080p
   entry is only meaningful with a 1080p source and reads "Native" until
   then. A setting for the default, a combo box beside "Virtual camera",
   under 1 ms a frame, works for both engines.
2. **"Camera battery died" -- a black frame with that text on the
   virtual camera, by hand and automatically.** By hand: a button in the
   window and a global hotkey (the meeting app is in front); pressing it
   again brings the camera back. Automatic: on the face lost (the engine's
   `_passthrough`, `is_placeholder=True` -- today his own face goes out),
   on a pipeline exception (`PipelineRunner._run` logs and returns, the
   camera freezes or blanks), on the extreme-pose hold past a few seconds.
   Shape: the black frame appears at once (a dead battery does not fade),
   stays at least ~3 s, and the avatar returns only after the engine has
   been stable for a second -- hysteresis, never a flicker. The preview
   shows the true output with a "BATTERY" indicator so he knows what the
   other side sees. The text: "Camera battery died" is fine English; a
   webcam-like alternative is "Camera disconnected" -- his choice. Sits on
   the same stage as the quality switch. About a day for both, with
   tests.

**Day 15 (2026-09-11, Fable 5.1 implementing on his "start the teeth
plan, commit both days"): days 13 and 14 committed (062c2f0, ffb97d9,
a4e63eb); the capture carries the session's own anchor; the lips'
parting has its own gain.**

*Step 1 -- a replay is the session itself.* `LivePortraitEngine.
anchor_state(avatar_id)` returns the frozen anchor as plain arrays --
`exp_d0`, `t_d0`, `scale_d0`, `R_d0`, `pose_d0`, and the rest values the
gates, the crop and the eye / brow / chin drives measure against
(`rest_ear` .. `rest_pitch`, `rest_gaze`; an unknown one is NaN) -- and
`seed_anchor(avatar_id, state)` makes them a fresh load's own, frozen:
the graph's tensors overwritten in place, the smoothing state primed to
the anchor as the freeze primes it, the per-frame state cleared as
`reset_anchor` clears it. `PipelineRunner.capture` asks the active
engine for it (duck-typed: the photograph engine has none and writes
nothing extra) and stores it beside the tracking as `anchor_*` arrays
with the avatar's id; `jaw_band.py --render` and the new
`scripts/probes/capture_replay.py` seed from it when it is there. The
window's dialog says the anchor is in the file.

Proof, on driving.mkv (a session run to frame N, then a fresh load fed
frame N's own reading and picture sixty times):

| frame N | replay | mouth / eyes / width / brow read on the composite | vs the session's composite |
|---|---|---|---|
| 1200 (talk) | the session | 0.131 / 0.356 / 0.592 / 0.144 | -- |
| | **seeded with its anchor** | 0.133 / 0.346 / 0.594 / 0.147 | 12% of pixels differ, mean 3.1 grey levels, max 56 |
| | learning its own | 0.128 / 0.341 / 0.559 / 0.185 | mean 13.9, max 160 -- and no anchor at all (60 of 60 rejected) |
| 925 (toothy smile) | the session | 0.190 / 0.266 / 0.655 / 0.158 | -- |
| | **seeded** | 0.189 / 0.264 / 0.658 / 0.159 | mean 1.5, max 15 |
| | learning its own | 0.187 / 0.315 / 0.611 / 0.164 | mean 13.0, max 160, no anchor |
| the file (npz) round trip | | | bit-identical to the seeded replay |

What remains between a seeded replay and the session is the smoothing
history (the session's EMA and One-Euro state at frame N against a
replay's, converged on a still picture) -- a grey level or three. What a
capture still cannot carry is the *picture* of any frame but the press's
own: the extractor reads pixels, and the ten seconds of tracking hold
none. So the smile of his 08:10 capture (t = 11.6-12.5 s) stays
unmeasurable in that file, and the file itself predates the anchor -- a
replay of it renders the portrait at rest (mouth 0.003 against the 0.174
he saw). **His next press is the first that can be replayed.** If the
moment before the press matters often, the next lever is a short ring
of camera frames in the capture (his face, gitignored like camera.png;
~4 s at full size is ~250 MB of RAM, a half-size ring a quarter of it) --
proposed, not built. Tests: five for the snapshot (nothing before the
freeze, plain copies with NaN for unknowns, a seeded engine equal to the
one that wrote it and reporting the same state back, the npz round
trip, refusals), two for the capture (the keys ride along; an engine
without an anchor, the photograph engine, no avatar: nothing extra).

*Step 2 -- `mouth_opening_gain`.* The y of keypoints 17 / 19 / 20
(`_MOUTH_OPENING`) weighted like `mouth_smile_gain` weights the smile's
component: default 1.0, bit-exact; the gate, the smile and the lateral
components untouched; multiplies with `expression_gain`. Swept on both
clips at 0.8 and 0.6 against the day-14 build (`crop_vy_ratio` -0.06),
avatar mouth ratio over his on frames where his is >= 0.05 (p50 / p90),
and the bars:

| | **1.0 (the build)** | 0.8 | 0.6 | bar |
|---|---|---|---|---|
| driving, toothy smile 868-985 | 1.02 / 1.08 | 0.89 / 0.94 | 0.74 / 0.79 | |
| driving, talk 1104-1341 | 0.97 / 1.15 | 0.81 / 0.94 | 0.65 / 0.78 | |
| driving, talk-only mar gain | 1.10 | 0.90 | 0.71 | >= 0.85 |
| driving, talk + open mar gain | 1.21 | 1.10 | 0.88 | >= 0.85 |
| driving, small openings on talk (his 0.05-0.15), p50 | 0.94 | 0.82 | 0.68 | |
| natural, smile 1307-1455 | 1.05 / 1.23 | 0.82 / 0.96 | 0.63 / 0.74 | |
| natural, talk 713-1306 | 0.96 / 1.26 | 0.78 / 0.98 | 0.56 / 0.71 | |
| natural, talk mar gain | 1.03 | **0.83** | 0.60 | >= 0.85 |
| blink / brow / turns / frown / still | unchanged at every value | | | |

The knob does exactly what it says and nothing else -- and it says the
same thing to a smile and to a word: 0.8 takes the smiles from 1.02-1.05
to 0.82-0.89 and the talk from 0.96-0.97 to 0.78-0.81, under the bar on
natural.mkv. There is no value that trims the smile and leaves the talk,
because **on the clips there is nothing smile-specific to trim**: the
toothy smile's opening is 1.02 of his at the median, the small openings
on talk 0.94-0.95 -- the "small openings over-shoot" reading of the
morning was the p90 tail (1.15-1.26, tracking noise on a 0.05 opening
as much as anything), not the median. The 1.76 of his capture is not in
either clip. So a curve of the opening is not justified by the clips
either, and `mouth_opening_gain` stays 1.0: **the option exists, the
measurement on his face does not yet.** What differs live from the clips
is what the anchor now carries -- his rest at that session, his crop at
that distance, the tracker at 22 fps -- and the seeded replay is the
instrument for it. Kept: nothing in his metadata.json; `mouth_opening_
gain` in the code at 1.0.

*What he should do:* smile with teeth as he did this morning and press
"That looked wrong" **during** the smile (the capture now carries the
anchor; only the press's own frame can be replayed). Then
`capture_replay.py --capture <it> --sweep mouth_opening_gain=1.0,0.8,0.6
--stills` renders that very frame at each value in one process, and the
value where his teeth are his is read off, not guessed.

*Tests:* 245 in the engine file (the anchor snapshot's five, the opening
gain's one), 26 in the runner's (two new); ruff and mypy clean on both.
Files: nothing of his face kept from today's replay checks (the
anchorless replay of the 08:10 capture was deleted); the CSVs are
`recordings/*-follow-day15-og*.csv`.

**Day 15, later (his "wonderful, I liked it -- now the next plans: the
video qualities, and the black screen with text when the machine
presumably is not working"): both built, Phase 3's first two.**

*Where they live.* One stage, `sha256/pipeline/conditioning.py`
(`OutputConditioner`), between `renderer.render` and the sink in
`PipelineController.process_once`; the frame it returns is what the sink
gets *and* what the preview shows, so what he sees is what the other side
sees. Wired in `PipelineComponents` (defaulted, so every earlier wiring
and test stands) from a new `[output]` settings section
(`OutputSettings`, `configs/default.toml`).

*The quality switch.* Native / 1080p / 720p / 480p / 360p in a combo
beside "Virtual camera". The picture is scaled down to that height
(INTER_AREA) and straight back up to the camera's own size (INTER_LINEAR):
the virtual camera's format never changes, Zoom and Meet see one stream,
and the picture looks like a cheaper webcam's. Native leaves it alone;
a height at or above the frame's own is native too, so 1080p and 720p
read as native on the 720p ZV-E10 until there is a 1080p source. Cost
on a 1280x720 frame: 480p 1.0 ms, 360p 0.7 ms, native nothing.

*The black screen.* A black frame with "Camera battery died" centred
in a quiet grey (`failure_text`, his to change; built once per size and
cached) in place of the avatar:

- **By hand:** the red "Camera battery died" button in the bar (checkable
  -- pressed, the screen is black; pressed again, the avatar is back at
  once), and a hotkey that works with the meeting app in front,
  `ctrl+alt+b` by default (`blackout_hotkey`): `sha256/ui/hotkey.py`
  binds it with Windows' own `RegisterHotKey` and hears `WM_HOTKEY`
  through Qt's native event filter, so a Qt shortcut's "only while
  focused" does not apply. The hotkey *is* the button (it toggles it),
  so the two never disagree.
- **On its own** (`auto_blackout`, default on): the frame that would
  have gone out is the camera's own picture -- the engine's passthrough,
  before a face is first found or after it is lost with nothing rendered
  yet -- and it blacks at once (that is the day-13 "2.5 s of his real
  face on the virtual camera" case, closed); or the face has been lost
  for `lost_face_seconds` (1.5) while a frozen avatar sat over a moving
  room. A dead battery cuts at once and does not come back in a second:
  an automatic black screen stays `min_blackout_seconds` (3) at least
  and lifts only after the face has been tracked for `recovery_seconds`
  (1) without a break -- hysteresis, never a flicker. The manual switch
  outranks all of it and survives a pipeline restart; the automatic
  state is forgotten on every start.
- **When the loop dies** (`PipelineRunner._run`'s exception path, which
  stops the loop): the controller's new `publish_failure` puts the black
  frame on the virtual camera once before the error is reported, so the
  other side sees a dead camera rather than the last avatar frame frozen
  for the rest of the call.
- The status line says `BLACK SCREEN -- the other side sees "Camera
  battery died" (by the button / face lost / ...)` while it is on, and
  the preview shows the black frame itself.

To tell the engine's passthrough from a placeholder an engine *drew*
(the null engine marks every frame `is_placeholder`), `RenderedFrame`
gained `is_passthrough`: set by both real engines' `_passthrough`, read
by the controller, false everywhere else. Without it the null-backend
app would have blacked every frame.

*A bug the tests caught before he could.* The first `RegisterHotKey`
call passed the window handle as a bare Python int, which ctypes hands
to C as a 32-bit int: the 64-bit handle was truncated and Windows
answered "invalid window handle" (1400). Declared argument types
(`HWND`) fixed it; checked on a real window (`hotkey_live.py`,
scratchpad: platform windows, bound, a posted `WM_HOTKEY` fired the
callback, unbound). The suite's own copy of that check skips itself on
the offscreen platform the tests run on, where a window has no Win32
handle.

*Tests:* `test_conditioning.py` (quality sizes and softness, the black
frame, the hysteresis on a hand-moved clock, the manual switch, reset,
auto off), `test_pipeline_controller.py` (new: one frame through fakes
-- the sink and the preview get the conditioned frame, the camera's own
picture never reaches the sink, the manual switch, a start forgets an
automatic black screen, `publish_failure`), the runner's failure path,
the `[output]` section, the hotkey parser and registration, the
window's combo and button and status line. 938 unit tests with the gui
ones deselected, 18 gui tests in the two ui files; ruff, format and
mypy clean on everything new (the warp engine's 11 pre-existing mypy
errors as before).

*What he should try live:* set 480p in the combo during a call and
look at the other side; press the button and the hotkey with Zoom in
front; walk out of the frame for two seconds and come back (black,
then the avatar after about three). Not committed; his word.

**Day 15, his look at both (2026-09-11, 09:50): "the teeth are still
wrong; and the qualities change imperceptibly -- I set 360p and honestly
nothing changed. I pressed That-looked-wrong again, look at the teeth."**

*The teeth: found, and it was never a gain.* His new capture carries
the session's anchor (the first that does), and the replay reproduces
what he saw -- `seen.png` read 1.68x his lip opening, the replay 1.60x
(mouth ratio 0.289 against his 0.181; the mouth zoom
`recordings/day15-replay-moment-20260911-095003-mouth_opening_gain1.0
-mouth.png`, two full rows of teeth against his one). The sweep on that
very frame: `mouth_opening_gain` 0.8 -> 1.36x, **0.6 -> 1.03x**, 0.4 ->
0.71x. So 0.6 fits his frame -- and the clips said 0.6 takes the talk
to 0.71 / 0.60. Same expression, same portrait, two answers: the
difference had to be in what the session carries and the clips do not.
Rendered his frame with four anchors:

| anchor | lip opening / his |
|---|---|
| the session's own (live) | **1.54-1.60** |
| one learned on driving.mkv frames 0-45 | 1.06 |
| the live one with only `exp_d0` from the clip | 1.03-1.06 |
| the clip one with the live rest_* values | 1.06 |

The anchor's `exp_d0` alone. The two differ by almost nothing (mean
0.002 over 63 components) except the lips' opening rows (`_MOUTH_OPENING`,
the y of 17 / 19 / 20): his live rest read (0.003, -0.007, 0.017), the
clip's (-0.001, 0.002, 0.008), and the portrait's own closed mouth,
`exp_s`, (-0.000, -0.004, 0.003). His lips were **pressed** at that
session's start -- what a person does looking at a screen -- and the
extractor reads a pressed mouth as closed beyond closed, so every
opening measured from there carries a constant extra: **+0.1 of mouth
ratio on both of his captures**, the 08:10 talk frame included (0.099 ->
0.174), which is why a multiplicative gain fitted one frame by accident
and could never fit the clips. With the anchor's opening rows taken from
the portrait's own closed mouth -- relaxed, read in the same units by
the same network -- his frame came out **0.98-1.02x his**; all five
mouth rows from the portrait, 1.02-1.07x.

**Built: `mouth_rest`** in engine_options, `"portrait"` (the default:
the anchor's lip-opening components are the portrait's, written in
place at every accepted anchor frame and when a captured anchor is
seeded, so an older capture replays under the new rule too) or
`"anchor"` (his own rest reading, the behaviour before). The opening
gate (his measured mouth ratio, day 9) keeps the mouth shut at rest
either way; the reference only matters once the mouth moves.
`mouth_opening_gain` stays 1.0 -- the wrong lever, kept as a knob.
Tests: four (`TestMouthRest`); the snapshot and smoothing-state tests
pin themselves to `"anchor"`, since their subject is the round trip.
Measured on his capture through the real option path and on both clips
whole:

| | `mouth_rest` anchor (before) | **portrait (kept)** |
|---|---|---|
| his 09:50 frame: lip opening / his | 1.54-1.60 | **0.98-1.02** |
| driving: talk-only mar gain | 1.10 | 1.11 |
| driving: toothy smile, opening / his p50 | 1.02 | 1.01 |
| driving: closed smile, width gain | 0.51 | **0.37** |
| driving: blink / brow / turns p90 / frown EAR | 0.90 / 0.72 / 21.7 / 0.291 | 0.90 / 0.72 / 21.9 / 0.291 |
| natural: talk mar gain | 1.03 | 1.07 |
| natural: smile, opening / his p50 / width gain | 1.05 / 1.04 | 1.02 / 0.95 |

The clips do not change where they were right (their anchors were
relaxed to begin with) and the live fault is gone. One cost, stated:
the closed smile's width is a little narrower (driving 0.51 -> 0.37,
natural 1.04 -> 0.95; the toothy smile's width 3% narrower too) -- his
relaxed rest reads as slightly *more* closed than the portrait's, and
the sliver of that the partly open gate lets through on a closed smile
pulls the lips together. `mouth_smile_gain` (bound 3) is the knob for
the closed smile if he misses it; his eye decides. Stills for him:
`recordings/day15-replay-moment-20260911-095003-mouth_opening_gain1.0
-mouth.png` (what he saw) and `-mouth_restportrait-mouth.png` (now),
his face; delete after.

*The qualities: right mechanism, invisible effect.* A plain down-and-up
scale of a softly lit face is nearly nothing to the eye: on his frame
the Laplacian sharpness fell 38 -> 6 at 360p but the mean difference
was under one grey level (`recordings/day15-quality-sheet.png`: native
and "360p linear" side by side, and he was right). What everyone
recognises as a poor camera is the block artefacts of a compressed
small picture, so each level now passes through a JPEG at its own size:
1080p / 90, 720p / 65, 480p / 50, 360p / 40 (`QUALITY_LEVELS`). On the
sheet 480p reads as an ordinary webcam and 360p as a bad connection;
1080p and 720p on a 720p camera do not resize, so the JPEG alone is
their step. Cost on a 1280x720 frame: 1080p 3.8 ms, 720p 3.5 ms, 480p
2.6 ms, 360p 1.6 ms (the full-size JPEG is the expensive one). The
sharpness test in `test_conditioning.py` now walks the ladder on a
face-like picture and demands each step be softer than the one above.

**Day 15, his third look (2026-09-11, 10:14): "put the teeth change back
as it was -- when I open my mouth slightly the AI's mouth does not open,
it just droops; when I close, it hides the lips, purses them; talking no
longer follows. The qualities are fine now. Part of the jaw shows again.
Fix these, and break nothing that was good."**

*Reverted at once:* `mouth_rest` is `"anchor"` again (the code default
and his metadata); `"portrait"` stays an option with the reason in its
comment. His description is the mechanism seen from the other side: with
the portrait's relaxed lips as the reference, his own rest reads as
"closing", so a slight opening only drooped the lower lip and a closing
pursed. The reference has to be his mouth.

*The fault, isolated on the clip.* What differs between his live
sessions and the clips is the anchor, so the clip was run three ways,
the avatar read on the composite as `measure_follow` reads it:

| driving.mkv, avatar lip opening / his (p50) | toothy smile | wide open | talk |
|---|---|---|---|
| the clip's own anchor (today's rule) | 1.03 | 1.10 | 0.97 |
| **his 09:50 live anchor seeded** (lips pressed at that session's start) | **1.58** | **1.59** | **2.20** (p90 3.6) |
| the same pressed anchor + lip rest tracking, EMA 0.03 | 1.07 | 1.11 | 1.10 |
| the same, EMA 0.01 with a 15-frame settle | 1.08 | 1.12 | 1.12 |
| the clip's own anchor + tracking (0.01, settle 15) | 1.09 | 1.11 | 1.10 |

So a pressed anchor over-opens *everything* on the clip too, talking
most -- exactly his live experience (the 08:10 talk frame 1.76x, the
09:50 smile 1.60x). The anchor's lip-opening rows are learned once, from
fifteen frames at the session's start, and his mouth then is not his
mouth minutes later: pressed while he sets up, relaxed once he talks.
And the reference matters at the thousandth -- 0.001 on these rows is
about 10% of a talking mouth (the last two lines: the tracked rest lands
a hair more closed than the first fifteen frames, and talk reads 1.10
instead of 0.97) -- so no one-time reading can be right for a session.

**Built: `lip_rest_tracking`** (frames, default 100, in his metadata; 0
is day 14's behaviour, exact). Once the anchor is frozen, on every frame
the day-9 gates call a resting mouth -- the opening and width gates both
shut, the face not held -- for more than 15 frames in a row, the anchor's
lip-opening rows move toward the extractor's reading of that frame by
1/100 (a time constant of ~4-5 s live), as one masked in-place op on the
static tensor the graph reads; nothing else of the anchor moves, and the
tracked rows ride along in a capture's anchor so a replay is still the
session (`_track_lip_rest`, `_LIP_REST_SETTLE`, `_LIP_REST_GATE_SHUT`;
six tests). What it does not do: help in the first seconds after a
pressed anchor, before enough resting frames have passed -- the anchor
is still the start.

*The jaw.* On his 10:14 capture (shaved, mouth shut) the avatar's
rendered chin sits 8.8 px above his, and below it his own bare chin
shows under the avatar's stubbled one (the day-12 lesson: a beard hid
this). `mask_chin_reach` now has room (day 14) and day 13's sweep said
0.04 is the value: on this capture the join moves from above his chin
line onto it (`recordings/day15-jaw-chin-zoom.png`, three reaches
zoomed; 0.08 runs onto the neck). Set to 0.04 in his metadata; the band
metric reads worse (12.7 -> 20.5 abs) because the band moves onto his
neck, as day 13 found -- the zoom decides, and his eye after it. The
deeper cause is the portrait's stubble over his shaved chin: a
clean-shaven portrait stays on the list.

*Measured, the build he will run (tracking 100, reach 0.04) against day
14's (`*-follow-day15-liprest.csv` vs `*-follow-day14-vy-0.06.csv`):*

| | day 14 | **this build** |
|---|---|---|
| driving: blink / talk mar / smile width / brow gain | 0.90 / 1.10 / 0.51 / 0.72 | 0.89 / 1.15 / 0.58 / 0.69 |
| driving: turns p90 / frown EAR / still spread | 21.7 / 0.291 / 0.005 | 21.8 / 0.290 / 0.005 |
| driving: lip opening / his, toothy / talk (p50) | 1.02 / 0.97 | 1.09 / 1.10 |
| natural: talk mar / smile width / brow gain | 1.03 / 1.04 / 0.80 | 1.01 / 1.04 / 0.81 |
| natural: lip opening / his, talk / smile (p50) | 0.96 / 1.05 | 0.92 / 1.03 |
| **the pressed anchor seeded on driving: toothy / open / talk** | **1.58 / 1.59 / 2.20** | **1.10 / 1.14 / 1.16** |

Everything that was right stays within a few percent (the thousandth
again: driving's talk a little more open, natural's a little less); the
fault -- a session started with pressed lips -- is gone on the clip. 255
engine tests. His live look is the last word, and the first seconds of a
session are still the anchor's.

**Day 15, his fourth look (2026-09-11, 10:57, two captures): "when I tilt
my head down a line appears under the jaw, noticeably; and on a long,
strong smile with the brows raised, something happens to the teeth --
the second press should show it. Fix these two and we have a really
strong application."**

*What the captures hold.* Neither has the moment: both were pressed
after it, with the mouth shut and the brows at rest for all 300 frames
(`capture_stats` on 105732 and 105750: mouth ratio <= 0.008, brow
0.20-0.23 throughout; 105750 holds a tilt to -22 degrees at t = 4-7 s,
the press at -6). So the smile-with-brows moment was analysed from the
clips and reproduced synthetically; the jaw line is in the press frames
themselves (pitch -12 and -6, shaved).

*The jaw line under a tilt: the portrait's own shadow.* Rendered at
chin reach 0 / 0.02 / 0.04 on both captures (`recordings/day15-jaw2-
reach-sweep-*.png`): the dark line under the avatar's chin is there at
every reach -- it is the portrait's jaw shadow, drawn by the network as
part of the face, pasted onto his lit under-chin; at rest it lands on
his own jaw shadow and hides, tilted down it does not. The band metric
agrees (105732 at reach 0: step +1.1; the reach moves the band onto the
neck). The edge colour match exists for exactly this and could not see
it: at an eighth-scale grid (8-px cells, a 24-px blur) a line ~8 px tall
is below its resolution. **Built: the correction's grid is quarter-scale
now** (`_CORRECTION_DOWNSCALE` 4; the brow-hair dilation, the zone's
softening and the fill's reach doubled in cells to keep their pixel
size; the edge blur stays 3 cells = 12 px, the finer match being the
point):

| capture (pitch) | jaw band abs / step, grid 8 (today) | **grid 4** | grid 4, blur 5 (the old 24 px) |
|---|---|---|---|
| 105732 (-12) | 18.8 / +9.8 | **11.0 / +3.7** | 13.5 / +5.6 |
| 105750 (-6) | 18.9 / +4.7 | **12.2 / +1.4** | 14.6 / +2.6 |
| 101434 (-3) | 20.4 / +6.0 | **13.2 / +1.9** | 15.9 / +3.6 |
| forehead abs, the three | 3.7 / 4.7 / 3.0 | **2.8 / 3.8 / 2.3** | 3.1 / 4.1 / 2.4 |

The zooms (`recordings/day15-jaw-fine-*.png`) show the line visibly
softer, the stubble and his skin unchanged. The day-13 forehead captures
(his brows raised, the toothy smile with hair over the forehead) are
re-measured below, since a finer residual is exactly what day 13 had to
guard against there.

*The teeth on a strong smile with the brows up: the brow raise leaks
into the mouth.* The extractor's mouth rows, per frame against his brow
ratio over driving.mkv's raise (mouth shut, 1642-1760): r = 0.83-0.98,
the lower lip's y (keypoint 20) +0.10 per unit of brow ratio, 17's y
+0.04, 7's y +0.015, the z of 17/20 -0.02 (natural.mkv's raise: +0.19,
+0.06, +0.03, -0.01/-0.03 -- the same signs, up to 1.8x the size; 19's y
and the smile row 14 disagreed between the clips and are left alone).
With the mouth shut the day-9 gate discards it; during a smile the mouth
gate is open and the leak rides in -- pointing the other way from an
opening. Reproduced: the toothy-smile frame 925 re-rendered with the
leak a +0.06 raise carries (injected through the anchor) **closed by a
third** (mouth ratio 0.190 -> 0.135; natural.mkv's smile 0.392 ->
0.266), and the brow drive itself (kp 2's y, the day-9 lift) narrowed
the smile 7% on top (`recordings/day15-leak-smile-*.png`, portrait
only). That is "something happens to the teeth". **Built:
`brow_mouth_leak`** (gain, default 1.0, in his metadata; 0 off): every
frame `_render` writes the leak his measured brow deviation predicts
into a static tensor (`mouth_leak`, the slopes in `_BROW_MOUTH_LEAK`),
and `_drive` subtracts it from the relative expression before the masks
and gates; three tests. The slopes are two clips' average of his own
face; a capture pressed *during* the smile-with-brows is what tunes the
gain -- the option is the knob.

*Measured, the build he will run* (`*-follow-day15-final.csv`: tracking
100, reach 0.04, the leak at 1.0, the quarter-scale match) against day
14's:

| | day 14 | **this build** |
|---|---|---|
| driving: blink / talk mar / smile width / brow gain | 0.90 / 1.10 / 0.51 / 0.72 | 0.90 / 1.14 / 0.58 / 0.72 |
| driving: turns p90 / frown EAR / still spread / still dy chin | 21.7 / 0.291 / 0.005 / -4.5 | 21.4 / 0.292 / 0.006 / -3.9 |
| driving: lip opening / his, toothy / talk (p50) | 1.02 / 0.97 | 1.06 / 1.15 |
| natural: talk mar / smile width / brow gain / ear r | 1.03 / 1.04 / 0.80 / 0.77 | 1.05 / 1.02 / 0.81 / 0.78 |
| natural: lip opening / his, talk / smile (p50) | 0.96 / 1.05 | 0.99 / 1.08 |
| day-13 forehead captures, forehead step: 021214 (brows raised) / 025329 (hair, the worst) / 025254 | -0.4 / +13.8 / -- | **-0.5 / +0.9 / +2.2**; jaw abs 6.3 / 5.6 / 6.0 (was 8.4) |
| paste, driving 950-1100, p50 / p90 | 8.4 / 9.7 ms (day 13) | 7.3 / 7.9 ms |

Nothing that was right moved beyond a few percent; driving's talk
opens 15% more than his where day 14 read 3% less (the tracked rest, the
thousandth again -- natural.mkv reads 0.99), and the finer match took the
worst forehead capture from +13.8 to +0.9 with the brow smear still
absent (`recordings/day15-fore-brows-zoom.png`). 258 engine tests; the
whole suite and the gui files pass; ruff, format and mypy clean.

*Files for him (his face; delete after):* `recordings/day15-jaw2-reach-
sweep-105732.png` (the tilt, three reaches -- the line at every one),
`day15-jaw-fine-105732.png` (the same frame, grid 8 vs 4: the line
softened), `day15-fore-brows-zoom.png` (his raised brows: no smear at
the finer grid), `day15-leak-smile-driving-925.png` (portrait only: the
smile, then with the brow raise's leak -- the teeth going), plus the
earlier `day15-jaw-chin-zoom.png` and the 09:50 mouth zooms.

*What he should try:* tilt down and look at the jaw (a fainter line,
not none -- the portrait's own shadow is still the portrait's); a long
toothy smile, then raise the brows while holding it (the teeth should
stay); talking with the brows moving. And **press "That looked wrong"
during the moment**: both captures today were pressed after it, and the
smile-with-brows leak's gain can only be tuned on a frame that has it.

**Day 15, his fifth message (2026-09-11, ~11:30): "good, wonderful --
two more things and I think we finish the project with this; maybe
refine details later." Two functions, two questions, one note.**

*Function 1 -- adding faces from the app, by him alone.* Built:
- `sha256/app/owner.py` (`OwnerAccess`): a passphrase, set on the first
  "Login" (typed twice, at least four characters), salted and stretched
  (PBKDF2-HMAC-SHA256, 200k rounds) into `configs/local.owner.json` --
  gitignored with the rest of `configs/local.*`. A gate in the window,
  not a vault: anyone with the machine's files can still copy a directory
  into the library by hand; it keeps his friends out of the button, which
  is what was asked.
- `sha256/avatar/importer.py`: a photograph in, an avatar out. The
  picture is inspected the way the engine will read it (MediaPipe, image
  mode) and refused for what the fifteen days say cannot work -- no
  face, turned or tilted more than 12 degrees (the anchor and the crop
  are frontal), a face under 200 px, shut eyes -- and noted for what
  works worse: off straight by more than 5 degrees, a face under 350 px,
  an open mouth (a grin grins at rest -- day 6's portrait 2), less than
  0.35 of a face height below the chin (day 14's wall). Then the
  directory: `portrait.png` (re-encoded) and `metadata.json` copied from
  the active neural avatar's manifest -- engine block, rendering, every
  engine option measured on this machine -- with a slug id unique in
  the library. Nothing half-made: a failure after the directory exists
  removes it.
- The window: "Login" (then "Owner", disabled) and, only once unlocked,
  "Add avatar": a file dialog, a name, the checks, the library re-read,
  the new avatar selected in the combo, a report of what the picture
  measured. Tests: the passphrase (six), the importer (eleven; one runs
  the real inspector on his portrait when the model and the portrait
  are on the machine), the window's wiring (five).

*Function 2 -- the black screen's words mirrored.* The preview mirrors
what it shows (a webcam feels wrong otherwise); the black screen carries
words, and mirrored words read backwards. The frame that goes out is the
right way round for the other side; only the window flipped it. Now that
one frame is shown unmirrored whatever the mirror is set to
(`PreviewWidget.show_frame(mirror=...)`, the window passing `False`
while the conditioner reports a black screen). Two preview tests, one
window test.

*Question 1 -- "on someone else's face: the same result, right?"* The
honest answer is: probably close, not certainly the same, and nobody has
measured it. What is his: the brow-to-mouth leak's slopes (his face, two
clips), the gates' thresholds (calibrated on his live jitter), the
crop's placement and the expression gain (chosen on his clips), the
anchor's habits (his pressed lips). What is general: the crop is sized
from the driver's own face, the mask follows the driver's oval, the
colour match is per frame, the lip rest tracking and the black screen
know nothing of who is driving. A friend with an ordinary frontal face
should get a day-13-to-15 avatar; the first session with one will show
two or three things to tune, and the tools -- a capture with its anchor,
the replay, the clip sweep -- make that a day's work, not fifteen. The
brow leak is the one to watch: `brow_mouth_leak` 0 turns it off if a
new face's brows do not leak the way his do.

*Question 2 -- "adding a new avatar: can problems come up?"* Yes, and
the import names them before writing anything. The ones it cannot
catch: a portrait whose lower face is proportioned unlike the driver's
(day 6: his jaw showed under a shorter portrait), stubble or a beard
the driver does not have (day 12), hair across the forehead (day 13's
reach), a photograph lit hard from one side (the plane corrects tone,
not a shadow's shape), glasses. A parted-lips portrait gives real teeth
(day 6); a closed-lip one hallucinates them. The engine options copied
from his avatar are the driver's as much as the portrait's -- right for
him driving any portrait, a starting point for anyone else.

*The note -- the whole forehead.* `mask_forehead_reach` is 0.85: the top
seventh of brow-to-hairline stays his. 1.0 puts the join at the hairline
itself, where the two heads' hair meets -- the reason the warp engine
kept it below 1 in August, and why day 13's 0.65 went the other way. His
ask is the whole face, later the avatar's hair too; that is a separate
piece of work (hair is what the mask has never been asked to carry),
parked at his own "later".

**Day 15, still open, in his order:** his look at the teeth (the
tracking, the leak), the jaw (the reach, the finer match), the
qualities, the login and the add-avatar flow; commit on his word; then
the forehead and the hair, the yaw pivot and the rest of the day-14 list.

### Phase 2 — live, days 6-9

- Engine in the app, live camera, the owner testing daily.
- Tuning by his eye: identity, mouth interior, eyes, head-turn range, lighting.
- Latency: pipeline the CPU crop with the GPU render if the budget is tight
  (+1 frame); 60 fps if the capture card arrives, including the preview timer.
- GPU check at startup; the photograph engine as automatic fallback.
- Exit: the owner uses it on a real call and does not report the mouth.

### Phase 3 — a package that installs, days 8-12 (overlaps phase 2)

- A shippable virtual camera backend: `MFCreateVirtualCamera` (Windows 11) or a
  small native DirectShow filter; pyvirtualcam stays for development only.
- `.exe` with the CUDA runtime; first install on a second machine that never
  saw the repository; a plain message when the GPU is missing.
- Enrolment: a portrait in, an avatar out, no scripts.
- Output conditioning (his two ideas of 2026-09-09, see "Parked on his
  ask" under day 13): the quality selector (Native / 720p soft / 480p /
  360p) and the "connection dropped" blur -- button, hotkey, automatic on
  a lost face, a pipeline error, a clipped mask, a hold.
- Exit: a user installs it from a file and is on a call in ten minutes.

### Phase 4 — protection and rollout, days 12-18

- Licence server with the kill switch, the download site, avatar encryption.
- Five portraits generated and enrolled; five installs; a written one-page
  guide with the camera settings (NTSC 30 fps, shutter, USB Streaming).
- Exit: five users on calls; the owner can switch one off remotely.

## The road to the finish (written 2026-09-11, the evening of day 15; work starts day 16)

**Where we stand.** The neural engine runs live on his face and has been
tuned against his eye through fifteen days: the crop, the anchor and its
lip rest, the gates, the brow drive and its leak into the mouth, the
paste mask and the colour match, the black screen and the quality
ladder, a login and an avatar import. Days 13-15 are committed locally
(seven commits ahead of origin). What is not done is what the delivery
is *for*: five other people, each with their own GPU, camera, face and
portrait, on calls their boss watches. Everything below serves that,
in the order that finds the unknowns soonest. His own words set the
constraint: quality over the date, packaging last.

**Block A -- a second face (days 16-17).** Everything measured so far is
his; the first thing to learn is what does not carry.

- **A2 first, then A1 in person** (his word, the same evening: the
  friend comes to him, and the app measures the face live). **A2.
  "Calibrate me".** A guided half-minute in the window -- sit still,
  blink, open the mouth, smile with teeth, raise the brows, turn left and
  right -- measured live by the same instruments and written to a
  *driver profile* (`configs/local.driver.json`, per person) that is laid
  over the avatar's engine options: the gates' jitter thresholds, the
  brow-to-mouth leak's slopes, the expression gain that brings the
  talking mouth to 1.0, the lip rest, the resting ratios. This is the
  fifteen days' hand work for the measurable parts, done by the app in
  thirty seconds for anyone. **And it records its own session** -- the
  clip and the tracking, the way a capture does, under `recordings/`
  (gitignored), with the person's consent and deleted when done -- so
  the same face can be re-measured offline after every change without
  the person present. *Done when* his own profile measures what day 15
  set by hand within a few percent, and the session file replays through
  `measure_follow.py`. One to two days, built before the friend comes.
- **A1. The friend's visit.** They sit where he sits, run "Calibrate
  me", make a five-minute call, and press "That looked wrong" *during*
  anything that looks wrong. What comes out: a profile, a session clip,
  captures with their anchors. Then the whole measurement on the clip:
  the day-15 bars, the brow-to-mouth probe, the anchor testbed. *Done
  when* there is a table of which bars a second face fails and by how
  much, with and without the profile. Half a day of measurement; his
  part is the visit.
- **A3. The split, written down.** Which options belong to the portrait
  (the mask reaches, the jaw shrink, the crop of the picture) and which
  to the driver (gains, gates, leak, rest), so a new person and a new
  portrait each get the right defaults without the other's. Half a day,
  mostly documentation and the import/profile code reading it.

**Block B -- new avatars without surprises (days 17-18).**

- **B1. Portrait preparation at import.** The picture is padded below
  the chin to at least 0.4 of a face height (the day-14 wall) and
  brought to a standard size (a 1024 canvas, the face about 450 px
  tall), the original kept beside it. *Done when* a photograph with the
  chin at its bottom edge imports and the chin drop's cap reads 50 px or
  more. Half a day.
- **B2. A portrait profile.** Measured from the picture itself: stubble
  or a beard (the chin's tone against the cheek's), hair over the
  forehead (brow to hairline against the face height), the lips' state
  -- written into the new avatar's options as `mask_jaw_shrink`,
  `mask_forehead_reach` and a note on the teeth. *Done when* his
  portraits 2, 3 and 4 import with sensible masks and no hand edit of
  `metadata.json`. Half a day.
- **B3. Portrait-to-driver proportions.** At the anchor, the driver's
  lower face against the portrait's (nose to chin over face height, day
  6's finding) sets the chin reach and the crop's vertical placement for
  that pair, once per session. *Done when* the avatar's chin lands within
  4 px of the driver's on his day-15 captures and on the friend's
  recording. One day.

**Block C -- what his eye still sees (days 18-20).**

- **C1. The whole forehead.** `mask_forehead_reach` to the hairline, with
  the join placed hairline-aware (the lower of the two hairlines, the
  colour match carrying the rest); measured on the day-13 forehead
  captures and his own look. Half a day. Then, separately and honestly
  the biggest piece left: **the avatar's hair** -- the network draws the
  whole 512 crop, hair included; what hides it is the face oval. A
  whole-head mode needs a matte of the portrait's head (a segmentation
  model, MediaPipe's selfie matte or similar), the portrait's hair pasted
  over his own and moving with his head, and the crop's edges (shoulders,
  background) kept out. Two to three days, odds fair, not certain: this
  is the one item that can look worse than what it replaces and must be
  a switch.
- **C2. The turn lag.** The yaw pivot (day 10's step 4): turns still lag
  21 px at the 90th percentile on every day-14 run. One day.
- **C3. The moments he pressed after.** A capture during a strong smile
  with the brows raised tunes `brow_mouth_leak`; the first seconds after
  a pressed anchor (tracking starts only once the mouth rests) get a
  faster start. Half a day, and needs his press *during* the moment.

**Block D -- the package and the five (days 20-24).** Last by his own
choice, and unchanged since the 2026-08-15 decision: no licence server,
no payments, no accounts; a courtesy kill switch.

- **D1. The kill switch.** A static file on a host he controls, read at
  start and hourly, fails open, never on the frame path. Half a day.
- **D2. The GPU check.** At start: an NVIDIA GPU with the VRAM the avatar
  needs, else a plain sentence and the photograph engine as the fallback.
  Half a day.
- **D3. The virtual camera.** pyvirtualcam is GPL and stays for
  development; the shipped app uses the OBS Virtual Camera device, which
  means each user installs OBS Studio once (free; it need not run). A
  native filter would cost days and is not worth them. Decision, not
  code; the guide says it.
- **D4. The `.exe`.** PyInstaller with the CUDA runtime and the model
  files; the first install on a second machine that never saw the
  repository; the one-page guide (camera settings: NTSC 30 fps, USB
  streaming, the shutter; Discord closed; OBS installed; Login and Add
  avatar for him only). Two to three days.
- **D5. Five people.** Their portraits (frontal, neutral, lips barely
  parted, even light, no glasses, hair off the forehead, room below the
  chin -- the import says which of these a picture misses), five
  installs, each runs "Calibrate me", each makes one call. *Done when*
  five people have been on a call and none reported the mouth.
- **D6. Push.** The seven commits and what follows go to origin when he
  says, under the anonymity rules that stand (author `sha256`, no URLs
  in committed files).

**What only he can provide, and when:** a friend's two recordings (A1,
first); his press *during* the smile-with-brows moment (C3, any time);
the five portraits (D5, by day 20); a second machine and OBS on it
(D4); the host for the kill-switch file (D1).

**The count.** Blocks A-C are about six working days, D about four:
ten from day 16, so the five are on calls around the 22nd to the 24th
of September if the friend's recording comes first and nothing in it is
a surprise. The two things that could stretch it: C1's hair (a switch
that may stay off), and whatever A1 finds that fifteen days on one face
never showed -- which is exactly why A1 comes first.

### Day 16 (2026-09-12), block A2: "Calibrate me" -- built, tested, measured on his two clips

**What was built (Fable 5.1, on his "start A2, do it fully, with tests").**
A driver profile and the run that writes it, on the plan of the evening
before, plus the split (A3) in code.

- **The split (A3, the code side).** `DRIVER_OPTION_KEYS` /
  `PORTRAIT_OPTION_KEYS` in `engine_liveportrait.py`: every engine
  option is one or the other, and a test checks his avatar's
  `metadata.json` names only known keys. The driver's: the gains
  (`expression_gain`, `eye_gain`, `eye_headroom`, `eye_scale`,
  `gaze_gain`, `brow_gain`, `brow_follow`, `mouth_*_gain`), the gates
  (`expression_gates`, `mouth_gate_mar`, `mouth_gate_width`,
  `mouth_activity_mar`, `mouth_activity_width`, `brow_deadband`), the
  leak (`brow_mouth_leak`, `brow_mouth_leak_slopes`), the rest
  (`mouth_rest`, `lip_rest_rows`, `lip_rest_tracking`, `anchor_frames`),
  the holds and the smoothing, `translation_gain_*`, `scale_gain`,
  `crop_centre`. The portrait's: the crop (`crop_scale`, `crop_vy_ratio`,
  `crop_border`), the mask (`mask_*`, `chin_follow`, `edge_match`), the
  networks (`stitching`, `*_retargeting`, `lip_gain`,
  `eye_expression_gain`, `eye_mask_output`, `retarget_zero_at_rest`),
  `mouth_tilt_trim`, `rotation_pivot`, `pivot_z`. The engine's
  `set_driver_options` accepts the first set only; `_Portrait` lays it
  over the manifest's options at load; the graph bakes them, so a
  profile applied to a loaded avatar needs a reload.
- **The hand constants became options** with the constants as defaults:
  the four gate thresholds and the brow dead band (`_GateThresholds`,
  read once at load, passed into `_mouth_gate` / `_mouth_activity` /
  `_brow_lift`), the leak table (`brow_mouth_leak_slopes` as
  `[[keypoint, axis, slope], ...]`, replacing the whole table), and
  `mouth_rest = "driver"` with `lip_rest_rows` -- the driver's own
  relaxed reading on the three lip-opening rows, written into the anchor
  the way "portrait" writes the portrait's (`_apply_mouth_rest`, at the
  freeze and on a seed). Nothing changes without a profile: 256 engine
  tests unchanged.
- **The instrument.** `LivePortraitEngine.frame_readings(avatar_id)`:
  anchor frozen / held, the three gates as applied, the rest values, the
  extractor's own reading of every mouth keypoint row (`exp_<kp>_<axis>`)
  and the anchor's lip rows -- two small tensors off the GPU, meant for a
  calibration, not the frame path.
- **The package `sha256.calibration`.** `script` (twelve steps, 46 s
  with the 3-s lead-in: sit still 5, blink 3, close the eyes 3, open the
  mouth 3, talk 7, smile closed 3, smile with teeth 3, raise the brows 4,
  glance 4, turn 5, sit still 3; the recorder's wording where the
  movement is the same, and a keyword table that maps the recorder's
  "full" and "natural" scripts onto the same steps); `samples` (his
  readings with the engine's own ratio functions, the avatar's read back
  by a second tracker on the composite); `measure` (below); `profile`
  (`configs/local.driver.json`); `session` (the live run: the runner's
  new `add_observer` hands every published frame to it on the pipeline
  thread; the camera frame goes to a writer thread -- FFV1, the
  recorder's codec -- and the composited frame to a reader thread with
  its own unfiltered tracker, so the pipeline pays nothing; the clip and
  its `tracking.npz` are aligned frame for frame, a dropped frame drops
  its row; `clip.json` is the recorder's sidecar format with `kind` per
  segment, so `measure_follow.py` reads a session as it reads
  `driving.mkv`); `offline` (a clip through the same measurement).
- **The window.** "Calibrate me" beside "Re-centre": refused without
  Start or without the neural engine; a consent dialog that names the
  recording's folder before anything is written; a large instruction
  with a countdown over the video; press again to stop (the recording is
  removed); at the end the measurement, `report.txt` and `profile.json`
  in the session folder, the profile written (the previous one kept as
  `local.driver.<its stamp>.json`), the engine given the profile and the
  avatar reloaded under it with the pipeline stopped and started again
  (a frame must not meet an unloaded avatar), and a dialog with the
  report and where the person's face went. Closing the window mid-run
  cancels and removes the recording. At start, `bootstrap.apply_driver_
  profile` lays the machine's profile over the engine before the default
  avatar loads.
- **Tools.** `scripts/calibrate_clip.py --clip | --session | --samples`
  (the last re-runs the measurement on saved per-frame samples in
  seconds, no GPU -- how the rules below were re-read three times in an
  afternoon); `measure_follow.py` finds its segments by keyword now.
- **Tests: 1074 non-gui (was 971) + the gui files run one per process
  (main_window 15, calibrate 8, shell 11).** The measurement is tested
  on synthetic sessions with known gains, jitter and leak slopes; the
  session on the runner's fake with a 64x48 clip really encoded; the
  offline path on a 30-frame MJPG clip with a sidecar. ruff and mypy
  clean (the 11 warp-engine mypy errors pre-exist).

**The rules, each traced to the hand value it reproduces** (the
docstring of `calibration/measure.py` carries them):

| profile item | rule | the hand value's own rule (day) |
|---|---|---|
| `mouth_activity_mar` / `_width` | still p99 of the per-frame change / 1.1, floors 0.002 / 0.008 | live p99 read 1.1 units at 0.004 / 2.5% (days 9-12) |
| `mouth_gate_mar` | 2 x the still top (p95 of his resting opening, both still steps), full 3 x start; [0.003, 0.012] | "twice the still reading's own top", full 3 x (day 9, x0.4 day 12) |
| `mouth_gate_width` | 3 x the within-still-step spread (p95), full +0.05; [0.02, 0.06] | "still spread +-1% -> 0.03, full 0.08" (day 9) |
| `brow_deadband` | p90 of his brow deviation over blink / eyes shut / open / talk / smiles / glance, frontal; [0.05, 0.25] | +9..13% over the same confounds -> 0.10 (day 9) |
| `expression_gain` | old / the slope of (avatar opening, his opening) over talk + open; dead zone 5%, change x[0.65, 1.5] | talk mar gain to 1.0 (days 12, 15) |
| `eye_gain` | old / the slope of (avatar EAR, his EAR) over blink + eyes shut; x[0.7, 1.4] | blink gain to 1.0 (days 8-12) |
| `brow_follow` | old / (avatar rise p95 / his rise p95) over the raise -- the whole raise, since the drive has a dead band; needs his p95 >= 0.15 | "+38% -> +34%" (day 9) |
| `gaze_gain` | old / (avatar glance range / his, p95-p5) over the glance, blinks out | gain 1.03-1.06 (day 11) |
| `brow_mouth_leak_slopes` | per row, the slope of the extractor's row on his brow deviation over the raise with the lips shut (opening within 2 x the gate start); \|r\| >= 0.6, else the row kept | the day-15 regression |
| `lip_rest_rows` + `mouth_rest="driver"` | median of the extractor's three lip rows over still frames with both mouth gates shut (>= 30) | the day-15 clip rest |
| reported, not written | the smile's width response; the rest values; the drift of the resting mouth between the two still steps | day 9 set `mouth_smile_gain` by eye |

The first cut of the rules (p99 for the distance gates and the dead band,
a slope past the dead band for the brows and the gaze) read his clips
wrong -- the width gate came out 0.08 (the closed smile would not have
opened it) because his resting width sits 2-4% wider a minute into a
clip, and the brow slope read 1.47 where the hand rule saw a whole raise
against a whole raise. The saved samples made the second reading a
matter of seconds: the rules above are the second.

**Measured on his two clips (the bar: his hand values of day 15 within a
few percent).** `calibrate_clip.py --clip recordings/driving.mkv` /
`natural.mkv`, `--no-profile`, the avatar's own options; ~6 min each on
the GPU; samples kept in `recordings/day16/*-samples.json`.

| option | hand (metadata) | driving.mkv | natural.mkv | reading |
|---|---|---|---|---|
| `expression_gain` | 1.8 | 1.606 (talk 1.12x, r 0.99, 356 fr) | 1.8 kept (1.00x, r 0.95, 594 fr) | the clips disagree by 12%, day 15 saw the same (1.15 / 0.99); the hand value is their middle |
| `eye_gain` | 1.0 | 1.0 kept (blink 0.96x, r 0.97) | no blink step | **within 5%** |
| `brow_follow` | 1.0 | 0.919 (1.09x, r 0.94) | 1.192 (0.84x, r 0.99) | +-16% either side of the hand value; day 9 found the call-posture response lower |
| `gaze_gain` | 1.0 | 0.918 (1.09x, r 0.98, 296 fr) | no glance step | 8%; day 11 read 1.03-1.06 on the same segment |
| `mouth_gate_mar` | 0.004 / 0.012 | 0.0046 / 0.0138 | 0.0086 / 0.0258 | driving **15%**; natural's rest opening is twice driving's |
| `mouth_gate_width` | 0.03 / 0.08 | 0.0313 / 0.0813 | 0.0506 / 0.1006 | driving **4%**; natural's still frames are noisier (1.7% spread) |
| `brow_deadband` | 0.10 | 0.072 | 0.099 | natural **1%** (call posture, where the confound is); driving's scripted talk moves the brows less |
| `mouth_activity_mar` / `_width` | 0.004 / 0.025 (live) | 0.002 / 0.008 (floors) | floors | a clip's still jitter is 3-7x below live (day 9 night); only a live run can set these -- expected, and the floors hold |
| leak (17,1) (20,1) (7,1) (17,2) (20,2) | 0.052 0.144 0.020 -0.014 -0.025 | 0.043 0.101 0.015 -0.017 -0.019 | 0.059 0.188 0.024 -0.011 -0.030 | **the two clips' mean 0.051 0.145 0.020 -0.014 -0.025 -- the hand table within 3% on every row** (it was the clips' average) |
| `lip_rest_rows` | clip rest (-0.001, 0.002, 0.008) (day 15) | (-0.0013, 0.0009, 0.0068) | (-0.0012, 0.0033, 0.0085) | within 0.0011 of the day-15 reading; the anchors' own rows 0.001 more open on both |
| smile width (reported) | -- | 0.60x (r 0.99) | 1.20x (r 0.85) | not written; the two clips disagree 2x |

So: the instrument reproduces what it was built from -- the leak table
to 3%, eye_gain and the width gate to 5%, expression_gain on the clip it
was set on -- and where his own two clips disagree (talk gain 12%, brow
response 30% between them) the calibration says which clip it is
reading. The bar "his profile within a few percent of the hand values"
is met on the driving clip for everything but the live-only motion
units and the opening gate (15%), and cannot be met from clips for the
motion units at all. **His live run is the one that counts**, and it is
the one thing this day could not do without him at the camera.

**A finding for later: the resting mouth drifts.** Between the first
still step and the last, his resting width read +3.7% (driving) and
+2.4% (natural) wider, and the opening +0.0007..0.0011 -- past the
width gate's start on driving. The gates compare to the anchor's frozen
rest, so late in a session the width gate can sit part-open at rest
(the extractor side has `lip_rest_tracking`; the gate side has no
rest-width tracking). The calibration reports it as a note; a
`rest_width_tracking` on the same pattern is the candidate fix, not
built.

**The replay bar.** A session's `clip.mkv` + `clip.json` are the
recorder's format with the calibration's instructions; `measure_follow`
finds its segments by keyword now (blink / close your eyes / mouth wide
/ talk / smile / eyebrows), which a unit test checks against both
scripts. There is no live session yet to replay -- the first is his;
`driving.mkv` through the changed `measure_follow.py` is the run that
proves the path (`recordings/day16/driving-follow-day16.csv`).

**What he sees live (the instruction for his look).** Start, the avatar
on, then "Calibrate me": a dialog names the folder
(`recordings/calibration-<stamp>/`) and asks; 46 seconds of
instructions in large type with a countdown; then "Measuring..." for a
few seconds, the pipeline stops and starts once (the reload), and a
dialog with the report. The profile is `configs/local.driver.json`;
`configs/local.driver.<stamp>.json` keeps the one before. To undo:
delete the file and restart, or set any option back by hand -- the
avatar's `metadata.json` is never touched. Then the three things to
watch on a call: talking (the gain is now his measured one), a slight
opening after sitting with the lips pressed (`mouth_rest = "driver"`:
the anchor starts from his relaxed rows -- the one item that can look
different in the first minute), and a brow raise (his own slopes).
Press "That looked wrong" during anything odd. The offline check
afterwards: `calibrate_clip.py --session recordings/calibration-<stamp>`
re-measures the same face after any engine change, no sitting required.

**Files of his face this day:** `recordings/day16/` (two sample JSONs --
readings, no imagery -- the logs, and the measure_follow CSV). Nothing
new with pixels.

### Day 16, block B: new avatars without surprises -- built, measured on his four portraits

**B1, the picture prepared at import** (`importer.prepare_portrait`,
`chin_room` in the engine). The inspection now keeps the face's points
(`PortraitReport.points`); with them the photograph is scaled down to a
450-px face when taller (INTER_AREA; never up), then padded with its own
edge rows (`BORDER_REPLICATE`, not black) until the engine's whole crop
square lies inside the picture and the chin has 0.40 of a face height
under it. `chin_room(points)` is the load's own arithmetic -- the crop's
chin row, the headroom less the oval's measured feather tail and the
clearance, the reach and the jaw drop's cap, and whether the crop lies
inside the picture -- as a pure function the importer asks before any
load; `load_avatar` reads the same function now. The original is kept as
`original.<ext>` (asset role `source_original`, optional), the numbers in
`portrait_profile.json`. **His four portraits**: all had 0.23-0.27 of a
face height under the chin (the crop at -0.06 wants 0.24); prepared they
have 0.40, the crop inside, the jaw drop 50-52 px: portrait 1 scaled
0.81 and padded 179/64/91/97 px (top/bottom/left/right) to 1021x1076;
portraits 3 and 4 (2048-px photographs) scaled 0.40; portrait 2 (642 px)
padded only. The ROADMAP's bar ("a photograph with the chin at its
bottom edge imports and the chin drop's cap reads 50 px or more") is a
unit test on a synthetic face with the chin 10 px above the edge, and
his portrait's own case is a test that runs when the model and the
portrait are on the machine. **A finding**: the cap is a crop-framing
number (52 px at `crop_vy_ratio` -0.06 for every face, since the crop
normalises the face height), not a picture number -- what the padding
buys is a crop with skin under the chin instead of a black band where
the mask's reach and drop paste.

**B2, the portrait profile** (`importer.portrait_profile`). Read from the
prepared picture: the chin's luminance against the cheeks' (points 50 /
280; a patch between the lower lip and the chin) -- `bare` under 8%
darker, `stubble` to 20%, `beard` above; the hairline, walked up from
the brow top along the face's own vertical at three columns, as the
fraction of the forehead (brow top to the mesh's own top) where the
luminance leaves the forehead skin's by 30% for 6 px; the lips (closed /
parted / open by the mouth ratio). What it writes into the new avatar's
`engine_options`: `mask_chin_reach: "auto"` (B3 below) and
`mask_forehead_reach` lowered to the hairline less 0.08 when the hair
starts below the default 0.85 -- never below a floor of the mask's own
softness (erosion + half the feather of the face width) plus the brow
hair's height (day 13's lesson at 0.65: a fraction has no floor above
the brows). The jaw shrink is *not* set from the portrait: day 12 found
it a decision about the pair (0.08 hid a bearded driver's jaw under a
bare portrait and put a ledge under a stubbled portrait on a shaved
one); the profile records the beard for that decision, and the
calibration's session frames could carry the driver's side later.
**His four**: stubble / stubble / beard / stubble (chin 18-20% darker
than the cheek -- the moustache and stubble he no longer has, day 12),
hairline at 1.52-1.58 of the forehead (above the mesh's top, so the
reach stays 0.85 -- his hand value), lips closed (0.005) / open (0.166,
0.187, 0.114 -- the importer's own warning). Imported into a temporary
library through the real path, all four validate, with `original.png` /
`original.jpg` beside `portrait.png` and no hand edit of
`metadata.json`: the bar met, on the mask side, by measurement; his eye
on the pasted result is the other half.

**B3, portrait-to-driver proportions -- measured, and the premise did not
hold.** The plan: the driver's nose-to-chin over face height against the
portrait's sets the chin reach at the anchor. Measured with MediaPipe on
the portraits' crops and on his captures' frontal frames: the portraits
read 0.449 / 0.464 / 0.468 / 0.458, he reads 0.446-0.454 -- a difference
of 0-4 crop px, real and small. Self-driven, each portrait's rendered
chin lands on its own chin point within 0.3-3 px (the network does not
move the chin), and on his day-15 captures the rendered face sits 2-6 px
above his as a whole (eyes -5, chin -6 on 10:14; the crop's placement,
not a proportion). Then the reach that added the anchor's own rest_chin
was tried on the captures: the live anchor's reach read 10 px longer
than his frontal median (a session starts at whatever posture) and put
the join 14-21 px under his chin; the hand 0.04 puts it +7..+12, reach 0
puts it 2-6 px *above* (`recordings/day16/join-probe.log`). What the
hand value is, in the mask's own terms, is the contour's clearance above
the chin point -- the erosion's radius plus half the feather, of the crop
face width: 9.4 px on his portrait against the hand's 9.3. So
`mask_chin_reach: "auto"` = that clearance, resolved at load like a
number (`_auto_chin_reach`, `chin_room`), the importer writes it for
every new avatar, and his metadata keeps 0.04 (the same to a tenth of a
pixel). The bar as written ("the avatar's chin within 4 px of the
driver's on his day-15 captures") is a placement bar, not a mask bar,
and stands at 2-6 px whole-face on the captures with a frontal anchor.

**Tests: 1101 -> 1104 non-gui** (`test_portrait_preparation.py` 11,
`test_engine_chin_reach_auto.py` 3; the importer's 15 unchanged), gui
files as before; ruff and mypy clean.

**What he sees live.** "Add avatar" on any of his photographs: the report
now says what was done to the picture ("prepared: scaled ..., padded
..."), what the picture holds ("portrait: stubble ..., hairline ...,
lips ...") and the folder has `original.<ext>` and
`portrait_profile.json` beside `portrait.png`. The chin join on a new
avatar is "auto" -- the same place his 0.04 puts it.

### Day 16, block C: what his eye still sees -- three levers built, two measured off, one waiting for his eye

**C2, the turn lag -- the premise measured, and it was mostly not a lag.**
The yaw pivot of day 10's Step 4 was built as `yaw_pivot_z` (a point on
the head's vertical axis behind the face, in keypoint units; the yaw
delta's own translation `p (I - R_y)`, added beside the pitch pivot's,
exactly zero on a pure pitch; default 0 = bit-exact) and swept on
driving.mkv's turns (frames 1342-1641, `measure_follow --last 1641`,
`recordings/day16/driving-yaw-*.csv`) against the day-16 baseline
(`driving-follow-day16.csv`):

| run | turn \|dx nose\| p50 / p90 / max | yaw [-90,-30) lag p50 | yaw [30,90) lag p50 | frontal dx p50 |
|---|---|---|---|---|
| baseline (0) | 12.6 / 21.3 / 38.5 | +19.9 | -3.3 | +13.4 |
| yaw_pivot_z 0.1 | 14.0 / 27.6 / 40.5 | -4.8 | -15.0 | +13.2 |
| yaw_pivot_z 0.3 | 15.3 / 59.0 / 72.7 | -29.9 | (worse) | +11.3 |

The baseline's "lag" is one-sided: +20..25 px on a turn to his right, -3..-9
on a turn to his left -- and **+13.4 px on the frontal frames**, where
there is no turn at all. Read per anchor on the day-16 measure_follow
table: the avatar's eyes sit +7.5 px, its mouth +7.6, its chin +6.1 and
its nose +14.2 to the screen-right of his at rest. So the day-14 "21 px
at p90" was a static sideways offset of the whole face (~7 px) plus the
portrait's own nose sitting further right than its eyes (~7 px more at the
nose), with 4-7 px of real, symmetric lag on top. A pivot shifts the face
by sin(yaw) in one direction and cannot take a constant off; at 0.1 it
moved every turn by ~25 px, five times the real lag. **Kept as an option,
default 0, not in his metadata.** What follows from the finding:
`crop_shift_x` (crop px along the face's horizontal, in
DRIVER_OPTION_KEYS): a constant shift of the driver's crop, which moves the
avatar by the same amount and nothing relative (the anchor learns in the
shifted crop). "Calibrate me" measures it -- the avatar's eye centre
against his on the still frames, `avatar_eye_dx` in the profile's rest --
and writes it (dead zone 2 px; on driving.mkv the reading would be +7.5
px -> -5.8 crop px). Unmeasured live; the first live calibration sets it.

**C3, the first seconds after a pressed anchor.** `lip_rest_tracking` now
moves the anchor's lip rows four times faster over the first 60 resting
frames after the freeze (`_LIP_REST_FAST_FACTOR`/`_FRAMES`), then at the
option's rate -- within 10% of the resting reading in two seconds instead
of eight; a re-centre or a seed starts the fast frames again. With
`mouth_rest = "driver"` from the profile the start is his relaxed rows to
begin with. Tests on the CPU fixture; the old lip-rest tests start past
the fast frames.

**C1a, the forehead to the hairline -- built, measured, not kept.**
`mask_forehead_reach` accepts numbers up to 1.6 (past 1.0 the oval's
forehead arc is lifted toward the ceiling -- all of it at the arc's top,
none at the brow line -- so the hull reaches above the mesh's own top
point) and `"hairline"`: the portrait's hairline is read at load
(`_hairline_fraction`, the importer's reader moved into the engine: a walk
up from the brow top along the face's vertical at three columns, hair
where the luminance leaves the forehead skin's by 30% for 6 px), the
driver's on frontal frames until the anchor freezes, and the reach settled
at the lower of the two less 0.08 (`_hairline_reach`, [0.3, 1.6]); the
portrait's mask and its box are rebuilt then; a re-centre measures again.
Measured with `jaw_band.py --capture --render` on four captures (forehead
band, grey levels): the two day-12 captures learn no anchor under --render
and read identically (8.67 / +0.94; 9.40 / +1.79); on 032824 (anchor
learned) abs 5.89 -> 5.50, step +0.45 -> +2.28; on 101434 (seeded) abs
2.56 -> 3.29, step -0.44 -> -0.07. No gain on the metric, and the band
moves with the edge so the numbers are not the same band; his eye is the
judge and the forehead was parked at his "later". **Default 0.85, his
metadata unchanged; "hairline" is there to try.** The hair switch (a
segmentation matte, the portrait's hair over his) is not built -- the
2-3 days it needs are block D's now.

**What he sees live (block C):** nothing changes by default. To try the
forehead: `"mask_forehead_reach": "hairline"` in his metadata, Start,
look at the camera for the second the anchor takes, and the join moves to
the lower of the two hairlines. The crop shift arrives with his first
"Calibrate me".

### Day 16, block D: the package and the five -- D1, D2, D4 built; D3 decided; D5, D6 his

**D1, the courtesy switch** (`sha256.app.kill_switch`, `[kill_switch]` in
the settings: `url` empty by default, `check_hours` 1, `timeout_seconds`
5). A static file at the URL, read at start and hourly on its own thread:
JSON `{"enabled": false, "message": "..."}` or the text `off` stops the
application -- the window stops the pipeline, disables Start and shows
the message once; everything else (no URL: no request at all; no host, a
slow host, a 404, an empty or strange file) is allowed. Never on the
frame path. Tests with a local HTTP server: the stop, the message, the
fail-open cases, the listener surviving its own failure. The guide says
what it is and that clearing the URL is the user's right.

**D2, the GPU check.** `bootstrap._start_engine_with_fallback`: when the
neural engine cannot start (no NVIDIA GPU / CUDA / weights -- the
`EngineError` it already raised, which used to stop the application from
opening), the photograph engine is started in its place and the window
says so once (`ApplicationContext.startup_notes`); a card under 6000 MB
gets a note too. Tests with a stub registry.

**D3, the virtual camera:** decided, not coded. pyvirtualcam (GPL) stays
for development and drives the OBS Virtual Camera device; the shipped
folder's `default.toml` sets it, the guide tells each user to install OBS
Studio once. A native filter is not worth its days.

**D4, the portable build.** `packaging/sha256.spec` (PyInstaller: the
frozen `sha256.exe` with torch, mediapipe, cv2, PySide6, pyvirtualcam;
the weights and avatars stay outside) and `scripts/build_exe.py` (freezes,
then lays `models/`, `avatars/` (the ids named), `configs/default.toml`
with the shipped backends and the switch URL, and `USER_GUIDE.md` beside
the executable, zips it on request). `AppPaths.resolve` knows the layout:
a frozen executable with a `models` folder beside it is its own root
(`portable_root`) -- a USB drive or a folder, everything together, which
is how he hands it out; without it, `%LOCALAPPDATA%` as before.
`docs/USER_GUIDE.md` is the one page: what is needed (the card, the
camera's settings -- NTSC, USB Streaming, Smartphone Connect off, a data
cable -- OBS once, Discord closed), the first run (Start, Calibrate me,
the virtual camera), the call (Re-centre, the black screen and its hotkey,
the quality, That looked wrong), adding an avatar (the owner), a table of
what goes wrong, and privacy. PyInstaller 6.22 was added to the dev
group and the build run on this machine: 141 s to freeze, the portable
folder 5.3 GB (the CUDA runtime is most of it), `sha256.exe` 44 MB. The
first frozen run died in PyInstaller's "Unhandled exception" dialog
within 10 s: the layout had appended the shipped backends to
`default.toml`, and TOML declares a table once ("Cannot declare
('engine',) twice") -- they go in `configs/local.toml` now, layered as a
machine's own file is. The second run opened the `sha256` window at 30 s
with the neural avatar loaded (405 MB on the GPU), the tracker ready, the
hotkey bound, no camera plugged in at the time (0 devices enumerated).
The portable layout was also started on the development interpreter with
`sys.frozen` faked (26 s to the active avatar). What remains D4's own bar
is a second machine that never saw the repository -- his test -- and the
zip's size on the download site.

**D5 (five people) and D6 (push) are his:** portraits, installs, calls;
the push when he says, under the anonymity rules.

**Tests at the end of the day: 1132 non-gui, the gui files one per
process (main_window 15, calibrate 8, block D 2, shell 11); ruff and mypy
clean (the 11 warp-engine errors pre-exist).**

**What he sees live (block D):** nothing on this machine unless the
switch URL is set. On a second machine: the folder from `dist/
sha256-portable/`, the guide's first-run steps; a machine without an
NVIDIA card opens on the photograph engine with a plain sentence.

### Day 16, his first look at "Calibrate me" (2026-09-12, 00:41): more time, a voice, and what his live profile read

**He ran it** (`recordings/calibration-20260912-004157`: 894 frames at the
live ~19 fps, none dropped, FFV1, 467 MB; `configs/local.driver.json`
written 00:42) and said: *"you give me very little time for the movements
-- I have to read and then do the movement; either a voice, or 3-5 more
seconds per task."* Both:

- **A read phase per step** (`CalibrationStep.read_seconds`, 4 s on every
  movement step; the lead-in and the first still step have none -- the
  anchor wants those frames at once). `step_at` returns a `StepPosition`
  (step, seconds left in the phase, whether it is the read phase); the
  session tags read-phase frames `ready` (their own sidecar segment, "Get
  ready: ..."; the keyword table maps them to `lead`, never a measured
  step); the window shows "Get ready... 4" then "NOW 3". About 85 s now.
- **A voice** (`sha256.ui.speech.Speaker`): Qt's own text-to-speech
  (SAPI / WinRT on this machine, both present) reads each instruction as
  its step begins, once per step; a machine without a voice is silent and
  the read phase carries it. Replaced by a fake in the window tests.
- The brows step is 5 s (the leak's thirty shut-lips frames need it).

**What his live profile read, and what the rules did with it.**

| item | hand | live (00:41) | re-measured with the rules below |
|---|---|---|---|
| talk gain -> `expression_gain` | 1.8 | 1.20x -> 1.497 | 1.20x -> 1.5 |
| blink -> `eye_gain` | 1.0 | 0.51x -> 1.4 (the change ceiling) | 0.48x -> 1.4 |
| brow raise -> `brow_follow` | 1.0 | 0.97x, kept | 0.98x, 0.867 with the band's correction |
| glance -> `gaze_gain` | 1.0 | 1.25x -> 0.81 | 1.25x -> 0.8 |
| avatar's eyes vs his at rest -> `crop_shift_x` | 0 | +9.8 px -> -5.9 | +9.9 -> -5.9 |
| `mouth_activity_mar` / `_width` | 0.004 / 0.025 | 0.002 / 0.0108 (the still step read 0.0008 / 1.2%) | 0.004 / 0.025 (floors) |
| `mouth_gate_mar` | 0.004 / 0.012 | 0.0061 / 0.0183 | 0.007 / 0.021 |
| `mouth_gate_width` | 0.03 / 0.08 | 0.06 / 0.11 (still spread 4.2%) | 0.06 / 0.11 (5.0%, the ceiling) |
| `brow_deadband` | 0.10 | 0.173 (confound p90 17%) | 0.066 (offline p90 6.6%; bounded [0.05, 0.15]) |
| lip rest rows | clip (-0.001, 0.002, 0.008) | (-0.0003, 0.0023, 0.0132) | not measured (26 shut-gate frames) |
| leak table | day 15 | kept (25 shut-lips frames < 30) | kept |

Read against the clips' numbers: the talk gain live (1.20x) is driving's
(1.12x) and more; the blink at 0.51x is *half* what the clips read (0.96x)
-- live at 19 fps a blink is two or three frames, and what the second
tracker sees of the avatar's lids on the composite is smaller than on a
30-fps clip; `eye_gain` went to the change ceiling (1.4x) on it, and
whether that over-closes at rest is his eye's call (the eye network's own
rest is subtracted, so 1.4 scales the blink's depth only). The sideways
offset is real and the same on the clips (+7.5 px there, +9.9 live). The
still step read far LESS mouth jitter than a call's rest (0.0008 against
the day-9 captures' 0.0043) and far MORE width spread (4-5% against the
clips' 1-2%): a deliberate stillness is not a call's stillness. Hence the
rule changes:

- **The motion units are floored at the day-12 live values** (0.004 /
  0.025): a still step cannot see a call's jitter, so the calibration can
  only raise the units, never make the gates twitchier at rest.
- **The first still step's first 30 usable frames are dropped** from the
  jitter and the rest (the tracker and the person settle; the width
  spread read 4.2% with them and 5.0% without -- his mouth genuinely moves
  at rest, so the width gate stays at its 0.06 ceiling: the closed smile
  (+8% median) opens it part-way; his eye decides whether that is worse
  than a gate that opens at rest).
- **The brow dead band is bounded [0.05, 0.15]**, and **the brow gain is
  corrected for the band's own change** (`extra_factor` in `_adjust`: the
  drive lifts by his rise less the band, so a band that moves would raise
  the avatar less at the same peak -- the gain is scaled by (peak - old
  band) / (peak - new band) so the measured peak ratio still comes out at
  1.0). Live the confound read 17% at p90 and offline 6.6% on the same
  frames -- a difference between the live 19-fps tracking and the clip's
  30-fps replay that is not understood; the bounds hold either.
- The leak's shut-lips criterion is at least 0.008 of opening
  (`LEAK_SHUT_MIN`); with the longer brows step the thirty frames should
  come.

**Not changed on his word:** the profile in force is the one his 00:41
run wrote (eye_gain 1.4, the width gate 0.06, the dead band 0.173, the
units 0.002 / 0.011). To go back to the hand values: delete
`configs/local.driver.json` and restart; the next "Calibrate me" writes
the profile under the rules above.

**The passphrase** he set (00:45): `configs/local.owner.json` holds a
32-hex salt, a 64-hex key, 200 000 iterations; the file reads as
configured and a wrong passphrase is refused. The passphrase itself is
his alone.

**"Add avatar" opens with the guide** (`sha256.avatar.photo_guide`,
`sha256.ui.photo_guide_dialog`): what the photograph must be (frontal,
eyes open, lips closed or barely parted, even front light, hair off the
forehead, no glasses, a large face, room below the chin, a plain
background, square 1024+), then two prompts to paste into an image
generator -- the portrait (with a line to describe the person) and the
optional lips-barely-parted variant (attach the portrait; day 6's teeth
lever) -- each with a Copy button; "Choose the photograph..." goes on to
the file dialog, Cancel stops. In English, as asked.

**Tests: 1135 non-gui, the gui files one per process (main_window 17,
calibrate 8, block D 2, shell 11); ruff and mypy clean.**

### Day 16, the menu (2026-09-12): the bar that set the window's width

**His report after the day-16 build:** the buttons along the bottom did
not fit when the window opened, and the window could not be made smaller.
His instruction: a dropdown menu; build it with tests, he tests later.

**Measured on this machine (2560x1440, Segoe UI 9) before the change:**
the bar's fifteen controls set a minimum width of **1846 px**, and a
status message added its own width on top of that (**2342 px** with the
black screen's line showing) -- a plain QLabel asks for the whole width
of its text. The window asked for 780 px of height on any screen, and
Windows does not move a window up to keep its bottom on the screen. So on
a screen with less than that free the bar was below the edge, and the bar
would not let the window shrink.

**Built (`sha256.ui.main_window`, `sha256.ui.widgets`):**

- **The bar keeps six controls:** Start, Stop, Camera, Avatar, Virtual
  camera, the red black-screen button -- what a call needs at one click --
  then the status line, the fps, and **Menu ▾** at the right end.
- **The Menu** (a QToolButton with a QMenu, the dark style of the combo
  lists): View ▸ (Avatar / Tracking (no avatar), exclusive), Dots, Mirror,
  Quality ▸ (Native / 1080p / 720p / 480p / 360p, exclusive), then That
  looked wrong, Re-centre, Calibrate me (its entry reads "Stop
  calibrating" while one runs), then Login / Owner and Add avatar (shown
  only unlocked). Every entry is a QAction the handlers and the tests
  drive alike; tooltips show in the menu.
- **`ElidedLabel`** for the status line: asks for no width of its own,
  cuts what does not fit with "…", keeps the full text in the tooltip and
  in `text()`.
- **The combos ask for 8 characters**, not their longest entry (the list
  still opens at full width; the chosen name is the tooltip).
- **`opening_geometry`**: the window opens at 1200x780 when the screen's
  free area has the room, else at what fits under the frame (40 px of
  width, 60 of height left for it), centred, never above the title bar's
  room. A 1080p laptop at 150% (1280x660 free) gets 1200x600.

**Measured after:** minimum width **1846 -> 1033 px**; with the status
line's longest message **2342 -> 1033** (the line reads "BLACK SCREEN -
the other…" at the 1200 opening width, the whole in the tooltip). Height
306 minimum.

**Tests:** main_window 24 (was 17): the bar's composition and the menu's
entries, the view entries choosing what the preview shows, dots and
mirror reaching the preview, a long status not widening the window, the
label's cut and tooltip, the opening geometry on three screens, the
window built at that size. The calibrate tests follow the action's text.
The guide names the moved controls as "Menu → …".

**Not yet:** his look at it -- the menu's look on his background, whether
"Camera battery died" should stay on the bar (it did, on purpose: one
click in a bad moment), whether 8 characters of a camera's name is
enough. Uncommitted, on his word.

### Day 16, the first other face (2026-09-12 evening): the anchor that never locked, and a name for each person

**His report:** a friend ran Calibrate me twice; "it did not fit the
avatar to the face afterwards and did nothing", nothing on screen said
the calibration had worked, and: is there a select somewhere for whose
calibration applies? His ask, sent while this was being measured: a name
field at the end of every calibration, and the name chosen from the menu
when the person sits down.

**Measured (the two sessions' own recordings, `tracking.npz` frame by
frame against the anchor gate, then `calibrate_clip.py --session`):**

- The anchor gate took the camera for the person's eye line: `|pitch|
  <= 4`, `|yaw| <= 4`, `|roll| <= 3` degrees around ZERO. The friend's
  head reads **pitch -11** sitting still in front of this camera (the
  owner's reads -3, and passes by a hair); yaw -4. In his 18:56 "sit
  still" step **0 of 106 frames** passed the gate; in the whole session
  36 of 1894, in the 19:01 session 3 of 1866. The log agrees: each anchor
  before the calibrations froze after 889-1195 rejected frames (30-60 s),
  inside the 18:56 session it froze during the BROWS step (a raise opens
  the eyes and lifts the head past the gate), inside the 19:01 session
  never. Every frame before the freeze is unusable to the measurement, so
  the 18:56 run read talk 0, blink 0, smile 0 frames and wrote a profile
  of two keys (crop_shift_x kept, gaze_gain 0.77) -- which, being the
  whole `local.driver.json`, dropped the owner's twelve measured values
  on the way -- and the 19:01 run wrote "Nothing measured" and left the
  anchor reset, the avatar a still picture until the friend's face
  happened to pass the gate: his "did nothing".
- The pose noise of a head that IS still: over 15 frames, yaw/pitch/roll
  spread p50 0.1-0.3 deg, p90 0.4-1.4 (the owner's and the friend's alike).
  Reaching for the mouse (the day-5 case the gate was built for) moves
  several degrees in that half second.
- The friend's second "sit still" step was him being told what to do:
  spread 6-10 deg, roll 10 -- no gate would take it. And his turn ran into
  the last still step (yaw +44 at the median, the eye ratio reading 1.07):
  those frames, once the anchor is frozen, were "still" to the
  measurement and put the jitter at every gate's ceiling.

**Built:**

- **The anchor gate reads stillness** (`_pose_still`, `_ANCHOR_STILL_FRAMES`
  15, `_ANCHOR_STILL_SPREAD` 3 deg; bounds widened to `|yaw| <= 10`,
  `|pitch| <= 15`, `|roll| <= 6`): a frame is worth learning from when the
  pose has held for the last half second at whatever angle this camera
  sees a resting head from. A move while the anchor is still being learned
  starts the accumulation over (`_restart_anchor`), so two still poses are
  never averaged into one anchor. The anchor's own yaw is kept
  (`rest_yaw`, an anchor scalar like `rest_pitch`) and **the extreme-pose
  hold measures from the anchor's rest pose**: at rest pitch -11 a nod of
  nine degrees no longer freezes the avatar. The status line reads "Sit
  still and look at the camera to centre the avatar (n/15)".
- **The measurement counts still frames only while facing the camera as
  the anchor did** (`_anchor_pose`, `_frontal` on both axes relative to
  the rest pose); the anchor's pitch/yaw are in `frame_readings`.
- **The still step reads first too** (4 s, spoken), and the anchor is
  reset when its measured phase starts, not when the instruction appears
  (`CalibrationSession._enter_phase`). Every "Get ready:" segment maps to
  the read phase offline. 89 s in all.
- **Named profiles** (`sha256.calibration.roster`, his ask): Calibrate me
  asks the person's name at the end; the profile goes to
  `configs/drivers/<name>.json` (gitignored) and, as a copy, to
  `local.driver.json` -- still the file the start reads, so the bootstrap
  is untouched. **Menu → Person** lists Nobody (the avatar's own values)
  and every name, the one in force checked; choosing one lays it over the
  engine and reloads the avatar; the status line reads "Profile: name"
  whenever nothing else needs saying. A `local.driver.json` from before
  names existed is adopted as "Unnamed (when)". The same person
  calibrating again keeps, for what the run could not measure, their
  earlier value -- so "left as it is" in the report is true, and a new
  person no longer inherits the last one's numbers.
- **The roster on this machine, seeded:** the owner's 00:42 profile as
  "Owner (2026-09-12 00:42)", back in force (the friend's two-key file had
  replaced it); the friend's 18:56 session re-measured under the new gate
  as "Friend (2026-09-12 18:56)". Both to be re-run under real names.

**Measured after, offline, the same three recordings (`calibrate_clip.py
--session ... --no-profile`, the avatar's own values as the base):**

| session | live run (old gate) | offline, new gate |
|---|---|---|
| friend 18:56 | talk 0, blink 0, brows 67 (r 0.30), gaze 85, smile 0 frames; still 22 | talk 212 (r 0.99), blink 128 (0.98), brows 105 (0.98), gaze 85 (0.98), smile 128 (0.99); still jitter 0.0001 / 0.45%; lip rest 52 frames; leak 4 of 5 rows |
| friend 19:01 | nothing measured | talk 209 (0.98), blink 128 (0.98), brows 106 (0.98), gaze 85 (0.97), smile 128 (0.99); lip rest 64 frames; leak 4 of 5 |
| the owner 00:41 | talk 1.20x, blink 0.51x, gaze 1.23x, gates [0.0061, 0.0183] / [0.06, 0.11], lip rest [-0.0003, 0.0023, 0.0132] | talk 1.20x (195), blink 0.49x (117), gaze 1.25x (77), gates [0.0056, 0.0168] / [0.06, 0.11], lip rest [-0.0003, 0.0024, 0.0127]: his own profile again within a few % |

The two friend sessions agree with each other: expression_gain 1.48 /
1.38, gaze_gain 0.90 / 0.94, eye_gain 1.16 / kept (0.95x), brow_follow
kept / 1.06, mouth_gate_mar [0.0034, 0.0102] / [0.0033, 0.0099],
mouth_gate_width [0.044, 0.094] / [0.040, 0.090], lip rest rows within
0.002, leak rows within 0.02. His smile reads 1.45x / 1.36x (reported,
not written); his crop_shift_x -9.9 / -12.4 (the avatar's eyes +14 / +18
px from his at rest -- a per-anchor number, as day 16 found). The
frontal filter alone took the 18:56 still jitter from 0.047 / 8.7% (every
gate at its ceiling) to 0.0001 / 0.45%.

**Tests:** 1165 non-gui + 82 gui, ruff clean, mypy the 11 pre-existing
warp errors. New: the gate's stillness and the friend's pose, the pose
window, the restart, the hold from the rest pose, the still step's read
phase and reset, the roster (slug, list, activate, adopt), the Person
menu, the name at the end, the same person again, a blank name, still
frames turned away. Uncommitted, on his word.

**Not yet:** his live look (the anchor now locks in about a second on a
still face; the friend's profile from the recording, his own restored);
the friend under his real name; whether 3 degrees of spread is right for
a camera on a shaky desk; the units' floors on a second face (both
sessions sat at the floors, as the owner's live run did).

### Day 16 evening, the second look (2026-09-12 19:45): "a slightly wider face" that was not, and the crop shift withdrawn

**His report:** the friend (roster name "baska") calibrated under the
fixed gate -- talk 214 frames (r 0.99), blink 128, brows 106, gaze 80,
smile 128, 124 still frames, the anchor locked in 6 s -- then pressed
That looked wrong three times (captures/moment-20260912-1944xx) and sent
a screenshot: "she has a slightly wider face than mine; analyse and fix".

**Measured:**

- **Width.** Face width over height, the tracker's own points, at rest:
  jaw 58-288 portrait 0.720 / owner 0.669 / friend 0.709; cheek 234-454
  0.818 / 0.782 / 0.812; the face oval in crop pixels 191 / 184 / 190.
  The friend's face is the PORTRAIT's proportions within 1%; the owner's
  is 4-7% narrower than the portrait. "Wider than mine" is true and is
  not the defect: the rendered face is sized by face height, and hers
  matches the portrait's width exactly.
- **The seam.** The zooms (seen.png beside camera.png, 3x) show the
  paste's join running down her cheek on the ear side, 13-20 px inside
  her face contour, with her own cheek showing between it and the ear --
  the "wider face" -- and the portrait's stubble ending on that line. The
  mask's own geometry, rebuilt from her landmarks and the portrait's
  (scratch script, no GPU): at ``crop_shift_x`` 0 the portrait's oval
  lands on hers within **0-2 px on both sides**, on all three of her
  captures and both of his (outline shift -0.3..-1.3 crop px). With her
  profile's -10.0 the whole rendered face moves 13 frame px: the portrait
  oval 608..851 against her 619..864 -- 11 px into her hair on one side,
  13 px short of her contour on the ear side. That is the strip.
- **Where the shift came from.** The calibration turned the avatar's eye
  offset at rest into the shift (day 16 C2). On her that offset reads
  +11..+22 px at a resting yaw of -7..-9 degrees; on him +2..+4 at yaw
  -1..-2 (his session anchor, at -3.7, read +9.8 -> -5.9). The avatar is
  drawn frontal at rest whatever the head's resting yaw, and the eyes of
  a turned head sit toward the turn: the offset is the rest yaw against a
  frontal render, not a misplacement of the face. Aligning the eyes
  misaligns the outline by the same amount, and the join follows the
  outline.

**Changed:** ``calibrate`` reports ``avatar_eye_dx`` with the rest yaw
and never writes ``crop_shift_x`` (the option stays, for a hand value);
the three roster profiles lost their shifts (-5.9 / -9.9 / -10.0, the
removal noted in each file). Nothing in the engine.

**Proven** on her capture 19:44:32, replayed with the session's own
anchor (``scripts/probes/capture_replay.py --sweep crop_shift_x=-10,0``):
at 0 the pasted face reaches her ear-side contour and the strip of her
own cheek is gone; a paler band remains along that join -- the portrait's
cheek against her darker one, the colour correction's residual, a
smaller matter. Stills: recordings/day16/baska-earside-before-after.png
and baska-face-before-after.png (her face; delete when looked at).

**What this does not fix, and cannot:** the portrait's stubble on a
woman's chin and the tone of its skin against hers are the portrait's
identity -- her own avatar (Add avatar, a photograph with her
proportions) is the answer the product was built for; and his own look
at crop_shift_x 0 (the C2 shift is withdrawn on his profile too; a hand
value can go back into configs/drivers/owner-*.json).

**Tests:** 1164 non-gui + 82 gui; ruff clean; mypy the 11 warp errors.
Uncommitted, on his word.

### Day 16 evening, the third look (2026-09-12 20:07): the avatar that looked away, and the anchor's yaw in the wrong degrees

**His report:** baska in force (crop shift gone), one That looked wrong
(captures/moment-20260912-200754): "look at the errors, analyse, fix".

**Measured (the capture's tracking, camera.png against seen.png by the
tracker, the replay probe with the session's own anchor):**

- Mouth, eyes, width: the avatar within a few % of hers (mar 0.87x,
  open 0.86x -- both shut; ear 0.333 vs 0.335; width 1.06x). Brows: the
  avatar's 0.153 vs her 0.209 -- the portrait's own lower brows at rest,
  not a drive. Pose: the avatar 5 deg more frontal than her on both axes,
  as a render relative to her anchor is.
- **The eyes:** the tracker reads her irises at **+0.138** of an eye width
  to the side and the avatar's at **+0.143** -- the drive is faithful --
  and the 3x zoom shows her looking to her left, at the SCREEN, not the
  camera. Over the capture's 15 s her gaze sits at -0.02 (the camera) for
  6 s, at +0.13..+0.16 (the screen) for 4 s and at the press. Her anchor
  learned the rest gaze at -0.018 while she looked at the camera as the
  status line asks; every look at the screen after that became a full
  sideways glance of the avatar (+0.11 drive, 0.811 gain), for as long as
  she looked. On the owner it never showed: his anchor happened to catch
  him looking at the screen.
- **The anchor's yaw:** the evening's `rest_yaw` was taken from the
  network's pose average (`pose_d0`), which is in other degrees than the
  tracker's: her anchor read yaw +3.97 there while the tracker read -8.5
  (the log's "anchor frozen -- pose 10.7/4.0" is the network's; the
  tracker's pitch was -12.5). The hold and the calibration measure in
  the tracker's degrees.

**Built:**

- **The rest gaze follows where the person looks** (`gaze_rest_tracking`,
  a driver option, default 300 frames as the time constant, 0 off): once
  the anchor is frozen, on frames with the eyes open and the face not
  held, the rest moves toward the gaze by 1/300 a frame. A two-second
  glance moves it 6% of the glance (under the dead band); a look held at
  the screen becomes the rest and the avatar looks at the viewer again.
  Simulated on her capture's own trace: the drive at the press +0.116 ->
  +0.077 after 15 s; a look held at +0.14 from the anchor's rest: drive
  +0.112 at 0 s, +0.031 at 10 s, 0 at 30 s (the capture ran at ~19 fps,
  so the constant is 10-16 s of wall time).
- **The anchor's pose by the tracker** (`pose_samples`, the tracker's
  (pitch, yaw) on accepted frames, medians at the freeze): `rest_yaw` is
  the tracker's now, `rest_pitch` too when the crop centre has not set it.

**Tests:** 1169 non-gui + 82 gui; ruff clean; mypy the 11 warp errors.
New: the tracking's default and bounds, a held look becoming the rest
and a glance not, no tracking before the anchor / while held / at 0,
the anchor's pose as the tracker's median. Uncommitted, on his word.

**Not yet:** his live look; whether 300 frames is the right constant on
a call (the option is in the profile); the brows' rest height
(the portrait's) against a driver whose brows sit higher -- reported by
the calibration, not driven.

### Day 16 evening, the fourth look (2026-09-12, the same capture): her brow printed through, and the line at the join

**His report:** "you should have noticed more: her brow shows in the
photo; don't you see the line between the AI face and the photo? The
eyes are not the problem." Right on both counts -- the eyes were measured
and the picture was not looked at.

**Measured (3x zooms of seen.png beside camera.png; the replay of the
same capture with ``edge_match`` 0 and 1; the correction's own cells):**

- **Her brow shows:** a dark band above the avatar's thin brows, where
  her thick, arched brows are. With ``edge_match`` 0 (the plane alone)
  the forehead is clean; with 1 the band is there. The day-13 leak,
  back: the brow zone cut out of the residual was the hull of the brows'
  LOWER row of landmarks (plus one inner upper point) dilated 4 cells,
  16 px -- his thin brows, not the upper two thirds of hers -- so what
  was left of them went into the residual as "the skin's difference at
  the edge" and was printed above the avatar's brows.
- **The line at the join:** with ``edge_match`` 1 a dark band ran along
  the avatar's cheek on the ear side and a smear sat at the temples; with
  0 both were gone and a plain tone step remained. Her long hair lies
  down both cheeks; the mask's feathered edge lies over it, and the match
  took the hair for the room's light. Per correction cell (quarter scale)
  against the face's own median lightness: hair under the edge 0.30-0.35
  at p10, her brow cells 0.25-0.35 at p10 (0.62-0.69 at p50 -- a 4-px
  cell mixes hair and skin), the darkest skin of the core (the eye
  sockets, the shadow under the jaw) 0.49-0.64 at p5, on her three
  captures and his two alike.

**Built (`_face_colour_correction`):**

- **The brow zone is the whole brow**: both rows of the brow landmarks
  (`_BROWS_UPPER` added to `_EDGE_FEATURES`), then the 4-cell dilation
  as before.
- **The match is made on skin** (`_dark_cells`, `_EDGE_DARK_RATIO` 0.55):
  a cell whose scene is darker than 0.55 of the face's core median
  (hair under the edge, a brow's core, lashes, nostrils) is unknown to
  the lighting plane AND to the edge residual, and the residual there is
  filled from the skin around it, as the feature zone already was.

**Proven** on the same capture, replayed with the session's own anchor
(recordings/day16/baska-brows-before-after.png, baska-left-edge-…,
baska-right-edge-…, baska-face-20-07-before-after.png -- her face,
delete when looked at): the band above the brows gone, the dark band
along the ear-side cheek gone, the temple smear gone. What remains is a
light tone step at the temples where the avatar's matte skin meets her
shadowed, shiny skin at the hairline -- smaller, and the next lever is
the residual's reach (`_EDGE_REACH`) or her own avatar.

**Tests:** 1172 non-gui + 82 gui; ruff clean. New: the brow zone takes
both rows, dark cells are hair and a brow's core and not shadowed skin,
hair under the edge no longer darkens the cheek beside it nor tilts the
plane. Uncommitted, on his word.

### Day 16 evening, the fifth look (2026-09-12 22:43, the new avatar): the teeth that grow on a strong smile

**His report:** a new avatar added from a photograph (woman_person, the
importer's checks passed: face 450 px, pose -2.4/-0.4/+1.0); "when I
smile strongly the teeth grow in height" -- the day-14 complaint on the
old avatar, back on a new one; one capture (captures/moment-20260912-
224332), a medium smile then a strong one, the press on the strong one.

**Measured (the capture's trace, the replay with the session's own
anchor, ``--sweep mouth_opening_gain=1.0,0.7,0.5``):** his mouth over
the capture: width +16% and opening 0.07 on the strong smile, +11% /
0.05 at the press (his talk sits at 0.06). The avatar at the press
frame: width 1.17x his, **lips parted 1.36x his** on the frame he saw
and **1.62x** on the replay (mar 1.29x) -- the teeth a tall band where his
are a thin line. ``mouth_opening_gain`` 0.7 -> 1.34x, **0.5 -> 1.00x**
(width 1.16x, unchanged). But that gain weights talking the same way
(day 15: linear on smiles and talk alike) and would leave speech half
as open.

**Built:** ``mouth_smile_opening`` (a portrait option, default 1.0,
bit-exact): the opening gate is multiplied by ``1 - (1 - g) * widening``
where ``widening`` is the width gate alone (``_mouth_widening``, factored
out of ``_mouth_gate``) -- nothing while he talks with the mouth at its
resting width, the full reduction on a wide smile, the width itself
untouched. ``woman_person`` runs with 0.5 (measured on this frame);
``my_avatar_neural`` is left at 1.0 -- his day-15 verdict on that one
was "a little", and it is one line in its metadata if he wants it.

**Proven:** the replay of the same capture with the option at 1.0 and
0.5 (recordings/day16/smile-teeth-before-after.png, his own face): the
parting at 0.5 reads 1.0x his, the width the same at both.

**Tests:** 1175 non-gui + 82 gui; ruff clean; mypy the 11 warp errors.
New: the option's bounds, the width gate alone, a wide smile parting the
lips half as much while talking at the resting width is unchanged.
Uncommitted, on his word.

## Day 17 (2026-09-13): the whole forehead, and the avatar's hair

His word at the start of the day, after the brows-on-a-blink probe had
begun: "last time we wanted the forehead filled fully and then the AI
avatar's hair put on too; that was the task -- leave the brows and eyes
alone." So: block C1, both halves, nothing else.

### C1a, the whole forehead -- measured, then made the default

**Where the join sat**, read off six captures (his face and baska's,
`camera.png` against `seen.png` along the face's own vertical through
the brow top; the forehead height is brow top to the mesh's top point):

| capture | driver | forehead height | join above the brows | the hairline |
|---|---|---|---|---|
| 20260911-101434 | him | 47 px | 0.64 | (not found in 1.8) |
| 20260911-105732 | him | 60 px | 0.77 | 1.65 |
| 20260911-105750 | him | 59 px | 0.63 | 1.59 |
| 20260912-194432 | baska | 57 px | 0.42 | 1.71 |
| 20260912-200754 | baska | 84 px | 0.34 | 1.68 |
| 20260912-224332 | him, woman_person | 55 px | 0.67 | 1.59 |

At the 0.85 default the top third of his forehead and the top two
thirds of hers stayed the driver's own skin -- on her, with the shine
the avatar's matte skin does not have (the zooms:
`recordings/day17/forehead-*.png`, her face and his; delete after).

**Replayed** (`capture_replay --sweep mask_forehead_reach=0.85,hairline,1.6
--stills`, the session's own anchor) on 105750, 200754 and 224332
(`recordings/day17/forehead-row-*.png`): at "hairline" the whole
forehead is the avatar's on all three, the join just under the hair;
1.6 looks the same (the hairline reads 1.5-1.6 on every portrait, the
mode lands within 0.08 of it). The mouth, eyes and brows read the same
to the third decimal across the three values -- the reach touches
nothing else.

**Kept:** `mask_forehead_reach` defaults to `"hairline"` now
(`_FOREHEAD_REACH_DEFAULT`), his avatar's and woman_person's metadata say
so, and the importer writes `"hairline"` on every new avatar instead of
a number (the engine reads the portrait's hairline itself at load with
the same reader, and the driver's at the anchor). A plain `_Portrait`
measures; a number never settles (test_engine_hairline updated).

### C1b, the avatar's hair -- built as a switch

**What it is.** The network draws the whole 512 crop, hair included;
the face oval hid it. Now, with `hair = "avatar"` (or Menu -> Avatar's
hair), every rendered crop is segmented by MediaPipe's multiclass selfie
segmenter (`selfie_multiclass_256x256`, Apache-2.0, fetched by
`scripts/fetch_selfie_segmenter.py`, recorded in
`models/metadata/selfie_multiclass.json`; classes background / hair /
body skin / face skin / clothes / accessories) and the hair class's
confidence becomes an alpha the paste takes the maximum of with the face
mask (`sha256/avatar/hair.py`; the engine's `_update_hair`, `_paste`).

**Measured first (2026-09-13, this CPU, the two segmenters on the
portrait crops and two camera crops):** the multiclass model 65 ms a
frame (image mode; 170-180 with the replay job loading the CPU beside
it), the hair-only `hair_segmenter` 34-38 ms. Both far past the frame.
Quality: the multiclass matte is clean on his portrait's curls and on
the woman's shoulder-length hair, with a separate face-skin class; the
hair-only one smears 0.1-0.4 over his beard and sideburns. The
multiclass one is used, on its own thread (`HairMatteWorker`): the frame
path submits the latest render and takes whatever matte is newest -- one
or two frames old at 30 fps, which hair does not show -- and an EMA
(`_HAIR_INERTIA` 0.5) halves its flicker. The frame path pays the EMA,
a maximum and a box (`hair_ms`), not the segmenter.

**What shapes the alpha** (`hair_alpha`): the confidence curved between
0.25 and 0.75 (the beard's "maybe" pastes nothing), a 5-px feather, and
a fade to nothing over the last 24 px at each crop edge -- the woman's
hair reaches the crop's bottom row (511) and his curls its row 109, and
a hairstyle the crop cuts must end in a fade, not a line across the
driver's picture. The paste's box is the face box grown to the hair's
(`hair_box`, only ever grows, so the correction's field keeps one frame
of reference); the colour match is still fitted on the face mask alone
and its plane -- the room's light -- is what the hair takes too.

**With the hair on, the forehead reach** is the portrait's own hairline
plus 0.1 (`_HAIR_FOREHEAD_OVERLAP`), so the oval overlaps the matte and
no strip of the driver's forehead shows between the two; the driver's
hairline is not read (his hair is under the avatar's). Off again, the
option's own reach comes back and a "hairline" reach is measured again.

**The switch:** `LivePortraitEngine.set_hair(avatar_id, on)` -- live, no
reload (the matte lives outside the graph), remembered per avatar
across loads (`_hair_wanted`), the portrait crop's own matte as the
first matte so the first frame already has hair. Menu -> Avatar's hair
(after Mirror; a wish thrown before an avatar is active is applied when
one is; an engine without hair, or the model missing, puts the switch
back and says so). Explained in the Instructions and the guide. Off by
default in every metadata: it is his eye's call, and the ROADMAP said
from the start this is the one item that can look worse than what it
replaces.

**What it cannot do:** the driver's own hair past the avatar's edge
stays the driver's (her long dark hair below the woman's brown; a
bigger hairstyle than the portrait's); the portrait's hair past the
crop fades out. Both are what a matte of the render can do, no more.

**Replayed** (`capture_replay --sweep hair=off,avatar --stills`, then
`--option hair=avatar` on the build below; `recordings/day17/head-row-*`
and `head-row2-*.png`, his face and hers, delete after his look). On his
105750 the red curls sit over his head, larger than his own hair, and
his hair shows only at the temples under the avatar's hairline; on
224332 (him driving woman_person) her shoulder-length hair over his head
reads as a head, his own dark hair at the parting and the temples; on
baska's 200754 the curls sit ON her long dark hair, which continues down
both sides -- a wig, as the ROADMAP's own warning said, and her own
avatar is the answer there, not the switch. Mouth, eyes and brows read
the same with the hair on and off (within the tracker's noise on a
closed mouth). The frame path: `hair_ms` 0.1; the paste 9.5 -> 17.9 ms
on his (the box grown to the curls), 17.1 -> 23.4 on hers, 9.1 -> 20.9
on the woman's -- the segmenter itself never on the frame path.

**Two things the first replay showed, one kept, one not.** A pale rim
around the curls: the render's own pixels at the hair's edge are hair
mixed with the portrait's background (white on his, grey on the
woman's), and a soft alpha over the mix pastes the mix -- so the curved
alpha is eroded 3 px before the feather (`_ERODE_PX`), inside where the
render is hair; kept. His own hair between the avatar's hairline and the
face oval at the temples and the parting: the render's face-skin class
was let into the paste above the eyes (`skin_weight`, `head_confidence`)
to cover it -- and the avatar's paler temple skin came out as a patch
over his hair beside the hairline, plainly worse than the hair showing
(`head-row2-*.png`, the right tile). Off: the `hair_skin` option (default
False) keeps it as a lever to measure against.

**Tests:** tests/unit/test_engine_hair.py (the option, the alpha's curve
/ erosion / feather / border, the skin weight and the head confidence,
the boxes, the worker thread with a gated fake segmenter, the paste
under a matte reaching past the face box and byte-identical outside the
union, the engine's switch and the skin lever on the CPU fixture). The
whole suite passes (1280 + 2 skipped); ruff and mypy clean on the new
code. Uncommitted, his word.

### Day 17, his first look (2026-09-14 01:16-01:18, three captures): the band of his own hair, and a hairline the engine never read

He pressed "That looked wrong" three times with woman_person and the
hair on, head down at the 13-15 degrees he sits at (the camera above the
screen), and asked for three things: what is wrong in the pictures; "a
bit more" of the forehead covered; and that a person with a SMALLER
forehead than his gets the join lower -- "Calibrate will fix that,
right?"

**What the pictures showed** (`recordings/day17/look-forehead-*.png`,
`look-head-*.png`; his face, delete after): between the avatar's
forehead and the woman's hair, a band of HIS black hair. The face oval
is the portrait's oval AND his -- and his oval stops at HIS hairline
(reach x his forehead), which is lower than the woman's; above it
nothing was pasted until the hair matte began. And at the temples his
hair under the woman's, the limit stated the day before.

**What the log showed:** `forehead reach settled at 1.60 (portrait
hairline ?, his ?)` -- and for his own avatar `(portrait hairline 1.52,
his ?)`. His hairline was never read live. The reader walked 1.8
foreheads up and stopped: head down, his forehead foreshortens along the
face's vertical and the hairline reads 1.78-1.84 of it (the six
captures, offline, with the walk let go to 3.0; 1.59-1.68 on the frontal
ones); the woman's portrait reads 2.06 (her brow sits low). So on every
live frame the driver read "no hair" and the pair's reach was the
portrait's alone less the clearance: 1.44 on his avatar, the ceiling 1.6
on hers. The frontal gate (|pitch| 12) also refused most of his frames.

**Built:**

- The walk goes to 3.0 foreheads (`_HAIRLINE_SEARCH`), the frontal gate
  takes |pitch| 20 (`_HAIRLINE_FRONTAL`), the settle wants 15 frontal
  frames (`_HAIRLINE_MIN_SAMPLES`), and the option's bound is 3.0
  (`_FOREHEAD_REACH_MAX`; the "nothing read" reach stays 1.6,
  `_HAIRLINE_CEILING`). His hairline now reads on every capture.
- `hairline_clearance` (portrait option, default 0.04, was the constant
  0.08): the join sits that much under the lower hairline -- his "a bit
  more". On his avatar 1.44 -> 1.48 of his forehead, before his own
  hairline is even counted.
- **A smaller forehead.** Two ways, both automatic: live, the engine
  reads the driver's hairline in the first frontal second and takes the
  lower of the two (that was always the design; it had never worked --
  see above); and **Calibrate me now measures it** over the still step
  (`frame_readings["hairline"]`, one frontal frame in six once settled;
  the median over the still frames -> `driver_hairline` in the profile,
  a driver key) and the engine settles from the profile's word at the
  first frame of every session, no live frames needed. So the answer
  to "Calibrate will fix that?" is yes -- and even without a
  calibration, Start does it.
- **With the hair on, no band:** the avatar's own hairline is read off
  its hair matte (`_matte_hairline`: up from the brow top at three
  columns, the first row where the hair confidence passes 0.5), the
  portrait's oval reaches that plus 0.1, and HIS oval is let up to the
  bound (`_frame_mask(driver_forehead_reach=3.0)`) -- his hairline no
  longer bounds the join when his hair is under the avatar's.

**Proven by replay** (`recordings/day17/look2-*`, `clear-*`, `small-*`;
his face, delete after): on all three of his captures the band is gone
-- the woman's forehead runs up into her hair (`look2-forehead-*`,
`look2-head-*`, "seen" against "now"); his own hair still shows at the
temples beside her hair, the limit of a matte. On 105750 (his avatar,
hair off) the clearance 0.08 -> 0.04 moves the join up by a few pixels
(`clear-forehead-*`), and `driver_hairline=1.2` -- a person whose hair
starts a fifth of a forehead lower than the mesh's top -- puts the join
there, with the skin above it their own (`small-forehead-*`). Mouth,
eyes and brows read the same on every run. Tests: 1289 + 2 skipped
(test_engine_hairline 16, test_calibration_measure +3).

**What he will see live:** Start, look at the camera for a second: the
log says `forehead reach settled at ... (portrait hairline 1.52, his
1.6x from 15 live frames, clearance 0.04)`. Calibrate me writes
`driver_hairline`; from then on the reach settles from it at the first
frame. Menu -> Avatar's hair: the woman's forehead meets her hair; his
own hair at the temples is what it was.

### Day 17, his second ask (2026-09-14 01:48): "head down and looking to the side, the face must not get lost" -- the hold replaced by a cap

His four captures (014756-014839, my_avatar_neural) were pressed once
he was back at the camera: every press frame is frontal (yaw within 3,
pitch -8..-12), the turns and the tilt are in the tracking only (yaw to
+39 / -24, pitch to -27 in the three seconds before). So the movement
itself was read off his recorded clips through a new probe
(`scripts/probes/turn_probe.py`: a segment through the app's own tracker
and engine with the profile laid over, stills at the first crossing of
each angle and at the farthest turn / deepest tilt, `camera | output`
head crops with the pose and the hold state written on, and an mp4 of
the segment; `--sweep` as capture_replay's).

**What happens today (steep_pose = "hold", day 8):** past 20 degrees of
pitch or 35 of yaw from the anchor's rest the network is skipped and the
last rendered face is frozen -- it still follows him and is masked at his
position, but its expression stops. On natural.mkv's tilt step (417-712,
"tilt your head slightly down, three times") that is **165 of 296 frames
frozen**: at pitch -25 and -30 the avatar keeps the half-shut, downcast
eyes of the frame the hold began on, and at -33, where he looks UP at the
camera from under his brows, the avatar's eyes are still down
(`recordings/day17/tilt-natural-steep_posehold-sheet.png`, his face,
delete after). That is the "lost" he sees: not the tracking, the freeze.

**Built: `steep_pose = "clamp"`.** The network draws every frame; what is
capped is the turn applied to the portrait -- the relative pitch/yaw
against the anchor's own pose, clamped to `hold_pitch_degrees` /
`hold_yaw_degrees` inside `_drive` (a static `pose_limit` tensor, so the
graph carries it) -- and the expression goes on: mouth, eyes, brows. The
EMA state keeps the real reading. On the same tilt step, 0 frames held:
at -25 the avatar's eyes are half-shut and down like his, at -33 they are
open and on the camera like his
(`tilt-natural-steep_poseclamp-sheet.png`). The head is drawn at the cap
while his goes further; the paste's oval is his, so the face still sits
on his head. On the calibration clip's turn step (735-831, only to yaw
+37 / -19) hold and clamp look the same at +37 -- one held frame's worth
of difference.

**The deeper turns** (driving.mkv 1342-1641, to yaw +38 / -48;
`drive-*`, `drivecap-*`): under the hold the -48 frame is the face
frozen at -35 with the eyes where they were when the hold began; under
the clamp with the expression left on it is worse in another way -- a
wall-eyed gaze (the tracker's iris at a near profile) and a smeared
mouth corner (the extractor's expression with half the face hidden).
The cap itself is not the problem: at caps of 45 and 55 the network
draws his -48 turn plausibly. So **two more things:** the yaw cap is 45
(`_DEFAULT_HOLD_YAW`, was 35) and the pitch cap 30 (`_DEFAULT_HOLD_PITCH`,
was 20; his -30 and -33 tilts drawn cleanly at 30, `tiltcap-*`), and past
the yaw cap the expression **fades to rest over 15 degrees**
(`_steep_fade`, `_STEEP_FADE_DEGREES`: the mouth and brow gates and the
brow lift scaled; the blink stays, the near eye reads; the drive's own
states go on), and **the gaze fades earlier, from 30 to 45 degrees**
(`_gaze_yaw_fade`, `_GAZE_YAW_FADE`): the iris is read against the eye
corners, and past 30 degrees the far eye's corners crowd together and the
reading over-shoots -- with the gaze faded only from the cap, the -48
frame was still wall-eyed (`final-driving-sheet.png`). At a full profile
the face is the head drawn at 45 with a calm face on his outline; when
the tracker loses him the black screen takes over as before.

**Kept as the default:** `steep_pose = "clamp"` (`_DEFAULT_STEEP_POSE`);
"hold" stays as the option and as what its own tests set (their
thresholds pinned at the day-8 20/35). Tests: tests/unit/test_engine_steep.py
(the option, the fade, never held, the cap in `_drive` against the
warping net's keypoints -- cos 58.5 becomes cos 35, the reading kept,
inside the cap bit-identical, relative to the anchor's yaw); the whole
suite 1296 + 2 skipped.

### Day 17, his third ask (2026-09-14 02:41): the blink -- "the eye shuts beautifully, only the lip must not move and the brow must not come down a little"

Three captures again pressed once back at rest (all three frontal,
eyes open at the press), so the blink was read off his calibration
clip's blink and eyes-shut steps (frames 153-269 of
calibration-20260912-004157) through a new probe
(`scripts/probes/blink_probe.py`: the segment through tracker, engine and
profile; on every frame his readings and the avatar's off the composite
-- the eye ratio, the upper brow points' height above the nose tip, the
mouth corners' height, the width, the opening -- and the engine's gates;
the shut frames (his ratio under 0.7 of rest) against the open (over
0.9), per `--sweep` value; a still of the deepest shut frame).

**Measured (18 shut frames, 94 open):**

| run | brows (avatar / his) | mouth corners (avatar / his) | gates on the shut frames |
|---|---|---|---|
| eye network on (eye_gain 1.4) | **-4.2 px** / +0.0 | **+1.4 px** / +0.4 | mouth 0.00, opening 0.00, brow 0.00 |
| eye network off (eye_gain 0) | -0.4 / +0.0 | +0.2 / +0.4 | all 0.00 |
| expression off (expression_gain 0) | -2.7 / +0.0 | +1.7 / +0.4 | all 0.00 |

The whole of it is the eye retargeting network: its delta on
EYE_KEYPOINTS, confined there since day 8, still pulls the warp's
neighbourhood when the lids shut -- the brows down, the mouth corners
up. The extractor's expression plays no part (off, nothing changes) and
every gate reads 0.00 on the shut frames, so neither the brow drive nor
the mouth gates are the mechanism (what the saved calibration samples had
already said on day 16's list: `gate_brow` 0 on every shut frame).

**Built: `blink_brow_hold`** (and `blink_mouth_hold`, a lever left at 0):
in the rows the brow drive already writes each frame (`brow_delta`),
keypoint 2's y is lifted by the option's units per unit of blink depth
(`_blink_depth`: his eye ratio's shortfall from the anchor's rest, 0
open, ~0.6 shut) -- the same channel a raise uses, the opposite way, so
the network's own pull is met where it starts. Nothing else touched: the
lids close exactly as before (the eye network's own delta is not
scaled), the gates, the drive and the anchor are as they were, and with
the option at 0 every frame is bit-identical. Swept on the same frames:
0.01 -> brows -1.1 px, corners +0.3 (his +0.4); 0.02 -> +1.1 / -0.6;
0.03 -> +3.8 / -1.2 -- keypoint 2 moves the corners a little with the
brows (day 9's nudge table said so), so the one lever settles both, and
the mouth lever is not needed. The smile axis itself (`_MOUTH_SMILE`,
keypoint 14's y) was swept too for the record: -0.02 -> corners -3.1 px,
+0.02 -> +7.8 px.

**Verified at the live configuration** (the profile laid over the
metadata as the app does it -- the first sweeps had let the avatar's own
driver keys over the profile, fixed in the probe; the eyes then shut
deeper, avatar ratio 0.125 against 0.20, and the pull is larger):

| `blink_brow_hold` | his avatar: brows / corners | woman_person: brows / corners |
|---|---|---|
| 0 | -4.0 px / +1.4 | -3.1 / +1.9 |
| 0.012 | -1.7 / +0.6 | |
| 0.013 | | -0.2 / +1.0 |
| 0.015 | -1.2 / +0.3 | |
| 0.018 | -0.5 / +0.3 | |
| 0.02 | **+0.3 / +0.1** | |

(his own on the same frames: brows +0.0, corners +0.4.) 0.02 was kept
first -- and **his second look said "now the brows go UP when I blink,
and the lips still smile a little"**. The medians over the shut frames
had hidden the blink's own fast phases; measured by depth (his ratio
over rest, bins from open to shut; the probe's `by depth` line):

| depth bin | pull, hold 0: brows / corners | linear 0.02: brows / corners |
|---|---|---|
| 0.7-0.9 (a fifth shut) | -1.7 / +0.4 | -0.9 / -0.5 |
| 0.5-0.7 | -3.1 / +1.0 | -0.6 / -0.4 |
| 0.3-0.5 | -4.1 / +1.4 | +0.1 / -0.2 |
| under 0.3 (shut) | -4.8 / +2.0 | +0.7 / +0.8 |

The pull is steep at first and flattens; a linear counter set for the
shut frames under-counters the fast phases and over-counters the shut
ones -- the brows up at the bottom of every blink, exactly his words --
and keypoint 2's pull on the corners saturates early, so the shut end
keeps its smile. **So the counters have a shape** (`_blink_brow_curve`
d(2 - d): 0.36 / 0.64 / 0.84 / 0.98 at those depths; `_blink_mouth_curve`
0 to half shut, whole at 0.85) and the mouth lever is used, -0.002 (a
negative value lowers the corners):

| depth bin | brow 0.015 shaped, mouth 0 | brow 0.015, mouth -0.002 |
|---|---|---|
| 0.7-0.9 | -0.6 / -0.8 | -0.7 / -0.8 |
| 0.5-0.7 | -0.1 / -0.6 | -0.0 / -0.5 |
| 0.3-0.5 | +0.4 / -0.1 | +0.4 / -0.5 |
| under 0.3 | +0.1 / +0.9 | +0.1 / +0.2 |

Everything within a pixel at every depth, the shut frames within 0.2.
**Kept:** his avatar's metadata `blink_brow_hold` 0.015 and
`blink_mouth_hold` -0.002, the code defaults the same. The lids shut as
before (the shut ratio 0.125 -> 0.09 with the lift, deeper if anything).

**woman_person, the same frames (him driving it):** its pull is weaker
(brows -3.1 px at the median, corners +1.9), and his 0.015 over-shot
it (+0.8 / +1.4 / +1.9 / +1.7 by depth). At `blink_brow_hold` 0.01 and
`blink_mouth_hold` -0.004: brows +0.1 / -0.2 / -0.0 / +0.3, corners
-0.5 / -0.0 / -0.1 / +0.4 -- its metadata says so. The number is the
portrait's (how the network's lid delta reaches that face), so a new
avatar starts from his and wants the probe run once
(`blink_probe.py --avatar <id>`); Calibrate me does not measure it yet.
**His third look (03:16, capture 031607, pressed a third of the way
into a blink -- his ratio 0.69 of rest):** "look at shut and open eyes,
tell me the error, then fix". On that frame the counters lift the
brows 1.3 px and lower the corners 1.4 px against the same frame with
them off (`capture_replay --sweep blink_brow_hold=0,0.015`; what he saw
matches the counters-on replay within 0.5 px). The error against the
open face, from the session by depth **with his own movement beside**
(the probe prints it now): brows -0.7 / -0.0 / +0.3 / +0.1 px (his
0.0 at every depth); corners -0.7 / -0.5 / -0.4 / +0.3 -- **his own
corners rise +0.3 at every depth** (his mouth eases as the lids shut),
so the avatar's sat 0.7-1.0 px under his through the fast phases: the
brow lever's own pull on the corners, largest where the lids only start
to shut. **`blink_mouth_lift`** (the third and last lever): the smile
axis lifted by a bump that is 0 open, largest around a third shut and
gone at 0.85 (`_blink_lift_curve`). Swept: 0.01 -> corners +0.2 / +0.7
/ +0.3 / +0.3 against his +0.3, brows -0.3 / +0.3 / +0.3 / 0.0; 0.015
over (+0.9 / +1.1 / +0.9 / +0.4). **Kept 0.01** in his metadata and
as the default: brows within 0.3 px and corners within 0.4 px of his
own at every depth of a blink. woman_person's lift measured on its own:
0.007 (corners +0.1 / +0.8 / +0.4 / +0.4 against his +0.3, brows
within 0.3; 0.01 over) -- its metadata says 0.01 / -0.004 / 0.007.
Stills: recordings/day17/blink-before-shut.png,
blink-after-shut.png, blinkw-after-shut.png (his face; delete after).
Tests tests/unit/test_engine_blink.py; the whole suite passes.
Uncommitted, his word.

**His side-look captures (024140, 024155):** pressed once back at the
camera again (yaw -2 / -3 at the press), so what he saw there is
unseen; the tracking says 024140 turned to +29 only and 024155 to +46 /
-50 with 79 frames past the 45 cap. He says "not much better"; what
that is needs a press IN the pose, or his words.

### Day 17, the version and the package (2026-09-14 03:45-04:10)

His word: "this detail of the project is good now, this version" --
committed and pushed (19a3b75 / 8c402fb / 77ce772). Then three asks
answered and two of them built:

- **"How do we upload the application?"** `scripts/build_exe.py --avatar
  my_avatar_neural --avatar woman_person --zip` rebuilt the portable
  folder with today's code and the segmenter model beside the landmarker:
  `dist/sha256-portable/` and `dist/sha256-portable.zip`, **3.8 GB
  zipped** (5.3 GB unpacked; the CUDA runtime is most of it). The zip
  goes on his download site, a drive or a USB stick; the user unpacks
  it, runs `sha256.exe` past SmartScreen's "More info -> Run anyway"
  (unsigned by choice, see the delivery context) and installs OBS Studio
  once. Two of his decisions stay open: the switch URL (`--switch-url`)
  and which avatars ship (his two, today).
- **"We must be able to turn it off at any moment, for them too."** The
  courtesy switch is what there is (D1, the 2026-08-15 decision: no
  licence server). It stops every copy that carries the same URL, within
  the check interval. Made quicker: `KillSwitch.request_check` wakes the
  thread and **every Start asks it**, and the shipped `local.toml` sets
  `check_hours` to five minutes beside the URL. What it cannot do, said
  to him plainly: a user who edits `configs/local.toml` (the URL is
  there) or blocks the host is allowed -- it fails open, so a call
  without internet still works. If he wants more: the URL baked into the
  executable (a config edit no longer removes it), per-copy ids in the
  file (one person switched off, the rest not), or a fail-closed grace
  (a copy that has not read the file for N days stops) -- each a small
  piece, each his decision, none built.
- **"Will a database be needed for adding avatars? We must be able to
  delete them too."** No database: an avatar is a folder under
  `avatars/` (the portrait, `metadata.json`, the profile), the library is
  the folder listing, Add avatar writes one. **Delete avatar built:**
  `remove_avatar` in the importer (the gated counterpart of the import;
  the repository stays read-only -- only a directory under the library
  whose manifest names that id is removed), Menu -> Delete avatar behind
  the owner's login, the avatar chosen in the bar, one question, the
  pipeline stopped first; the manager lets it go and the bar is refilled.
  Tests: tests/unit/test_avatar_remove.py, the window's delete flow, the
  switch's request; the instructions and the guide name it.

**Google Meet:** the OBS Virtual Camera device is registered on this
machine (B004 re-run: 1.4 ms a frame to send); the steps are in the
Instructions (gear -> Video -> Camera -> OBS Virtual Camera). His test.

## Day 19 (2026-09-17): "the face is lost when the head turns" -- the plan, and its first two steps built

The client's one complaint on day 18's version, and his order of 2026-09-17:
a plan that certainly holds a turn of 30, better 40 degrees, **and nothing
of what the face, the eyes, the brows and the mouth do today may break.**
Day 18's turn views (a second photograph per direction) were reverted in
full on his word before this; what they measured is kept in the notes
and is not repeated here.

**The plan, in the order the steps are worth:**

0. *Measure* -- his 39-second turn clip (`--script turn`, restored to the
   recorder: slow full turns each way, held profiles, quick glances, a
   half turn held), read by `turn_probe` (which now prints where the
   tracker let go and where it found the face again) and by the new
   `redetect_probe`.
1. *A loss while turned never blacks the call* -- the conditioner holds
   the last frame that went out.
2. *The tracker's reach* -- a second look in a crop where the video
   landmarker finds nothing.
3. *The picture at 30-45* -- the stills from his clip decide; the front
   portrait's own render (the cap is 45 since day 17) first, the turned
   photographs only if the front smears at 40, and then by day 18's rules
   (a switch only past 30 with a minimum time between switches, each
   view's offset and scale measured and compensated, the mask following
   the drawn face).
4. His live look.

Steps 1 and 2 are built and measured; step 0 waits on his clip; step 3
waits on what the clip shows. Everything built lives off the drive path:
the recorder, the probes, the output conditioner, the tracker on frames
where **no** face was found, the status line and the instructions. On
every frame the tracker finds a face, its output and the engine's are the
same bytes as at b5397e7 (the tracker's second landmarker is never
consulted there; the conditioner passes a face frame through as before).

### Step 1: a face lost while turned away holds the last frame (built, tested)

What happened on his live look of 2026-09-15: the tracker lost his face
for 143 of 300 frames and the black screen ("Camera battery died") took
the call for 22 seconds. The loss was not a tracker fault on a face in
front of the camera; the head was past what a webcam tracker reads. On a
call that is a look at someone in the room.

`OutputConditioner.condition(turned_away=)`: the controller keeps the
head's turn (the larger of |yaw| and |pitch|) off the last frame with a
face; a loss that begins past `TURNED_AWAY_DEGREES` (25) is a turned-away
loss, decided on the loss's first frame and kept for its length. For such
a loss the conditioner returns **the last frame that went out with a
face** -- a copy, in one refilled buffer, because the engine's frame is
the camera's buffer with the face pasted on and the capture backend may
reuse it -- for `lost_face_seconds_turned` (**8 s**, schema, default.toml,
bootstrap) before the black screen (reason "face lost, turned away"),
and the avatar is back the moment the face is (no minimum black, since no
black happened). A frontal loss is unchanged: the engine's own frame for
`lost_face_seconds` 1.5, then black. Day 18's patience showed the engine's
frame (the last face pasted on the LIVE picture) for those 8 seconds; a
head that keeps turning would have put his own ear or cheek past the
mask's edge, so the held frame is the safe form. Status line "Face not
found -- turned away, the last frame is held for the other side";
`OutputConditioner.holding`; the instructions and the user guide say it.
Tests: `TestTurnedAway` in test_conditioning.py (10) and in
test_pipeline_controller.py (4), the config default.

### Step 2: the second look in a crop (built, measured, tested)

MediaPipe's video landmarker tracks a face it has from frame to frame;
once it has let go it must *detect* again, and its detector -- a
short-range one that sees the whole frame scaled to 128 pixels -- finds a
face turned far round only when the face is large and near the front. So
a head that went to 60 degrees is found again only near the front, which
is the "coming round" of his 2026-09-15 log.

`MediaPipeFaceTracker(redetect_in_crop=True)` (settings
`tracking.redetect_in_crop`, default on; off in the constructor so every
still reader and every existing test is untouched): a second, IMAGE-mode
landmarker built at start (detection confidence `redetect_confidence`
0.3). On a frame the video landmarker finds nothing, the crop --
`_REDETECT_SCALE` 1.8 times the last face's *height* (a turn narrows a
face, it does not shorten it), the frame's own aspect ratio so the task's
camera model still holds, shifted inside the frame, never shrunk; 2.6
after 30 lost frames -- is read by the second landmarker, and its result
is mapped back into the frame: the points through the window
(`_crop_to_frame_points`), the pose by `_pose_from_crop`. The crop
follows the face it finds. `extras["redetected"]` marks such frames.
Cost: one landmarker read per lost frame, the same as a tracked frame;
frames with a face cost nothing more (measured 12.7 ms/frame either way
on driving.mkv).

**The pose from a crop, measured before it was trusted**
(`scripts/probes/redetect_probe.py`, his driving.mkv):

- The task's fit is only weakly perspective. The same frame read again
  moved 160 and 320 px sideways (a ray angle of 15 and 29 degrees at the
  task's 63-degree camera) moves the fitted yaw by **0.23 of the ray
  angle** at the median on the full-size face (0.20 frontal, up to 0.69
  at 50 degrees of yaw), 0.47 on the half-size face; pitch the same. So
  `_RAY_SHARE` 0.25 of the ray angle is undone, not all of it: a face
  within 160 px of the frame's centre is read within 3 degrees either
  way, one at the frame's edge within 13.
- Depth scales nearly with the picture the face is measured against:
  `(height / h) ** 0.92` (`_DEPTH_EXPONENT`; a face read at 2.09 times
  its size came out 2.01 times nearer, at 1.03 times 1.03).
- The crop's landmarks land on the whole frame's within 0.5-2.4 px at the
  median (up to 15 px on the far side of a 53-degree turn).

**The reach, measured** (a fresh read of every second frame of the turn
segments, no tracking state -- what the video landmarker has once it has
let go):

| his face in the frame | whole frame finds it | the crop finds it |
|---|---|---|
| 54% of the height (driving.mkv as recorded) | 100% to -55 | (the crop would be the whole frame) |
| 27% (the frame shrunk to half: a call distance) | 100% to -50, 70% at -55..-50 | 100% at every angle, -55..-50 included |
| 19% (shrunk to a third) | nothing, at any angle, frontal included | nothing (no first face) |

The video tracker itself never lost his face on these segments (to -55)
at full or half size. The third row is a limit of the detector, not of
the turn: a face under a fifth of the frame's height is not found at all,
turned or not. **What his 2026-09-15 loss was** -- a turn past 60, a
small or dark face, a look down -- the clip will say; the tools are ready
for it.

Tests: `TestSecondLook` (11), `TestRedetectWindow`, `TestCropToFrame` in
test_tracking_mediapipe.py; the fake landmarks have a `spread` now (a
face with a size). Whole suite 1345 + 2 skipped; ruff clean; mypy the 11
pre-existing errors in engine_warp only.

### Step 0, his turn clip (2026-09-17 23:57): where the tracker lets go, and what the picture does

`recordings/driving-turn.mkv`, 1159 frames, 38.6 s, 720p, his face 54% of
the frame's height (a close seat). Both probes over every segment
(`redetect_probe` per segment with the second look off and on;
`turn_probe --yaw-marks 20,30,35,40,45,50,60 --video`, sheets and clips
in `recordings/day19/turn-{119-472,473-769,770-1068}*` -- his face,
delete after his look).

**The tracker (the app's settings, the second look off):**

| segment | frames lost | where it let go | where it found the face again |
|---|---|---|---|
| slow full turns each way (119-472) | **0** | never (his turn read +45 / -50) | -- |
| held LEFT profile (473-622) | 64 | after f554/557 at yaw **+57..+65** | f565 at +26 for one bad frame, then lost 57 frames to the segment's end (the profile held) |
| held RIGHT profile (623-769) | 95 | after f698 at yaw **-58..-60** | f648 at +55 (the left profile of the segment before, 0.8 s); f757 at **-70..-72** after 1.9 s |
| quick glances (770-889) | 30, all before the first glance | (still in the profile of the segment before) | f801 at **-63..-65**; **not one glance lost** |
| half turns held (890-1068) | **0** | never (+45 max) | -- |

So on this camera and at this size the face is held to **about 60
degrees** and let go only at a real profile; found again at 53-72, not
"near the front" -- the fresh detector finds his face at 100% in every
bin to +-55 (whole frame and crop alike). The second look found him on
4-5 frames at -77 / +65 during the profiles and nothing else: at a full
profile (70-78) neither the whole frame nor the crop reads a face ("lost
by VIDEO": whole frame 0%, crop 3-9% of 34-35 frames), which is the face
model's limit, not the detector's. It costs nothing on a tracked frame
(5.8-6.3 ms either way) and stays on for the cameras that need it (the
half-size measurement above). The crop's pose held on his frames: yaw
within 0.2-1.7 degrees of the whole frame's read, depth within 0.3%, the
ray share 0.23-0.35 again.

**What his 2026-09-15 loss was, then:** a head held past 60 -- a real
profile, several seconds -- and under the old rule (1.5 s, then black)
every profile held longer than a moment blacked the call; the held
profiles here lasted 1.9-2.7 s and would have. Under step 1 the other
side sees the last frame instead, and the avatar is back at 53-72
degrees, on the way round.

**The picture (the stills):** at +21..+48 and -20..-51 the avatar follows
his turn and the face is clean -- both eyes drawn, nothing smeared, the
quick glance of 543-549 (+23 to +42 in four frames) included. Past the
cap (his +52..+57, -70) the head is drawn at 45: a three-quarter face
on a profile head, coherent, with the mask's edge showing on the cheek
where his stubble goes on past it. At a full profile that is what a
frontal photograph can give.

**Step 3, decided by the stills: the cap alone.** The front portrait
draws 40 cleanly; the yaw cap stays 45 (day 17); the turned photographs
are not needed for 30-40 and are not built. What is left for a fuller
profile is a second photograph per side, and its price is known from day
18; not now.

Not touched, on purpose: the paste's edge (the white band of day 18 was
measured at 117 px worst on the clip with the views off -- invisible --
and only grew with the views' crossfade), the drive, the render.

### His two screenshots (2026-09-18 00:13): the strip beside the eye on a held turn, the white streak on a fast shake

His live look came back with two pictures: a head turned and held, a
pale strip ringed on the cheek **beside the eye, on the nose's side**; and
`fast.png`, a quick shake, a white streak down the same side from the
temple to the chin -- "the machine cannot keep up and follow my head".

**What it was, measured on his turn clip** (`scripts/probes/band_probe.py`:
the paste hooked, the render, the mask and the paste as it goes out cut
out at the mask's box, the near and the far side zoomed; the zooms of
his face in `recordings/day19/near-before-after-*.png`, delete after):

- The far cheek was covered -- the join there sits on skin against his
  hair and stubble, as on day 16.
- **The near side was cut straight through.** The oval is the convex hull
  of the face's RIM, and at a turn the rim's near points project inside
  the face: the hull runs straight from the temple to the chin and the
  nose, the lips and the cheek in front of the eye fall OUTSIDE the paste
  -- his own skin, lit from the window, shows there, and the avatar's nose
  is cut off (f452, yaw -47: the hull's edge 10-25 crop px inside the
  drawn face). That is the strip he ringed. The fast shake is the same
  cut with motion blur on top (f804, 7 degrees in one frame): the mask
  where the landmarks are, the near side open.
- The white in `fast.png` is what the paste shows once the mask does reach
  past the drawn face: the photograph's own flat background (his avatar's
  is (243, 243, 237) BGR), day 18's finding.

**Built, off the drive path, provably nothing on a frontal frame:**

1. **The midline in the oval** (`_MIDLINE`: brow ridge, nose bridge and
   tip, the lips, the chin -- 26 mesh points -- go into the hull with the
   rim; every statistic of the oval, the rim's top and the mean the jaw
   shrink pulls toward, stays the rim's own). On a frontal face each of
   these is inside the rim, so the hull is the same polygon: proven on
   **his own recorded landmarks** (`TestMidlineOnHisFace`, the calibration
   session of 2026-09-12, 894 frames, gitignored so it skips elsewhere) --
   every frame under 12 degrees of yaw gives the identical mask to the
   byte, and no frame's mask ever shrinks; on his face the nose leaves the
   rim past about 30 degrees. At a turn the near side follows the profile
   and the avatar's nose is pasted (f452 / f459 / f804 / f1042 after).
2. **The background guard** (`background_guard`, default on):
   `_background_colour` reads the photograph's flat background from its
   border at load (the top ring and the upper halves of the sides, 3% of
   the picture; none when the border is not one colour -- a room -- and
   then nothing changes); `_background_keep` cuts the mask, at every
   paste, where the render is within 30 of that colour AND joined to the
   box's border (a highlight on the nose is that colour and is not joined
   to anything), reaching 2 px into the drawn face's own edge (the render's
   face-to-background pixels are a mix that reads pale on his cheek) with
   a 5-px feather. **Keyed to the turn** (`_background_weight`: nothing
   under 12 degrees of yaw from the rest, all of it from 20): on a frontal
   frame the mask meets the render's own edge at the jaw for a pixel or
   two and an ungated guard trimmed those (f30: 810 px changed over 8
   levels) -- gated, every frontal frame of the clip differs between the
   guard on and off by the GPU's own run-to-run noise alone (at most 2
   grey levels, none over 8; f30, 60, 90, 110, 338, 499, 900, 1150),
   while f1100 at yaw +36 is what the guard is for (3021 px).

After: no near-white pixel is pasted inside the mask on any of the turn
or fast frames (f298, 452, 459, 802, 804, 805, 1042; f1042 had 8
before), and the raw hull now reaches 250-1500 px of background at a
turn that the guard removes.

Tests: `TestMidlineInTheOval`, `TestMidlineOnHisFace`, `TestBackgroundGuard`,
`TestBackgroundWeight` in test_engine_liveportrait.py (the synthetic face's
midline is pinned on the midline now, as on a face).

**Next: his live look** -- the held turn and the fast shake again, and a
held profile on a call with the black screen no longer taking it; then
commit on his word.

### His look at that build (2026-09-18 ~01:00), and the plan for day 20

His words: turning the head "everything sits well", but the eye on the
side he turns to "still shows as mine", the nose needs more when he turns
harder; and a quick turn left-right "zooms the face out" -- the avatar's
face shrinks on his and the photograph's white background shows as a line
-- "the AI face cannot follow at the pace needed, it reacts slowly".

**Measured before planning** (his turn clip; the scratch probe of the
near eye and the fast glance):

- *The near eye at a held turn is covered.* At +38..+46 and -47..-49 the
  paste mask reads 1.00 at every point of the near eye, the portrait's
  oval and his own alike; the eye in the paste is the avatar's (its iris,
  not his) -- but blurred, drawn at the 45 cap while his head is past it,
  and looking at the camera (the gaze goes to rest from 30 to 45, day 17)
  while his looks into the turn. Past 50 the mask's edge -- the hull eroded
  by 4% of the face width (13 px) and feathered -- sits 10-15 px inside
  the profile, and there a strip of his own face along the profile shows:
  the outer corner of the near eye, the nose's tip. "More work for the
  nose when I turn harder" is that strip.
- *The zoom-out after a fast turn is the DEPTH reading.* The crop's size
  is his resting face height scaled by the pose's translation z (day 6),
  filtered at 1 Hz. On the fast glance his z reads 31.0 frontal and
  **32.3 at -65** (the face model fitted to half a face reads it farther):
  the crop shrinks 794 vs 821 px, the avatar with it, **3-4% smaller**,
  and the 1-Hz filter holds that for the frames after he is back --
  frontal again, the guard off (it is keyed to the turn), his outline
  wider than the drawn face by 10-14 px, the white line along the far
  side. On the slow turn the same drift is there (+58: 30.6) and small.
- *The reaction.* Live the pipeline ran 19.8-23.9 fps on his captures
  against the camera's 30 (the tracker 6-13 ms and the paste 7 ms on the
  CPU, the network 26 ms on the GPU, in series); the tracker's landmark
  filter is a 0.35 EMA (half a frame late by design, day 3); the drawn
  head sits 15-25 px behind his at a turn (day 4). Together a fast turn's
  avatar is a frame and a half behind his face.

**The plan, in order of what it buys against what it risks:**

1. **The depth held through a turn** (the zoom-out's root). Past 35
   degrees of yaw from the rest the crop's size stops following z and
   keeps the last frontal value (a turn in place does not change the
   distance), released on the way back; measured on the clip as the
   avatar's face width against his per frame through the fast glances
   (a second tracker on the output, as measure_follow reads it) -- the
   3-4% dip must go, and no frame under 35 degrees may change (the crop
   is bit-identical there by construction). Off the drive path.
2. **The guard's grace.** The guard stays on for half a second after the
   yaw came back under 12 degrees, so whatever the size filter still
   holds is cut rather than pasted white; steady frontal frames unchanged
   by construction. Small; belt to (1)'s braces.
3. **The profile without the erosion.** A midline point that lies outside
   the rim's own hull (only at a turn) is pushed outward by the erosion's
   radius before the hull, so after the erosion the mask's edge lands ON
   the profile, not 13 px inside it; the frontal hull is untouched
   (nothing is outside the rim there). Measured as the strip's width at
   the profile on f452/f459 and at his hardest turns. The near eye's
   outer corner and the nose tip come under the paste. Off the drive path.
4. **The eyes at the cap** -- his call, on the drive path: past the cap
   the avatar's gaze could lead into the turn by the degrees the head
   is short (a person whose head stops turning looks on with the eyes),
   instead of the rest gaze at the camera; stills first, at 50 and 60.
5. **The reaction speed**, measured before touched: (a) where the frame's
   time goes today (capture, tracker, network, hair, mask, paste, sink --
   `last_timings` and the tracker's own time, on a live run), then the
   tracker on its own thread one frame ahead of the network (the
   pipelining the architecture left for later), which alone is the
   difference between 20 and 30 fps on this machine; (b) the landmark
   filter: a One Euro on the landmarks in place of the 0.35 EMA follows a
   fast turn without the half-frame lag and still steadies a resting
   face -- on the tracked-frame path, so replayed first (rest jitter, fast
   lag) and only on his word; (c) the drawn head's 15-25 px lag at a turn,
   measured per yaw with the output tracker and undone in the paste as a
   shift of the render (the day-18 idea for the views, applied to the
   front portrait) -- paste-side, off the drive.

What is not in the plan: a second photograph per side (the views); the
network's own speed (26 ms is the graph, TensorRT would be a week for
maybe 8 ms).

## Day 20 (2026-09-18): the depth held through a turn -- the plan's step 1, built and measured

His order: the day-20 plan from its first step, and nothing of the face,
the eyes, the brows and the mouth may break. Step 1 is the zoom-out's
root: the crop's size follows the pose's translation z every frame, and z
is a rigid fit of a whole face model that wanders once half the face is
gone.

### What z does at a turn, measured first (the tracker as the app runs it, his turn clip)

z over the clip's still start (rest yaw -1.6, z 31.9); 1.0 is his seat:

| yaw from rest | frames | z, median (min..max) | so the crop is |
|---|---|---|---|
| 0-10 | 561 | 0.995 (0.97..1.08) | his own leans, up to 8% |
| 10-20 | 56 | 0.989 | trusted |
| 20-30 | 63 | 0.976 (0.94..1.15) | 2.4% too large |
| 30-35 | 53 | 0.959 | 4% too large |
| 35-45 | 173 | 0.976-0.980 (left turns 1.01) | 2% too large; 1% too small to the left |
| 45-55 | 54 | 0.96-0.98 | |
| 70-90 (the profiles) | 6 | 0.935 (0.89..0.95) | 7% too large |

Not monotonic and not one-sided: nearer at 20-35, farther at 36-47 on a
left turn, nearer again at a profile. **The fast glance** (f770-889): the
face is found again at f801, yaw -63, with z 1.012 -- 4% farther than
the frames right after (0.971-0.984 on the way in, 0.994 settled) -- and
the crop's 1-Hz size filter, reset by the loss, starts from that frame:
the crop 1.8% under its settled size at f801, the avatar with it, then
1.0-1.5% over for f803-f809 as the return sweep reads "nearer". That is
what he saw. His real leans move z by at most 1.6% in one frame (p99
0.8%); the slow turns drift the same way and he did not mind them.

### Built: `depth_hold_yaw` (driver option, default 35; 0 off), off the drive path

`_held_depth` inside `_driver_face_height` -- the one place the crop's
size reads z. At or under 35 degrees of yaw from the rest the depth is
the frame's own, as before; past it the crop is sized from **the depth he
sits at**: the median of the last five z readings on frames within
`_DEPTH_TRUST_YAW` (20) degrees (`z_recent`, `z_frontal` on the portrait;
reset with the anchor, kept across a loss, so a face found again at 60
degrees after a profile is drawn at the size it left with). Released the
moment the head is back under 35 -- the live z again, so the frames at
20-35 are what they were. The first version held the *last frame's* z at
or under 35, and the clip caught it: the second look found a face for
one frame (f565, yaw +25, pitch +29, z 16% off) and the next profile was
drawn **15.5% too small for six frames**; hence the median of five within
20, where the fit is trusted -- one bad frame cannot reach the hold.
Tests: `TestDepthHold` (10, the poisoning included); suite 1367 + 2
skipped, ruff and mypy clean.

`scripts/probes/depth_probe.py`: the clip through the tracker and the
engine as the app runs them; per frame z, the depth used, the crop's raw
and filtered size, and the avatar's face height against his by a second
unfiltered tracker on the output and a third on the camera frame;
`--sweep depth_hold_yaw=0,35` prints every window (the first frame found
after a loss, each return under 35) against where the size settles, and
the proof below. Sheets `recordings/day20/depth-driving-turn-*.csv`
(numbers only).

### Measured, the hold off against on (the crop against its settled size)

| window | hold off | hold on |
|---|---|---|
| f801, the fast glance's first frame (-63, after 26 lost) | -1.8%, then +1.0..+1.5% to f809; 1 frame more than 1% small | +0.5% flat to the release at f807; 0 |
| f757, the left profile found (-70, after 44 lost) | +4.1..+10.8% | 0.0% |
| f769 (-76, after 5 lost) | +4.9..+5.3% | +0.5% |
| f648, the right profile found (+55, after 82 lost) | +5.2..+8.0% | -0.0% (the first version: -15.5%) |
| f654, the return from it (+34 down) | -1.4%, 7 frames small | -0.6%, 0 |
| f311 / f472 / f1102, the slow turns' returns (all under 35) | +3.4 / +1.4 / +3.2% | +3.0 / +1.1 / +3.2% |
| every frame past 35 (239) | -1.7..+11.8%, median +1.5 | -0.6..+3.3%, median +0.6 |

The avatar's height against his, read off the output through the glance
(f803-f805, his head past the cap): -2.4..-2.2% off, -2.0..-2.3% on --
that is the cap (his head at 52, the avatar drawn at 45), not the size.

**The proof of the untouched face:** the raw crop size is the same bytes
on every frame at or under 35 degrees (0 of 972 differ); the filtered
size differs by 0.0001 px past 45 frames after a held one and by nothing
past 51 (the 1-Hz filter's tail); the outputs of the eight frontal check
frames (f30, 60, 90, 110, 338, 499, 900, 1150) differ by 0-3 grey levels,
none over 8 -- the GPU's own run-to-run noise, day 19's rule. The drive
and the render are not in the change.

**Left as it is, on purpose:** the 20-35 wander (the crop 3-5% large for
half a second on a slow turn's way back, f654-f658 and f311-f320) is
under the threshold and is today's behaviour, and it errs large -- the
avatar inside his own oval, never his skin outside it. Sizing the 20-35
frames from the remembered depth too would change frames under 35, so it
is a question for him, not a change.

### His two screenshots (2026-09-18, `new.png` / `new2.png`) and his speed question -- analysed

Turned to each side, ringed: his own black brow's end and eye corner
beside the avatar's along the profile, and on the other side "the nose
doubled" -- his nose tip beside the drawn one. Both are the day-19
measurement: past 50 degrees the eroded and feathered edge of the oval
sits 10-15 frame px inside his profile, and along that band his own face
shows; a black brow and a nose tip are what the eye catches. That is step
3 below. Past ~50 the head is drawn at the 45 cap, so the drawn nose is
always a little short of his -- the honest limit without the views.

"The AI face cannot keep up with a slightly quick turn -- is it the 20-21
fps?" Measured on the turn clip (a scratch probe: his yaw by an unfiltered
tracker on the camera frame, the app's yaw, the avatar's yaw by an
unfiltered tracker on the output, and each one's nose tip):

| moment | avatar's turn against his | the drawn nose off his |
|---|---|---|
| sitting still | -- | 6-7 px |
| slow turns (f244-330, f355-480) | 0.91-0.95 x, 0-1 frame behind | 5-9 px while moving |
| the fast return (f801-830, 3 deg/frame) | follows within a frame | **20-32 px** |
| the same with `size_smoothing` 0.3 instead of 0.85 | 0.96 x, 0 frames | 8 px |

The tracker's 0.35 EMA is one frame; the head's TURN follows within a
frame; the head's POSITION lags about six frames, and the cause is
`size_smoothing` 0.85 -- day 9's EMA on the extractor's scale and
translation against the "breathing" scale while sitting still. The scale
itself has not been applied since day 13 (`scale_gain` 0), so only the
translation is left carrying the heavy smoothing. At 20 fps six frames is
0.3 s of the ~0.4 s a fast turn is behind; the serial pipeline (tracker
6-13 ms, network 26, paste 7) is the other 50 ms, plus the camera's own.
So fps matters twice (every frame-counted filter lasts longer in seconds,
and the pipeline's one frame is 50 ms rather than 33), but the bulk is the
0.85. It is on the drive path: his call, after the sitting-still jitter is
measured at 0.85 against 0.3-0.5 on the still clips. Alternatives: a
speed-aware filter on the translation (steady still, fast when moving);
the tracker on its own thread for 30 fps (the plan's step 5a).

### Step 3: the profile without the erosion -- built, measured, his eye next

`_push_profile` in `_oval_mask` (`_PROFILE_PUSH` 1.0, erosion radii): a
midline point that lies outside the rim's own convex hull -- which only a
turn does; on a frontal face every midline point is inside the rim, proven
again on his 894 recorded frames (`test_the_profile_push_moves_nothing_on_
a_frontal_frame`) -- is moved outward along the ray from the rim's centre
by the erosion's radius before the hull is taken, ramped over the first
radius outside so the frame the nose leaves the rim moves the edge by
nothing. The erosion then lands the mask's edge on the profile instead of
a radius inside it. Tests: `TestProfilePush` (5); suite 1373 + 2 skipped,
ruff and mypy clean.

`scripts/probes/profile_probe.py` (the frame's mask rebuilt under each
push value from the hooked crop points; the profile = the midline points
outside the rim that frame; from each, inward along the ray to the rim's
centre, where the mask reaches 0.5 and 0.9; zooms of the paste per push
value, `recordings/day20/profile-f*.png` -- his face, delete after his
look). His hardest frames, crop px (1 crop px = 1.6 frame px here):

| frame, yaw | profile points outside the rim | push 0 (before): 0.5 / 0.9 reached, max | push 1: | push 2: |
|---|---|---|---|---|
| f298 +48 | 17, by 4-21 px | 3.5 / 6.5 | 0.5 / 3.5 | 0.0 / 0.5 |
| f452 -45 | 17, 5-22 | 3.0 / 6.0 | 0.0 / 3.5 | 0.0 / 0.0 |
| f553 +55 | 23, 2-33 | 4.0 / 7.5 | 1.0 / 4.5 | 0.0 / 1.5 |
| f697 -56 | 22, 5-31 | 2.0 / 6.0 | 0.5 / 4.0 | 0.0 / 0.5 |
| f804 -45 (fast) | 19, 2-24 | 2.5 / 5.5 | 0.0 / 2.5 | 0.0 / 0.0 |
| f1042 +40 | 13, 2-17 | 3.5 / 6.5 | 0.5 / 3.5 | 0.0 / 1.0 |

(The medians are 0 at every setting: most midline points lie in the
hull's concave stretches, inside it; the maxima are the vertices -- the
brow ridge, the nose tip, the chin -- where the strip is.) The zooms say
the same: at f553 the half-way contour ran inside his nose tip and brow
ridge (his own skin between the drawn nose and the room -- the doubled
nose), at push 1 it lands on the profile points, at push 2 it crosses 2-4
px onto the room behind him. So 1.0, the plan's number. The frontal
frames f30, 60, 90, 110, 338, 499, 900: the mask identical to the byte at
every push value. What is left along the profile at push 1: the feather's
inner half, 2.5-4.5 crop px from 0.5 to 0.9 -- his brow can tint that
band faintly; his eye decides whether 1.5 is wanted (the constant).

### His look at the day-20 build (04:00-04:03, from the source tree, both steps in): "nothing changed at all"

His words: exactly the same problems, nothing new, the fps still 21 --
"would 30 or 60 fix it?" The log confirms the build he ran had steps 1 and
3 (the engine file's last change 03:51, the avatar loaded 04:00:53). His
verdict is right about what the eye sees: step 1 removed a 1.8% dip after
a glance and step 3 moved the paste's edge 5-6 frame px along the profile;
what dominates a fast turn -- the drawn head 20-32 px behind his (30-45
live at 21 fps, the filter counts frames) -- was left as it was, on the
drive path, waiting for his word. The wrong order: that number should have
been put in front of him first.

**Measured for the decision (replay, no code changed):**

| | `size_smoothing` 0.85 (today) | 0.3 |
|---|---|---|
| the drawn nose off his on the fast return (turn clip f801-830) | 20-32 px | 8 px |
| the avatar's turn against his | 0.34 x, 6 frames (unexplainable fit) | 0.96 x, 0 frames |
| sitting still at the camera (natural.mkv f40-118): avatar-minus-his per frame p50 / p90 | 0.14 / 0.40 px | 0.12 / 0.29 px |
| sitting still at the screen (f119-416) | 0.25 / 0.76 px | 0.18 / 0.57 px |
| the drawn nose's own movement per frame, still, p90 | 0.49 / 1.90 px | 0.45 / 1.96 px (his own 0.49 / 1.94) |
| expression following (driving.mkv): ear r/gain, mar, mouth width, brow | 0.94/0.97, 0.99/1.17, 0.99/0.67, 0.91/0.69 | 0.94/0.97, 0.99/1.17, 0.99/0.68, 0.91/0.69 |
| lip thickness, position on every frontal segment | -- | the same to 0.1 px |
| the photograph's background under the mask on the return, f805 (yaw -42, the guard on) | 1789 px | 673 px |

So the 0.85 buys nothing today -- the scale it was set against is off since
day 13 -- and costs the whole lag. 0.3 changes nothing sitting still and
nothing in the expression; the head's position follows within a frame.
On the fps: 30 is reachable (the tracker on its own thread; the network
26 ms + the paste 7 leave ~30) and shortens every frame-counted lag by a
third and the pipeline's own frame from 50 to 33 ms; 60 is not reachable
on this machine (the network alone is 26 ms) nor from this camera (the
ZV-E10 streams 30). Neither is the bulk; the 0.85 is.

**The brow and the nose at a hard turn, honestly:** past ~50 degrees the
head is drawn at the 45 cap, the drawn nose stops short of his, and his
own nose tip stays visible beside it -- the guard cuts the render's
background there and his skin shows through. Step 3 covered the 5-px
strip along the profile and nothing more. What is left is the cap's
limit: accept it; or raise the cap to 50-55 and look at the smear
(measured on days 17 and 19: the front portrait draws 40 cleanly, 45 at
the edge); or the second photograph per side, day 18's approach, which
broke the face when it switched inside a movement -- it would have to
switch only on a held turn. His call, after the lag is fixed, because the
lag makes every turn look worse than it is.

**His word (04:2x): "yes, change it to 0.3."** Done: `_DEFAULT_SIZE_SMOOTHING`
0.3 (the code default, `_size_smoothing`), and the two avatars' own
`metadata.json` (his neural avatar and woman_person both carried 0.85
explicitly; the importer copies his manifest, so new avatars inherit
0.3). The test's default expectation follows; suite 1373 + 2 skipped,
ruff and mypy clean. The build as it now runs, on the turn clip: the
drawn nose 7.4 px off his on the fast return (was 21.5), 1.5 / 6.1 px on
the slow turns (was -4.6 / +8.6); the avatar's turn 0.93-0.96 x his with
0 frames of lag on every window (was 1-2, and the fast return
unexplainable). Next: his live look; then the tracker on its own thread
for 30 fps; the cap question after that.

### His look at 0.3 (04:34): "it follows well now, beautiful, I like it" -- and his two presses: the far brow's end at a turn

His words: one problem left -- turning past about 20 degrees either way,
his own brow shows a little, and part of the eye. He pressed "That looked
wrong" once per side (`captures/moment-20260918-043431`, yaw +27.6 from
rest, held; `-043450`, -17.4, held). On both, at the far side of the turn
(the profile side), his black brow's outer end and the eye's outer
corner show past the paste's edge.

**Measured on the two presses** (the frame replayed as the session,
the capture's anchor seeded; the paste's reach at the far brow's outer-end
landmark, along the ray from the rim's centre, crop px; 1 crop px = 1.6
frame px):

- The far brow's end lies **outside the rim's own hull** at a turn: 6-8 px
  at +28 (anterior to the temple, it moves with the turn more than the rim
  does), 2 px at -17. The midline push of step 3 does not reach it: the
  brows were not in the hull.
- What cut it was not the driver's oval but **the portrait's static
  oval** -- the frontal photograph's, built once at load: at +28 it ends
  11 px short of his far brow's end (14 to the 0.9 level), at -17 7.5 px
  short; the driver's oval alone, with the brows pushed, would reach
  within 2.5 / 1.5 px. And the far brow's end sits at the very edge of the
  portrait mask's box (crop x 355 of a box ending at 358).
- The drawn face itself **does** reach it: at +28 the render's far temple
  ends 1 px beyond his brow's end, at -17 8 px beyond (the row of the
  landmark, the render's pixels not within 30 of the photograph's
  background). So the gap was the mask's, not the head's.

**Built, off the drive path (the paste's mask only):**

1. `_BROW_LINE` (the 12 brow points) joins the pushed hull set beside the
   midline; the push's hull test and direction now use the rim **as read**
   (`_push_profile(reference=)`), before the mask's own forehead ceiling,
   chin drop and jaw shrink, so what is pushed depends on the face alone
   (a chin drop no longer moves a pushed point). On his 894 frontal frames
   the mask with the brows is the mask without, to the byte
   (`test_the_brows_in_the_hull_change_nothing_on_a_frontal_frame`); the
   synthetic test face's brows sit 1-5 px outside its own rim, unlike a
   real face's, so the synthetic identity test excludes them.
2. **The room past the portrait's oval** (`_turn_room`, `_frame_masks`
   returning the two ovals, `extra` carried in the held tuple and given to
   `_paste`): at a turn the paste may add the driver's oval where the
   portrait's falls short of it, up to the room's weight -- and that
   addition is pasted **only where the render is the drawn face**, by the
   guard's keep map in full, so the photograph's background stays out as
   before, by the guard instead of the static oval. The weight
   (`_room_weight`, `_ROOM_YAW` 12 to 16 degrees from the rest): nothing
   under 12 -- `extra` is `None` and `_paste` is the byte it was -- all of
   it from 16; the guard's own ramp (12 to 20) left the -17 press two
   thirds covered. A photograph without a flat background has no guard and
   no room, as before.
3. `_MASK_TURN_ROOM` 0.08: the portrait mask's box widened to either side
   by 8% of its width at load (154..358 -> 138..374 on his avatar), since
   the paste warps only into the box and the drawn face's far side at +28
   ended 2 px inside the old edge. The paste's area grows by that much.
4. `_paste` with the hair on now takes the guard and the room too (the
   hair union is built on the guarded patch); frontal frames with hair are
   the same values as before.

Tests: `TestTurnRoom` (5: the room's algebra, the ramp, the box, the
paste adding only over the drawn face -- a checkerboard face, since the
lighting plane matches a flat face into a flat scene exactly, with the
correction's memory reset per paste); suite 1379 + 2 skipped, ruff and
mypy clean. The probes' paste hooks take `extra`; `band_probe` and
`profile_probe` hook `_frame_masks` now.

**After, on the presses** (the paste's reach at the far brow's end; negative
is short of it):

| press | before: to 0.5 / to 0.9 | after |
|---|---|---|
| +27.6 (left turn, held) | -11.0 / -14.0 px | -3.5 / -6.0 |
| -17.4 (right turn, held) | -7.5 / -10.0 px | -1.5 / -4.0 |

What is left at +28 is the drawn head's own far edge: the render ends 1
px past his brow's end there, and the guard's feather (5 px, reaching 2
into the render's edge) holds the paste that far back from it -- the
avatar's temple simply ends where his brow ends at that turn. The zooms
(his face, in the session's scratch folder, gone with it) show the dark
strip along the far side gone under the drawn temple. The turn clip's
frontal check frames: the mask identical across the push values, as
before.

### His look (04:5x): "perfect" -- committed and pushed; and does it hold for another face?

His word: commit and push, this is the version he likes. Days 19 and 20
went up as four commits (4359d00 the turned-away hold and the second
look; 4b26f11 the engine: the oval to the profile, the room, the depth
held, the smoothing; b7b4cde the three deleted test avatars; ee5d837 the
docs) -- origin/main ee5d837, with day 17's last two commits that had
been waiting. The zooms of his face under recordings/day19 and day20 are
deleted; the day-20 CSVs (numbers) stay.

His question, "will it work on someone else's face once they calibrate?"
-- measured rather than assumed, on the friend's calibration recording of
2026-09-12 19:40 (her clip through the engine with her own avatar):

- Every one of her 624 recorded frontal frames (under 12 degrees) gives
  the same mask to the byte with the day-20 rules (the midline and the
  brows in the hull, the push) as without; her 26 turned frames only grow.
- Her turn step (+18 to -34 degrees): at -34 the profile push does on her
  face what it does on his -- the strip along the profile 3.0 crop px to
  0.9 before, 0.0 after (max 6.0 to 2.5). Her turns never passed 35, so
  the depth hold was not exercised on her; when it is, it reads her own
  resting depth from her own anchor.
- Nothing in the turn work carries a number of his face: every rule
  measures from the person's own rest (the anchor learned at each Start,
  the calibration's profile) and from the tracker's landmarks; the degree
  thresholds (12, 16, 20, 35) are about the tracker's geometry. The
  translation smoothing is the avatar's (both avatars 0.3; a new avatar
  copies his manifest). What is the same for everyone: the 45-degree cap
  and the drawn head's own far edge.

**Next:** the tracker on its own thread (30 fps); then the cap question.

## The delivery, safely: the lock, the two builds, the payments (a plan, 2026-09-18, nothing built)

His ask after the push: "show me the full roadmap for uploading this
application completely safely, so that nobody can steal it -- I fear a
scam, that I work for nothing and they take the app and do not pay."

### What is true today (checked, not assumed)

| | today |
|---|---|
| the source | private repository (the GitHub API answers 404 to a stranger); the package holds no `.py` -- PyInstaller's bytecode inside `sha256.exe`; the models are public LivePortrait / MediaPipe weights, the value is the 20 days of engine work |
| the switch | a **courtesy**: it fails open (no network, a blocked host, a deleted line -- all "allowed"), its URL sits in `configs/local.toml` as plain text, and **the build of 2026-09-14 carries no URL at all** (built without `--switch-url`), so that package cannot be switched off |
| the client's copy | identical for everyone: no per-copy id, no expiry; a copy runs forever offline |
| the exe | unsigned by choice (anonymity); SmartScreen's warning is accepted |
| the deal | staged payment was the protection (the 2026-08-15 decision: no licence server, no accounts) |

So the honest answer to "can they take it and not pay": with the package
as built, yes -- they unplug the network or never see a switch, and the
application works. The 08-15 decision traded this for time on quality.
That time is spent; the question is open again, and this is what can be
done, in the order it is worth.

### Step 1 -- the courtesy switch becomes a lock (one day)

The same file on a host he controls, but:

1. **Fails closed.** No answer within the timeout, no file, a wrong file
   -- the application does not Start, with a plain message ("the
   application could not confirm its licence; check the internet and try
   again"). A call already running is not cut mid-call: the check every
   five minutes only refuses the *next* Start once it has failed for N
   minutes (a grace of, say, an hour), so a flaky line does not kill a
   meeting.
2. **The URL and a public key inside the executable**, not in a text
   file. A config edit removes nothing.
3. **The file is signed** (Ed25519: he keeps the private key, the exe
   carries the public one). Nobody can forge "on" by pointing the host
   name elsewhere or serving their own file; the only way round is to
   change the executable itself.
4. **A hard expiry in every build** (30 days from the build date, say):
   after it the build refuses to Start whatever the file says. Every
   month they need a build from him -- until he decides otherwise. The
   file can also say "allowed until <date>", so an extension is one line
   on the host, no rebuild.
5. **A per-copy id** compiled into each package (client A, client B, the
   demo): the file lists which ids run. One copy switched off, the rest
   untouched; a leaked copy is switched off by name.
6. The guide says it plainly: the application confirms its licence at
   Start and every five minutes; it needs the internet for that.

What a thief must then do: extract the Python bytecode from the exe and
patch out the check -- a skilled person's afternoon-to-day with Python
3.12 (decompilers are poor for it, but patching a check does not need
readable source). That is step 2's job.

### Step 2 -- the executable made hard to open (one to two days)

PyArmor over the PyInstaller build (the bytecode obfuscated and bound to
a runtime), or Nuitka (Python compiled to C). Either turns "patch a
check" from an afternoon into days-to-weeks of expert work on a 5 GB
bundle with torch and MediaPipe inside -- more than the application is
worth to steal for most. Torch and MediaPipe must be tested under either
(a known source of trouble; a day is budgeted for it). No client-side
lock is unbreakable: a person with a debugger on their own machine wins
in the end. The aim is cost above the payment, not impossibility.

### Step 3 -- two builds, and the payments tied to them

- **The demo build**: the lock on, a 14-day expiry, its own id, the
  avatars he chooses. They test it on real calls. Nothing is owed for it
  beyond what was agreed for the demo stage.
- **The final build**: after the payment for the delivery -- the lock
  still on (expiry 30 days, renewed by a line on the host while the
  deal holds), so the last payment is still protected.
- **The unlocked build** (or the host's file set to "allowed, no
  expiry"): only after the full payment. His choice which.
- **The source code is never part of the package**, and not part of the
  deal unless priced separately -- the exe is the product.
- Something written, even an e-mail exchange, that says what is
  delivered, when, for what -- through the friends' company, in whatever
  name he chooses; his anonymity constraint makes a court a poor
  fallback, which is why the lock, not the paper, is the protection.

### Step 4 -- the upload itself

The zip (3.8 GB) on the download site he builds and on the USB stick,
with its SHA-256 checksum published beside it, so a client can tell his
package from a tampered one. The locked package is safe to hand out:
whoever downloads it needs the host's word to run it. The host for the
switch file: any static host on the anonymous account (a GitHub Pages
page of the private organisation, a small object store) -- since the
file is signed, the host need not be trusted, only reachable. The private
key: kept off the machine that runs the client's build, backed up twice
-- lose it and every copy stops at its expiry, which is the one way the
lock can hurt him.

### What no step can prevent, said plainly

- The idea. Anyone may build "a live avatar for calls"; what took twenty
  days of measurement cannot be copied from a screen.
- A screen recording of the output.
- A client who pays, gets the unlocked build, and passes it on: only the
  per-copy id (a name inside the file) says whose copy it was.
- An expert who decides to spend weeks: step 2 makes it weeks, not
  hours.

### The order, and his decisions

1. Step 1 first (one day; it turns the fear into a switch he holds).
2. The demo build with it; their look; the first payment.
3. Step 2 before the final build, if the first payment came without
   trouble and the deal looks real; if it did not, nothing more is spent.
4. Steps 3 and 4 as written.

His decisions: the expiry lengths (14 / 30 days); whether the last
build is unlocked at all; the host; whether step 2 is worth its two days
against the sum at stake. Nothing here is built until his word.

### Option (b) in plain words, and the host (2026-10-09, still a plan, nothing built)

He asked whether the application could be "deployed" so that a server
switches it off directly. Two readings were put to him: (a) the whole
application runs on a rented GPU server and the client gets only the
video -- safest, but a GPU server per client every month, the network's
delay inside every call, weeks of work and a paid account in a name;
(b) the application stays on the client's PC and asks a host for
permission -- step 1 above. He chose to have (b) explained; this is the
explanation he asked for, written for a fifteen-year-old.

One refinement proposed on top of step 1, **his decision, not yet
taken**: tell two cases apart. *No answer* from the host (bad internet)
keeps the hour's grace, so a call is never cut by a flaky line. *A signed
"off"* from the host stops the application within five minutes, even in
the middle of a call. Step 1 as first written only refuses the next
Start.

**The process, step by step (the text he can hand on):**

*The idea in one sentence.* The app runs on the client's computer, but
before it starts it reads a small note on our website that says "you may
run" or "you may not run". We control the note, so we control the app.

*Part 1 -- one-time setup, done once by us.*

1. We make two keys, born together as a pair. The **private key** is
   like a stamp: only we have it, kept in two safe places, and only it
   can sign the note. The **public key** is like a picture of the stamp:
   it is built into every copy of the app, which uses it to check that
   the note was really stamped by us. It cannot stamp anything, so it is
   not secret.
2. We choose a **host**: a place on the internet where the note lives.
   Because the note is stamped, nobody can fake it, even if they get
   into the host.

*Part 2 -- making a copy for a client.*

1. We give the copy a neutral name, for example `copy-03`.
2. We build the app. Inside it we lock in its name, the address of the
   note, the public key and an end date (for example 30 days).
3. We add one line to the note: `copy-03: allowed until <date>`.
4. We stamp the note with the private key and upload it.
5. We send the client the app and the usage rules.

*Part 3 -- what happens on the client's computer.*

1. **Start:** the app downloads the note, checks the stamp, finds its own
   name and checks the date. All three fine -- it starts. Otherwise it
   shows "Could not confirm the licence. Check the internet."
2. **While running:** every five minutes it checks the note again,
   quietly. The face, the speed and the quality do not change.
3. **If the internet drops:** the call keeps going; one hour of grace.
   After that the app does not start again until it can reach the note.

*Part 4 -- what we can do from our side.*

- **Extend:** change the date in the note, stamp, upload. No new app
  needed, until the end date built into the app.
- **Switch one client off:** write `off` next to their name, stamp,
  upload. Their app stops (at the next Start, or within five minutes if
  the refinement above is taken). Everyone else keeps working.
- **Find a leaked copy:** every copy carries its own name, so we know
  whose copy it is and switch off only that one.
- **An end date in every version:** even if the note says "allowed",
  each version stops on its own end date; a new version comes from us.

*Part 5 -- if someone tries to cheat.*

- They turn the internet off: the app does not start.
- They write their own note saying "allowed": the stamp does not match,
  the app does not start.
- They change the app itself: the only way around it, which is why the
  app is also scrambled (step 2, PyArmor or Nuitka) -- days or weeks of
  expert work instead of an afternoon.

*What can go wrong for us.*

- The host is down: calls continue for an hour, new Starts wait until
  the host is back -- so the host must be a reliable one.
- We lose the private key: no new note can be stamped, and every copy
  stops at its end date. That is why it is kept in two places.

**The host -- what it is and how it is made.** A host is only a place on
the internet from which one file is read by its link; no server is
bought or rented, and one file is free to keep. Recommended: **GitHub
Pages on the anonymous account he already uses** (no new sign-up, no new
trace), about ten minutes:

1. On GitHub, a new repository, for example `licence`. It must be
   **public** -- Pages from a private repository is not free -- which is
   harmless: it holds only the signed note, never code.
2. Upload the note to it.
3. Settings -> Pages -> the `main` branch -> Save.
4. The note appears at `https://<account>.github.io/licence/licence.json`;
   that address is what goes into the executable.

Cloudflare Pages or Netlify work the same way and are free too. One rule
follows from the note being public: **copies get neutral names**
(`c7f2`, `copy-03`), never a client's real name; who is which stays in
his own notes only.

**Where the client downloads it, and who adds avatars (answered and
built 2026-10-09).** The 3.8 GB zip goes on a cloud link on the anonymous
account (Google Drive or Cloudflare R2; GitHub Releases refuses a file
over 2 GB) or a USB stick; with the lock built, a leaked link runs
nothing. A thought for his decision on the expiry: a 30-day expiry *in
the build* means a 3.8 GB download every month -- a longer build expiry
(six months, say) with the monthly control in the note's date costs
nobody a download. **Avatars are the clients' own now, on his word:**
Menu -> Login and the passphrase are gone; Add avatar and Delete avatar
are in the Menu for everyone. An avatar is a folder under `avatars/` next
to `sha256.exe`, on the client's computer only. The cost, said to him: a
client can also delete the avatars shipped with the package, and a poor
photograph gives a poorer avatar (the importer's checks and the photo
guide catch what can be measured; the rest comes back to him as a
message).

## The fast version: the same face in fewer milliseconds (planned and built 2026-10-08; his live look pending)

His ask (2026-10-08): "write the plan for what the maximally optimised
version gives us in this application, with the face and the expressions
working as they do now -- nothing spoilt, everything the same." And: will
it run on an i5-14400F with an RTX 4060 Ti 16 GB, 32 GB, 1 TB. The ask came
after a generic "Principal AI Systems Architect" prompt from another
assistant (zero-copy, multi-process, TensorRT, adaptive scaling, 60 fps on
a 3060); what of it applies here is below, what does not is at the end.

### What is true today (this machine, the dates given)

| stage | where | ms | measured |
|---|---|---|---|
| tracker (MediaPipe Face Landmarker, VIDEO mode) | CPU | 12.6-12.7 | day 11 live; day 19 on driving.mkv |
| crop (one warpAffine to 256) | CPU | 0.6 | day 11 |
| model (three nets, fp16, one CUDA graph, the upload inside) | GPU | 17-27 | 25.5 on 2026-09-04 with sync; ~17 in the day-11 render script; 26.7 in the paste table |
| the 512 face downloaded, RGB to BGR | GPU to CPU | under 1 | |
| mask (`_frame_masks`) | CPU | 1.2-1.5 | day 13 |
| paste (three warpAffines, the quarter-scale colour plane, one fused blend) | CPU | 7.6-9.5; 17.9 with the hair on | 2026-09-04; day 17 |
| virtual camera send | CPU | 1.4 | B004 re-run |
| **sum, one thread, hair off** | | **42-55** | **18-23 fps, his counter** |

Everything runs on one thread (`PipelineRunner._run` calling
`PipelineController.process_once`): read, track, animate, render,
condition, send, in that order. The GPU waits while the CPU tracks and
pastes; the CPU waits while the GPU draws. The camera gives a frame every
33.2 ms. The stages' sum equals the frame time (day 11: 43 ms, 23 fps, his
number), so there is no hidden interpreter or VS Code cost: the exe will
not be faster than the source tree by more than a frame or two.

Not measured anywhere yet: the stages with the app's own `last_timings`
for 60 s with him in frame; the portable build against the source tree;
any GPU but the 5070 Ti; any CPU but the 14700KF.

### The rule every step is held to

The drive path -- landmarks, gates, `ratios` / `brow_delta` / `mouth_leak`,
`graph.replay()`, `static_out` -- is not touched. The tracker reads the same
1280x720 frame, the crop is the same warpAffine, the graph is the same
graph. Proven the way days 19-20 were: `capture_replay.py` on his captures
and the friend's recording, `static_out` byte-identical, frame for frame.

What may change is where the drawn face is put into the frame (the paste)
and when each stage runs (the threads). The paste's change carries a
tolerance, because GPU bilinear sampling rounds differently from
OpenCV's: at most 2/255 at p99 and 0.5/255 mean against today's CPU paste
on 20 stills across three avatars, frontal and turned, hair on and off;
the frame outside the box byte-identical, as `TestPaste` already requires.

Nothing adaptive. No frame skipping of the tracker, no smoothing, no
resolution ladder, no quality switch mid-call (the output quality selector
of day 15 stays what it is, the user's own choice).

### Step 0 -- the measurement protocol (half a day)

A probe (`scripts/probes/frame_budget.py`, or a `--budget` flag on the
app) that runs the app's own pipeline for 60 s with him in frame --
talking, a smile, a turn each way -- and writes per stage p50/p90/p99 and
the fps counter: the tracker (the monitor's TRACKING stage), the engine's
`last_timings` (crop/model/hair/mask/paste), the send, end to end. A CSV
under `recordings/budget/` (not committed) and a table here. Twice: hair
off, hair on. Once more from the portable build (`dist/sha256-portable`),
to settle "is the exe faster" with a number (expected: the same within
1-2 ms).

This protocol is what every later step is measured by, before and after:
`driving.mkv` through replay for the stages that replay, the live camera
for the fps counter.

### Step 1 -- two threads: the tracker on the next frame while the GPU draws this one (one to two days)

`pipeline/runner.py` grows a second thread. Thread A (capture):
`camera.read()`, `tracker.process(frame)`, the pair `(frame, tracking)`
into a one-slot mailbox -- latest wins; a pair nobody took is counted as
dropped, the runner's own "drop, never queue" rule. Thread B (render):
takes the newest pair, `engine.animate`, `renderer.render`, the day-19
turned-away reading, `conditioner.condition`, `virtual_camera.send`,
`_publish`. The snapshot carries the pair's own frame and tracking, not
`controller.last_*` -- two threads must not share a "last". `process_once`
stays as it is, the single-thread diagnostic path the tests and
`capture_replay` use, built from the same two halves
(`_capture_and_track`, `_render_and_send`) so the two paths cannot drift.

What it gives: the frame time becomes the longer of (CPU after tracking +
GPU) and (capture + tracking), instead of their sum. On this machine: GPU
17-27 + mask 1.5 + paste 7.6 = 26-36 ms against the camera's 33.2 -- 30 fps
most of the time, step 2 for the rest. On a slower CPU the tracker's
15-20 ms stops costing anything so long as it stays under the GPU's time.

What it costs: one more frame of latency, 33 ms, on a chain that is about
260 ms glass to glass today (227 of it the camera). He has cared about
latency; this is his call, said before it is kept.

What stays the same: the engine's temporal state (`last_dt` from the
camera's timestamps, the EMAs, `_pose_still`, the anchor) reads the same
timestamps and sees frames in the same order; the tracker's VIDEO mode
runs on one thread with monotonic timestamps as now; the hair worker is
already its own thread; UI calls into the engine (Calibrate me, options,
Reset, capture) run from the UI thread today without a lock and keep
doing so -- thread B is the one thread that calls `animate`, as the
pipeline thread is today.

Keep bar: replay byte-identical (the same frame sequence, the same
`static_out`, the same output); live, the fps counter at 29 or more for
60 s with him in frame, hair off; p90 frame time on thread B under
33.2 ms; no frame ever published with another frame's tracking (asserted
on `frame_id` in the snapshot). Live, the dropped-frame pattern changes
with the timing, so a live session is not byte-comparable to the old
build: the replay is the proof, the live run the measurement.

### Step 2 -- the paste on the GPU (one to two days)

Today `_render` downloads the whole 512x512 face, then `_paste` warps the
mask box's patch of it into the frame with three `cv2.warpAffine`s (face,
mask, the hair blend), fits the lighting plane at quarter scale, and
blends: 7.6-9.5 ms, 17.9 with the box grown to the hair.

New: `static_out` stays on the GPU. The mask box's patch of the face, this
frame's mask (`_frame_masks`' output, uploaded as one small float tensor,
about 13% of the square), `extra` and the day-19 `keep` go through one
`affine_grid` + `grid_sample` (bilinear; `align_corners` chosen to match
OpenCV's pixel-centre convention, the match measured, not assumed) into
the frame box only; the blend in fp32 on the GPU; the box downloaded once
as uint8 and written into `scene.copy()` as now. The lighting-plane fit
(`_face_colour_correction`) keeps its quarter-scale arithmetic, either on
a downloaded quarter-scale ring (the same numpy, a 1/16 download) or
ported to torch -- whichever differs less from today, measured. The hair
matte's warp joins the same `grid_sample` as a second channel, so the box
grown to the curls costs GPU pixels, not CPU passes.

The hair worker still needs the face on the CPU: that download becomes
asynchronous (a pinned buffer, `non_blocking`, consumed by the worker a
frame later; it already runs one or two frames behind). The day-19 hold
(`loaded.last`) keeps the face it needs the same way.

Expected: paste 7.6-9.5 to 2-4 ms, hair on the same as off; the CPU side
of thread B down to mask + download + send, about 4-5 ms.

Keep bar: the tolerance above on 20 stills -- his captures' replay, the
friend's recording, baska's, woman_person, frontal and at +-30 yaw, hair
on and off; the zooms of every edge looked at before the number is
believed (the day-16 lesson); paste p50 at most 4 ms, hair on at most 5.

### Step 3 -- the GPU's housekeeping, exact (half a day)

Small, byte-identical things: the crop's upload through a pinned buffer
with `non_blocking=True` on the graph's stream; step 2's one download the
only synchronisation point in the frame; `cudnn.benchmark` stays (already
on); the mask's landmark half (`_frame_masks`' driver oval) built on
thread A beside the tracker, since it needs only the landmarks, leaving
thread B the portrait half and the intersection. Expected 1-3 ms off
thread B. Kept only if replay is byte-identical.

### Step 4 -- re-measure (an hour)

Step 0's protocol again, source tree and portable build. Expected on this
machine: the fps counter at the camera's 30, p90 under 33.2 ms, hair on
and off. If step 1 alone reaches it, step 2 still goes in: it is what
makes a slower CPU irrelevant.

### Step 5 -- the first measurement on a slower machine (a day, when one is at hand)

The portable build and step 0's protocol on a 4060 Ti / 5060 Ti-class
machine -- the one he is considering, or one of the five users'. The first
real number for any GPU but the 5070 Ti; every estimate in the table below
stands until it exists. It decides whether phase 2 is needed at all.

### Phase 2 -- TensorRT for the two pixel networks (after the delivery, only if step 5 is under 25 fps)

The warping module and the SPADE decoder (31.5 of B005's 38.4 ms) compiled
to one TensorRT fp16 engine; the motion extractor and every line of
`_drive` stay in torch, so the keypoints -- the expression -- are
bit-identical and only the drawn pixels change, under step 2's tolerance.
The cost is B005's list: the `grid_sample_3d` plugin built against this
TensorRT, the CUDA runtime matched, the engine built on each user's GPU at
first run (minutes), about 1 GB more in the package, and a new way for a
user's machine to fail. Expected 1.5-2x on the model. Not for this
delivery.

**Measured 2026-10-09, on the `tensorrt` branch (a copy of the repo), in
`docs/benchmarks/B006-tensorrt.md`.** B005's three blockers are gone:
TensorRT 11.2.1+ runs 3-D GridSample natively (no plugin), the cu12 pip
wheels (11.3.0.99) load beside torch's cu128 DLLs, Python 3.12 has its
bindings. The two networks exported as one fused fp16 ONNX (opset 20, 204 MB)
and built strongly typed in 45 s on the 5070 Ti; the TensorRT enqueue
captures into the engine's own CUDA graph. The graph's GPU time 25.8 to
14.3 ms a frame (1.81x; the two networks alone 23.5 to 12.1, 1.94x); the
render thread 34 to 21.5 ms, 30 fps at the camera with 12 ms to spare. The
keypoints handed to the warping network are byte-identical over all 601
frames (the drive path is the same path); the composite differs by fp16
rounding -- at most 6 grey levels, 99% of the differing pixels within 2,
mean 0.02 over the frame, 4-5% of the frame's pixels touched -- less than
torch's own fp16 differs from fp32. Package: about 0.75 GB with prebuilt
plans per GPU family, about 1 GB building on first run. Stage 2 (the
integration, the torch path kept as fallback) waits for his word; the
probe is `scripts/probes/tensorrt_probe.py`.

**Stage 2 built the same evening (B006 section 9):** `engine.pixel_backend`
(`auto` by default: the TensorRT engine when one is built for this GPU and
TensorRT version under `models/tensorrt/`, the torch networks otherwise;
`torch` never looks; `tensorrt` insists), `sha256.avatar.liveportrait_tensorrt`,
`scripts/build_tensorrt_engine.py`. The integrated path is the probe's
measured path to the byte (601 of 601 on the same plan); the torch path is
the reference's (601 of 601). The hairline read is cut to the window it
touches, exactly (125 of 125 readings the same), ~17-20 to ~6 ms. On this
machine: the render thread 21.5 ms p50 / 23.4 p99, no frame over 33 ms after
warm-up, 30.0 fps at the camera, latency 37 ms. Two builds of the same ONNX
differ by fp16 rounding (0.08 mean), so a new plan is proven by the probe's
run on it, not by a stored hash.

### Not in the plan, and why

- **The tracker on a half-size frame** (day 11's 2a): MediaPipe's landmarks
  move by sub-pixel amounts, and the landmarks are the drive path's input
  -- the gates, the crop, the anchor. After step 1 the tracker's time is
  hidden behind the GPU on every machine where the GPU is the slower side,
  which is every machine this plan is for. Not worth the one thing he
  asked to keep.
- **Tracking every other frame, temporal smoothing, an adaptive
  resolution or quality ladder**: each changes what the face does or how
  it looks mid-call. Excluded by the rule.
- **Multiprocessing, shared memory, zero-copy camera to GPU**: the frame
  to the GPU is a 256x256 crop (196 KB); OpenCV, MediaPipe and torch
  release the GIL in their C++; a second process would cost the
  PyInstaller build more than it gives.
- **Nuitka for speed**: the Python layer is thin (the stages' sum already
  equals the frame time); 1-2 ms at most. It stays in the delivery plan
  for the lock, not for fps.
- **The model's input size**: 256 is LivePortrait's own; another size is
  another model.

### What the fast version gives, machine by machine (estimates, except the first row's today)

| machine | today's code | after steps 1-3 | after phase 2 |
|---|---|---|---|
| his: i7-14700KF + RTX 5070 Ti | 18-22 fps (measured) | 30, the camera's ceiling, p90 under 33 ms | 30, with room for 60 if a 60 fps source ever exists |
| i5-14400F + RTX 4060 Ti 16 GB, 32 GB | 10-15 (CPU 30-38 ms serial + model 35-65) | 18-28, model-bound; 30 not promised | likely 30 |
| i5-14400F / Ryzen 5 7500F + RTX 5060 Ti | 12-17 | 22-30 | 30 |
| 6-core 2020 + RTX 3060 12 GB (the written minimum) | 8-13 | 15-25 | 25-30 |

How the 4060 Ti row is reached: its compute is about half the 5070 Ti's
and its memory bus a third (288 against 896 GB/s; Ada's 32 MB L2 softens
that for 256x256 maps), so the model is estimated at 2-2.5x: 35-65 ms.
The i5-14400F's single-thread speed is about 80-85% of the 14700KF's, so
the tracker is about 15-16 ms -- under the model's time, hence hidden by
step 1. Its 16 GB of VRAM buys nothing here (the model peaks at 0.7 GB);
32 GB of RAM and 1 TB are above the written 16 GB / 6 GB. It runs; after
steps 1-3 it is usable; 30 fps on it needs phase 2 or a faster GPU. Step 5
replaces every number in that row.

### The order, and his decisions

Step 0, 1, 2, 3, 4 -- about four to five days; step 5 when a machine
exists; phase 2 after the delivery, on step 5's number. His decisions: the
one frame of latency (step 1); whether a 4060 Ti machine is bought before
step 5 can be run on it, or a user's machine is measured first. Nothing
here is built until his word.

### Built the same day (2026-10-08): steps 0-3, measured, proven

His word in the afternoon: "start the optimisation from beginning to end,
report the result at the end; the application's quality must not break --
the face and so on all work well." Steps 0 to 3 are built and measured on
`recordings/driving.mkv` (frames 300-900, his face, 1280x720); step 4's
live run with him in frame, and step 5, are still to come. Nothing is
committed.

**Step 0, the measurement -- `scripts/probes/frame_budget.py`.** The clip
through the application's own `PipelineController` (and, in `--mode
threads`, through `PipelineRunner` itself), every published frame hashed,
the stages from the monitor and the engine's `last_timings`. Three pacings:
`asap` (stage timings), `lockstep` (the next frame only once the previous
is published -- no drops, so two runs' frame sequences are the same and
their hashes must be), `realtime` (a frame every 33.3 ms, the latest due
one, as a camera hands it -- the frame rate). `--compare` says, frame by
frame, whether two runs agree to the byte; `--deterministic` turns
`cudnn.benchmark` off so they can (see below); `--cpu-paste` / `--cpu-fit`
run the old paths for the comparison; `--profile` is cProfile over the
loop. `scripts/probes/still_diff.py` reads two output frames' difference
in grey levels. What the first runs showed, against the plan's table:

| stage, one thread, the clip | plan's figure | measured 2026-10-08 |
|---|---|---|
| tracker | 12.6-12.7 | **6.5 or 12.7**, bimodal between runs of the same code -- not understood; MediaPipe's threads, or the CPU's clocks |
| model (the CPU waiting the graph out) | 17-27 | **28.0 p50, 37 p90** |
| mask | 1.2-1.5 | **2.1-3.1** (two ovals a frame since the chin reach) |
| paste | 7.6-9.5 | **10.1-15.7** (the day-16 dark cells, day-19 keep, day-20 room) |
| the p90 spikes | -- | **the hairline read**: `_luminance` over the whole 1280x720 frame, every 6th frontal frame, 17-20 ms, counted in `model_ms` |
| frame, one thread, 30 fps camera | 42-55 | **50-66 ms, 18.8 fps** -- his 18-22 |

The profile (cProfile, 271 frames) put the paste's cost where the plan did
not expect it: not the three warps (0.9 ms) but the colour correction's
full-box float arithmetic (9.4 ms: the mask-weighted subtract and multiply
over the box, the area resizes, the astype conversions), and the GPU's own
time in `.cpu()` at 25.8 ms a frame.

**Step 1, two threads -- built as planned.** `PipelineController.process_once`
is now `capture_and_track` then `render_tracked`, the same code in two
halves; `PipelineRunner` runs them on `sha256-capture` and `sha256-render`
with a mailbox of one between them (`TrackedFrame`: the frame and its
tracking together; the newest wins, drops counted as `dropped_at_capture`);
a failure on either thread ends both, with the black screen as before.
`performance.overlap_tracking = false` in `configs/local.toml` is the one
thread again. Ten new tests (`tests/unit/test_pipeline_runner_overlap.py`):
the pairing of frame and tracking, the mailbox never queuing, both threads
ending on either's failure, the prompt stop, the single-thread fallbacks.

**Step 2, the paste on the GPU -- built, further than planned.** `_render`
hands the face back as the tensor it is (`on_device=True`); nothing waits for
the graph until the paste's last download. `_paste` dispatches on the face's
type: a numpy picture takes the CPU paste exactly as it was (the tests, the
CPU fixtures, `--cpu-paste`); a tensor takes `_paste_gpu`: the mask and the
hair blend warped on the CPU by the very `cv2.warpAffine` the CPU paste
uses (so the alpha is the CPU paste's to the bit), the face warped by one
`grid_sample` on the inverse of the same affine at integer pixel centres,
the mask-weighted difference and room shrunk to the correction's cells with
OpenCV's own `INTER_AREA` weights as two small matrix products
(`_area_weights`, the `computeResizeAreaTab` rule), and the whole colour
fit on the GPU (`_fit_field_gpu`): the dark cells with numpy's median read
by a sort and a device-side gather, the plane's sums in float64 and its 3x3
solve in closed form with the singular fallback chosen by the determinant,
the edge residual's Gaussian blurs as banded reflect-101 matrices (against
`cv2.GaussianBlur`: 2e-7), the inertia on a device-side field. The CPU
keeps what needs no pixel of the face -- the mask's sum and its "too little
mask" answer, the cell-size coverage by the same `cv2.resize`, the distance
transform, his features' cells -- and uploads it through pinned memory
(torch's caching host allocator; a copy from pageable memory makes the
driver drain the stream first, i.e. wait for the graph). The hairline read
moved after the render's launch, so its 17-20 ms run while the GPU draws.
The hair worker gets its picture from a pinned copy the GPU fills
meanwhile, submitted after the paste; the fold of its newest matte stays
before the mask, as it was. One wait per frame, at the box's download.

The tolerance, measured on five stills across the clip (frames 330, 450,
600, 700, 850), the GPU paste against the CPU paste with identical graph
output: **17-24 pixels a frame differ, every one by 1 grey level**, mean
difference 0.000, nothing outside the box. The plan allowed 2 at p99 and
0.5 mean. For scale: the application's *own* run-to-run difference, from
`cudnn.benchmark` picking kernels anew each start, is ~18,000 pixels a frame
by 1-3 levels (2% of the frame) -- which is why the hash proofs below run
with `--deterministic`, and why two live sessions of the unchanged code
were never byte-identical either.

**Step 3, housekeeping -- folded into step 2.** The pinned uploads, the one
synchronisation, the CUDA events that time the graph without waiting
(`model_ms` is now the graph's own time; `launch_ms` the CPU's launch;
`wait_ms`, `paste_pre/fit/post_ms`, `upload_ms` say where the paste's time
goes). Not done: the renderer's fresh-canvas copy (0.6 ms, exact to skip
for a same-size frame, but it is the renderer's race guarantee and stays);
the drive tensors' ~30 scalar writes vectorised (0.7 ms, on the drive path,
not worth the risk); a pinned upload of the 256 crop (0.1 ms).

**The proofs, all 601 frames, `--deterministic`:**

| what | result |
|---|---|
| the CPU-paste path after every change, against the untouched code's run | **601 of 601 byte-identical** |
| two threads in lockstep against one thread, CPU paste | **601 of 601** |
| two threads in lockstep against one thread, GPU paste | **601 of 601** |
| the GPU paste against the CPU paste, five stills | 17-24 px by 1 level, mean 0.000 |
| the 440 CPU unit tests | pass |

**The measurement after, the same clip, a 30 fps camera simulated:**

| | one thread, before | one thread, GPU paste (`overlap_tracking = false`) | **two threads, GPU paste** | two threads, hair on |
|---|---|---|---|---|
| frame rate | 18.8 fps | 19.4 fps | **29.4 fps** (interval p50 33.3 ms, p90 36.3) | 28.2 fps |
| render thread per frame | 50-66 ms | 50 ms | **30.5 ms p50, 33.4 p90** | 31.3 / 36.3 |
| of which the GPU's graph | 28 (waited out) | 26.0 | 26.0 (`model_ms`, events) | 26.0 |
| CPU before the wait: launch + mask + paste prep + fit enqueue | -- | -- | 2.3 + 2.6 + 1.8 + 5.1 = 11.8, under the graph | the same |
| the wait and the download | -- | -- | 17.3 | 17.3 |
| latency, capture to render (the clip; the camera's own 227 ms is on top) | 48.5 ms | 48.5 | **55.3 ms** p50, 70 p90 | 64 |
| frames dropped at the source | 1 in 3 | 1 in 3 | 100 of 600 (29.4 against 30) | 122 of 600 |

So: the GPU paste alone changes little on one thread (the tracker's 12.7 ms
still adds to the frame); the two threads alone gave 21.9 fps (the render
thread was 38 ms); together, 29.4 -- the render thread fits the camera's
33 ms most of the time. Hair on costs 1.2 fps now, not the 6.6 it did
before the pinned copy. The latency grew by about seven milliseconds on the
clip, not the full 33 of the plan: the render starts the moment its frame
is tracked, and the tracking of the next overlaps the GPU's wait.

**What stands between 29.4 and a locked 30, on this machine:** the GPU's
graph is 26.0 ms of the 33.3 and the CPU's serial remainder (crop, launch,
download, the renderer's and the conditioner's copies, the sink's send)
about 7 -- the render thread is at the budget's edge, so a slow frame drops
one. Two things would buy the margin, neither exact, neither done without
his word: `channels_last` for the warping and SPADE networks alone (cuDNN
spends 2.7 ms a frame converting layouts inside the graph; the motion
extractor -- the expression -- would stay as it is, the pixels would change
by fp16 rounding, like the paste's tolerance), and phase 2's TensorRT. The
exact 0.6 ms of the renderer's copy is there too.

**Observations worth keeping.** The tracker's cost is bimodal across runs
of identical code (6.5 or 12.7 ms) -- environmental, unexplained; it sits
on the capture thread now, where it costs no frame rate either way. A
staging buffer reused by name under a pending transfer corrupted the cached
blur matrices for one build (two uploads of the same name in one frame);
torch's pinned allocator tracks the stream and does this right -- never
hand-roll it. `render_liveportrait.py` unpacks `loaded.last` as three
items; it has been four since day 20 and is a GPU tensor's face now --
the script is stale, not touched here.

**Still open from the plan:** step 4's live run with him in frame (the
counter, hair on and off, and his word on the latency); the portable
build against the source tree (the expectation stands: the same within a
frame or two -- the stages are C++ and CUDA either way, and the exe is
PyInstaller's bytecode of the same interpreter); step 5 on a 4060 Ti /
5060 Ti-class machine. The stills under `recordings/budget/` were his face
and are deleted; the JSONs (hashes and timings) stay, gitignored.

## Day 21 (2026-10-14): the friend test's two faults -- the skin's tone, the forehead to the person's own hairline, the head guide

**His report (2026-10-14).** The fast package ran at 27-30 fps on the
friend's RTX 4070. On his own face everything sat well. On the two
friends, uncalibrated -- one bald, one with the hair pulled back -- a
stripe on the forehead: the skin colour differed from the avatar's, and
the forehead's size was not fitted, theirs being taller. His ask:
auto-calibration in both versions from now on, no difference between the
avatar's skin and the person's. His decisions on the plan: the avatar's
hair is out ("frozen, easily noticed"); the forehead may be stretched
only moderately, "a medium height"; the forehead should end slightly
below where the person's hair starts; and during calibration a green
outline where the head goes, so nobody sits too far or too close -- "a
simple process for everyone". Then: "start, in order".

**What was true, read off the code and measured first.** The forehead
reach took the LOWER of the two hairlines less a clearance of 0.04, so
with the person's hair higher than the portrait's it was always the
portrait's own hairline: the join sat on the avatar's hair roots, in the
middle of their bare forehead, their own skin above. The roots are a
band of darkening skin -- on his portrait the luminance leaves the
forehead's by 5% at 1.40 of the forehead, 10% at 1.44, 30% (the reader's
word for hair) at 1.52; on the woman's 10% at 1.90, 30% at 2.06 (the
reader's own columns, `scratch`). That is the stripe. And the colour
match was the room's light alone: a plane (a brightness and a gradient
per channel) over the whole face, the residual at the join faded over a
fifth of the box -- a face paler or pinker than the portrait's kept the
avatar's own tone in its middle. The baseline on the woman's capture of
2026-09-12 20:07 (`jaw_band --capture --render`, the top fifth of the
mask's rows, grey levels): forehead abs 2.70, step +1.50; on his of
2026-09-11 10:57: 7.51 / +4.42. No recording of the two friends exists
yet; the woman's captures (her forehead 84 px against his 47-60) stand
in for the taller forehead.

### Built 1: the skin's tone (`skin_match`)

Before the plane, the render's skin is given the person's: on the cells
of the correction's grid that are skin on BOTH sides (the mask's core;
not the scene's dark cells, not the render's -- its brows, eyes,
nostrils -- not his feature zone) the mean and the spread of each
channel are read for the room and the render, and the render gets, per
channel, a CAST `mean_room / mean_render` and a CONTRAST about the
room's skin mean `spread_room / (cast * spread_render)` (both within
0.7..1.4; the contrast 1 when either spread is under 2 levels). The
plane then fits the light's gradient on its own cells as before; the
residual as before. The reading is one frame in four (2 ms of the render
thread's CPU, measured; the apply 0.2 ms a frame), an EMA of 0.66 a
reading carries the nine numbers (the cast, the contrast, the room's
skin mean), the first reading taken whole, nothing read while the face
is held. On the GPU paste it is tensor sums and a device-side `where`
(the "too few cells" rule, the first-reading rule), nothing waiting; the
CPU paste the same in numpy. **Proven:** at `skin_match 0` nothing of it
runs and the frame is the untouched tree's to the byte -- 601/601 on his
clip (`frame_budget --deterministic --compare`,
`recordings/budget/tone-base` against `tone-off`).

**Where it is applied took five tries, each on her capture.** (1) A gain
and an additive offset per channel, on every pixel: her skin matched,
and the avatar's brown irises turned blue-grey -- the channels' offsets
differ, and what is a tone on skin is a hue on an iris. (2) Weighted by
his feature zone (grown and softened): the tone was left off exactly the
cells it had been read from, so the plane put it back, everywhere. (3)
The whole tone spared on the dark pixels (each pixel's lightness against
the render's skin median): the irises brown again, but the avatar's
stubble left pale over hers (the jaw band's step +5.8 -> +11.1) and the
brows pale. (4) The offset's luminance everywhere, its chroma by
lightness: the irises and the stubble right, and her avatar's beard
ORANGE -- her warmer skin's chroma, added whole to the beard's mid-lit
hairs. (5) **Kept:** the room's white balance -- the cast MULTIPLIED into
every pixel, which moves a dark pixel by little and a skin pixel by the
whole of it, the physics of a colour cast -- and the contrast by each
pixel's lightness (nothing under 0.7 of the skin's, all of it from 0.9:
a brown brow's lit hairs read 0.5-0.7 and took her flat-lit face's
contrast under 1 as a fading). The irises, the brows, the beard keep
their look; the skin takes the room's tone.

### Built 2: the forehead to the person's own hairline (`_forehead_plan`)

At the settle (the anchor, or the profile's `driver_hairline`), with the
driver's own forehead height in the crop against the portrait's: the
driver's oval goes to their hairline less `hairline_clearance`; the
avatar's clean skin is its hairline less `hair_root_margin` (0.15 of its
forehead, the measured shadow); the render's rows above 0.3 of a
forehead over the brow line -- past any brow raise, so the brows are
never stretched -- are stretched by what it takes for the clean skin to
reach the join, within [1, `forehead_stretch_max`], 1.25 by default
("medium"); the portrait's oval goes to the stretched clean skin's top,
the paste to the lower of the two ovals as ever. A person whose hairline
cannot be read (bald, or hair past the walk) gets the whole moderate
stretch and no bound from their side. The stretch is the paste's alone:
`_warp_face` remaps the render with the rows above the anchor row
sampled closer to it (the CPU), the grid does the same on the GPU;
nothing below the row moves, the drive is untouched (the avatar's mouth,
eyes and brows read the same on both captures' replays, within the
output tracker's noise). On his face: hairline 1.6, the portrait's 1.52
-- the join just under his hair, the upper forehead stretched 1.13. On
the woman's: the cap, the join at 1.31 of her forehead on the avatar's
clean skin, her own skin above it, matched.

**The clearance under the person's hairline, swept** (the same band):

| `hairline_clearance` | the woman's 200754 | his 105750 |
|---|---|---|
| untouched build (the old rule) | 2.70 / +1.50 | 7.51 / +4.42 |
| 0.04 (the old default) | 4.08 / +3.13 | 8.89 / +6.24 |
| 0.08 | 3.50 / +2.55 | 8.42 / +5.52 |
| **0.12 (the default now)** | 2.99 / +1.99 | 7.48 / +4.57 |

At 0.04 the avatar's clean skin lay over the person's own roots; 0.12
of a forehead is six to ten pixels, still "slightly below where the hair
starts". His eye may put it back (a portrait option).

The settled build's bands (the cast tone, the clearance 0.12): the
woman's forehead 2.99 / +1.98, her jaw 10.29 / +8.73 (the untouched
build 2.70 / +1.50 and 9.88 / +5.77); his forehead 7.47 / +4.41, his jaw
20.32 / +12.99 (7.51 / +4.42 and 21.70 / +13.17). The band reads the ten
pixels inside the mask's edge; what the tone changes is the face's
inside, which the zooms show and the band cannot.

### Built 3: the head guide in Calibrate me (`sha256.ui.head_guide`)

Where the head should be, read off the four good sessions of 2026-09-12:

| session | face (of the frame's height) | centre |
|---|---|---|
| his (004157) | 0.547 | 0.50 x 0.51 |
| the friend's two (185642, 190114) | 0.472, 0.474 | 0.51 x 0.43, 0.49 x 0.42 |
| the woman's (194044) | 0.435 | 0.57 x 0.42 |

So the target is a mesh 0.48 of the frame's height (0.80..1.25 of it
allowed: farther is "Come closer", nearer "Move back"), centred within
0.12 of (0.50, 0.46) ("Put your head in the oval"), frontal within 15
degrees of yaw and 20 of pitch ("Face the camera"); the oval drawn is a
head, 0.62 by 0.44 of the frame's height. The preview draws it over
either view in the fit's colour -- amber with the one line of what to
fix, green with "Good. Hold still." -- and the window's placing phase
makes the session at the press but starts its clock only once the fit
has been good for a second; the line is said once, the steps follow.
The fit is a pure function (`fit_head`), tested without a window.

### Tests, docs, the frame rate

`test_engine_skin_tone.py` (the cells, the tone, the EMA, the paste at
0 and 1, the GPU against numpy), `test_engine_forehead_stretch.py` (the
plan on the day's faces, the stretch in the paste, the GPU against the
CPU, the settle with a taller forehead), `test_head_guide.py`, the UI
tests adapted (the placing phase, the oval's pixels, the guide riding
with the frame); `test_engine_hairline`'s settle tests to the new rule.
**1394 passed.** `jaw_band.py`'s capture re-render read the engine's
four-part `last` again (day 20 had broken it). CHANGELOG, USER_GUIDE
step 3, the in-app instructions (section 2 step 4, the menu, the status
line), SYSTEM_ARCHITECTURE's paste sentence.

**The frame rate, honestly.** The tone's own cost is measured: a reading
2.0 ms of the render thread's CPU, the apply 0.22 ms, so 0.65 ms a frame
at one reading in four (`scratch`, 200 calls each); the stretch is one
`where` on the grid. The whole-loop numbers of the evening are muddier:
the morning's run of the first tone build read the render thread at 30.7
ms p50 (threads, realtime, his clip; 30.5 before the day) with the
tracker at 8.6 ms; the evening's runs of the settled build read 32-35 ms
and 26-28 fps with the tracker in its slow mode (12.6-13.1 ms, the
bimodal cost noted on 2026-10-08) and every stage, the GPU graph
included, a tenth slower -- the machine's state, not a stage of ours.
`recordings/budget/final2-*.json`, `cast-alone-*.json`. A clean
measurement on a fresh machine state is owed; his fps counter on a live
call is the real number.

### What he will see, and what is open

Start, any face: the log says `skin tone matched -- gain [...], offset
[...]` at frame 60 and `forehead settled -- the portrait's oval to X of
its forehead, the driver's to Y of theirs, the render stretched xZ`.
Calibrate me: the oval, amber then green, then the steps. Open, his
eye: the stretch on his own forehead (1.13), the clearance 0.12 under
his hair, the tone on his face in his room; the two friends' recordings
(Calibrate me's own session folder is the easy ask) for the real proof;
then his word on the commits, the merge of main into the clone's
`tensorrt` branch (one touchpoint: the live hairline read, which the
clone cut to a window) and the two packages.

## Day 22 (2026-10-15): the calibration "that will surely work on anyone" -- the width fit (always on) and the slow session

His order, the night before: the calibration must "simply put itself in
the person's face shape"; the guided process SLOW and dead simple ("on
the count of three, laugh"; a text to read for twenty seconds); both in
main and in the tensorrt clone; no glasses, no big beards among the
users. And the day's goal past that: expressions 1:1 (a 40% blink is a
40% blink, a slight smile a slight smile), and the mouth's SHAPE on "O"
like the person's own. The plan, in order: (1) the always-on width fit,
(2) the session his way, (3) the 1:1 curves per expression, (4) the
mouth's rounding, (5) the merge, (6) the packages and a friend's sitting.
Parts 1 and 2 were built today; nothing committed, nothing pushed.

### Part 1, built: the face's width fitted to the person's

**The measurement first** (`scripts/probes/face_width.py`: the rim's
extent along the face's own horizontal over the face height, the
tracker's 2D points, frontal frames only, the median):

| face | rim width / height | against his portrait (0.822) |
|---|---|---|
| his 2026-09-12 00:41 session (845 frontal frames) | 0.789 | **x0.959** |
| the friend's 19:40 session (1911) | 0.815 | x0.991 |
| the woman's capture 200754 (265) | 0.824 | x1.002 |
| his day-9 clip `driving.mkv`, read live by the engine | -- | **x0.935** |

The 2D rim is NOT yaw-invariant: 5-7% per 10 degrees of yaw, and
one-sided (the index peaks at a negative yaw, the tracker's zero being
off the geometric frontal), so the live reading takes still frames
within 8 degrees of yaw and 10 of pitch, like the hairline's. A 3D
measure from the mesh's z was tried and is no better (the side points'
3D distance drifts 6-9% over the same yaws). The person's resting yaw
is their habit and the render turns with it, so a reading at rest
against the portrait is what the paste needs.

**The fit** (engine): once the anchor is frozen, the median ratio of the
still frontal frames (or `driver_face_width` from the profile, when
Calibrate me has written one -- part 3's job) is the horizontal scale
of the render and of the portrait's own oval, about the portrait rim's
centre column, within 0.85..1.15, times `face_width_fit` (0..1, 1). The
render is resampled once right after the network (`_squeeze_face_gpu`:
one `grid_sample` over the 512 crop; `_squeeze_face` on the CPU path,
one `warpAffine`); the portrait's points are squeezed for the masks
(`mask_points`) and the mask rebuilt at the settle; the driver's oval,
the drive, the network and the crop are untouched. A difference under
0.5% is not applied at all. `frame_readings["face_width"]` is the
frame's own ratio, `face_width_scale` the scale in force.

**The proofs** (`frame_budget.py --deterministic --option skin_match=0`,
`driving.mkv` 300..900, 601 frames, against the untouched engine's run):

| run | result |
|---|---|
| `face_width_fit=0` | **601 of 601 byte-identical** |
| the default (x0.935 on this clip) | 28 identical (the frames before the settle), 573 differ |
| the stills at 450/600/800, fit on against off | 12.4% of the frame's pixels differ, p99 of them 38 levels, mean 0.6 -- the whole paste box moved inward |

Looked at (his face, the zooms deleted): the mouth and the moustache sit
narrower, nearer his own proportions; the cheeks' join is under his
beard on this clip either way. His live look is the real judge.

Tests: `tests/unit/test_engine_width_fit.py` (the options, the scale's
bounds and dead band, the geometry, the CPU squeeze, the GPU squeeze
against the CPU one within a grey level, the settle from the samples /
the profile / too few samples, the readings). Full suite 1452 passed.

### Part 2, built (Opus 5.5 from the spec; verified by the suite and a read of the diff): the session his way

`calibration/script.py`: three phases per step -- the instruction (3 s,
on screen and spoken), the count "3", "2", "1" (one a second, drawn
large, spoken), the measured window; "On three: ..." wording; the new
steps `eyes_half`, `open_small`, `read` (20 s, `READING_TEXT`, English,
full of o/a/e/m sounds; replaces `talk`), `smile_small`, `laugh`,
`brows_half` -- the graded levels part 3 measures the 1:1 curves from;
186 s in all. `session.py`: the green oval judged every frame
(`guide_ok` in the samples and `tracking.npz`; the clock never pauses);
a step whose movement was too small (the `*_MIN_RANGE` thresholds, half
for the "a little" steps, a smile +5% of the resting width) is repeated
once, right after it, "Once more: ...". `main_window.py`: the card with
the count, the text, the amber guide line. `measure.py`: `read` read as
`talk` (one change). The name is still asked at the end.

### His first look (18:00-18:14), three presses, and what each was

Screenshots first: a pale smeared patch at the top of the paste on his
forehead; at a turn to his right his own brow and half an eye beside the
avatar's (to the left it worked); a band under the chin; and "lower than
360p would be good". Then three "That looked wrong" presses (181314: hands
on the head pulling the hair back, frontal-ish; 181328: head turned to his
right and down; 181432: a 40-degree turn -- a capture's `pose` is stored
(yaw, pitch, roll), and reading it as (pitch, yaw, roll) cost an hour of
"tilt" rules that the press never exercised). The replay could not plan
the forehead or the width from these (turned frames read no hairline, and
the capture did not carry the settle) -- `--option driver_hairline=1.71
--option driver_face_width=0.969`, the live log's own values, stood in,
and the capture carries them from now on (`settle_hairline`,
`settle_width`).

| fault | measured | built |
|---|---|---|
| the forehead patch | with `edge_match=0` the patch is a DARK streak: the avatar's hair roots. On the portrait crop the row's p5 leaves the skin by 7% at 1.40 F, 20% at 1.45; the reader's hairline 1.52, the clean top 1.37; the mask's feather tail reaches +0.04 F past the reach (alpha 0.01 at 1.41) | `_roots_onset` at load; the clean top = min(hairline - margin, roots - 0.10), a source row; the stretch carries it. Replay 181314: the patch gone, the join shows his own shine above it |
| the hairline raised by his hand | the settle is once; `hairline_last` was read every 6th frame and used by nothing | the forehead follows: 15 readings kept, a new plan past 0.15 F of drift; the width settles once |
| the band under the chin (a 40-degree turn) | NOT the photograph's background (the composite's last paste rows read (159,155,148) over his (38,46,66); nothing within 40 of the background (243,243,237)): the portrait's chin underside, a light shadow, over his dark one, drawn by the reach under the chin (`mask_chin_reach` 0.04 = the blob's 12 rows) | the chin reach fades out with the turn or the tilt (12..20 deg from the rest pose); the jaw shrink by the pose (0.06) was tried and dropped: the join went 26 px up, his stubble under the avatar's chin (the day-12 ledge); the guard on a tilt too; the chin's own keep for a real background blob (every frame, no download) |
| the far eye at a turn to the right | not reproduced: 181328 at widths 1.0 / 0.969 / 0.9 covers his eye corner; the screenshot was likely mid-motion | nothing; his HELD press in that pose is the next measurement |
| lower than 360p | -- | 240p (240 rows, JPEG 30) |

Proof (`frame_budget.py --deterministic --option skin_match=0`, 601
frames, the final engine against the untouched one): 26 identical, 575
differ -- the 26 are the frames before the settle, less frames 300 and
317 where the chin keep found background under the chin; `face_width_fit=0`
the same 26/575 now (the roots' clean top moves everyone's join). Unit
suite 1468 passed.

### His second look (19:00-19:10): "left and right are mixed up", and his first session on the slow script

His words: the fix went to the opposite side's turn, it was fine before;
and the session's turn (and some steps) give too little time -- five
seconds more on every step. The seconds: done (every measured window +5,
about 266 s in all; his own turn window had ended with his head still at
-39 degrees). The sides: nothing built today is side-specific -- the
width fit is about the rim's centre, the roots are the portrait's, the
chin reach fades on |yaw - rest yaw| either way. Measured on his own
session's turn step (frames 3080/3096 at +30/+51 yaw -- his LEFT, the
face toward the image's right on the unmirrored camera -- and 3115/3124 at
-25/-39, his RIGHT), the untouched engine against today's, deterministic,
`frame_budget --stills`: on both sides the two builds draw the same face
to the eye (197 of 225 frames differ by the day's paste changes, none of
them a side). What IS visible at his RIGHT turn, in BOTH builds: the
avatar's eyes pushed to the far corner, the whites showing -- the gaze
drive over-driven at a turn -- and that is older than today (day 21's
build he never saw live; day 20 was "PERFECT" on the far brow, not on
the eyes). His new profile (`roy`, blink 1.00x, gaze 0.99x, brow 0.93x,
talk 1.29x -> expression_gain 1.163, driver_hairline 1.819) does not
touch it either (gaze_gain kept). Not reproduced by a replay yet: a
HELD press at his right turn is still the measurement to take; the
capture now carries the settle, so the replay will be the session.
Stills deleted.

### His held press at the right turn (19:18): the avatar turns less than he does

The press (191804, tracker yaw -42, held still, the capture carrying the
settle so the replay IS the session: 5659 px differ from `seen.png`):
no sliver of his brow at the far side -- but at the NEAR side his own eye
shows through the paste's feather beside the avatar's, 40 px inward of
his. The avatar's head is turned LESS than his, measured by the output
tracker on the renders:

| frame | him | avatar |
|---|---|---|
| the held press | -45.1 | -37.0 |
| session f3115 / f3124 (his right) | -36.9 / -42.9 | -29.6 / -35.8 |
| session f3070 (his left) | +37.6 | +36.9 |
| session f3080 / f3096 (his left) | +54.3 / +54.9 | +40.0 / +36.9 |

1:1 to about 35 degrees, then it saturates. The clamp is not the limit
(hold_yaw_degrees 45, 60, 90 all give -37 on the press): the network
answers the extractor's delta with less turn past ~35. Built, both OFF
by default (bit-identical): `turn_boost` (the yaw delta past 30 degrees
driven harder: delta + boost x (|delta| - 30), before the clamp) and
`turn_cap_degrees` (the clamp's own yaw cap, apart from hold_yaw_degrees
which stays the expression fade's start -- day 17's smeared mouth at -48
must keep fading). Measured on the press and the session:

| run | the press (-45) | session right -37 / -43 | session left +54 / +55 |
|---|---|---|---|
| boost 0.5, cap 45 | -38 | -32.9 / -37.5 | 39.9 / 36.9 (capped) |
| boost 1.0, cap 75 | **-44.4** (the near eye on his) | -35.2 / **-45.4** | 64.2 / BROKEN (a near-profile, the portrait's hair smeared over the far cheek: the network past a delta of ~70) |
| boost 1.5, cap 75 | -49.4 | -- | -- |

So the boost wants a cap the network can carry: 56 of delta was whole,
~75 was not. Measured with the cap at 60 and the boost at 1.0 on the
session: the right -34.9 / -45.5 at his -36.9 / -42.9, the LEFT +50.4 /
+47.0 at his +54.3 / +54.9 -- whole, the far cheek clean (the tracker
reads the drawn head pitched -11..-15 there, a lean the eye half sees).
SET IN HIS AVATAR'S metadata.json: `turn_boost` 1.0, `turn_cap_degrees`
60; the engine defaults stay off (the woman's avatar untouched until a
recording of hers at a turn says the same). Why the left "worked" for him: at his left turn the hair
falls over the near temple and his own eye under the feather is dark on
dark; at the right turn the temple is bare.

### His goal, stated (19:40): the face must sit at larger turns, no border, no hair above the forehead

His words after the boost: on the side he turns to, his eye, brow, cheek
and lip still show -- both sides; "this is our goal: fix it in every
variant; the avatar must sit so that no border-like line gives it away;
sometimes a little of the avatar's hair shows above the forehead".

**The turn, measured again.** The boost was live (the avatar reloaded at
19:38 with turn_boost 1.0 / cap 60) and the press's replay IS the
session. What remains at -45: the drawn face's near (profile) edge sits
inside his by 6 px at the median and 19 px at the brow row (the
prototype `scripts/probes/outline_fit.py`, the drawn silhouette = the
render's non-background, non-border columns, against the paste's own
extent per row). Between the two edges the guard keeps nothing (the
render is the photograph's background) and his skin shows; within the
feather his eye shines through beside the avatar's.

**Built: the outline fit** (`outline_fit`, default on; the paste only).
Per row between the brows and the chin, the render is scaled
horizontally about the paste's far edge so its near edge lands on the
paste's near edge: scale within 0.75..1.35, smoothed over 21 rows, faded
over 40 rows beyond, eased over frames (half the previous map), weighted
by the room's turn gate (nothing within 12 degrees of the rest yaw).
On the device: one (H,) download per turned frame (the near edge per
row by an argmax, looked for within 48 px outside the paste's own near
edge -- the photograph's edge against the warp's black border read as
"drawn" 45 px into the crop before the window); the map folded into the
GPU grid's x and the CPU remap's. The prototype on the press: the
avatar's near eye moved onto his, the ghost of his eye at the feather
went with it.

Two faults on the first engine run, both measured on the session's turn
frames: (1) a thin WHITE line along the near cheek on the moving turn --
the guard's keep (the background cut) was computed on the unwarped
render and multiplied into the mask, while the render was sampled
through the map, so the background the warp pulled in at the profile
was not cut; the keep now goes through the same per-row map. (2) At his
left profile (+51) the avatar's nose drew large: his nose stands out
further than the avatar's, and widening the near side to his outline
(18% there) widened the nose with it; the scale is bounded to 14% of
widening now (0.88..1.15), the rest of the gap left to the keep -- his
own nose tip at the profile, not the avatar's blown up. The maps on the
settled frames: right turn gaps 3-13 px, scales 0.91-1.05; left 6-25 px,
0.85-0.93 before the bound.

**Verified** (the keep through the map, the bounds): his held press --
the avatar's near eye on his, no ghost; the session's turn frames both
sides -- the white line gone, the nose at the left profile the avatar's
own size, the profile on his. Frontal: 601 of 601 identical to the
paste-only build (the fit is gated by the turn). Still open from his
goal: the avatar's HAIR at the top of the far side at a turn (the paste
reaches his hairline there and the render has the portrait's hair --
the drawn skin's own top edge should bound it, as the near edge now
bounds the side); the seam's invisibility (texture / grain at the join);
and his live look at all of today.

### His third look (20:17): "now it is broken at the side, the face is lost sooner -- fix it or go back to the pushed version"

No error and no lost face in the log after 20:17; the fault is visual.
The whole turn of his session rendered through the two-thread pipeline
(`frame_budget --mode threads --pace lockstep`, frames 2950..3130,
stills every 10) showed it: on the far side of every turn past ~25
degrees, and worst coming back from +50 (frame 3100), a SMEAR of the
avatar's hair and temple over his forehead and far cheek. The cause is
two days meeting: day 21 lifted the person's oval to THEIR hairline
(1.7-1.8 of his forehead, above the portrait's clean skin at 1.3-1.5),
and day 20's room pastes the person's oval beyond the portrait's at a
turn wherever the render is not the background -- the render's hair
included. Frontal, the room is off and nothing showed; the day-20
build never had the lifted oval, so it never had the smear. Built: the
room reaches no higher than 0.2 of a forehead above the brow line
(`_ROOM_BROW_MARGIN`). Also the outline fit widens only (the first
build narrowed the render at moderate turns, where the drawn edge lies
outside the oval's eroded edge, uncovering him) and fits to the oval's
hull. The strip after: the smear gone both sides, the face on his at
every frame from -4 through +50 and back to -47. Frontal 601 frames
identical to the previous final. Suite 1477.

### His fourth look (20:29-20:30): the nose at a turn

Two presses, +32 and -32. Turning left the avatar's nose lost its shape
(a smooth blob); turning right a white sheen lay on his nose. Replayed
with `outline_fit` 1 and 0 (`recordings/day22-nose-*`, deleted): with
the fit off the nose is the avatar's own on both; with it on, the blob
and the sheen. On the nose's rows the near edge the fit reads is the
avatar's nose tip and the target is his -- a difference of SHAPE, not of
turn -- so the fit stretched the nose and, at the tip, sampled the
photograph's background. Built: the band from under the eyes
(the eye corners' row + 4) to under the nostrils (landmark 2 + 4) reads
the identity, ramped over 12 rows; the cheek, eye and brow rows above
and the jaw below stay fitted. The replays of the two presses and of the
held -45 press are the proof (the near eye still covered at -45).

### His fifth look (20:50): dark blotches at the join while turning -- parked

His screenshot (near frontal, turning left to right): dark, shadow-like
blotches at the paste's border -- the forehead's top at the hairline on
both sides, and the right temple -- never in the middle of the face. His
dots: the avatar's skin light and pink, a thin darker line (the join),
then his own skin, darker. Seen and confirmed at zoom. The analysis (not
measured yet): his face is lit from the front with nothing from the
sides, so his temple and hairline sit in shadow while the render is lit
evenly; the edge match's plane fits the whole face's mean and the
residual is read from a 10 px band on a quarter-scale grid, neither of
which follows a side-shadow gradient at the edge, and the correction's
inertia (day 11) drags the previous frames' field over a moving head --
hence "in moments while turning". A phone light from the side did not
help much. His decision: at 240p and 360p it does not show, so it is
bypassed for now and he will come back to it.

The plan for then, cheapest first: (1) a wider feather at the sides and
the top (30-40 px for 16); (2) the edge residual on a finer grid near
the edge; (3) a gradient-domain (Poisson) blend at the join, the "heavy
option" of day 21 -- the avatar's interior brightness following his edge
brightness smoothly; (4) light from the side on his end. Measured on the
session's turn frames with `jaw_band.py`'s forehead band (and a temple
band to add), (1) and (2) by replay first.

### What is next (parts 3-6)

3. **The 1:1 curves**: per expression, the avatar's response at the
   session's levels (eyes half / shut, mouth a little / wide, smile
   small / closed / teeth, brows a little / high) read by the output
   tracker on the render, against the person's -- a few-point curve per
   expression in the profile in place of one gain, applied in the drive
   with the replay proof; also `driver_face_width` written by
   `measure.py` from `frame_readings["face_width"]` over the still step.
   Honest limit: the output tracker's own noise is a few %, so "1:1"
   means within ~5%.
4. **The mouth's rounding on "O"**: on the read step, the output
   tracker's mouth width AND opening per frame against his; first the
   question whether the drive carries the corners pulled in at all.
5. The merge into the tensorrt clone (the calibration code is shared,
   no TensorRT touchpoint; the engine's squeeze is torch on either).
6. Both packages; a friend sits for the proof.

## Day 23 (2026-10-16): his ideas for the avatar -- the skin colour from the camera, the shaved head

He recorded his ideas so as not to lose them and pasted the transcript.
Five points; each answered against the code before anything was built,
then built on his answers (Gemini or ChatGPT make the photographs; the
application does NOT call a generator -- the button gives the colour and
the prompt carries it; he will check yesterday's +5 s himself).

### His five ideas, and what each is worth

1. **Five seconds more per calibration task.** Done on day 22 evening on
   his word (266 s). If the current build is still tight, another 5 s is
   one line.
2. **"The skin colour and the lighting must not change" in the avatar's
   prompts.** The old 22-variant prompts (`AVATAR_PROMPTS.txt`, the
   photograph engine) already say it. The neural avatar is ONE
   photograph, made from words, and its prompt described the person in
   a bracketed line -- so the idea lands exactly there: the placeholder
   becomes the MEASURED colour. The honest limit: the avatar's colour is
   the starting point, not the end -- the room's light and the camera's
   white balance differ per call, so day 21's live `skin_match` stays;
   but the nearer the avatar's colour, the less the cast has to do
   (bounded 0.7..1.4; a large cast moves the irises and brows too). The
   blotches of his fifth look are a side-lit gradient and texture, not
   colour -- no prompt fixes those.
3. **The same avatar, completely bald.** The strongest idea. The three
   hair faults of days 21-22 -- the pale patch (the roots), the smear over
   the far cheek at a turn, the hair above the forehead -- all come from
   hair the portrait HAS. The engine joins the avatar's face to the
   person's forehead under THEIR hairline and keeps their hair, so the
   avatar's hair is never drawn on purpose, only by accident. With a
   shaved portrait nothing of it can show, and the forehead plan has the
   whole scalp as clean skin: `_forehead_plan` with no hairline read uses
   the ceiling 1.6 F (less the margin 0.15), the roots reader has
   nothing to find. His refinement: make the face WITH hair first (the
   generator makes a better face from words that way) and shave it
   after, from that photograph, so it stays the same face; the
   parted-lips variant from the shaved one.
4. **A button that reads the person's skin colour and fills the
   prompt.** The pieces existed: the engine reads the room's skin mean
   every fourth frame for the tone match; the Add avatar window already
   copies prompts; `check_avatar_photos.py` reads a photograph's skin.
   Built, see below. The reading is the CAMERA's, light and white
   balance included -- the right thing to match -- so it must be taken
   in the light the calls will have.
5. **The face's measurements in the prompt, so the avatar's eyes, mouth
   and forehead match the person's.** As numbers: no -- generators do
   not follow numeric proportions ("eye spacing 0.46 of the face width").
   As WORDS ("a broad forehead, wide-set eyes, a narrow jaw, full lips")
   they follow roughly. The realistic form: the importer and the
   calibration already measure the proportions (his face 0.96 of his
   portrait's width; the eye spacing, nose length, mouth width, jaw
   width of the checker) -> words in the prompt -> three or four
   candidates generated -> the importer scores each against the person's
   numbers and ranks them. The gain is real (the width fit and the
   forehead stretch, both artefact sources, shrink). What no prompt
   gives: the expressions 1:1 -- that stays part 3 of day 22 (the
   response curves). NOT BUILT -- his word next.

### Built (uncommitted with the day-22 tree)

- `sha256/avatar/skin_tone.py`: `read_skin` (both cheeks, points 50/280,
  and the forehead's middle, 151; patches of a twentieth of the face
  width; cells under 0.55 of the patch's median lightness and above its
  95th percentile dropped; per-channel medians), `SkinSampler` (the
  median over 30 frames), `SkinReading` (sRGB hex; CIELAB; the ITA
  angle `atan((L-50)/b)` into the bands very light / light /
  light-medium / medium (tan) / brown / deep brown; the undertone from
  the hue angle: warm above 60 deg, cool under 45, neutral between --
  rough, words for a prompt), `fill_prompt` (every `[your skin color]`),
  `compare` (CIE76 distance; close <= 8, noticeably different <= 16, far
  beyond: "generate again").
- `photo_guide.py`: three prompts -- the face with hair (from words, the
  skin line "Render the skin in exactly this colour, under neutral white
  balance"), the SAME face shaved (attach prompt 1's photograph; "no hair
  at all on the scalp, no stubble on the scalp, no hairline anywhere; the
  skin of the scalp the same colour as the forehead"), the shaved one
  with the lips barely parted (attach prompt 2's). The guide says which
  one to add (the shaved one) and why.
- `photo_guide_dialog.py`: "Read my skin from the camera". With the
  pipeline running the window hands in `runner.latest` and the head
  guide; the button polls at 30 Hz, reads only frames whose head sits in
  the guide (the placing phase's gate: close enough, centred, facing),
  30 readings or 10 s; then the whole text re-rendered with the words in
  place, the Copy buttons copy the filled prompts ("Copied, with your
  skin."). Without the camera: "Start the camera first". Nothing steady
  in 10 s: told what to do (half the frames still give a reading).
- `importer.py`: `PortraitReport.skin` by the same reader on the
  photograph, in the report's text; the window's "added" message
  compares it with the dialog's reading when one was taken.
- Tests: `test_skin_tone.py` (23), the window tests (the three prompts,
  the reading from a synthetic frame + 478 points in the guide, the
  no-camera and the timeout paths), the importer's describe. Docs:
  CHANGELOG, USER_GUIDE, instructions.py.

### What to check when his first shaved avatar exists

- The importer's hairline reader on a bald scalp (`_hairline_fraction`):
  None is the good answer (the engine takes the ceiling); a false
  "hairline" at the scalp's edge against the background would cap the
  forehead's clean skin -- read the import report's line.
- The roots reader at load (`_roots_onset`): on a bald scalp it must
  find nothing; if the scalp's shading at the top reads as "roots", the
  log at load says so and the plan must ignore it.
- The forehead stretch: with the whole scalp as source, the stretch
  should come out near 1.0 for him (his hairline 1.82 F of his forehead;
  the portrait's clean top now ~1.45 F) -- read `settle` in the log.
- The hair's smear at turns (his third look of day 22): gone by
  construction; confirm on the session's turn strip.
- The colour: the import report's distance against his reading; then
  the tone log at frame 60 (the cast should sit nearer 1.0 than with the
  old portrait).

### His first shaved avatar (02:21-02:33): the switch that kills the pipeline, and the brows 20 px low

**The load failure first.** "engine failed to load 'baldnew': CUDA error:
operation failed due to a previous error during capture", then the same
for woman_person. The log has it on 2026-09-05 and 2026-09-14 too: an
avatar chosen WHILE THE PIPELINE RUNS. The UI thread loads the new avatar
and records its CUDA graph (`_capture`, the default global capture mode);
the render thread keeps drawing the old avatar, and its next GPU call
("operation not permitted when stream is capturing", today at
`_drawn_near_edge_gpu`'s `torch.as_tensor`) kills the render loop AND
invalidates the recording; the process's graph recording stays poisoned,
so every later load fails too. Never worked; an avatar chosen before
Start (or after Stop) loads fine. THE FIX, on his word, not built: one
lock in the engine shared by `animate` and `load_avatar`/`unload_avatar`,
so the render thread waits the one or two seconds of the load (pixels
untouched). The workaround told him: close, open, choose, then Start.

**His press (02:33, `baldnew2`, yaw -3.5 / pitch -5.6).** His words: "my
brow and its brow are apart; its face is probably shorter; its brow is
lower than mine". Measured on the capture (MediaPipe on camera.png and
seen.png, frame rows; `scripts/probes/face_rows.py` re-renders the same
numbers from the capture, "fit off" below):

| row | his | avatar (fit off) | avatar, fit x1.047 |
|---|---|---|---|
| brow | 278 | 298 (+20) | 286 (+8) |
| eye | 318 | 327 (+9) | 317 (-1) |
| nose | 412 | 409 (-3) | 403 (-9) |
| mouth | 453 | 451 (-3) | 446 (-7) |
| chin | 530 | 525 (-6) | 523 (-7) |

The lower face on his, the upper face lower. The crop is sized by the
mesh's face HEIGHT (forehead top to chin), so the two faces share a
height; a portrait whose forehead takes more of it puts its brows and
eyes lower at the same chin. `scripts/probes/face_height.py`: his brow
top to chin 0.843 of the face height over 300 still frames, the
portrait's 0.805 -> x1.047 (his own avatar x1.032, the woman's x1.017);
the eyes x0.997 -- the avatar's own brows sit close to its eyes (0.109
of the face height against his 0.138; the photograph's). Checked that
the offset is the pair's geometry and not the moment's anchor: the
capture re-rendered with an anchor learned from its own frame instead of
the session's gives the same rows (+19 / +9 / -2 / -4 / -4).

**Built: the height fit** (`_brow_chin`, `_height_scale`, `_lift_points`;
`face_height_fit` 0..1 default 1, bounds 0.90..1.10, `driver_face_height`
from a profile; readings `face_height`, `face_height_scale`; the capture
carries `settle_height`). The width fit's vertical twin: the ratio of
the driver's brow-top-to-chin to the portrait's, still frontal frames,
the median, settled with the width; the render and the portrait's points
scaled vertically about the portrait's CHIN ROW in the same resample as
the width (`_squeeze_face` / `_squeeze_face_gpu` take both axes); the
forehead's plan reads the LIFTED brow and forehead (its stretch row, its
height), since that is where the render's rows are. The off path is the
old one to the bit by construction (the resample skipped when both
scales are 1, the plan fed the unlifted points). The basis is the BROWS,
not the eyes: the brow row is where the join starts and where his own
brow shows beside the avatar's; the eyes then land within a few px
either way (this pair: x1.047 puts the eyes on his and the brows 8 px
low -- the rest is the photograph's own brow; his own avatar on the same
capture: fit off brow +10 / eye +2.5, fit x1.032 brow +2 / eye -4). The
nose and mouth rise 4-9 px above his under the paste (unseen), the chin
2-7 px (watch the jaw join live). Tests `test_engine_height_fit.py`
(15); suite 1513. A replay cannot read the fit (its frames are one frame
repeated; the still gate gives one sample): give it
`--option driver_face_height=...` from face_height.py.

**Built: the shaved forehead.** The roots reader DID find "roots" on the
shaved scalp (0.90 of the forehead -- the scalp's shading) and capped the
clean skin at 0.93, stretched x1.25, so the avatar's forehead ended ~25
px under his hairline with his own skin above it. A portrait with no
hairline read now ignores the roots reader: the plan takes 1.45 of its
forehead (the ceiling 1.6 less the margin), no stretch, and with the
lift the portrait's forehead (47 -> 49 crop px) reaches his oval's 1.70
of his (38-40 px). His word was "the forehead is cut too much, enlarge
it": this is that.

**NOT built (his word: "it's the photo's fault"):** the brow band lift
for the last 8 px, and the prompt line "eyebrows set high above the
eyes". The lock for the avatar switch waits for his word too.

### His look at the height fit (03:1x): "fixed" -- and the turn: "on 09-18 the face followed me further"

His words: on the 09-18 push, looking to the sides sat very well; now
the face does not follow as far when he turns; fix only that and it is
ideal. Bisected by measurement, not memory: the 09-18 push (ee5d837) in
a worktree with the models and the portrait linked in, and the SAME
movement through both trees -- his session's turn step
(calibration-20261015-190122, frames 3024-3200, left to +51 then right
to -52), the output read by MediaPipe every fourth frame
(`scripts/probes/turn_motion.py`; a held press: `turn_compare.py`).

| his yaw | 09-18 drawn | today, boost 1.0 cap 60 | today, boost off |
|---|---|---|---|
| +24 | +28 | +31 | +29 |
| +30 | +33 | +42 | +35 |
| +35 | +37 | +54 | +37 |
| +40 | +37 | +54 | +37 |
| +45 | +39 | face lost | +39 |
| +50 | +37 | +47..+49 | +40 |
| -39 | -35 | -46 | -34 |
| -46 | -36 | -57 | -36 |
| -52 | -35 | -55 | -38 |

The 09-18 drive: 1:1 to about 30 degrees, saturating at 37, monotonic,
the pitch within a degree or two of his. Day 22's `turn_boost` 1.0 +
`turn_cap_degrees` 60 (his metadata, copied into every avatar imported
since -- baldnew, baldnew2): the avatar runs AHEAD of him at 30-40
degrees (x1.4-1.5), its pitch dips to -9 at his -5, the tracker loses
the drawn face at 45, and at 50 it has come BACK to 48 -- a turn that
overshoots and retreats while he keeps turning, which is what "does not
follow me further" looked like. Day 22 measured the endpoints only (his
-43/-55) and missed the band. On the held press 191804 (his -42): 09-18
-36 / pitch -5, today's boost -46 / pitch -9, today's boost off -37 /
pitch -6; the height fit off or on changes none of it.

DONE: the two keys removed from the three avatars' metadata (the engine
defaults were off all along); today's tree without them reproduces the
09-18 curve on the same frames, with the shaved avatar too (37-40 at
45-50, the pitch within two degrees). The paste at a 50-degree turn
reaches past his rim by 46-58 px today against 49-55 on 09-18 (the eye
row), the brow row the same -- day 22's outline fit and room stay. The
worktree and its junctions removed after the measurement.

If he wants more range than the network's own 37 later: a gentle gain
measured over the WHOLE band with turn_motion.py (monotonic or nothing),
not the endpoints.

### His two presses (03:25): "the brow and eye stretch when I turn right; my eye and lashes remain when I turn left"

Presses 032522 (his yaw -23, a turn to his right) and 032542 (+34, to
his left), both on his own avatar, after the boost's removal. Measured
on each (MediaPipe on camera.png and on the re-rendered output; the
paste's own alpha at his eye points: `scripts/probes/eye_alpha.py`,
which takes `--src` so the 09-18 worktree answers the same questions):

**The right turn.** The avatar turns 0.68 of his (-15.6 at -22.6) -- the
same on the 09-18 tree (-15.0) and on the shaved avatar (-25.9 at -31.6
on the session): his side, not the portrait's. Its eyes sit 5-11 px to
the image's right of his. Day 22's outline fit then scaled every row
from the brows down toward the paste's edge: the brow row wanted 19 px,
the eye row 6, and the brow smeared sideways (the zoom shows it; with
`outline_fit=0` it is gone). FIXED: the fitted band starts at the eyes'
top (points 159/386 less 4 rows), not the brows; the 40-row fade above
leaves the brow little of the eyes' scale. A per-side yaw gain was built
and measured (`turn_gain_right` 1.3: the session's right turn 0.96-0.98
of his up to 39 degrees then the network's own saturation, monotonic;
the press -20.7 at -22.6, the eyes within 5 px of his; 1.5 overshoots)
-- but at 1.3 the avatar's far eye corner renders darker (the network's
own; faint on the 09-18 tree, visible at 1.3), so it is NOT in his
profile: his word on the 09-18 turns was good, and this is that drive.
One line in configs/drivers/roy.json turns it on. (A first build applied
the gain to the wrong side: the extractor's yaw is signed the other way
round from the tracker's -- a delta under 0 is his left -- found by
setting one gain and watching which side moved on turn_motion.py.)

**The left turn.** The avatar turns 1:1 (+34.9 at +34.4) and its eyes
sit within 4 px of his; what remained was his far eye's OUTER CORNER and
lashes, in the paste's feather beyond the drawn face's edge: alpha 0.88
at the corner, 0.59 five px out, 0.08 ten; the drawn face ends 5-10 px
inside his lashes there (the avatar's face is narrower than his at the
far eye at this angle), the same on the 09-18 tree (0.98 / 0.66 / 0.10),
which he liked at the time. The width fit's 3% squeeze of the portrait's
oval took 4-5 px more off the corner (pasted to 768 against 773). The
room (day 20) pastes the driver's oval only where the render is the
drawn face, and there the render is the photograph's grey background.
Tried: ONE scale for the outline fit's whole band (the median) -- it
smeared nothing, and did nothing for the eye either (the band's median
is 1); dropped. BUILT: the profile fill (`profile_fill_px`, default 12,
0 off; `_profile_fill_np` / `_profile_fill_gpu` in `_fit_outline`, before
the map): at a turn, on every row the background columns within 12 px
beyond the drawn face's profile edge take the edge column's own colour
(the avatar's cheek skin, read 3 columns inside the edge, and the edge's
own 3 columns overwritten with it -- `_PROFILE_FILL_INSET`: the edge
pixels are the network's pale blend with the grey background; carried
out as they were they drew a pale streak down the far temple, left in
place under the strip a pale jagged line along it), the width ramped
with the turn gate, so the
guard sees drawn face there and the room pastes it, inside the driver's
oval as before; frontal frames untouched (gate 0 -> the same object).
Measured on the press: the lash box (18x26 px at his far corner) mean
luminance 70 before, 92 after with the inset (119 with the pale edge
colour; his lashes 68, the avatar's skin ~150); the outer corner now
pasted; a faint dark remnant stays at 775-780 where the feather is
0.59 -> 0.08. On the session's turn the paste reaches 13
px further on the profile side at 50 degrees and nothing else moved.
NEXT if his eye still sees the remnant: the feather on the profile side
narrowed or pushed outward at a turn (the day-20 `_PROFILE_PUSH`), so
the filled strip lands at full alpha. Tests: test_engine_turn_gain.py
(7: the gains, the fill CPU/GPU, a row without an edge).

## Road C (2026-10-17, 04:10): a one-shot face swap -- this tree, sha256-tensorrt-clone

### Why this tree exists

Day 24 (2026-10-16 night to 10-17 04:00, in sha256-tensorrt) built road B:
two turned photographs of the avatar beside the portrait, each a source of
its own, the switch past 28-32 degrees with a crossfade. It measured and
worked (the turn followed 1:1 to 54 degrees, frontal frames bit-identical),
and his five live looks ended with "it is very noticeable that the
photograph changes" -- measured: the generator's turned photograph is a
second RENDITION of the face (17 grey levels apart from the front's drawing
at the switch, colour 2-3 of it, the rest creases and light), and no fade,
colour match or feature blend makes two renditions one face; the generator
could not give him the same face three times. His word 03:50: sha256-tensorrt
back to the pushed state he liked (a384738; the day's work in `git stash`
"day 24 (2026-10-17)..." and recordings/day24-reverted.patch), this exact
copy made (no dist, no recordings but driving.*), and road C started here.
Branch `road-c`; nothing is pushed from this tree without his word.

### What road C is, plainly

LivePortrait DRAWS the avatar's photograph and moves it with his face: the
output is the photograph, so a turn past what one photograph holds has to be
invented. A one-shot face swap goes the other way: it takes HIS face as it
is in the camera frame (his pose, his expression, his turn, his light) and
paints the avatar's IDENTITY onto it -- one embedding of the avatar's
portrait, computed once at load, and a small network that re-renders his
aligned face crop with that identity. There is no photograph to turn, so a
turn is whatever his face does (to about 70 degrees, where detection ends);
no calibration (the expression is his own); any face, any person. The
price: the output is his face shape with the avatar's features -- a
likeness of about 70%, not the photograph -- and the crop is 128 px (256
in the newer models), so an enhancer runs after it or the face is soft.
This is what DeepFaceLive's "Insight" mode is (inswapper_128 + an
enhancer), the look he named as the goal.

### The models, and the licences (checked 2026-10-17; sources in the day's notes)

| model | what | licence | use |
|---|---|---|---|
| InsightFace inswapper_128 (+ buffalo_l: the detector and the ArcFace embedding) | the swap DeepFaceLive uses | the CODE is MIT, the WEIGHTS are non-commercial research; InsightFace sells a commercial licence for inswapper_128 | the quality he wants; the test can run on research terms, the delivery needs the licence (his decision, with the client) |
| GHOST (ai-forever, 2022) | a one-shot swap, Apache-2.0 | Apache-2.0 code and weights; its identity encoder is an ArcFace -- whose weights' licence must be read before a commercial use | the free candidate; quality and speed unmeasured |
| ReSwapper | an open re-implementation of inswapper (128/256) | AGPL-3.0, and an open issue on whether training against inswapper carries its terms | no: AGPL on a closed app, and the provenance question |
| SimSwap, AlphaFace, Deep-Live-Cam's models | | non-commercial | no |
| GFPGAN 1.4 | the enhancer (512) | Apache-2.0 (third-party parts to read) | yes, after a read of its parts |
| CodeFormer | the enhancer | S-Lab, non-commercial | no |

The fact that decides the client's version: EVERY good open swap today is
trained on InsightFace's ArcFace embedding, and InsightFace's pretrained
weights are research-only. So the client's delivery is one of: (1) the
InsightFace commercial licence for inswapper_128 (and the recognition
model), the simplest and the DeepFaceLive quality; (2) GHOST, if its
ArcFace weights clear and its quality holds (to be measured); (3) a
swapper trained here on a clean identity encoder (weeks, data licences).
His call, with the client; the tests below run on research terms.

### The pipeline, in this app's words

Per frame, on the render thread (the tracker thread stays as it is):

1. ALIGN: the five points (eyes, nose, mouth corners) from the MediaPipe
   landmarks the tracker already gives -> the ArcFace similarity to a
   128x128 (or 256) crop of his face. No new detector on the frame path.
2. SWAP: the swapper network (TensorRT, FP16): the aligned crop + the
   avatar's embedding (computed at load from portrait.png, cached in the
   avatar folder) -> the swapped crop. inswapper_128 is small: a few ms.
3. ENHANCE: GFPGAN (TensorRT, 512, ~15-25 ms FP16 on his GPU by published
   figures) on the swapped crop -- the one step that threatens 30 fps; the
   levers: 256 input, every other frame, or a lighter restorer.
4. PASTE: the inverse similarity and the mask, THIS app's paste (the
   feather, the edge colour match, the hairline join, the background
   guard: days 13-23's work stays, it is engine-agnostic).

Budget on his PC (33 ms a frame, the render thread today 21.5 p50 with
LivePortrait): swap 5-8 + enhance 15-25 + paste 9 = 29-42 ms -> the
enhancer decides; measured, not guessed, before any promise.

### The steps

- C0 (now): this plan; his licence decision; his permission to download
  the test models (inswapper_128.onnx ~530 MB, buffalo_l ~300 MB,
  GFPGANv1.4.pth ~350 MB).
- C1: a still test on his own presses (captures/moment-*: camera.png is
  his face at +40, -32, -52): swap + enhance, the likeness and the turn
  quality seen before any pipeline work. GHOST the same, side by side.
- C2: `engine_swap.py` behind the same engine interface (load_avatar =
  the embedding; animate = align, swap, enhance, paste); the avatar's
  manifest says which engine draws it; the UI untouched.
- C3: TensorRT plans for the swapper and the enhancer; frame_budget on
  driving.mkv; the 30-fps proof.
- C4: his live look; the package.

### C0-C3 built and measured (2026-10-17, 05:40-07:20)

C0. The clone's `.venv` had lost torch, TensorRT, MediaPipe and cv2 to a
`uv sync` four minutes after the copy (uv prunes what the lock's default
groups do not name); the packages were copied back from sha256-tensorrt's
`.venv` (7.98 GB). RULE: never `uv run` or `uv sync` in these trees; start
the application with `start-swap.bat` (`.venv\Scripts\sha256.exe`). The test
models downloaded on his word, sizes exact, sha256 in
models/downloads/swap/SHA256SUMS.txt: inswapper_128.onnx (InsightFace's own
release link still serves it), buffalo_l.zip (only w600k_r50.onnx is used),
gfpgan_1.4.onnx (facefusion's export). GHOST not downloaded: its identity
encoder is InsightFace's ArcFace too, so it buys nothing on the licence.
The captures' yaws are +40, -32 and +34 (the "-52" above was wrong).

C1. The stills (scripts/probes/swap_still.py): the turn holds at all three,
the avatar's identity on his face in his light; ArcFace cosine to the avatar
0.88-0.89 after the swap, 0.80-0.85 after GFPGAN, to him ~0 (before: ~0).
TensorRT (scripts/probes/swap_trt_time.py, each net alone, p50 of 200): fp32
inswapper 8.6 ms, GFPGAN 14.6; **fp16 inswapper 2.9, GFPGAN 5.7, ArcFace
0.7 (load only)**, within 2-3 grey levels of the fp32 outputs on his face.
TensorRT 11 has no FP16 flag (strongly typed networks only), so fp16 means an
fp16 ONNX; onnxruntime's and onnxconverter_common's converters mis-wire a
partially converted graph (21 dB even with every op kept fp32) -- our own
converter lives in `sha256.avatar.swap_models.to_fp16`, with the nodes that
overflow fp16 on a real face (inswapper's instance-norm variances and
GFPGAN's StyleGAN2 demodulations, up to 3e6: models/metadata/
swap_fp16_blocks.json, from scripts/probes/swap_fp16_range.py) and their
consumers kept fp32.

C2. `sha256.avatar.engine_swap.SwapEngine` (registry id "swap", `[engine]
backend = "swap"` in local.toml) behind the engine protocol: load_avatar =
the portrait's five points from the tracker's landmarker, the ArcFace 112
crop, the embedding through inswapper's emap; animate = the tracker's five
points -> the 128 ArcFace crop -> inswapper -> paste inside the face oval
(MediaPipe's silhouette: nothing on his hair) -> the FFHQ 512 crop -> GFPGAN
-> paste inside the oval; a lost face passes the camera frame through. The
paste blends only the crop's rows and columns with cv2.blendLinear (numpy's
float blend was 9 ms, this 0.7); the networks' pre- and post-processing on
the GPU. Avatars bidzo_swap and my_avatar_swap (portrait.png copies, options
in metadata.json). Through the engine on his captures: 13-15 ms a frame.
**His first live look (07:00): "the rest is ideal, the best"; two faults:
the mouth does not follow well, and the teeth on a smile.** Measured on
driving.mkv's smile, open-mouth and talk frames
(scripts/probes/swap_clip_stills.py): inswapper itself draws his talking
mouth a third open (the portrait's closed mouth leaks through the
identity), GFPGAN closes it further and invents teeth; a crop with more chin
(`swap_zoom`, `swap_shift`) changes nothing. The fix, default: his OWN mouth
through a hole in both masks -- `lips_keep` 1.0 (the outer-lip ring) and
`mouth_hole` 1.0 (the inner ring), feathered, nothing when the lips are
shut; `enhance_strength` 0.75. The smile, the open mouth and the talk match
the original; the avatar sits in the eyes, nose, brows and skin.

C3. frame_budget --engine swap, threads, realtime, driving.mkv (1850
frames): **30.0 fps** (publish p50 33.34 / p90 34.49 ms), 1847 published, 2
dropped at capture; animate p50 17.1 / p90 18.5 / p99 19.7 ms (align 0.7 +
swap 3.5 + paste 1.5 + enhance 6.3 + paste 4.7), tracker 6.3, end-to-end
latency p50 25.3 / p99 31.0. LivePortrait's render thread was 21.5 / 23.4:
~13 ms of headroom for the client's weaker cards.

**His second live look (07:40): "the mouth follows now, but the avatar
has MY lips, and below the red line it is MY jaw" (his drawing: the chin
and the jaw under the lips) -- "the avatar's lips as in the photograph must
stay, on every avatar, mandatory".** Measured on his capture: the swap's
128 crop reaches 41 px below the chin and the oval covers the jaw; what
reads as his jaw is his STUBBLE, which inswapper keeps (it preserves the
target's texture) and GFPGAN sharpens. Built, the defaults now:

- `lips_source = "avatar"`: the PORTRAIT's lips on his lip geometry. The
  lip band (MediaPipe's outer and inner lip rings, 40 triangles) is warped
  piecewise-affine from the portrait's 512 crop onto his, after GFPGAN,
  three pixels inside the ring and feathered (the photograph's lip edge
  drew a dark line otherwise), its colour moved by the Lab-median
  difference between the portrait's oval and the restored face's, pasted
  whole; the opening stays his own teeth (`mouth_hole`). The lips are the
  avatar's and move exactly as his.
- `shave` (1.0 for bidzo_swap, 0 for the bearded my_avatar_swap): the oval
  below the nose's base, the lips left out, smoothed by a half-size median
  and tinted to the skin's Lab median (the lightness only 60%, the chin
  keeps its shadow) BEFORE GFPGAN, which then draws clean skin there; the
  restorer's share is whole in that zone. Chin and moustache clean on the
  clip and on his capture.
- The cost, 42 ms at first, brought to 22-25 ms a frame: the lips' work
  confined to their box, the medians on a stride, the oval mask at half
  size (cProfile: the 3.4 ms "download" per network is the GPU wait, not
  CPU). frame_budget --engine swap, threads, realtime, driving.mkv: **30.0
  fps** (publish p50 33.30 / p90 34.81 ms), 1847 published, 2 dropped;
  animate p50 25.2 / p90 27.1 / p99 29.2 ms, tracker 6.4, latency p50
  33.9 / p99 38.4. Four to eight milliseconds of headroom on his card; the
  client's RTX 3060 (the networks ~2x slower) will want the 512-stage
  compositing moved to the GPU -- the next lever, if it is needed.

**His third look (08:10): "now the avatar looks good and works well" --
then "can I not use the other avatars?" and "we need the .exe, the whole
application working, to show a friend".** Built:

- Every avatar with a portrait drives the swap engine as it is
  (`SwapEngine.supports`: a `source_portrait` asset is all the swap
  needs), so the library's LivePortrait and photograph avatars -- bidzo,
  bidzo1, bidzo_new, my_avatar, my_avatar_neural, woman_person -- load in
  the swap session without a copy; only the swap's own option keys are
  read from a foreign manifest.
- `shave = "auto"` (the default): the portrait decides. `beard_signal` is
  the chin's lightness speckle over the cheeks' in the portrait's 512 crop
  (stubble is a few-pixel speckle the cheeks lack): the clean-shaven
  avatars read 1.18-1.54, the bearded my_avatar 1.77, the threshold 1.6.
  A manifest's 0 or 1 overrides it.
- `scripts/build_exe.py --engine swap`: the package with the three fp16
  networks (512 MB), inswapper's cached emap and this card's plans, no
  fp32 originals and no LivePortrait weights (1.2 + 1.6 GB stay home); the
  shipped local.toml starts the swap engine. The friend's card builds its
  own plans at the first start (~2 min, counted on the start-up window).
  The run for the friend test: `--engine swap --avatar bidzo_swap
  --avatar bidzo1 --avatar bidzo --avatar my_avatar_swap --avatar
  woman_person --zip`.
- The work committed in this clone's `road-c` branch and pushed to a new
  repository, `sha256-tensorrt-clone`, on his word; `main` and
  sha256-tensorrt untouched.

Open: the friend test; C4's rest (the InsightFace licence for the client,
the 3060 measurement, the 512-stage compositing on the GPU if the 3060
needs it).

### C4 (2026-10-18): his library, the skin, the forehead, the package

Pushed to github.com/director0000000001/sha256-tensorrt-clone (road-c) at
682eccf in the morning. He added lalo, lima and saul_goodman with Add avatar
and deleted four: the flow writes a LivePortrait manifest (the template is
the first liveportrait avatar), the swap reads only its own keys from it
and runs it with the defaults, so nothing had to change. His look: "the
app gives the avatar a tan". Measured per stage on his presses (Lab
medians of the face oval): inswapper keeps HIS lightness (his room light
58-63 L* against the portraits' 65-72) and adds +2 a*; GFPGAN is neutral;
the shave +1.5 L*. A first fix moved the face toward the portrait's
median; live he called it worse ("see the forehead"): the portraits' skin
is warm in numbers (a* +12..+15, b* +17..+21) although it reads white, and
the swap paints the forehead paler and flatter than his, which no uniform
colour change mends. Built: `forehead_fade` 0.8 (the hull falls off from
the brow line to the oval's top), and `skin_match` with the person's own
skin as its base (0 = his colour with the cast corrected, 1 = the
portrait's, "off" = nothing), both layers landing on one goal colour by a
per-channel gain table (`gain_lut`, the shadows kept), eased 0.2 a frame.
The first version was a Lab offset at 2.8 ms and cost the 30 fps (28.4);
the table costs 0.3 ms and `lab_median` on the strided picture gave a
millisecond back. Numbers: the hairline step at the camera's own (+6.7
against +7.1), the cheek's a*/b* his (+6/+15). frame_budget, threads,
realtime, driving.mkv: 30.0 fps, animate 25.1 p50 / 28.4 p99, skin 1.2
ms. His word: "ideal". The package: `ensure_plans` required the fp32 ONNX
the package does not ship -- fixed (`fp16_source`); sm75's builder
resource added for the RTX 20 series. Built with `scripts/build_exe.py
--engine swap --avatar bidzo_swap --avatar saul_goodman --avatar lima
--avatar lalo --avatar bidzo1 --avatar woman_person --avatar my_avatar_swap
--zip` for his friend's machine.

## What is left, in his order (written 2026-09-12 night, after the push)

Everything above is on origin/main at c5eda18. What he wants next, in
the order he gave it, then what the code side sees.

### 1. Tomorrow: the whole forehead filled, and the hair (block C1, again) -- DONE day 17

Built: "hairline" is the forehead's default and the avatar's hair is a
switch (see "Day 17" above). What is left of it is his look, live, and
the two limits stated there.

### 2. The brows dip on a blink (his report, 2026-09-12 night) -- DONE day 17

Measured and countered (see "Day 17, his third ask" above): the eye
network's own pull, three levers per portrait, within half a pixel of
his own movement at every depth. Also done on day 17, beyond the list:
the steep poses (the clamp in place of the hold). Added to the list by
his question of 03:40: **a hand in front of the face** -- a hand matte
(MediaPipe's hand model on a thread), the paste cut under the hand,
the covered parts of the face and the pose held; 2-3 days, after the
calls. What the report said:

"When I blink or just close my eyes the brows go down too, and come
back up when I open them -- not easy to notice, but not natural." Two
candidate mechanisms, both measurable on the calibration sessions
already recorded (their `blink` and `eyes_shut` steps carry his brow and
the avatar's per frame):

- the eye retargeting network: its 63 outputs are not confined to the
  lids (day 6's finding); the delta was confined to `EYE_KEYPOINTS` on
  day 8 -- check that the brow keypoints are outside that set on a
  full shut, not only on a widening;
- the brow drive: MediaPipe's brow ratio drops on a blink (the lids'
  landmarks move with the eye), so `_brow_lift` reads a lowering; the
  dead band was set from exactly this confound (his live 0.173, at the
  band's cap; baska's 0.055, where a blink's 5.5% is at the edge).

Split them with the replay probe: `--sweep brow_follow=1,0` on a shut
frame; whichever drops the avatar's brow is the one. The likely fix is
to hold the brow drive while the eye ratio is under the blink hold
(`_GAZE_BLINK_HOLD` already does this for the gaze).

### 3. Then: the virtual camera on real calls

Google Meet, Discord, Zoom, Teams, one after the other, on this machine
first: the OBS Virtual Camera chosen in each, the picture's orientation
(mirrored in the preview only), the black screen's text readable on the
other side, the frame rate the call actually gets, and Discord's own
camera hold (it must be closed before Start today). One page of what
each app needs, in the guide.

### 4. The requirements, and an Instructions button -- BUILT the same night

Menu -> Instructions (`sha256.app.instructions`, `ui/instructions_dialog`):
eight sections -- what you need, the first time, the bar, the menu, the
status line, on a call, if something goes wrong, what stays on this
computer -- every control named as it is on the screen, tested to stay
so (tests/unit/test_instructions.py). The requirements as written there,
from what this machine and the recordings measured:

- Windows 10 (64-bit, 21H2 or newer) or Windows 11. Not yet run on a
  Windows 10 machine -- the .exe on a second machine is still his test.
- An NVIDIA card with 6 GB of memory free: RTX 3060 at the least; this
  machine's RTX 5070 Ti runs 19-25 fps live at 720p, so an RTX 4060 or
  better is what "comfortable" means. Without an NVIDIA card the plain
  photograph avatar runs instead -- a weaker picture.
- A processor of the last four years with six cores or more (the face
  tracker runs on the CPU), 16 GB of memory, 6 GB of disk for the
  package.
- A camera at 720p 30 fps at the least (1080p 30 is fine): a Sony ZV-E10
  over USB Streaming, or a good webcam of the Logitech C920 class. At
  eye height, the face a third of the picture's height (300-450 px tall
  at 720p is what every good session had), soft light from the front,
  no window behind. Sit still and look at the camera for the first
  second.
- OBS Studio installed once, for its virtual camera device. No internet
  needed for a call.

### 5. What the code side sees, besides his list

- The join's remaining tone step at the temples on baska (the residual's
  reach, `_EDGE_REACH`), measured on her captures.
- `mouth_smile_opening` measured by Calibrate me from the toothy-smile
  step, per avatar, instead of set by hand.
- Calibrate me asking the name FIRST and starting a new name from
  Nobody, so a new person's gains are corrected from the avatar's own
  values and not from whoever was in force.
- The pipeline's frame rate under load: the captures of tonight ran at
  19-22 fps (300 frames in 13.5-15.4 s) against the 30 the camera gives;
  where the time goes (the tracker on the CPU, the correction, the
  paste) has not been measured since day 11.
- The instructions, the calibration's voice and the dialogs are English;
  the five are not. Georgian text in the dialogs is a day's work; a
  Georgian voice is not there in SAPI.
- The kill switch is unconfigured (inert) -- his decision, before the
  five get the package.

## Honest odds

- **The picture:** high. LivePortrait is proven and it fits this GPU; what is not
  yet seen is its quality in a real room at 720p. Day 4 answers that.
- **"Hard to tell it is an avatar" on a normal call window:** realistic within
  phases 1-2. Under close inspection or a large screen: not in three weeks.
- **Five users:** depends entirely on their GPUs and cameras. Without an NVIDIA
  GPU a user gets the photograph engine, which is what exists today.
- **Eleven days from today:** phases 1-2 fit; phase 3 partly; phase 4 does not.
  A quality delivery to five people is about **three weeks**, four with margin.

## What stops

No more work on the photograph engine beyond keeping it running as the
fallback: no refiner, no new clips, no third axis, no morphing between rungs.
Every day spent there is a day the neural engine is not being built.
