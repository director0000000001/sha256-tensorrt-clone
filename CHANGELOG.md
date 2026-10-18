# Changelog

All notable changes to sha256 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions below `1.0.0` carry no API stability guarantee.

---

## [Unreleased]

### Added

- **Road C (2026-10-17): the one-shot face swap engine, in the
  sha256-tensorrt-clone tree.** A new engine, `swap` (`[engine] backend =
  "swap"`): the camera's face as it is -- his pose, expression, turn and
  light -- with the avatar's identity painted on it by inswapper_128 and
  restored by GFPGAN 1.4, both as TensorRT fp16 plans built at the first
  start from fp16 ONNX files the application writes itself (TensorRT 11
  builds strongly typed networks only; the public converters mis-wire the
  graph, so `sha256.avatar.swap_models` carries its own, with the nodes
  that overflow fp16 on a real face kept fp32). No photograph to turn, no
  calibration: turns to where the tracker loses the face. The avatar stays
  inside the person's face oval (nothing on their hair); the opening of
  the mouth stays the person's own teeth (`mouth_hole`) because the
  restorer invents teeth; the LIPS are the portrait's, warped onto the
  person's lip geometry every frame (`lips_source = "avatar"`), because
  inswapper draws a talking mouth a third open; and a clean-shaven avatar
  can have the person's stubble smoothed and tinted away before the
  restorer draws clean skin (`shave`, "auto" reads the portrait). Every
  avatar with a portrait drives it, the library's LivePortrait and
  photograph avatars included; `bidzo_swap` and `my_avatar_swap` carry
  the options spelled out. `scripts/build_exe.py --engine swap` packages
  it with the three fp16 networks and no LivePortrait weights. Measured
  on the RTX 5070 Ti over driving.mkv: 30.0 fps, the engine 25.2 ms p50 /
  29.2 p99 a frame. InsightFace's weights are research-only without a licence:
  the client's copy needs InsightFace's commercial licence for
  inswapper_128 and the recognition model (docs/development/ROADMAP.md,
  "Road C").
- **Road C, C4 (2026-10-18): the skin the person's own, the forehead
  theirs, the library from the application.** His look of the night: "the
  app gives the avatar a tan", and after a first fix toward the portrait,
  "worse, see the forehead". Measured on his presses: inswapper keeps the
  person's lightness and adds red and yellow (+2..+6 a*/b* on the cheek),
  GFPGAN sharpens a forehead the swap paints paler and flatter than his,
  and the portraits' skin is warm in numbers (a* +12..+15) though it reads
  white on a white background, so pulling the face toward it made it
  pinker. Now `skin_match` (default 0): the face's skin colour lands on the
  person's own, read off their crop before the swap, the swap's and the
  restorer's cast corrected, the face the colour of their neck; 1 is the
  portrait's, "off" leaves the networks' colour. Both layers (the swap's
  and the restorer's) land on the one goal colour by a per-channel gain
  table (exposure-like, the shadows stay dark; 0.3 ms), eased over frames
  so nothing flickers. `forehead_fade` (default 0.8): the paste falls off
  from the brows up to the hairline, so the person's own forehead shows
  through and the band at the hairline is gone. `lab_median` converts the
  strided picture only (the lips' and the shave's medians a millisecond
  cheaper together). Add avatar works with the swap engine as it is (the
  manifest copies a LivePortrait template; the swap reads its own keys):
  lalo, lima and saul_goodman were added from the application today; bidzo,
  bidzo_new, my_avatar and my_avatar_neural were deleted in it. Measured
  over driving.mkv: 30.0 fps, the engine 25.1 ms p50 / 28.4 p99, the skin
  work 1.2 ms.
- **Fixed (2026-10-18): the packaged swap could not build its plans on
  another graphics card.** `ensure_plans` asked for the fp32 ONNX before
  looking for the fp16 copy, and `build_exe.py --engine swap` ships the
  fp16 copies only; on any card without a shipped plan the first start
  died with "network is missing". Now the fp16 copy is what a plan is
  built from and the fp32 original is needed only to write it
  (`fp16_source`). The package also carries TensorRT's builder resource
  for the RTX 20 series (sm75), beside the 30 and 40 series.

- **Day 23 (2026-10-16): the avatar in the person's own skin colour, and
  shaved (the owner's ideas, from his recording).** Add avatar's window now
  has three prompts, in the order they are made: the face with hair from
  words alone; the SAME face with the head shaved, made from the first
  photograph -- the one to add as the avatar; and, optional, the shaved
  one with the lips barely parted. The avatar's hair is never wanted: the
  application joins the avatar's face to the person's own forehead and
  keeps their own hair (day 21), and every hair fault of days 21-22 (the
  pale patch of the roots on the forehead, the smear over the cheek at a
  turn, hair above the forehead) came from hair the portrait had; with
  none in the picture, none can show. Every prompt says "[your skin
  color]": the new button "Read my skin from the camera" (with the camera
  running) reads the person's skin off thirty frames whose head sits in
  the green guide -- both cheeks and the middle of the forehead, the dark
  cells and the highlights dropped, the medians -- and puts its words and
  code in the placeholder's place in all three prompts ("light-medium skin
  with a warm, golden undertone, sRGB #C6966E": the lightness band from
  the ITA angle, the undertone from the hue, both in CIELAB), since a
  generator follows words better than a code. The reading is the
  camera's, room light and white balance included, which is what the
  avatar must match. When the photograph is added, the same reader gives
  its skin colour (in the report) and, if a reading was taken in the
  window, the distance between the two in words: close, noticeably
  different, or far enough to generate again
  (`sha256.avatar.skin_tone`).
- **Day 23 (2026-10-16): the face's height fitted to the person's, and the
  shaved head's whole forehead.** His first shaved avatar, his press at
  02:33: the avatar's nose, mouth and chin sat on his, but its eyes 10 px
  and its brows 20 px lower -- the crop makes the two faces the same
  height (forehead top to chin) and a portrait with a taller forehead puts
  its brows and eyes lower at the same chin. The height fit is the width
  fit's vertical twin: the person's brow-top-to-chin over the portrait's
  (both of the face height, along the face's own vertical, in the crop, on
  still frontal frames, the median -- or `driver_face_height` from a
  profile) scales the render and the portrait's points vertically about
  the portrait's chin row, in the same resample as the width, paste only;
  the forehead's plan then reads the lifted brow and forehead
  (`face_height_fit`, 0..1, default 1; bounds 0.90..1.10; the readings
  `face_height`, `face_height_scale`; the capture carries `settle_height`).
  On his press: the eyes from 9 px low to 1 px, the brows from 20 to 8 (the
  rest is the avatar's own brow set close to its eyes). The shaved head:
  the roots reader found "roots" at 0.90 of a scalp that has none (its
  shading) and capped the clean forehead at 0.93, stretched x1.25; a
  portrait with no hairline read now ignores the roots reader, so the
  whole forehead is the source (1.45 of its height, no stretch).
  Probes: `scripts/probes/face_height.py` (the ratio from a capture or a
  recording against portraits), `scripts/probes/face_rows.py` (a capture
  re-rendered, the drawn rows against the person's).

### Changed

- **2026-10-16, his ask: Add avatar and Delete avatar behind the owner's
  Login again, "temporarily".** A copy of the application went missing on
  a friend's computer, so the two entries that change the library are
  back behind a passphrase (`sha256.app.owner`, as on day 15: salted and
  stretched, kept in `configs/local.owner.json`, never in the repository).
  What differs from day 15: the window never SETS a passphrase -- a copy
  without the file simply has no Login (it says so) -- the owner sets it
  on his machine with `sha256 --set-owner` (asked twice, unseen), and the
  portable build copies the file into the package, so every copy carries
  the same gate. A gate in the window, not a vault: the files can still
  be copied; the lock plan in the ROADMAP is the real answer.
- **Day 23 (2026-10-16), his two presses at 03:25: the brow that stretched
  at a turn to his right, and the eye and lashes that remained at one to
  his left.** (1) The stretch was the outline fit's per-row map (day 22)
  on the BROW rows: at his -23 the brow row wanted 19 px of widening and
  the eye row 6, and the brow smeared sideways. The fitted band now starts
  at the eyes' top, not the brows (`_fit_outline`; the fade above still
  eases the eyes' scale out, so the brow takes little of it). (2) His far
  eye's outer corner and lashes sat in the paste's feather beyond the
  drawn face's edge: the avatar's face is 5-10 px narrower than his there
  at +34 (alpha 0.88 at the corner, 0.59 five px out, 0.08 ten;
  `scripts/probes/eye_alpha.py`), the same on the 09-18 tree, and the room
  pastes only where the render is the drawn face -- the photograph's
  background there. New: the profile fill (`profile_fill_px`, default 12,
  0 off): at a turn, on every row the background columns just beyond the
  drawn face's profile edge, and the edge's own three columns, take the
  colour three columns inside the edge (the edge pixels are the network's
  pale blend with the background: carried out they drew a streak, left in
  place a pale line), ramped with the turn gate, so the room pastes the
  avatar's cheek skin over his lashes, inside the driver's oval as
  before. On the press the lash box's mean luminance went from 70 to 92
  (his lashes 68); a faint remnant stays where the feather fades.
  Frontal frames untouched. (3) The turn
  is not symmetric on him (right turns drawn 0.7-0.8 of his angle, left
  1.0-1.1, on the 09-18 tree and the shaved avatar alike): a per-side
  yaw gain exists now (`turn_gain_left` / `turn_gain_right`, driver
  options, default 1.0 = off, bit-identical; measured monotonic at 1.3 on
  his session's right turn, the eyes within 5 px of his at -23) but is
  NOT set in his profile: at 1.3 the avatar's far eye corner darkens at
  the turn (the network's own rendering of it), and his word on 09-18's
  turns was good -- his call. The extractor's yaw runs the other way from
  the tracker's (a delta under 0 is a turn to his left), found by setting
  one gain and watching which side moved.
- **Day 23 (2026-10-16): the turn boost withdrawn from his avatars' metadata
  -- the turn as it was on the 09-18 push.** His words: "on 09-18 the face
  followed me further when I turned; now it does not". Measured on his
  session's turn step (frames 3024-3200 of calibration-20261015-190122)
  through today's tree and the 09-18 tree in a worktree, the drawn face's
  yaw read by MediaPipe on every fourth output (`scripts/probes/turn_motion.py`;
  one held press: `turn_compare.py`): the 09-18 drive turns 1:1 to about
  30 degrees and saturates at 37 (his 50 drawn 38), monotonic, the pitch
  tracking his. Day 22's `turn_boost` 1.0 + `turn_cap_degrees` 60 (set in
  his metadata, copied into every avatar imported since) over-turn the
  avatar to 54 degrees while he is at 35-40 (x1.4-1.5), dip its pitch to
  -9 at his -5, lose the face to the tracker at 45, then bring it back to
  48 at his 50 -- a turn that runs ahead of him and then retreats, which
  is what "does not follow" looked like; day 22 had measured the
  endpoints only. Today's tree with the two keys off reproduces the
  09-18 curve on the same frames (37-41 at 45-50 degrees, the pitch
  within two degrees), with the shaved avatar too. The keys are removed
  from `my_avatar_neural`, `baldnew` and `baldnew2`; the engine's
  defaults were off all along. Day 22's paste-side work at turns (the
  outline fit, the room below the brow line) stays: the paste's reach
  past his rim at a 50-degree turn is within 6-10 px of the 09-18 tree's.
- **Day 22 (2026-10-15): Calibrate me, slow and simple (the owner's order).**
  Every movement now comes in three phases: the instruction (three seconds,
  on screen and read aloud), a count of three drawn large in the middle of
  the card ("3", "2", "1", one a second, spoken "three", "two", "one" where a
  voice exists, each cutting off what was still being said), then the
  movement itself. The wording says it: "On three: ...". The script is new
  and longer (about 266 s; his first session on it asked for 5 s more on every movement, and got it): still, blink, eyes HALFWAY, eyes shut, mouth a
  LITTLE ("oh"), mouth wide, READ a short text aloud for 20 seconds (the text
  replaces the instruction on the card, large: `READING_TEXT`, full of "o"
  and "m"), smile a little, smile closed, smile with teeth, LAUGH, eyebrows a
  little, eyebrows high, glance, turn, still again -- the new kinds
  (`eyes_half`, `open_small`, `read`, `smile_small`, `laugh`, `brows_half`)
  are the contract for the per-level measurement still to be written; the
  talking mouth's fit reads `read` as it read `talk` (kept for older
  recordings). The green oval stays on the picture for the whole session:
  while the head is out of it during a movement, its one line ("Come
  closer", ...) shows under the instruction, the clock goes on, and every
  frame carries whether the head was in it (`guide_ok`, in the samples and
  in `tracking.npz`) so the measurement can drop those frames. A movement
  too small to measure -- read off the step's own frames against
  `measure.py`'s thresholds (half of them for the "a little" steps; a
  smile's +5% of the resting width) -- is asked once more right after it,
  "Once more: ...", at most once per step; the still steps and the turn
  never are.
- **Day 22 (2026-10-15): the face's width fitted to the person's (always on,
  no calibration needed).** The crop is sized by the face's height, so in it
  the two faces differ in width alone, and the paste's mask -- the portrait's
  oval AND the person's -- cut the render's cheeks off on a narrower face:
  the join on the avatar's own cheek skin, its features set wider than the
  person's (his resting face reads 4-6% narrower than his portrait's;
  `scripts/probes/face_width.py`). Now, once the anchor is frozen, the
  person's rim width over the portrait's (along the face's own horizontal,
  on still frontal frames, the median) scales the render and the portrait's
  oval horizontally about the portrait's rim centre -- in the paste alone
  (one resample of the rendered crop, GPU or CPU; the drive, the network
  and the crop untouched), within 0.85..1.15. `face_width_fit` (portrait
  option, 0..1, default 1; 0 is the old picture to the byte) and
  `driver_face_width` (a driver key, Calibrate me's measured ratio, in
  place of the live reading). `frame_readings` carry `face_width` and
  `face_width_scale`.
- **Day 22 (2026-10-15), his three presses after the first look.** (1) The
  pale smeared patch on his forehead was the avatar's own HAIR ROOTS: the
  hairline reader calls hair a 30% drop of the median, and the portrait's
  loose strands begin a tenth of a forehead under that, where the mask's
  feather tail still drew them and the edge residual brightened them. The
  portrait's roots are now read at load (`_roots_onset`: the first row
  whose 5th percentile over a wide band is 7% under the skin), the clean
  skin ends 0.10 of a forehead under them (or at the hairline less the
  roots' margin, whichever is lower), and that is a SOURCE row the stretch
  carries to the join (the margin used to be taken off after the stretch,
  which shrank it). (2) The forehead FOLLOWS the person's hairline: his hand
  pulling the hair back raised it and the settled join stayed, a band of his
  own skin above it; the last fifteen readings are kept and the plan is made
  again when their median leaves the settled hairline by 0.15 of a forehead
  (the width keeps what it has). (3) The pale blob under the drawn chin at a
  40-degree turn was the avatar's chin underside, a light shadow on a white
  photograph, pasted over his dark one by the reach under the chin
  (`mask_chin_reach`, his 0.04 -- the blob's own 12 rows): the reach now
  fades out with the turn or the tilt from the rest pose (12..20 degrees,
  the guard's gates). Pulling his oval's jaw in as well took the join 26 px
  up and showed his stubbled chin under the avatar's (the day-12 ledge), so
  it was not kept. Also: the background guard comes on with a tilt as it
  does with a turn, and the rows under the portrait's chin have a keep of
  their own on every frame for the photograph's background (`_chin_keep`,
  on the device too, no download). (4) A capture carries what the forehead and the
  width were settled from (`settle_hairline`, `settle_width` in the anchor
  state), so a replay plans the same forehead; his three presses of today
  predate it and were replayed with the session's values given by hand.
  (5) The output quality has a 240p rung under 360p (JPEG 30), his ask.
- **Day 22 evening, his held press at a turn: `turn_boost` and
  `turn_cap_degrees` (both off by default).** The network turns the head
  1:1 to about 35 degrees and less past it (his -45 drawn at -37, his
  +55 at +37..40), so at a strong turn the avatar's near-side eye and
  brow sit inward of his and his own show through the paste's feather
  beside them. `turn_boost` drives the yaw delta past 30 degrees harder
  (delta + boost x (|delta| - 30)) before the clamp; `turn_cap_degrees`
  is the clamp's own yaw cap, apart from `hold_yaw_degrees` (which stays
  where the expression starts to fade). Measured: boost 1.0 with the cap
  at 60 draws his -45 at -44; a cap of 75 lets the left turn break (a
  near-profile with the portrait's hair over the far cheek). Set per
  avatar in its metadata; the defaults change nothing.
- **Day 22 evening, his goal "the face must sit at larger turns": the
  render's profile side fitted to his outline (`outline_fit`, default on,
  nothing frontal).** At a turn the drawn head is narrower on the profile
  side than his, and between its silhouette and his own profile the paste
  showed his skin, his eye through the feather beside the avatar's. Now on
  every row between the brows and the chin the drawn face's near edge
  (the first column from the profile side that is neither the photograph's
  background nor the crop's black border, read on the device) is taken onto
  the paste's own near edge (his oval with the day-20 room) by a per-row
  horizontal scale about the paste's far edge -- bounded to a widening of
  14% and a narrowing of 13% (his nose at a left profile stands out further
  than the avatar's; more widening drew the avatar's nose large), smoothed
  over 21 rows, faded over 40 rows outside, eased over frames by half --
  folded into the paste's sampling (the GPU grid and the CPU remap), and
  the guard's keep is read through the same map, so the background the
  warp would sample at the profile is cut where it lands (without that, a
  thin white line ran along the near cheek on a moving turn). It comes on with the turn (the room's gate, 12..16 degrees from
  the rest yaw), so a frontal frame's paste is untouched. Prototyped on his
  held press in `scripts/probes/outline_fit.py` (the near-side gap -6 px at
  the median, 19 at the brow; the avatar's eye landed on his).
- **Day 22 evening, his second look ("now it is broken at the side, the
  face is lost sooner"): the turn room kept below the brow line, and the
  outline fit widens only.** Since day 21 the person's oval reaches THEIR
  hairline, above the portrait's clean skin; at a turn the day-20 room
  (the person's oval past the portrait's, where the render is not the
  background) pasted that strip -- the render's HAIR, which the guard
  keeps -- as a smear over the far temple and forehead, worst coming back
  from a turn. The room is for the far brow and eye: it now reaches no
  higher than 0.2 of a forehead above the brow line. And the outline fit
  never narrows the render: at a moderate turn the drawn edge lies
  outside the oval's eroded edge, and pulling the render in there only
  uncovered him; it fits to the oval's own hull (alpha over 0.05) and
  widens at most 14%. Measured on the whole turn of his session through
  the two-thread pipeline (frames every 10 from -4 to +50 and back to
  -47): the smear gone on both sides, the face on his at every frame.
- **Day 22 evening, his two presses at a turn (20:29, 20:30): the nose
  left out of the outline fit.** Turning left, the avatar's nose lost its
  shape; turning right, a white sheen lay on it. Replayed with the fit on
  and off: the fit. On the nose's rows the profile IS the nose, and his
  nose and the avatar's differ in shape, not in turn -- fitted, the
  avatar's nose was stretched into a blob and the tip sampled the
  photograph's background. The band from under the eyes to under the
  nostrils now reads the identity, ramped over 12 rows on either side;
  the cheeks, the eye and the brow above and the jaw below are fitted as
  before.
- **The skin's tone matched (2026-10-14, after the friend test).** Before the
  lighting plane, the neural engine gives the rendered face the person's own
  skin tone: on the cells that are skin on both sides (the mask's core, less
  the dark cells of either picture and the person's brows, eyes and mouth) the
  mean and the spread of each channel are read for the room and for the
  render, and the render gets, per channel, a CAST -- the room's mean over
  the render's, the room's white balance multiplied into every pixel, which
  moves a dark pixel by little and a skin pixel by the whole of it, so the
  irises, the brows and a beard keep their look while the skin takes the
  room's tone -- and a CONTRAST about the room's skin mean that gives the
  skin the room's spread, applied to each pixel by its own lightness against
  the render's skin (none under 0.7 of it, all from 0.9), both within
  0.7..1.4. (A gain and an additive offset on every pixel had turned the
  woman's avatar's brown irises blue-grey; the tone spared on dark pixels had
  left the avatar's stubble pale over hers; an additive chroma had turned the
  beard orange: five replays of her capture.) The plane then fits the light's
  gradient as before, the edge residual as before. Read one frame in four
  (the reading is 2 ms of the render thread's CPU, the apply 0.2), carried by
  an EMA, held with a held face. `skin_match` in `engine_options` (0..1,
  default 1; 0 is the paste before this day, to the byte -- proven 601/601
  on the owner's clip). On the GPU paste it is tensor sums on the device,
  nothing waiting; `tests/unit/test_engine_skin_tone.py`.
- **The forehead to the person's own hairline (2026-10-14).** The join sits
  slightly below where the PERSON's hair starts, and the avatar's upper
  forehead (above 0.3 of a forehead over the brows, past any brow raise) is
  drawn taller to reach it -- moderately, `forehead_stretch_max` 1.25 by
  default, "a medium height" -- its own hair roots' shadow kept out of the
  join by `hair_root_margin` (0.15 of its forehead). A person whose hairline
  cannot be read (bald, or hair past the walk) gets the whole moderate stretch
  and no bound from their side. The stretch is the paste's alone (the render
  sampled closer to the anchor row above it, on the CPU and the GPU alike);
  the drive is untouched. `_forehead_plan`, settled at the anchor with the
  driver's own forehead height against the portrait's;
  `tests/unit/test_engine_forehead_stretch.py`.
- **The head guide in Calibrate me (2026-10-14).** An oval on the preview
  where the head should be -- the distance of every good session (the face
  0.38..0.60 of the frame's height), centred, facing the camera -- amber with
  one line of what to do ("Come closer", "Move back", "Put your head in the
  oval", "Face the camera"), green with "Good. Hold still."; the session's
  clock starts once it has been green for a second (`sha256.ui.head_guide`,
  the preview's `guide`, the window's placing phase). The person is told the
  one thing to fix, so the still step and everything after it are measured
  from the right place. `tests/unit/test_head_guide.py` and the UI tests.

### Changed

- **The forehead join's rule.** Until today the reach took the LOWER of the
  two hairlines less a clearance, so a person with a taller forehead than the
  portrait's got the join on the avatar's hair roots, in the middle of their
  bare forehead -- the stripe of the friend test. Now the driver's oval goes
  to their own hairline less `hairline_clearance`, the portrait's to its
  clean skin (see above), and the paste to the lower of the two as ever. The
  engine's log says both reaches and the stretch at the settle.
  `hairline_clearance` defaults to 0.12 of a forehead now (was 0.04): under
  the person's own hair roots, swept on two captures with the forehead band
  (0.04 / 0.08 / 0.12 -> 4.1 / 3.5 / 3.0 grey levels on the woman's, 8.9 /
  8.4 / 7.5 on his; the untouched build 2.7 / 7.5).
- **`scripts/probes/jaw_band.py`** reads the engine's four-part `last` again
  (the turn's room of day 20 had broken its capture re-render).

- **The fast engine on the user's machine, and a start-up window (2026-10-09,
  the second package; `docs/benchmarks/B006-tensorrt.md` section 10).** The
  portable build carries TensorRT's runtime, the ONNX export and the builder
  resources for RTX 30 / 40 / 50 cards. At the first start on a computer with
  no engine for its graphics card, the application builds one -- one to three
  minutes, once -- while a small start-up window counts the seconds; every
  later start loads it in a fraction of a second. A build that fails is
  recorded beside the plan (`*.failed.json`, delete to retry) and the torch
  networks draw the face as before. The window also covers the ordinary
  twenty-second start, which showed nothing until now. `scripts/build_exe.py`
  ships the export and every engine built here, and leaves an avatar's
  `superseded*` folders out.
- **The face's pixels on TensorRT (2026-10-09, `docs/benchmarks/B006-tensorrt.md`).**
  LivePortrait's warping module and SPADE decoder -- the networks that draw
  the pixels -- run as one TensorRT fp16 engine when `engine.pixel_backend`
  (`auto`, the default) finds one built for this GPU and TensorRT version
  under `models/tensorrt/` (`scripts/build_tensorrt_engine.py`, about a
  minute, once); the torch networks otherwise, unchanged. The motion
  extractor and the expression stay in torch, bit for bit (the keypoints
  byte-identical over 601 frames); the drawn pixels differ by fp16 rounding
  (at most 6 grey levels in the composite, mean 0.02). On the owner's RTX
  5070 Ti the model's GPU time 25.8 to 14.3 ms a frame, the render thread
  32.5 to 21.5 ms, 30 fps at the camera with no frame over the budget,
  latency 21 ms shorter. The hairline read is taken from the window it
  touches instead of the whole frame -- the same reading to the bit, 17-20
  to about 6 ms.
- **Add avatar and Delete avatar without a login (2026-10-09, the owner's
  ask).** Both entries are in the Menu for whoever uses the application; the
  owner's passphrase (`sha256.app.owner`, Menu -> Login) is removed, with its
  tests. An avatar stays a folder under `avatars/` on the user's own computer;
  nothing is uploaded. A `configs/local.owner.json` left from before is no
  longer read.
- **The frame rate (2026-10-08, "The fast version").** The real-time loop runs
  on two threads -- the camera and the tracker on one, the avatar, the output
  conditioning and the virtual camera on the other, a mailbox of one between
  them (`PipelineRunner`, `PipelineController.capture_and_track` /
  `render_tracked`; `performance.overlap_tracking = false` restores the single
  thread). The neural engine keeps the generated face on the GPU from the
  graph to the paste: the face is warped, colour-matched and blended there,
  with the same arithmetic as the CPU paste (17-24 pixels a frame differ, by
  one grey level), and the CPU waits for the GPU once per frame. On the
  owner's machine under a 30 fps camera: 18.8 to 29.4 fps; one more frame of
  latency, about seven milliseconds measured on a clip. The face, the eyes,
  the brows and the mouth are untouched -- proven byte-identical by replay
  (`scripts/probes/frame_budget.py`, `--compare`).

### Added

- **Virtual camera output.** `PyVirtualCameraSink` publishes rendered frames to
  an existing Windows virtual camera device, so the avatar is visible in Meet,
  Zoom, Discord and OBS rather than only in sha256's own preview. Verified by
  round trip: a DirectShow consumer reads back the exact pixels published
  (`docs/benchmarks/B004-virtual-camera.md`).
- Publishing is a checkbox in the window and can be switched during a call —
  `PipelineController.set_publish_output()`. It is off until asked for.
- `sha256.virtual_camera.devices` reports which virtual camera devices are
  registered, by reading the DirectShow capture category rather than by opening
  a device, so the UI can say what to install without briefly advertising a
  camera to every application on the machine.
- `virtual_camera.device` setting: bind to one named device when a machine has
  several.

### Fixed

- The virtual camera sink was never configured by the composition root, so
  `virtual_camera.auto_start = true` failed on the very check that exists to
  catch that ordering mistake. Sinks are now configured where they are wired.
- `docs/development/AVATAR_QUALITY_BACKLOG.md` had been overwritten by a copy of
  `PROJECT_STATUS.md`. Recovered, and brought up to date.

### Notes

- `pyvirtualcam` is **GPL-2.0** and stays in the optional `vcam-gpl` extra. It is
  a development backend and cannot be distributed; the shipping decision is due
  at the day 7 scope review (ADR-0005).

---

## [0.1.0] — 2026-08-08

Project foundation. **No avatar functionality works yet** — this release is the
structure the real pipeline will be built into.

### Added

**Core contracts** (`sha256.core`)
- Component protocols: `CameraProvider`, `FaceTracker`, `AvatarEngine`,
  `Renderer`, `VirtualCameraSink`, `PerformanceMonitor`, `AvatarRepository`.
- Vendor-neutral data models: `VideoFrame`, `FrameMetadata`, `LandmarkSet`,
  `HeadPose`, `BlendshapeSet`, `FaceTrackingResult`, `RenderedFrame`.
- Avatar manifest schema with strict validation and path-traversal rejection.
- Metric result types, exception hierarchy, shared enumerations.

**Application**
- Component registry: backends selected by config string, instantiated lazily.
- Composition root wiring the full object graph.
- `PipelineController` for component lifecycle with roll-back on partial start,
  plus single-frame stepping. **No real-time loop.**

**Avatar system**
- Filesystem library, one directory per avatar, discovered at startup.
- Three-level validation: manifest, asset presence, optional SHA-256 integrity.
- Residency cache: LRU eviction, active-avatar pinning, VRAM budget that refuses
  rather than attempts an oversized load.
- `AvatarManager` coordinating discovery, loading, selection, and observers.

**Placeholder backends** — working implementations, so the app runs end to end
and the wiring is testable without hardware or model weights.
- Synthetic camera, synthetic tracker, procedural engine, CPU letterbox
  renderer, discarding virtual-camera sink.

**Performance**
- Rolling-window metrics with nearest-rank percentiles, timestamp-derived FPS,
  dropped-frame counting, thread-safe snapshots.
- System resource probe interface reporting "not measured" rather than zero.

**Configuration and infrastructure**
- Layered configuration: defaults → TOML → local TOML → env/`.env` → overrides.
- Development/installed path resolution.
- Stdlib logging with rotation and optional JSON output. No network handlers.
- PySide6 shell: avatar selector, camera selector, Start/Stop, developer panel.
- `uv` + Python 3.12, ruff, mypy (strict), pytest, pre-commit, GitHub Actions CI
  on Windows with a gitleaks history scan.
- `scripts/check_gpu.py` — reports real GPU capability and launches an actual
  kernel, because import-time checks do not prove execution works.

**Documentation**
- System architecture, avatar system, open-source evaluation, security, privacy,
  performance plan, six ADRs, contributing guide, foundation status.

### Verified

- 180 tests pass.
- `mypy --strict` clean across 59 source files.
- `ruff check` and `ruff format --check` clean.
- `python -m sha256 --check` wires the application and discovers both sample
  avatars.
- GPU report confirms RTX 5070 Ti, 16303 MiB, compute capability 12.0, driver
  610.62.
- Package availability for Python 3.12 on Windows confirmed by dependency
  resolution for `mediapipe`, `onnxruntime-gpu`, `pyvirtualcam`,
  `opencv-python`, and CUDA 12.8/12.9/13.0 builds of PyTorch.

### Not included — deliberately

Face tracking, avatar animation, camera capture, virtual camera output, the
real-time processing loop, CPU/GPU/VRAM sampling, benchmark results, and the
avatar gallery UI. No performance figure is claimed anywhere in this release.

### Known limitations

- `LICENSE` is a restrictive placeholder; the project license is undecided.
- The virtual camera approach is unresolved (ADR-0005) and bears on that choice.
- LivePortrait's model-weight license is **unclear**, and its InsightFace
  dependency ships non-commercial-only weights. See
  `docs/OPEN_SOURCE_EVALUATION.md`.
- The residency policy is unit-tested against simulated pressure but has never
  evicted a real model.
- Avatar loading still runs on the UI thread (marked `TODO(threading)`).

