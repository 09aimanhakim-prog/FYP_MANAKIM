# SOKA Mini PC serial test controller

This tool continuously sends the selected `CTRL` command at 20 Hz, comfortably
inside the ESP32 firmware's 300 ms watchdog interval. It also displays responses
and sends `STOP` during normal shutdown.

## Windows setup

Install Python 3 on the robot Mini PC and select the installer option **Add Python
to PATH**. Then open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python serial_test_controller.py --list
python serial_test_controller.py --port COM5
```

Replace `COM5` with the ESP32 port shown by `--list`. Close Arduino Serial Monitor
before running the controller because only one program can own the COM port.

Keep the wheels raised for the first test. At the `soka>` prompt, enter:

```text
zero
calibrate
ctrl 0 0 70
status
stop
ctrl 60 80 0
stop
quit
```

Pressing Ctrl+C also exits through the safe shutdown path. A USB disconnect still
causes the ESP32's independent 300 ms watchdog to stop the drivetrain.

## Goalkeeper mode

The web interface has a separate **Goalkeeper** tab and safety-arm button. It
uses the selected ball HSV detector and commands lateral X movement only:
`CTRL,X,0,0`. The target speed is proportional to horizontal pixel error, from
X command 24 up to 70, with an 18 px center tolerance, a 30 px release band,
and an EMA to suppress camera jitter. The robot therefore moves strongly for a
far-off-center ball and slows near the camera center line. It commands X=0 on
the first frame without a ball, keeps Y=0, holds zero heading, and does not run
the dribbler or kicker. Test with raised wheels before placing it on the goal
line. The browser now uses a green-pitch soccer theme; the controller itself
continues running on the Mini PC after it is armed and is not driven by browser
joystick updates.

## Cone Course mode

The **Cone Course** tab is an independent SAFE-by-default mode for a robot that
starts facing the colored goal. Arming sends `ZERO`, holds target heading 0°, and
uses only omni X/Y translation. `cone_course.py` thresholds dark HSV pixels in an
adjustable lower floor ROI, rejects tiny/wide shadow-like blobs, and retains a
valid cone for 280 ms across brief detection dropouts. A seven-candidate local
corridor planner inflates each cone by a perspective-dependent robot clearance,
prefers the remembered goal direction, and penalizes lane switching. Y reduces
as cones approach. Two adjustable red lower-corner danger zones now cover cones
that are already beside the robot as well as those in the forward corridor. A
cone crossing either zone forces Y=0 and latches a command-58 strafe away for at
least 0.9 seconds plus five clear frames. Similar close threats on both sides stop
the robot instead of choosing a blind escape direction. The danger line defaults
to 72% image height and each side zone defaults to 34% image width.

When Cone Course is armed, the current encoder-odometry X position is saved as
the course centerline. Cone avoidance always has priority. Once no cones remain,
an offset of at least 0.12 m activates a limited X return while Y is reduced to
18; correction stops inside 0.06 m. This hysteresis prevents left/right chatter.
Inside the center band, clear-path X is held at zero so an occluded or noisy goal
centroid cannot gradually pull the robot away from its original route. If
odometry is unavailable, the original camera corridor planner remains the fallback.

The preferred close-cone behavior is now a visual orbit rather than a blind
strafe. When the nearest cone crosses the adjustable magenta orbit line (default
60%), the controller selects the clearer pass side and locks that cone. It keeps
the cone near 28% or 72% of the camera width by changing gyro heading while a
field-relative forward-plus-sideways vector moves the chassis around it. Because
translation is transformed through current heading, the robot can look toward
the cone without losing its intended direction across the field. Very close
cones reduce or stop forward motion and increase the away component.

The cone is considered passed after the configured mapped forward distance
(default 0.55 m) and improved visual clearance, after a larger forced distance,
or after a sustained loss. Translation then stops while heading returns to 0°.
The controller resumes cruise/recentering and can lock the next cone. The older
red side zones remain an emergency fallback during cooldowns or ambiguous cases.

Course arming records the starting odometry Y. When mapped progress reaches the
adjustable threshold (default 75% of field length), close cones still have first
priority, but otherwise the robot stops and scans heading -38° to +38° for the
selected goal. Once currently visible, the goal bearing controls heading; forward
approach is allowed only when it is reasonably centered. Four strict frames past
the goal stop line still stop the course. If odometry is stale, map-triggered goal
search remains disabled rather than guessing that the robot is near the goal.

Goal detection remains secondary to collision avoidance. A strict goal detection
updates memory quickly; when cones split the colored interior, up to three aligned
color fragments provide a slow partial update. If the goal is hidden completely,
the last bearing is retained for four seconds, after which the initial straight
heading is used. Four strict goal frames beyond the existing orange shoot-distance
line stop the robot; this mode never kicks or runs the dribbler.

Tune while SAFE until real cones receive red `BLACK CONE` boxes and floor shadows,
tile lines, wheels, and goal structure do not. Start with the defaults: black V
cutoff 70, ROI top 38%, minimum area 180, and conservative clear Y command 30. Test raised
wheels, then one cone at low forward speed before attempting the full course.

The controller hides repetitive `OK,CTRL` acknowledgements so continuous 20 Hz
transmission does not interfere with typing. Status lines, errors, and safety
events remain visible.

For heading tuning, enter `log on` before a test. The controller requests STATUS
at 10 Hz and writes it to a timestamped `heading_telemetry_*.log` file beside the
script without printing those lines over the prompt. Enter `log off` after the
turn sequence and share that file for tuning review.

## Front-camera ball tracker

`ball_tracker.py` detects round HSV blobs without a trained model. It includes
presets for yellow, blue, red, and orange, live trackbars, click-to-sample hue,
shape filtering, target lock, and JSON setting persistence.

Start in vision-only mode:

```powershell
python ball_tracker.py --camera 0
```

Start with ESP32 communication available but robot motion initially safe/off:

```powershell
python ball_tracker.py --camera 0 --serial COM3
```

Window controls:

- `1`, `2`, `3`, `4`: yellow, blue, red, orange
- Click the ball: sample its hue into the selected preset
- `S`: save current tuning to `ball_colors.json`
- `C`: run the ESP32 one-minute gyro calibration (when serial is connected)
- `Space`: arm/disarm robot ball tracking
- `Q` or Escape: stop and quit

When armed in the web controller, the camera converts horizontal pixel error into
an estimated ball-bearing angle using a starting 70-degree C920 horizontal field
of view. It commands that heading in one turn, waits until the measured gyro is
within 2 degrees of the target, settles for 120 ms, and averages three frames for
a final correction. It holds the resulting heading during the forward approach.
If the ball leaves the wider release region for five frames, forward motion pauses
while the robot realigns. Losing the target immediately sends `STOP` and disarms
tracking. Keep the wheels raised during initial direction verification.

## Browser camera and joystick

`web_control.py` provides the annotated camera, ball-color selection,
autonomous arm/stop controls, actuator tests, and a touch/mouse X/Y joystick in
one local webpage. Close Arduino Serial Monitor and the desktop tracker first so
that this program can own COM3 and camera 0.

Install the updated requirements, then start it on the robot Mini PC:

```powershell
python -m pip install -r requirements.txt
python web_control.py --camera 0 --side-camera 1 --serial COM3
```

Omit `--side-camera 1` when the second camera is disconnected. Windows camera
indices can change after moving USB ports; if the feeds are reversed, swap the
`--camera` and `--side-camera` numbers.

On the robot, open `http://127.0.0.1:8080`. From another phone or computer on the
same Wi-Fi network, open `http://ROBOT-IP:8080`; obtain `ROBOT-IP` by running
`ipconfig` on the robot Mini PC. If Windows asks, allow Python on **Private
networks** only.

The page always starts in SAFE mode and has separate **Autonomous**, **Manual
Control**, **Ball Calibration**, and **Goal Calibration** sections. Manual mode
has an X/Y joystick and a spring-return rotation control. Rotation changes the
commanded gyro heading at up to 90 degrees/second, so translation and turning can
be used together.
Releasing the rotation control holds the newly requested heading. Releasing the
X/Y joystick sends zero translation, and the ESP32's independent 300 ms watchdog
remains active.

Manual mode also provides a press-and-hold D-pad for exact straight commands.
Forward/back always send X=0, and left/right always send Y=0, using the adjustable
20-100 button-speed value. Releasing a direction immediately sends zero
translation. The rotation pad remains usable while a D-pad direction is held.

The Ball Calibration page learns HSV thresholds without machine learning. Select
yellow, blue, red, or orange, press **Start new calibration**, then click the
literal center of the ball in the live picture. Each click collects a 12-frame
burst from a small circular patch. Move the ball to a different distance or
lighting area and repeat; three samples are recommended. **Save & apply** combines
the samples, rejects grey/dark pixels and off-hue outliers, calculates a hue
center/tolerance and minimum saturation/value, and saves the result to
`ball_colors.json`. The normal tracker uses the new values immediately. The robot
is forced to SAFE during this process, and the sample pixels are discarded after
the settings are calculated.

Goal calibration is independent. Select blue or yellow, start a goal calibration,
then drag a box fully inside the solid colored goal interior. Avoid including the
floor, walls, or posts. Each box collects a 12-frame subsampled HSV burst; combine
two or three viewpoints and press **Save & apply**. Values persist in
`goal_colors.json`. The goal detector then looks for a large, wide connected
region with suitable rectangularity plus left, right, and lower edge support. A
same-colored goal candidate is excluded from the ball candidate list when its
size makes it clearly the goal rather than a ball inside the goal mouth.

Starting calibration switches the robot to SAFE mode and places a `CALIBRATING`
progress bar over the live camera using the ESP32's `CAL` and `CALP` status values.
Keep the robot completely still until `CALIBRATION COMPLETE` appears.

Autonomous mode is continuous. A brief target dropout pauses translation while
holding heading. A longer loss searches through six 60-degree gyro targets for a
full 360-degree left rotation, then six targets for a full 360-degree right
rotation, repeating until reacquisition. Seeing a ball starts the dribbler at PWM
210. The existing direct-bearing controller aligns and approaches it. At the near-
camera trigger row, or when a previously close ball disappears under the camera,
the robot moves forward at command 55 for the user-tuned 1.50 seconds. Measure
actual floor travel and tune the duration for acceleration and slip. It then enters goal search with
the dribbler running. Press STOP to end the behavior and stop the dribbler.

Possession goal search deliberately does not rotate in place. It repeats
`CTRL,0,27,H` while advancing H left at 22 degrees/second, producing a tighter rolling
circle that keeps the captured ball pressed into the front dribbler. After three
goal frames, the camera bearing changes the gyro target while Y=22 creates a
shallow alignment arc. Centered goal approach uses Y=42. Goal loss briefly
continues the arc and then resumes the rolling-circle search.

The orange goal shoot line is adjustable from 5–99% image height in SAFE mode.
The robot becomes shoot-ready only after the goal bottom crosses this line and
the goal remains within 24 pixels of center for six frames. **Automatic kick is
OFF by default.** With it off the robot stops at `SHOOT READY`. With it explicitly
enabled before arming, the robot settles for 250 ms, sends one `KICK`, waits 700
ms, turns off the dribbler, and returns to ball search. Validate detection and
distance with automatic kick off first. If the goal bottom moves 5 percentage
points beyond the shoot line, `GOAL BACKOFF` reverses at Y=-28 while retaining
camera-to-heading alignment. It exits at 1.5 points beyond the line, providing
hysteresis so normal detection noise does not cause forward/reverse chatter.

The **Side Safety Camera** page uses the optional second/top camera at a requested
640x480 and 15 FPS in a separate capture thread, so it cannot block the front-
camera loop. Only two green rectangles—left and right—are active. Everything else
is visibly darkened/red and ignored. SAFE-mode sliders independently adjust the
shared top/bottom boundaries and each side's inner/outer edge. Constraints prevent
inverted or overlapping zones, and the six percentages persist under `side_mask`
in `web_settings.json`.

Floor calibration uses OpenCV Lab color because its lightness and chroma channels
are more useful than hue for pale/grey floors. While SAFE, start floor calibration
and click clean floor inside either green rectangle. Each click collects a 12-frame
circular patch from the raw image (not the red overlay). Take samples from both
sides and from several brightness levels while avoiding grout/tile lines, strong
reflections and shadows. Save & apply writes `floor_color.json`. The live preview
then removes pixels matching the learned floor, cleans thin noise morphologically,
and draws `FOREIGN OBJECT` around sufficiently large remaining blobs.

Movement is independently armed by **Enable autonomous opposite-side strafe** and
defaults OFF. Four consecutive obstacle frames activate a side; seven clear frames
release it. In autonomous mode, left blocked overrides motion with `CTRL,+35,0,H`,
right blocked uses `CTRL,-35,0,H`, and both sides blocked use `CTRL,0,0,H`. A side-
camera disconnect fails safe by stopping translation. An active side obstruction
also cancels an automatic shot before KICK is sent. Preview detection with movement
OFF first, then verify wheel direction raised before a careful floor test.

This remains color-difference avoidance: an object nearly identical to the floor
may not be detected, and a lighting change can require recalibration. It does not
replace the planned general obstacle/robot detector.

Inside the approximate 30 cm image row, alignment changes from rotation to a
gyro-held lateral strafe. Horizontal error is EMA-filtered with center hysteresis.
Strafe command ranges from 30 to 55; large error pauses forward travel, moderate
error uses Y=25, and centered travel uses Y=50. The normal omni equations still
apply, so M3 supplies the largest lateral component while M1 and M2 each supply
the required half component. Capture begins only when the close ball is centered;
an off-center disappearance does not automatically assume possession.

The Autonomous panel has SAFE-mode sliders for the near/strafe and capture lines.
Their values are image-height percentages measured from the top. The capture line
must remain at least five percentage points below the near line. The overlays and
the actual autonomous thresholds update together, and values persist in
`web_settings.json` across restarts.

## Field map and encoder odometry

The **Field Map** page draws a top-down soccer pitch, the robot's estimated pose,
heading nose, speed, wheel RPM values, and a cyan travel trail. Enter the measured
field width and length and save them while the robot is SAFE. Place the robot at
the center of the starting edge facing the far goal, then press **Set current pose
as start**. This resets the map to `(0, 0)` and sends `ZERO`, so that physical
direction becomes field heading 0 degrees.

`odometry.py` converts the existing RPM1-RPM3 telemetry back through the tested
three-wheel omni equations, converts wheel speed using the 60 mm wheel diameter,
rotates that robot-relative velocity using the MPU heading, and integrates it at
each STATUS timestamp. Common equal wheel speed from pure rotation cancels from
the translation estimate. No ESP32 firmware upload is required for this page.

The map is relative dead reckoning: tyre slip, wheel-size error, uneven floors,
and gyro drift accumulate into position error. Calibrate the two persisted scale
values separately. After resetting the pose, drive exactly 1.00 m forward and set
`Forward Y scale = 1.00 / displayed_distance`; repeat sideways for X. For example,
if a 1.00 m test displays 0.80 m, use scale 1.25. Reset and repeat the measurement
after changing a scale.

The map now supports one showcase waypoint on a clear floor. While SAFE, click a
point inside the pitch and then explicitly press **Arm selected waypoint**. The
yellow target and dashed route appear on the canvas. `waypoint.py` converts the
field-position error into robot-relative X/Y using current gyro heading, holds
target heading 0 degrees, slows within 0.65 m, and stops after entering the
adjustable arrival radius (default 0.15 m). Maximum command defaults to 45.

Waypoint mode stops if encoder telemetry becomes stale and preserves the ESP32
watchdog, but it intentionally does not run cone or side-camera avoidance. Use it
only in a cleared showcase area, begin at low speed with raised wheels, and keep
the web STOP button immediately accessible. Accumulated wheel slip can still make
the mapped target differ from the physical target.
