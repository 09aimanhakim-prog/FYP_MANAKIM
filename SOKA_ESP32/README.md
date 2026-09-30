# SOKA autonomous soccer robot — ESP32 firmware

Arduino IDE project for ESP32 board package 3.3.7. Open `SOKA_ESP32.ino` and
compile for the installed ESP32 Dev Module target. The firmware uses only Arduino
core APIs (`Wire`, GPIO, interrupts, and the core 3.x LEDC API).

## Implemented modules

- `MotorControl`: safe BTS7960 bidirectional PWM
- `OmniDrive`: continuous X/Y/rotation mixing and normalization
- `EncoderManager`: rising-edge counts and signed RPM telemetry
- `HeadingController`: MPU6050 raw I2C, ±1000 deg/s yaw integration, P hold
- `DribblerController`: reversed motor direction and variable PWM
- `KickerController`: non-blocking 50 ms relay pulse and cooldown
- `SerialProtocol`: bounded newline-delimited command parser
- `SafetyManager`: 300 ms drive-command watchdog
- `WheelSpeedController`: per-wheel RPM feedforward plus PID and anti-windup

The heading controller retries transient reads and automatically reinitializes the
I2C bus and MPU6050 after repeated communication failures. `STATUS` fields `IMUF`
and `IMUR` report total failed read/recovery operations and successful automatic
recoveries. Recovery preserves the boot-time gyro bias and target heading; yaw
movement during the outage cannot be reconstructed.

While the drivetrain, dribbler, and kicker are inactive, the controller waits for
two seconds below 2 deg/s filtered angular rate and then slowly updates residual gyro bias
with a 30-second time-constant EMA. It stops learning immediately when motion or
an actuator command occurs. `GB` reports bias in deg/s and `BL=1` indicates active
bias learning. A separate short EMA smooths gyro-rate noise before integration;
neither filter provides an absolute yaw reference.

`CALIBRATE` stops all actuators and rejects actuator commands while it runs. It
reports progress in 10% steps, calculates a fresh one-minute mean gyro bias, and
sets the current orientation to heading zero when complete. A gyro read failure
or detected movement above 5 deg/s aborts calibration instead of saving a
contaminated bias.

## Serial commands (115200 baud)

- `MOVE,45,80`
- `HEADING,70`
- `CTRL,45,80,0`
- `DRIBBLER,ON` (default PWM 210), `DRIBBLER,OFF`, or `DRIBBLER,0..255`
- `KICK`
- `STOP` (drive, dribbler, and active relay pulse off)
- `STATUS`
- `ZERO`
- `CALIBRATE` (2 s settling plus 60 s stationary calibration)
- `HEADING_OFF`

Send `MOVE` or `CTRL` continuously faster than every 300 ms while motion is
desired; 20 Hz is recommended. A timeout stops the three drive motors and emits
`EVENT,WATCHDOG_STOP`.

## Wheel-speed control

The normalized omni wheel commands are converted to targets using 400 RPM as the
initial rated-speed ceiling. Each drive wheel applies a configurable feedforward,
minimum effective PWM, proportional/integral correction, saturation protection,
and a coast-before-reverse guard. The watchdog and zero target reset all PID
state. Set `WHEEL_PID_ENABLED=false` in `Config.h` for immediate open-loop
fallback.

Translation intent has a robot-level vector slew limiter before omni kinematics.
`DRIVE_TRANSLATION_ACCEL_COMMAND_PER_S=300` makes 0 to command 100 take about
0.33 seconds; `DRIVE_TRANSLATION_DECEL_COMMAND_PER_S=450` returns 100 to zero in
about 0.22 seconds. X and Y ramp together along the requested vector, and a
direction reversal passes through zero. Rotation from heading correction is not
delayed. Explicit STOP, heading disable, calibration stop, and the 300 ms
watchdog call `drive.stop()` and bypass the ramp immediately. Set
`DRIVE_TRANSLATION_SLEW_ENABLED=false` only for comparison testing.

`STATUS` now reports requested `X/Y` and ramped `AX/AY`. During a repeated
`CTRL,0,100,0` test, Y remains 100 while AY rises smoothly toward 100 and the TR
values rise with it. An explicit STOP should still make AX/AY and PWM zero
immediately.

Heading correction uses a PD command while wheel PID is enabled because rotation
is interpreted as target velocity rather than raw PWM. Current starting values
are Kp 2.0, gyro-rate damping Kd 0.25, minimum command 14, maximum command 100,
and a 2-degree deadband. Large errors therefore request up to about 157 RPM on
all three rotation wheels, while proportional reduction and gyro-rate damping
slow the chassis near the target. When gyro-rate damping exceeds proportional
effort, the rotation target coasts at zero instead of forcing the minimum command
and adding momentum. The original heading constants remain available for
open-loop fallback.

`STATUS` reports measured RPM (`RPM1..3`), target RPM (`TR1..3`), and controller
output (`PW1..3`). It also reports heading error (`HERR`), filtered gyro rate
(`HRATE`), raw PD output (`RRAW`), limited rotation command (`RLIM`), and tolerance
state (`HIT`). The updated heading gains require raised-wheel validation before
floor testing.

## Heading tuning test

Use a controller that continuously refreshes each `CTRL` command faster than the
300 ms watchdog. With raised wheels first, test targets in this order:

```text
ZERO
CTRL,0,0,0
CTRL,0,0,30
CTRL,0,0,45
CTRL,0,0,90
CTRL,0,0,-90
STOP
```

Request `STATUS` several times during each turn and after it settles. Do not judge
camera alignment until these direct heading steps show fast initial rotation,
controlled slowdown, an accurate stop, and little or no repeated overshoot.

## First powered test

Keep the wheels raised and the chassis stationary for approximately 2 seconds
after reset while gyro bias calibration runs. Confirm `IMU=OK`, test translation
at low command values, then verify heading correction direction. Do not alter the
kinematic equations to fix an individual reversed wheel; direction signs are
isolated in `Config.h`.

The drive encoder conversion uses 206.8 rising-edge counts/output revolution.
The dribbler gearbox ratio was not specified, so its CPR remains an explicitly
marked placeholder until measured or confirmed.
