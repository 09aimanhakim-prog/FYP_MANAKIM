#pragma once

#include <Arduino.h>

namespace Config {
// Exact hardware assignments from the robot wiring specification.
constexpr uint8_t M1_RPWM = 25;
constexpr uint8_t M1_LPWM = 26;
constexpr uint8_t M1_ENC_A = 34;
constexpr uint8_t M1_ENC_B = 35;

constexpr uint8_t M2_RPWM = 27;
constexpr uint8_t M2_LPWM = 14;
constexpr uint8_t M2_ENC_A = 4;
constexpr uint8_t M2_ENC_B = 16;

constexpr uint8_t M3_RPWM = 12;
constexpr uint8_t M3_LPWM = 13;
constexpr uint8_t M3_ENC_A = 36;
constexpr uint8_t M3_ENC_B = 39;

constexpr uint8_t M4_RPWM = 32;
constexpr uint8_t M4_LPWM = 33;
constexpr uint8_t M4_ENC_A = 18;
constexpr uint8_t M4_ENC_B = 19;

constexpr uint8_t MPU_SDA = 21;
constexpr uint8_t MPU_SCL = 22;
constexpr uint8_t KICKER_PIN = 23;

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t PWM_FREQUENCY = 20000;
constexpr uint8_t PWM_RESOLUTION = 8;
constexpr int DRIVE_MAX_PWM = 255;
constexpr int DRIBBLER_DEFAULT_PWM = 210;

// Direction correction is separate from the tested kinematic equations.
constexpr int8_t M1_DIRECTION = 1;
constexpr int8_t M2_DIRECTION = 1;
constexpr int8_t M3_DIRECTION = 1;
constexpr int8_t M4_DIRECTION = -1;

constexpr float DRIVE_COUNTS_PER_REV = 11.0f * 18.8f;
// Replace after confirming the gearbox ratio of the 1000 RPM dribbler motor.
constexpr float DRIBBLER_COUNTS_PER_REV = 11.0f;
constexpr uint32_t RPM_SAMPLE_MS = 100;

// Drive-wheel velocity control. Set false for immediate open-loop fallback.
constexpr bool WHEEL_PID_ENABLED = true;
constexpr float WHEEL_MAX_TARGET_RPM = 400.0f;
constexpr float WHEEL_PID_KP = 0.35f;
constexpr float WHEEL_PID_KI = 0.60f;
constexpr float WHEEL_PID_KD = 0.00f;
constexpr float WHEEL_PID_INTEGRAL_LIMIT = 250.0f;
constexpr float WHEEL_DIRECTION_CHANGE_RPM = 20.0f;
constexpr float M1_FEEDFORWARD_GAIN = 0.55f;
constexpr float M2_FEEDFORWARD_GAIN = 0.55f;
constexpr float M3_FEEDFORWARD_GAIN = 0.50f;
constexpr int M1_MIN_EFFECTIVE_PWM = 55;
constexpr int M2_MIN_EFFECTIVE_PWM = 55;
constexpr int M3_MIN_EFFECTIVE_PWM = 55;

// Robot-level translation slew limiting. X/Y intent is ramped before omni
// kinematics so all three wheel targets change together without bending the
// requested travel direction. STOP/watchdog call drive.stop() and remain
// immediate. Heading correction is deliberately not slowed here.
constexpr bool DRIVE_TRANSLATION_SLEW_ENABLED = true;
constexpr float DRIVE_TRANSLATION_ACCEL_COMMAND_PER_S = 300.0f;
constexpr float DRIVE_TRANSLATION_DECEL_COMMAND_PER_S = 450.0f;
constexpr float DRIVE_TRANSLATION_SLEW_MAX_DT_S = 0.05f;

constexpr float HEADING_KP = 3.0f;
// Heading tuning. With wheel PID enabled these values are normalized omni
// rotation commands, later converted to wheel RPM by WheelSpeedController.
// A proportional term gives strong large-error turning, while gyro-rate damping
// reduces output as the chassis approaches the target at speed.
constexpr float HEADING_TOLERANCE_DEG = 2.0f;
constexpr float MIN_ROTATE_PWM = 55.0f;
constexpr float MAX_ROTATE_PWM = 110.0f;
// With wheel PID enabled, rotation input becomes a velocity target rather than
// direct PWM. These lower values prevent the old PWM tuning from commanding an
// excessive rotational wheel speed.
constexpr float PID_HEADING_KP = 2.0f;
constexpr float PID_HEADING_KD = 0.25f;
constexpr float PID_MIN_ROTATE_COMMAND = 14.0f;
constexpr float PID_MAX_ROTATE_COMMAND = 100.0f;
constexpr uint16_t GYRO_CALIBRATION_SAMPLES = 1000;
constexpr uint8_t IMU_READ_RETRIES = 3;
constexpr uint8_t IMU_FAILURE_THRESHOLD = 3;
constexpr uint32_t IMU_RECOVERY_RETRY_MS = 250;
constexpr uint32_t IMU_WAKE_TIME_MS = 100;
constexpr float GYRO_RATE_FILTER_TAU_S = 0.015f;
constexpr float GYRO_BIAS_EMA_TAU_S = 30.0f;
constexpr float GYRO_STATIONARY_RATE_DPS = 2.0f;
constexpr uint32_t GYRO_STATIONARY_DWELL_MS = 2000;
constexpr uint32_t GYRO_MANUAL_CAL_SETTLE_MS = 2000;
constexpr uint32_t GYRO_MANUAL_CAL_DURATION_MS = 60000;
constexpr uint32_t GYRO_MANUAL_CAL_SAMPLE_US = 2000;
constexpr float GYRO_MANUAL_CAL_MAX_RATE_DPS = 5.0f;

constexpr uint32_t DRIVE_COMMAND_TIMEOUT_MS = 300;
constexpr uint32_t KICK_PULSE_MS = 50;
constexpr uint32_t KICK_COOLDOWN_MS = 1000;
constexpr bool KICKER_ACTIVE_HIGH = true;
}
