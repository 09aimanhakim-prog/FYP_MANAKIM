#include "HeadingController.h"

#include <Wire.h>
#include <math.h>

#include "Config.h"

namespace {
constexpr uint8_t MPU_ADDRESS = 0x68;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_GYRO_ZOUT_H = 0x47;
constexpr float GYRO_SENSITIVITY = 32.8f;  // +/-1000 deg/s
}

bool HeadingController::begin() {
  if (!startBusAndWakeSensor()) return false;
  delay(100);  // Setup-only sensor wake time; never used in the control loop.
  if (!configureGyro()) return false;

  int64_t sum = 0;
  uint16_t valid = 0;
  for (uint16_t i = 0; i < Config::GYRO_CALIBRATION_SAMPLES; ++i) {
    int16_t raw;
    if (readGyroZ(raw)) {
      sum += raw;
      ++valid;
    }
    delay(2);  // Calibration occurs once at boot while motors are safely off.
  }
  if (valid < Config::GYRO_CALIBRATION_SAMPLES * 9 / 10) return false;
  gyroBiasZ_ = static_cast<float>(sum) / valid;
  filteredRateDps_ = 0.0f;
  calibrated_ = true;
  healthy_ = true;
  consecutiveFailures_ = 0;
  lastMicros_ = micros();
  return true;
}

bool HeadingController::startBusAndWakeSensor() {
  Wire.end();
  Wire.begin(Config::MPU_SDA, Config::MPU_SCL);
  Wire.setClock(400000);
  Wire.setTimeOut(20);
  return writeRegister(REG_PWR_MGMT_1, 0x00);
}

bool HeadingController::configureGyro() {
  return writeRegister(REG_GYRO_CONFIG, 0x10);  // FS_SEL=2, +/-1000 deg/s
}

bool HeadingController::writeRegister(uint8_t address, uint8_t value) {
  Wire.beginTransmission(MPU_ADDRESS);
  Wire.write(address);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool HeadingController::readGyroZ(int16_t& rawZ) {
  for (uint8_t attempt = 0; attempt < Config::IMU_READ_RETRIES; ++attempt) {
    Wire.beginTransmission(MPU_ADDRESS);
    Wire.write(REG_GYRO_ZOUT_H);
    if (Wire.endTransmission(false) == 0 &&
        Wire.requestFrom(MPU_ADDRESS, static_cast<uint8_t>(2)) == 2) {
      rawZ = static_cast<int16_t>((Wire.read() << 8) | Wire.read());
      return true;
    }
    while (Wire.available()) Wire.read();
    delayMicroseconds(100);
  }
  return false;
}

void HeadingController::update() {
  if (manualCalibrating_) {
    updateManualCalibration();
    return;
  }
  if (!healthy_) {
    updateRecovery();
    return;
  }
  const uint32_t now = micros();
  const float dt = (now - lastMicros_) * 1.0e-6f;
  lastMicros_ = now;
  if (dt <= 0.0f || dt > 0.1f) return;
  int16_t rawZ;
  if (!readGyroZ(rawZ)) {
    recordReadFailure();
    return;
  }
  consecutiveFailures_ = 0;
  const float rawRate = (rawZ - gyroBiasZ_) / GYRO_SENSITIVITY;
  const float rateAlpha = dt / (Config::GYRO_RATE_FILTER_TAU_S + dt);
  filteredRateDps_ += rateAlpha * (rawRate - filteredRateDps_);

  // Learn residual temperature/bias error slowly, but only after the main
  // program has declared all actuators idle and the filtered gyro rate has
  // remained quiet. Using the filtered value prevents normal sample noise from
  // repeatedly resetting the stationary dwell timer.
  const uint32_t nowMs = millis();
  if (biasLearningAllowed_ &&
      fabsf(filteredRateDps_) <= Config::GYRO_STATIONARY_RATE_DPS) {
    if (stationarySinceMs_ == 0) stationarySinceMs_ = nowMs;
    if (nowMs - stationarySinceMs_ >= Config::GYRO_STATIONARY_DWELL_MS) {
      biasLearning_ = true;
      const float biasAlpha = dt / (Config::GYRO_BIAS_EMA_TAU_S + dt);
      gyroBiasZ_ += biasAlpha * (static_cast<float>(rawZ) - gyroBiasZ_);
    }
  } else {
    stationarySinceMs_ = 0;
    biasLearning_ = false;
  }

  headingDeg_ = wrap180(headingDeg_ + filteredRateDps_ * dt);
}

bool HeadingController::startCalibration() {
  if (!healthy_ || manualCalibrating_) return false;
  manualCalibrating_ = true;
  manualCalibrationSucceeded_ = false;
  manualCalibrationStartedMs_ = millis();
  manualCalibrationLastSampleUs_ = micros();
  manualCalibrationSamples_ = 0;
  manualCalibrationSum_ = 0;
  requestedEnabled_ = false;
  enabled_ = false;
  biasLearning_ = false;
  stationarySinceMs_ = 0;
  filteredRateDps_ = 0.0f;
  return true;
}

void HeadingController::updateManualCalibration() {
  const uint32_t nowMs = millis();
  if (nowMs - manualCalibrationStartedMs_ < Config::GYRO_MANUAL_CAL_SETTLE_MS) {
    return;
  }

  const uint32_t nowUs = micros();
  if (nowUs - manualCalibrationLastSampleUs_ < Config::GYRO_MANUAL_CAL_SAMPLE_US) {
    return;
  }
  manualCalibrationLastSampleUs_ = nowUs;

  int16_t rawZ;
  if (!readGyroZ(rawZ)) {
    ++failureCount_;
    manualCalibrating_ = false;
    manualCalibrationSucceeded_ = false;
    return;
  }

  const float rate = (rawZ - gyroBiasZ_) / GYRO_SENSITIVITY;
  if (fabsf(rate) > Config::GYRO_MANUAL_CAL_MAX_RATE_DPS) {
    manualCalibrating_ = false;
    manualCalibrationSucceeded_ = false;
    return;
  }

  manualCalibrationSum_ += rawZ;
  ++manualCalibrationSamples_;

  const uint32_t samplingElapsed =
      nowMs - manualCalibrationStartedMs_ - Config::GYRO_MANUAL_CAL_SETTLE_MS;
  if (samplingElapsed < Config::GYRO_MANUAL_CAL_DURATION_MS) return;

  if (manualCalibrationSamples_ < 1000) {
    manualCalibrating_ = false;
    manualCalibrationSucceeded_ = false;
    return;
  }

  gyroBiasZ_ = static_cast<float>(manualCalibrationSum_) /
               static_cast<float>(manualCalibrationSamples_);
  filteredRateDps_ = 0.0f;
  headingDeg_ = 0.0f;
  targetDeg_ = 0.0f;
  requestedEnabled_ = true;
  enabled_ = healthy_;
  calibrated_ = true;
  manualCalibrating_ = false;
  manualCalibrationSucceeded_ = true;
  lastMicros_ = micros();
}

uint8_t HeadingController::calibrationProgress() const {
  if (!manualCalibrating_) return manualCalibrationSucceeded_ ? 100 : 0;
  const uint32_t elapsed = millis() - manualCalibrationStartedMs_;
  if (elapsed <= Config::GYRO_MANUAL_CAL_SETTLE_MS) return 0;
  const uint32_t samplingElapsed = elapsed - Config::GYRO_MANUAL_CAL_SETTLE_MS;
  const uint32_t progress =
      samplingElapsed * 100 / Config::GYRO_MANUAL_CAL_DURATION_MS;
  return static_cast<uint8_t>(progress > 99 ? 99 : progress);
}

void HeadingController::recordReadFailure() {
  ++failureCount_;
  if (++consecutiveFailures_ < Config::IMU_FAILURE_THRESHOLD) return;
  healthy_ = false;
  enabled_ = false;
  biasLearning_ = false;
  stationarySinceMs_ = 0;
  recoveryState_ = RecoveryState::WAIT_RETRY;
  nextRecoveryMs_ = millis();
}

void HeadingController::updateRecovery() {
  if (!calibrated_) return;  // A valid gyro bias must come from stationary boot.
  const uint32_t now = millis();
  if (static_cast<int32_t>(now - nextRecoveryMs_) < 0) return;

  if (recoveryState_ == RecoveryState::WAIT_RETRY) {
    if (startBusAndWakeSensor()) {
      recoveryState_ = RecoveryState::WAIT_WAKE;
      nextRecoveryMs_ = now + Config::IMU_WAKE_TIME_MS;
    } else {
      ++failureCount_;
      nextRecoveryMs_ = now + Config::IMU_RECOVERY_RETRY_MS;
    }
    return;
  }

  if (recoveryState_ == RecoveryState::WAIT_WAKE) {
    int16_t rawZ;
    if (configureGyro() && readGyroZ(rawZ)) {
      healthy_ = true;
      enabled_ = requestedEnabled_;
      consecutiveFailures_ = 0;
      ++recoveryCount_;
      recoveryState_ = RecoveryState::IDLE;
      filteredRateDps_ = 0.0f;
      lastMicros_ = micros();
    } else {
      ++failureCount_;
      recoveryState_ = RecoveryState::WAIT_RETRY;
      nextRecoveryMs_ = now + Config::IMU_RECOVERY_RETRY_MS;
    }
  }
}

void HeadingController::setTarget(float degrees) {
  targetDeg_ = wrap180(degrees);
  requestedEnabled_ = true;
  enabled_ = healthy_;
}

void HeadingController::enable(bool enabled) {
  requestedEnabled_ = enabled;
  enabled_ = enabled && healthy_;
}

void HeadingController::zero() {
  headingDeg_ = 0.0f;
  targetDeg_ = 0.0f;
}

void HeadingController::setBiasLearningAllowed(bool allowed) {
  biasLearningAllowed_ = allowed;
  if (!allowed) {
    stationarySinceMs_ = 0;
    biasLearning_ = false;
  }
}

float HeadingController::correction() {
  const float error = wrap180(targetDeg_ - headingDeg_);
  lastInsideTolerance_ = fabsf(error) <= Config::HEADING_TOLERANCE_DEG;
  if (!enabled_ || !healthy_ || lastInsideTolerance_) {
    lastRawRotationCommand_ = 0.0f;
    lastLimitedRotationCommand_ = 0.0f;
    return 0.0f;
  }
  const float kp = Config::WHEEL_PID_ENABLED ? Config::PID_HEADING_KP
                                             : Config::HEADING_KP;
  const float minimum = Config::WHEEL_PID_ENABLED
                            ? Config::PID_MIN_ROTATE_COMMAND
                            : Config::MIN_ROTATE_PWM;
  const float maximum = Config::WHEEL_PID_ENABLED
                            ? Config::PID_MAX_ROTATE_COMMAND
                            : Config::MAX_ROTATE_PWM;
  const float kd = Config::WHEEL_PID_ENABLED ? Config::PID_HEADING_KD : 0.0f;

  // Gyro rate has the same sign as chassis rotation. Subtracting it damps a
  // turn already moving toward the target and helps prevent overshoot.
  lastRawRotationCommand_ = kp * error - kd * filteredRateDps_;
  const float errorDirection = error > 0.0f ? 1.0f : -1.0f;
  // Only retain raw effort that points toward the target. When gyro-rate damping
  // is larger than P, coast instead of forcing the minimum command and adding
  // more momentum. The minimum remains active when forward corrective effort is
  // actually required, preventing a low-speed stall near the target.
  const float effortTowardTarget =
      lastRawRotationCommand_ * errorDirection;
  float outputMagnitude = 0.0f;
  if (effortTowardTarget > 0.0f) {
    outputMagnitude = constrain(effortTowardTarget, minimum, maximum);
  }

  // Damping never actively reverses the requested turn; any corrective reversal
  // comes from the heading error itself crossing zero outside the deadband.
  lastLimitedRotationCommand_ = errorDirection * outputMagnitude;
  return lastLimitedRotationCommand_;
}

float HeadingController::headingError() const {
  return wrap180(targetDeg_ - headingDeg_);
}

float HeadingController::wrap180(float degrees) {
  while (degrees > 180.0f) degrees -= 360.0f;
  while (degrees <= -180.0f) degrees += 360.0f;
  return degrees;
}
