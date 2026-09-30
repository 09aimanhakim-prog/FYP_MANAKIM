#pragma once

#include <Arduino.h>

class HeadingController {
 public:
  bool begin();
  void update();
  void setTarget(float degrees);
  void enable(bool enabled);
  void zero();
  bool startCalibration();
  void setBiasLearningAllowed(bool allowed);
  float correction();
  float heading() const { return headingDeg_; }
  float target() const { return targetDeg_; }
  float headingError() const;
  float rawRotationCommand() const { return lastRawRotationCommand_; }
  float limitedRotationCommand() const { return lastLimitedRotationCommand_; }
  float gyroRateDegreesPerSecond() const { return filteredRateDps_; }
  bool insideTolerance() const { return lastInsideTolerance_; }
  bool enabled() const { return enabled_; }
  bool healthy() const { return healthy_; }
  uint32_t failureCount() const { return failureCount_; }
  uint32_t recoveryCount() const { return recoveryCount_; }
  float biasDegreesPerSecond() const { return gyroBiasZ_ / 32.8f; }
  bool biasLearning() const { return biasLearning_; }
  bool calibrating() const { return manualCalibrating_; }
  uint8_t calibrationProgress() const;
  bool calibrationSucceeded() const { return manualCalibrationSucceeded_; }

 private:
  enum class RecoveryState : uint8_t { IDLE, WAIT_RETRY, WAIT_WAKE };

  bool startBusAndWakeSensor();
  bool configureGyro();
  bool writeRegister(uint8_t address, uint8_t value);
  bool readGyroZ(int16_t& rawZ);
  void recordReadFailure();
  void updateRecovery();
  void updateManualCalibration();
  static float wrap180(float degrees);
  float headingDeg_ = 0.0f;
  float targetDeg_ = 0.0f;
  float gyroBiasZ_ = 0.0f;
  float filteredRateDps_ = 0.0f;
  uint32_t lastMicros_ = 0;
  uint32_t nextRecoveryMs_ = 0;
  uint32_t stationarySinceMs_ = 0;
  uint32_t failureCount_ = 0;
  uint32_t recoveryCount_ = 0;
  uint32_t manualCalibrationStartedMs_ = 0;
  uint32_t manualCalibrationLastSampleUs_ = 0;
  uint32_t manualCalibrationSamples_ = 0;
  int64_t manualCalibrationSum_ = 0;
  uint8_t consecutiveFailures_ = 0;
  bool enabled_ = false;
  bool requestedEnabled_ = false;
  bool healthy_ = false;
  bool calibrated_ = false;
  bool biasLearningAllowed_ = false;
  bool biasLearning_ = false;
  bool manualCalibrating_ = false;
  bool manualCalibrationSucceeded_ = false;
  float lastRawRotationCommand_ = 0.0f;
  float lastLimitedRotationCommand_ = 0.0f;
  bool lastInsideTolerance_ = true;
  RecoveryState recoveryState_ = RecoveryState::IDLE;
};
