#include "WheelSpeedController.h"

#include <Arduino.h>
#include <math.h>

#include "Config.h"

WheelSpeedController::WheelSpeedController(MotorControl& motor,
                                           float feedforwardGain,
                                           int minimumEffectivePwm)
    : motor_(motor),
      feedforwardGain_(feedforwardGain),
      minimumEffectivePwm_(minimumEffectivePwm) {}

void WheelSpeedController::setCommand(float normalizedCommand) {
  normalizedCommand = constrain(normalizedCommand, -255.0f, 255.0f);
  targetRpm_ = normalizedCommand * Config::WHEEL_MAX_TARGET_RPM / 255.0f;

  if (!Config::WHEEL_PID_ENABLED) {
    outputPwm_ = lroundf(normalizedCommand);
    motor_.set(outputPwm_);
  } else if (fabsf(targetRpm_) < 1.0f) {
    stop();
  }
}

void WheelSpeedController::update(float measuredRpm, float dtSeconds) {
  if (!Config::WHEEL_PID_ENABLED || fabsf(targetRpm_) < 1.0f ||
      dtSeconds <= 0.0f) {
    return;
  }

  // Coast briefly before reversing a wheel that is still spinning rapidly in
  // the previous direction. This avoids abrupt active braking through BTS7960.
  if (targetRpm_ * measuredRpm < 0.0f &&
      fabsf(measuredRpm) > Config::WHEEL_DIRECTION_CHANGE_RPM) {
    integral_ = 0.0f;
    previousError_ = 0.0f;
    outputPwm_ = 0;
    motor_.stop();
    return;
  }

  const float direction = targetRpm_ > 0.0f ? 1.0f : -1.0f;
  const float feedforwardMagnitude = max(
      static_cast<float>(minimumEffectivePwm_),
      fabsf(targetRpm_) * feedforwardGain_);
  const float feedforward = direction * feedforwardMagnitude;
  const float error = targetRpm_ - measuredRpm;
  const float derivative = (error - previousError_) / dtSeconds;
  float candidateIntegral = constrain(
      integral_ + error * dtSeconds, -Config::WHEEL_PID_INTEGRAL_LIMIT,
      Config::WHEEL_PID_INTEGRAL_LIMIT);

  float unsaturated = feedforward + Config::WHEEL_PID_KP * error +
                      Config::WHEEL_PID_KI * candidateIntegral +
                      Config::WHEEL_PID_KD * derivative;
  const bool saturatedHigh = unsaturated > Config::DRIVE_MAX_PWM;
  const bool saturatedLow = unsaturated < -Config::DRIVE_MAX_PWM;
  if ((!saturatedHigh && !saturatedLow) ||
      (saturatedHigh && error < 0.0f) || (saturatedLow && error > 0.0f)) {
    integral_ = candidateIntegral;
  }

  unsaturated = feedforward + Config::WHEEL_PID_KP * error +
                Config::WHEEL_PID_KI * integral_ +
                Config::WHEEL_PID_KD * derivative;
  // Do not command reverse torque merely to correct overspeed.
  if (direction > 0.0f) unsaturated = constrain(unsaturated, 0.0f, 255.0f);
  else unsaturated = constrain(unsaturated, -255.0f, 0.0f);

  outputPwm_ = lroundf(unsaturated);
  motor_.set(outputPwm_);
  previousError_ = error;
}

void WheelSpeedController::stop() {
  targetRpm_ = 0.0f;
  integral_ = 0.0f;
  previousError_ = 0.0f;
  outputPwm_ = 0;
  motor_.stop();
}
