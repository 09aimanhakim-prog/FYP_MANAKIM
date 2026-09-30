#pragma once

#include "MotorControl.h"

class WheelSpeedController {
 public:
  WheelSpeedController(MotorControl& motor, float feedforwardGain,
                       int minimumEffectivePwm);
  void setCommand(float normalizedCommand);
  void update(float measuredRpm, float dtSeconds);
  void stop();
  float targetRpm() const { return targetRpm_; }
  int outputPwm() const { return outputPwm_; }

 private:
  MotorControl& motor_;
  float feedforwardGain_;
  int minimumEffectivePwm_;
  float targetRpm_ = 0.0f;
  float integral_ = 0.0f;
  float previousError_ = 0.0f;
  int outputPwm_ = 0;
};
