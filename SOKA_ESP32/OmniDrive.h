#pragma once

#include "MotorControl.h"
#include "WheelSpeedController.h"

class OmniDrive {
 public:
  OmniDrive(MotorControl& m1, MotorControl& m2, MotorControl& m3);
  void command(float x, float y, float rotation);
  void update(float rpm1, float rpm2, float rpm3, float dtSeconds);
  void stop();
  float targetRpm(uint8_t index) const;
  int outputPwm(uint8_t index) const;
  float appliedX() const { return appliedX_; }
  float appliedY() const { return appliedY_; }

 private:
  void updateTranslationSlew(float targetX, float targetY);

  WheelSpeedController wheel1_;
  WheelSpeedController wheel2_;
  WheelSpeedController wheel3_;
  float appliedX_ = 0.0f;
  float appliedY_ = 0.0f;
  uint32_t lastTranslationUpdateUs_ = 0;
};
