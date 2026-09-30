#pragma once

#include "MotorControl.h"

class DribblerController {
 public:
  explicit DribblerController(MotorControl& motor) : motor_(motor) {}
  void set(int pwm);
  void stop();
  int command() const { return command_; }

 private:
  MotorControl& motor_;
  int command_ = 0;
};
