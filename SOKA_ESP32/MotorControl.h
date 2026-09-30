#pragma once

#include <Arduino.h>

class MotorControl {
 public:
  MotorControl(uint8_t rpwmPin, uint8_t lpwmPin, int8_t direction = 1);
  void begin();
  void set(int command);
  void stop();

 private:
  uint8_t rpwmPin_;
  uint8_t lpwmPin_;
  int8_t direction_;
};

