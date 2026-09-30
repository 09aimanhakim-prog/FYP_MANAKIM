#pragma once

#include <Arduino.h>

class KickerController {
 public:
  void begin();
  bool trigger();
  void update();
  void stop();
  bool active() const { return active_; }

 private:
  void writeRelay(bool on);
  bool active_ = false;
  bool hasKicked_ = false;
  uint32_t pulseStartedMs_ = 0;
  uint32_t lastKickMs_ = 0;
};
