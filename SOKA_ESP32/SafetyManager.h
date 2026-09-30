#pragma once

#include <Arduino.h>

#include "OmniDrive.h"

class SafetyManager {
 public:
  explicit SafetyManager(OmniDrive& drive) : drive_(drive) {}
  void begin();
  void noteDriveCommand();
  bool update();
  void stopDrive();
  bool driveEnabled() const { return driveEnabled_; }

 private:
  OmniDrive& drive_;
  uint32_t lastDriveCommandMs_ = 0;
  bool driveEnabled_ = false;
};
