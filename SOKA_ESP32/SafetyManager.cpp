#include "SafetyManager.h"

#include "Config.h"

void SafetyManager::begin() {
  drive_.stop();
  driveEnabled_ = false;
  lastDriveCommandMs_ = millis();
}

void SafetyManager::noteDriveCommand() {
  lastDriveCommandMs_ = millis();
  driveEnabled_ = true;
}

bool SafetyManager::update() {
  if (!driveEnabled_) return false;
  if (millis() - lastDriveCommandMs_ <= Config::DRIVE_COMMAND_TIMEOUT_MS) {
    return false;
  }
  stopDrive();
  return true;
}

void SafetyManager::stopDrive() {
  drive_.stop();
  driveEnabled_ = false;
}
