#include "KickerController.h"

#include "Config.h"

void KickerController::writeRelay(bool on) {
  digitalWrite(Config::KICKER_PIN,
               (on == Config::KICKER_ACTIVE_HIGH) ? HIGH : LOW);
}

void KickerController::begin() {
  pinMode(Config::KICKER_PIN, OUTPUT);
  writeRelay(false);
}

bool KickerController::trigger() {
  const uint32_t now = millis();
  if (active_ || (hasKicked_ && now - lastKickMs_ < Config::KICK_COOLDOWN_MS)) {
    return false;
  }
  active_ = true;
  hasKicked_ = true;
  pulseStartedMs_ = now;
  lastKickMs_ = now;
  writeRelay(true);
  return true;
}

void KickerController::update() {
  if (active_ && millis() - pulseStartedMs_ >= Config::KICK_PULSE_MS) stop();
}

void KickerController::stop() {
  writeRelay(false);
  active_ = false;
}
