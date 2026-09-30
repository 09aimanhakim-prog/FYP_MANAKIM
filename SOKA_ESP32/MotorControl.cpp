#include "MotorControl.h"

#include "Config.h"

MotorControl::MotorControl(uint8_t rpwmPin, uint8_t lpwmPin, int8_t direction)
    : rpwmPin_(rpwmPin), lpwmPin_(lpwmPin), direction_(direction) {}

void MotorControl::begin() {
  pinMode(rpwmPin_, OUTPUT);
  pinMode(lpwmPin_, OUTPUT);
  // Establish OFF before attaching PWM; this is a safety-critical startup step.
  digitalWrite(rpwmPin_, LOW);
  digitalWrite(lpwmPin_, LOW);
  ledcAttach(rpwmPin_, Config::PWM_FREQUENCY, Config::PWM_RESOLUTION);
  ledcAttach(lpwmPin_, Config::PWM_FREQUENCY, Config::PWM_RESOLUTION);
  stop();
}

void MotorControl::set(int command) {
  command = constrain(command * direction_, -Config::DRIVE_MAX_PWM,
                      Config::DRIVE_MAX_PWM);
  // Always turn the inactive half bridge off first.
  ledcWrite(rpwmPin_, 0);
  ledcWrite(lpwmPin_, 0);
  if (command > 0) {
    ledcWrite(rpwmPin_, command);
  } else if (command < 0) {
    ledcWrite(lpwmPin_, -command);
  }
}

void MotorControl::stop() {
  ledcWrite(rpwmPin_, 0);
  ledcWrite(lpwmPin_, 0);
}

