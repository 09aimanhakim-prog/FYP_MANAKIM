#include "DribblerController.h"

#include <Arduino.h>

void DribblerController::set(int pwm) {
  command_ = constrain(pwm, 0, 255);
  motor_.set(command_);
}

void DribblerController::stop() {
  command_ = 0;
  motor_.stop();
}
