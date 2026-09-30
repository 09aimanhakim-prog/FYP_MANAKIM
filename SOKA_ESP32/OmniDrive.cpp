#include "OmniDrive.h"

#include <Arduino.h>
#include <math.h>

#include "Config.h"

OmniDrive::OmniDrive(MotorControl& m1, MotorControl& m2, MotorControl& m3)
    : wheel1_(m1, Config::M1_FEEDFORWARD_GAIN, Config::M1_MIN_EFFECTIVE_PWM),
      wheel2_(m2, Config::M2_FEEDFORWARD_GAIN, Config::M2_MIN_EFFECTIVE_PWM),
      wheel3_(m3, Config::M3_FEEDFORWARD_GAIN, Config::M3_MIN_EFFECTIVE_PWM) {}

void OmniDrive::command(float x, float y, float rotation) {
  x = constrain(x, -255.0f, 255.0f);
  y = constrain(y, -255.0f, 255.0f);
  updateTranslationSlew(x, y);

  // Corrected, tested robot-centric translation mapping.
  float w1 = 0.8660254f * appliedY_ + 0.5f * appliedX_ + rotation;
  float w2 = -0.8660254f * appliedY_ + 0.5f * appliedX_ + rotation;
  float w3 = -appliedX_ + rotation;

  const float largest = max(fabsf(w1), max(fabsf(w2), fabsf(w3)));
  if (largest > Config::DRIVE_MAX_PWM) {
    const float scale = Config::DRIVE_MAX_PWM / largest;
    w1 *= scale;
    w2 *= scale;
    w3 *= scale;
  }

  wheel1_.setCommand(w1);
  wheel2_.setCommand(w2);
  wheel3_.setCommand(w3);
}

void OmniDrive::updateTranslationSlew(float targetX, float targetY) {
  if (!Config::DRIVE_TRANSLATION_SLEW_ENABLED) {
    appliedX_ = targetX;
    appliedY_ = targetY;
    lastTranslationUpdateUs_ = micros();
    return;
  }

  const uint32_t nowUs = micros();
  if (lastTranslationUpdateUs_ == 0) {
    lastTranslationUpdateUs_ = nowUs;
    return;
  }

  float dtSeconds = static_cast<float>(nowUs - lastTranslationUpdateUs_) * 1.0e-6f;
  lastTranslationUpdateUs_ = nowUs;
  dtSeconds = constrain(dtSeconds, 0.0f,
                        Config::DRIVE_TRANSLATION_SLEW_MAX_DT_S);

  const float deltaX = targetX - appliedX_;
  const float deltaY = targetY - appliedY_;
  const float deltaMagnitude = sqrtf(deltaX * deltaX + deltaY * deltaY);
  if (deltaMagnitude < 0.001f) {
    appliedX_ = targetX;
    appliedY_ = targetY;
    return;
  }

  const float currentMagnitude = sqrtf(appliedX_ * appliedX_ +
                                       appliedY_ * appliedY_);
  const float targetMagnitude = sqrtf(targetX * targetX + targetY * targetY);
  const float directionDot = appliedX_ * targetX + appliedY_ * targetY;
  const bool slowingOrReversing =
      targetMagnitude < currentMagnitude || directionDot < 0.0f;
  const float rate = slowingOrReversing
                         ? Config::DRIVE_TRANSLATION_DECEL_COMMAND_PER_S
                         : Config::DRIVE_TRANSLATION_ACCEL_COMMAND_PER_S;
  const float maximumStep = rate * dtSeconds;

  if (deltaMagnitude <= maximumStep) {
    appliedX_ = targetX;
    appliedY_ = targetY;
  } else if (maximumStep > 0.0f) {
    const float scale = maximumStep / deltaMagnitude;
    appliedX_ += deltaX * scale;
    appliedY_ += deltaY * scale;
  }
}

void OmniDrive::update(float rpm1, float rpm2, float rpm3, float dtSeconds) {
  wheel1_.update(rpm1, dtSeconds);
  wheel2_.update(rpm2, dtSeconds);
  wheel3_.update(rpm3, dtSeconds);
}

void OmniDrive::stop() {
  wheel1_.stop();
  wheel2_.stop();
  wheel3_.stop();
  appliedX_ = 0.0f;
  appliedY_ = 0.0f;
  lastTranslationUpdateUs_ = 0;
}

float OmniDrive::targetRpm(uint8_t index) const {
  if (index == 0) return wheel1_.targetRpm();
  if (index == 1) return wheel2_.targetRpm();
  if (index == 2) return wheel3_.targetRpm();
  return 0.0f;
}

int OmniDrive::outputPwm(uint8_t index) const {
  if (index == 0) return wheel1_.outputPwm();
  if (index == 1) return wheel2_.outputPwm();
  if (index == 2) return wheel3_.outputPwm();
  return 0;
}
