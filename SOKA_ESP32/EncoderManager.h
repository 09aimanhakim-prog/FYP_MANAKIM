#pragma once

#include <Arduino.h>

class EncoderManager {
 public:
  void begin();
  bool update();
  float rpm(uint8_t index) const;
  int32_t count(uint8_t index) const;
  float samplePeriodSeconds() const { return lastElapsedMs_ * 0.001f; }

 private:
  static void IRAM_ATTR isr0();
  static void IRAM_ATTR isr1();
  static void IRAM_ATTR isr2();
  static void IRAM_ATTR isr3();
  static volatile int32_t counts_[4];
  int32_t previousCounts_[4] = {};
  float rpm_[4] = {};
  uint32_t lastSampleMs_ = 0;
  uint32_t lastElapsedMs_ = 0;
};
