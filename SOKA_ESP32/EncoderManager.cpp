#include "EncoderManager.h"

#include "Config.h"

volatile int32_t EncoderManager::counts_[4] = {};

static inline void IRAM_ATTR updateEncoder(volatile int32_t& count,
                                           uint8_t pinB) {
  count += digitalRead(pinB) ? 1 : -1;
}

void IRAM_ATTR EncoderManager::isr0() { updateEncoder(counts_[0], Config::M1_ENC_B); }
void IRAM_ATTR EncoderManager::isr1() { updateEncoder(counts_[1], Config::M2_ENC_B); }
void IRAM_ATTR EncoderManager::isr2() { updateEncoder(counts_[2], Config::M3_ENC_B); }
void IRAM_ATTR EncoderManager::isr3() { updateEncoder(counts_[3], Config::M4_ENC_B); }

void EncoderManager::begin() {
  const uint8_t pinsA[4] = {Config::M1_ENC_A, Config::M2_ENC_A,
                            Config::M3_ENC_A, Config::M4_ENC_A};
  const uint8_t pinsB[4] = {Config::M1_ENC_B, Config::M2_ENC_B,
                            Config::M3_ENC_B, Config::M4_ENC_B};
  for (uint8_t i = 0; i < 4; ++i) {
    // GPIO34-39 have no internal pull resistors. Encoder outputs must be driven
    // or externally pulled to 3.3 V if required by the particular motor board.
    pinMode(pinsA[i], INPUT);
    pinMode(pinsB[i], INPUT);
  }
  attachInterrupt(digitalPinToInterrupt(pinsA[0]), isr0, RISING);
  attachInterrupt(digitalPinToInterrupt(pinsA[1]), isr1, RISING);
  attachInterrupt(digitalPinToInterrupt(pinsA[2]), isr2, RISING);
  attachInterrupt(digitalPinToInterrupt(pinsA[3]), isr3, RISING);
  lastSampleMs_ = millis();
}

bool EncoderManager::update() {
  const uint32_t now = millis();
  const uint32_t elapsed = now - lastSampleMs_;
  if (elapsed < Config::RPM_SAMPLE_MS) return false;

  int32_t snapshot[4];
  noInterrupts();
  for (uint8_t i = 0; i < 4; ++i) snapshot[i] = counts_[i];
  interrupts();

  for (uint8_t i = 0; i < 4; ++i) {
    const int32_t delta = snapshot[i] - previousCounts_[i];
    const float cpr = i == 3 ? Config::DRIBBLER_COUNTS_PER_REV
                             : Config::DRIVE_COUNTS_PER_REV;
    rpm_[i] = delta * 60000.0f / (cpr * elapsed);
    previousCounts_[i] = snapshot[i];
  }
  lastSampleMs_ = now;
  lastElapsedMs_ = elapsed;
  return true;
}

float EncoderManager::rpm(uint8_t index) const {
  return index < 4 ? rpm_[index] : 0.0f;
}

int32_t EncoderManager::count(uint8_t index) const {
  if (index >= 4) return 0;
  noInterrupts();
  const int32_t value = counts_[index];
  interrupts();
  return value;
}
