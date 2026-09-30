#pragma once

#include <Arduino.h>

enum class CommandType {
  NONE, MOVE, HEADING, CTRL, DRIBBLER, KICK, STOP, STATUS, ZERO_HEADING,
  DISABLE_HEADING, CALIBRATE, INVALID
};

struct Command {
  CommandType type = CommandType::NONE;
  float x = 0;
  float y = 0;
  float heading = 0;
  int value = 0;
};

class SerialProtocol {
 public:
  bool poll(Command& command);
 private:
  static constexpr size_t BUFFER_SIZE = 96;
  char buffer_[BUFFER_SIZE] = {};
  size_t length_ = 0;
  bool parse(Command& command);
};
