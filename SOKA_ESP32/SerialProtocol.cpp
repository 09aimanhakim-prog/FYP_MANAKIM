#include "SerialProtocol.h"
#include "Config.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

namespace {
bool parseFloatStrict(const char* text, float& value) {
  if (!text) return false;
  char* end;
  value = strtof(text, &end);
  return end != text && *end == '\0';
}
}

bool SerialProtocol::poll(Command& command) {
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;
    if (c == '\n') {
      if (length_ == 0) continue;
      buffer_[length_] = '\0';
      const bool result = parse(command);
      length_ = 0;
      if (!result) command.type = CommandType::INVALID;
      return true;
    }
    if (length_ < BUFFER_SIZE - 1) {
      buffer_[length_++] = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    } else {
      length_ = 0;
      command.type = CommandType::INVALID;
      return true;
    }
  }
  return false;
}

bool SerialProtocol::parse(Command& command) {
  command = Command{};
  char* save = nullptr;
  char* name = strtok_r(buffer_, ",", &save);
  if (!name) return false;

  if (!strcmp(name, "MOVE")) {
    command.type = CommandType::MOVE;
    return parseFloatStrict(strtok_r(nullptr, ",", &save), command.x) &&
           parseFloatStrict(strtok_r(nullptr, ",", &save), command.y) &&
           strtok_r(nullptr, ",", &save) == nullptr;
  }
  if (!strcmp(name, "HEADING")) {
    command.type = CommandType::HEADING;
    return parseFloatStrict(strtok_r(nullptr, ",", &save), command.heading) &&
           strtok_r(nullptr, ",", &save) == nullptr;
  }
  if (!strcmp(name, "CTRL")) {
    command.type = CommandType::CTRL;
    return parseFloatStrict(strtok_r(nullptr, ",", &save), command.x) &&
           parseFloatStrict(strtok_r(nullptr, ",", &save), command.y) &&
           parseFloatStrict(strtok_r(nullptr, ",", &save), command.heading) &&
           strtok_r(nullptr, ",", &save) == nullptr;
  }
  if (!strcmp(name, "DRIBBLER")) {
    command.type = CommandType::DRIBBLER;
    char* arg = strtok_r(nullptr, ",", &save);
    if (!arg || strtok_r(nullptr, ",", &save)) return false;
    if (!strcmp(arg, "ON")) command.value = Config::DRIBBLER_DEFAULT_PWM;
    else if (!strcmp(arg, "OFF")) command.value = 0;
    else {
      char* end;
      const long value = strtol(arg, &end, 10);
      if (end == arg || *end || value < 0 || value > 255) return false;
      command.value = value;
    }
    return true;
  }
  if (!strcmp(name, "KICK")) command.type = CommandType::KICK;
  else if (!strcmp(name, "STOP")) command.type = CommandType::STOP;
  else if (!strcmp(name, "STATUS")) command.type = CommandType::STATUS;
  else if (!strcmp(name, "ZERO")) command.type = CommandType::ZERO_HEADING;
  else if (!strcmp(name, "CALIBRATE")) command.type = CommandType::CALIBRATE;
  else if (!strcmp(name, "HEADING_OFF")) command.type = CommandType::DISABLE_HEADING;
  else command.type = CommandType::INVALID;
  return strtok_r(nullptr, ",", &save) == nullptr;
}
