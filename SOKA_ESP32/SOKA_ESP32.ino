#include "Config.h"
#include "DribblerController.h"
#include "EncoderManager.h"
#include "HeadingController.h"
#include "KickerController.h"
#include "MotorControl.h"
#include "OmniDrive.h"
#include "SafetyManager.h"
#include "SerialProtocol.h"

MotorControl motor1(Config::M1_RPWM, Config::M1_LPWM, Config::M1_DIRECTION);
MotorControl motor2(Config::M2_RPWM, Config::M2_LPWM, Config::M2_DIRECTION);
MotorControl motor3(Config::M3_RPWM, Config::M3_LPWM, Config::M3_DIRECTION);
MotorControl motor4(Config::M4_RPWM, Config::M4_LPWM, Config::M4_DIRECTION);
OmniDrive drive(motor1, motor2, motor3);
DribblerController dribbler(motor4);
KickerController kicker;
EncoderManager encoders;
HeadingController heading;
SerialProtocol protocol;
SafetyManager safety(drive);

float requestedX = 0.0f;
float requestedY = 0.0f;
bool calibrationWasRunning = false;
uint8_t lastCalibrationProgressStep = 255;

void stopDriveIntent() {
  requestedX = 0.0f;
  requestedY = 0.0f;
  safety.stopDrive();
}

void reportStatus() {
  Serial.printf(
      "STATUS,MS=%lu,X=%.1f,Y=%.1f,AX=%.1f,AY=%.1f,H=%.2f,HT=%.2f,HE=%d,IMU=%d,"
      "HERR=%.2f,HRATE=%.2f,RRAW=%.1f,RLIM=%.1f,HIT=%d,"
      "DRIVE=%d,IMUF=%lu,IMUR=%lu,GB=%.4f,BL=%d,CAL=%d,CALP=%u,"
      "RPM1=%.1f,RPM2=%.1f,RPM3=%.1f,RPM4=%.1f,"
      "TR1=%.1f,TR2=%.1f,TR3=%.1f,PW1=%d,PW2=%d,PW3=%d,"
      "DRIB=%d,KICK=%d\n",
      millis(), requestedX, requestedY, drive.appliedX(), drive.appliedY(),
      heading.heading(), heading.target(),
      heading.enabled(), heading.healthy(), heading.headingError(),
      heading.gyroRateDegreesPerSecond(), heading.rawRotationCommand(),
      heading.limitedRotationCommand(), heading.insideTolerance(),
      safety.driveEnabled(),
      heading.failureCount(), heading.recoveryCount(),
      heading.biasDegreesPerSecond(), heading.biasLearning(), heading.calibrating(),
      heading.calibrationProgress(), encoders.rpm(0),
      encoders.rpm(1), encoders.rpm(2), encoders.rpm(3),
      drive.targetRpm(0), drive.targetRpm(1), drive.targetRpm(2),
      drive.outputPwm(0), drive.outputPwm(1), drive.outputPwm(2),
      dribbler.command(), kicker.active());
}

void acknowledge(const char* text) { Serial.printf("OK,%s\n", text); }
void reject(const char* text) { Serial.printf("ERR,%s\n", text); }

void handleCommand(const Command& command) {
  if (heading.calibrating() && command.type != CommandType::STATUS &&
      command.type != CommandType::STOP) {
    reject("CALIBRATING");
    return;
  }

  switch (command.type) {
    case CommandType::MOVE:
      requestedX = constrain(command.x, -255.0f, 255.0f);
      requestedY = constrain(command.y, -255.0f, 255.0f);
      safety.noteDriveCommand();
      acknowledge("MOVE");
      break;

    case CommandType::HEADING:
      heading.setTarget(command.heading);
      if (heading.healthy()) acknowledge("HEADING");
      else reject("HEADING,IMU_UNAVAILABLE");
      break;

    case CommandType::CTRL:
      requestedX = constrain(command.x, -255.0f, 255.0f);
      requestedY = constrain(command.y, -255.0f, 255.0f);
      heading.setTarget(command.heading);
      safety.noteDriveCommand();
      acknowledge("CTRL");
      break;

    case CommandType::DRIBBLER:
      dribbler.set(command.value);
      acknowledge("DRIBBLER");
      break;

    case CommandType::KICK:
      if (kicker.trigger()) acknowledge("KICK");
      else reject("KICK,COOLDOWN");
      break;

    case CommandType::STOP:
      stopDriveIntent();
      dribbler.stop();
      kicker.stop();
      acknowledge("STOP");
      break;

    case CommandType::STATUS:
      reportStatus();
      break;

    case CommandType::ZERO_HEADING:
      heading.zero();
      heading.enable(true);
      acknowledge("ZERO");
      break;

    case CommandType::DISABLE_HEADING:
      heading.enable(false);
      stopDriveIntent();
      acknowledge("HEADING_OFF");
      break;

    case CommandType::CALIBRATE:
      stopDriveIntent();
      dribbler.stop();
      kicker.stop();
      if (heading.startCalibration()) {
        calibrationWasRunning = true;
        lastCalibrationProgressStep = 255;
        Serial.println("EVENT,CALIBRATION_STARTED,DURATION_S=60");
      } else {
        reject("CALIBRATE,IMU_UNAVAILABLE");
      }
      break;

    default:
      reject("BAD_COMMAND");
      break;
  }
}

void setup() {
  // Each BTS7960 PWM input is explicitly driven LOW before PWM is attached.
  motor1.begin();
  motor2.begin();
  motor3.begin();
  motor4.begin();
  kicker.begin();
  safety.begin();

  Serial.begin(Config::SERIAL_BAUD);
  encoders.begin();
  const bool imuReady = heading.begin();
  Serial.printf("READY,SOKA_ESP32,IMU=%s,WATCHDOG_MS=%lu\n",
                imuReady ? "OK" : "ERROR", Config::DRIVE_COMMAND_TIMEOUT_MS);
}

void loop() {
  heading.setBiasLearningAllowed(!safety.driveEnabled() &&
                                 dribbler.command() == 0 && !kicker.active());
  heading.update();
  const bool rpmUpdated = encoders.update();
  if (rpmUpdated) {
    drive.update(encoders.rpm(0), encoders.rpm(1), encoders.rpm(2),
                 encoders.samplePeriodSeconds());
  }
  kicker.update();

  Command command;
  while (protocol.poll(command)) handleCommand(command);

  if (heading.calibrating()) {
    const uint8_t progress = heading.calibrationProgress();
    const uint8_t progressStep = progress / 10;
    if (progressStep != lastCalibrationProgressStep) {
      lastCalibrationProgressStep = progressStep;
      Serial.printf("EVENT,CALIBRATING,%u\n", progress);
    }
  } else if (calibrationWasRunning) {
    calibrationWasRunning = false;
    if (heading.calibrationSucceeded()) {
      Serial.printf("EVENT,CALIBRATION_COMPLETE,BIAS=%.4f\n",
                    heading.biasDegreesPerSecond());
    } else {
      Serial.println("EVENT,CALIBRATION_FAILED,KEEP_ROBOT_STILL");
    }
  }

  if (safety.update()) {
    requestedX = 0.0f;
    requestedY = 0.0f;
    Serial.println("EVENT,WATCHDOG_STOP");
  }

  if (safety.driveEnabled()) {
    drive.command(requestedX, requestedY, heading.correction());
  }
}
