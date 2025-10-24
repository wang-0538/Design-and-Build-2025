#ifndef MPU6500_H
#define MPU6500_H

#include "stm32f4xx_hal.h"

typedef struct {
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
} MPU6500_Data;

uint8_t MPU6500_Init(I2C_HandleTypeDef* hi2c);
void MPU6500_ReadData(I2C_HandleTypeDef* hi2c, MPU6500_Data* data);

#endif

