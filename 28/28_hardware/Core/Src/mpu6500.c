#include "mpu6500.h"
#include "math.h"

#define MPU6500_ADDR 0xD0

uint8_t MPU6500_Init(I2C_HandleTypeDef* hi2c) {
    uint8_t check;
    uint8_t data;

    HAL_I2C_Mem_Read(hi2c, MPU6500_ADDR, 0x75, 1, &check, 1, 100);
    if (check != 0x70) return 1;

    data = 0x00;
    HAL_I2C_Mem_Write(hi2c, 0xD0, 0x6B, 1, &data, 1, 100); // Wake up

    data = 0x07;
    HAL_I2C_Mem_Write(hi2c, 0xD0, 0x19, 1, &data, 1, 100); // Sample rate divider

    data = 0x00;
    HAL_I2C_Mem_Write(hi2c, 0xD0, 0x1A, 1, &data, 1, 100); // 260Hz bandwidth

    data = 0x00;
    HAL_I2C_Mem_Write(hi2c, 0xD0, 0x1B, 1, &data, 1, 100); // ¡À250 dps

    data = 0x00;
    HAL_I2C_Mem_Write(hi2c, 0xD0, 0x1C, 1, &data, 1, 100); // ¡À2g

    return 0;
}

void MPU6500_ReadData(I2C_HandleTypeDef* hi2c, MPU6500_Data* data) {
    uint8_t buf[14];
    HAL_I2C_Mem_Read(hi2c, MPU6500_ADDR, 0x3B, 1, buf, 14, 100);

    int16_t ax = (int16_t)(buf[0] << 8 | buf[1]);
    int16_t ay = (int16_t)(buf[2] << 8 | buf[3]);
    int16_t az = (int16_t)(buf[4] << 8 | buf[5]);
    int16_t gx = (int16_t)(buf[8] << 8 | buf[9]);
    int16_t gy = (int16_t)(buf[10] << 8 | buf[11]);
    int16_t gz = (int16_t)(buf[12] << 8 | buf[13]);

    data->accel_x = ax / 16384.0f;
    data->accel_y = ay / 16384.0f;
    data->accel_z = az / 16384.0f;
    data->gyro_x = gx / 131.0f;
    data->gyro_y = gy / 131.0f;
    data->gyro_z = gz / 131.0f;
}
