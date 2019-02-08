#pragma once
#include <stdio.h>

#define log_err(f_, ...) printf((f_), ##__VA_ARGS__)
#define log_deb(f_, ...) printf((f_), ##__VA_ARGS__)
#define log_inf(f_, ...) printf((f_), ##__VA_ARGS__)