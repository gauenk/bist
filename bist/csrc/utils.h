
#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"

// -- basic --
template <typename T>
bool parse_argument(int &i, int argc, char **argv, const std::string &arg,
                    const std::string &option, T &value) {
    if (arg == option) {
        if (i + 1 < argc) {
            ++i;
            if constexpr (std::is_same<T, int>::value) {
                value = std::stoi(argv[i]);
            } else if constexpr (std::is_same<T, float>::value) {
                value = std::stof(argv[i]);
            } else if constexpr (std::is_same<T, const char *>::value || std::is_same<T, std::string>::value) {
                value = argv[i];
            }
        } else {
            std::cerr << option << " option requires an argument." << std::endl;
            return false;
        }
    }
    return true;
}


// -- standard functions --
// void show_usage(const std::string &program_name);
// superpixel_options get_sp_options(int sp_size,float i_std,
//                                          float beta, float alpha_hasting);


