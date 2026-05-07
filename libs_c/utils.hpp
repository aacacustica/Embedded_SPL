#pragma once
#include <tuple>
#include <NumCpp.hpp>

extern float PREF;
extern float EPS;

float get_level_db(const nc::NdArray<float>& x, float C);

std::tuple<nc::NdArray<float>, nc::NdArray<float>>
lfilter_np(const nc::NdArray<float>& b,
           const nc::NdArray<float>& a,
           const nc::NdArray<float>& x,
           const nc::NdArray<float>& zi);

std::tuple<nc::NdArray<float>, nc::NdArray<float>>
sosfilt_np(const nc::NdArray<float>& sos,
           const nc::NdArray<float>& x,
           const nc::NdArray<float>& zi);