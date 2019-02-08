#pragma once
#include <vector>
#include "neuron.hpp"

namespace ai
{

class Net
{
private:

    // Note(max): m_layers[layer_num][neuron_num]
    std::vector<layer_t> m_layers;
    double m_error;
    double m_recent_average_error;
    static double m_recent_average_smooth_factor;
    
public:
    Net(const std::vector<unsigned> &topology);
    ~Net();

    void feed_forward(const std::vector<double> &input_vals);
    void back_prop(const std::vector<double> &target_vals);
    void get_result(std::vector<double> &result_vals) const;
    double get_recent_average_error() { return m_recent_average_error; }
};

}