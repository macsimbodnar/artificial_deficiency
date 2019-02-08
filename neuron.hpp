#pragma once
#include <vector>
#include <cstdlib>
#include "log.hpp"


namespace ai
{

class Neuron;
typedef std::vector<Neuron> layer_t;

struct connection_t
{
    double weight;
    double delta_weight;
};


class Neuron
{
private:
    static double eta;      // [0.0..1.0] overall net training rate 
    static double alpha;    // [0.0..n] multiplier of last weight change (momentum)
    unsigned m_my_index;
    double m_output_val; 
    // TODO(max):
    double m_input_val;
    double m_gradient;
    std::vector<connection_t> m_output_weights;

    static double random_weight() { return std::rand() / double(RAND_MAX); }
    static double transfer_function(double x);
    static double transfer_function_derivative(double x);
    double sum_DOW(const layer_t &next_layer) const;


public:
    Neuron(unsigned num_outputs, unsigned my_index);
    ~Neuron() {};

    void set_out_value(double val) { m_output_val = val; }
    double get_output_val() const 
    { 
        //log_deb("\nout: %f\n", m_output_val);
        return m_output_val; 
    }
    void feed_forward(const layer_t &prev_layer);

    void calc_output_gradients(double target_val);
    void calc_hidden_gradients(const layer_t &next_layer);

    void update_input_weights(layer_t &prev_layer);
};

}