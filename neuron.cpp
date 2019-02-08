#include <cmath>
#include "neuron.hpp"
#include "log.hpp"


double ai::Neuron::eta = 0.15;//0.33;      // overall net learning rate
double ai::Neuron::alpha = 0.5;//0.805;     // momentum, multiplier of last delta weight


ai::Neuron::Neuron(unsigned num_outputs, unsigned my_index)
{
    log_inf("Constructing Neuron with %u outputs\n", num_outputs);

    for (unsigned c = 0; c < num_outputs; c++)
    {
        m_output_weights.push_back(connection_t());
        m_output_weights.back().weight = Neuron::random_weight();
    }

    m_my_index = my_index;
}


void ai::Neuron::feed_forward(const layer_t &prev_layer)
{
    double sum = 0.0;
    /******************************************************************************
     *  NOTE(max): 
     *  Sum the previous layer's outputs (which are our inputs)
     *  Include the bias node from the previous layer.
     *****************************************************************************/

    for (unsigned n = 0; n < prev_layer.size(); n++)
    {
        sum += prev_layer[n].get_output_val() * 
                    prev_layer[n].m_output_weights[m_my_index].weight;
    }

    // TODO(max):
    m_input_val = sum;
    m_output_val = Neuron::transfer_function(sum);
}


double ai::Neuron::transfer_function(double x)
{
    /******************************************************************************
     *  NOTE(max): 
     *  We can use any functions.
     *  There we will use a tanh function with output range [-1.0..+1.0]
     *****************************************************************************/

    return tanh(x);
}


double ai::Neuron::transfer_function_derivative(double x)
{
    /******************************************************************************
     *  NOTE(max): 
     *  We can use one trick there.
     *  the derivate of hyperbolic tangent can be approximated with (1 - x^2)
     *****************************************************************************/

    // TODO(maz):
    return 1.0 - tanh(x) * tanh(x);
    //return 1.0 - x * x;
}


void ai::Neuron::calc_output_gradients(double target_val)
{
    double delta = target_val - m_output_val;
    // TODO(max):
    m_gradient = delta * Neuron::transfer_function_derivative(m_input_val);
    //m_gradient = delta * Neuron::transfer_function_derivative(m_output_val);
}


void ai::Neuron::calc_hidden_gradients(const layer_t &next_layer)
{
    double dow = sum_DOW(next_layer);
    // TODO(max):
    m_gradient = dow * Neuron::transfer_function_derivative(m_input_val);
    //m_gradient = dow * Neuron::transfer_function_derivative(m_output_val);
}


double ai::Neuron::sum_DOW(const layer_t &next_layer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < next_layer.size() - 1; n++)
    {
        sum += m_output_weights[n].weight * next_layer[n].m_gradient;
    }

    return sum;
}


void ai::Neuron::update_input_weights(layer_t &prev_layer)
{
    // The weights to be updated are in Connection container in the neurons in the 
    // previous layer

    for (unsigned n = 0; n < prev_layer.size(); n++)
    {
        Neuron &neuron = prev_layer[n];
        double old_delta_weight = neuron.m_output_weights[m_my_index].delta_weight;

        double new_delta_weight = 
            // Individual input, magnified by the gradient and train rate:
            eta
            * neuron.get_output_val()
            * m_gradient
            // Also add momentum = a fraction of the previous delta weight
            + alpha
            * old_delta_weight;
        
        neuron.m_output_weights[m_my_index].delta_weight = new_delta_weight;
        neuron.m_output_weights[m_my_index].weight += new_delta_weight;
    }
}