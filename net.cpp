#include <cassert>
#include <cmath>
#include "net.hpp"
#include "log.hpp"

// Number of training samples to average over
double ai::Net::m_recent_average_smooth_factor = 100.0; 

ai::Net::Net(const std::vector<unsigned> &topology)
{
    log_inf("Constructing the Net\n");

    unsigned num_layers = topology.size();
    for (unsigned layer_num = 0; layer_num < num_layers; layer_num++)
    {
        // Note(max): add new layer
        log_inf("\nAdded Layer to the Net\n");
        m_layers.push_back(layer_t());

        unsigned num_outputs = (layer_num == (topology.size() - 1) ? 0 
                                        : topology[layer_num + 1]);

        // Note(max): fill the layer with neurons
        for (unsigned j = 0; j <= topology[layer_num]; j++)
        {
            m_layers.back().push_back(Neuron(num_outputs, j));
        }

        // Force the bias neuron's output value to 1.0. It's the last neuron created
        m_layers.back().back().set_out_value(1.0);
    }

    for (int i = 0; i < m_layers.size(); i++)
    {
        layer_t &layer = m_layers[i];
        log_deb("Layer n: %d, size: %d\n", i, (unsigned)layer.size());
        for(int j = 0; j < layer.size(); j++)
        {
            log_deb("    Neuron n: %d, val: %f\n", j, layer[j].get_output_val());
        }
    }
}


ai::Net::~Net()
{}


void ai::Net::feed_forward(const std::vector<double> &input_vals)
{
    assert(input_vals.size() == m_layers[0].size() - 1);

    // NOTE(max): Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < input_vals.size(); i++)
    {
        m_layers[0][i].set_out_value(input_vals[i]);
    }

    // NOTE(max): Forward propagate
    for (unsigned layer_num = 1; layer_num < m_layers.size(); layer_num++)
    {
        layer_t &prev_layer = m_layers[layer_num - 1];
        for (unsigned n = 0; n < m_layers[layer_num].size() - 1; n++) 
        {
            m_layers[layer_num][n].feed_forward(prev_layer);
        }
    }
}


void ai::Net::back_prop(const std::vector<double> &target_vals)
{
    /******************************************************************************
     *  NOTE(max):
     *  There is the place where the net learn!
     *****************************************************************************/

    // Calculate overall net error (RMS - Root Mean Square Error of output)
    layer_t &output_layer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < output_layer.size() - 1; n++)
    {
        double delta = target_vals[n] - output_layer[n].get_output_val();
        m_error += delta * delta;
    }
    m_error /= output_layer.size() - 1;     // Get average error squared
    m_error = sqrt(m_error);                // RMS

    // Implement a recent average measurement for debug
    m_recent_average_error = 
        (m_recent_average_error * m_recent_average_smooth_factor + m_error) / 
            (m_recent_average_smooth_factor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < output_layer.size() - 1; n++)
    {
        output_layer[n].calc_output_gradients(target_vals[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layer_num = m_layers.size() - 2; layer_num > 0; layer_num--)
    {
        layer_t &hidden_layer = m_layers[layer_num];
        layer_t &next_layer = m_layers[layer_num + 1];

        for (unsigned n = 0; n < hidden_layer.size(); n++)
        {
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }

    // For all layers from output to first hidden layer, update connection weights
    for (unsigned layer_num = m_layers.size() - 1; layer_num > 0; layer_num--)
    {
        layer_t &layer = m_layers[layer_num];
        layer_t &prev_layer = m_layers[layer_num - 1];

        for (unsigned n = 0; n < layer.size() - 1; n++)
        {
            layer[n].update_input_weights(prev_layer);
        }
    }
}


void ai::Net::get_result(std::vector<double> &result_vals) const
{
    result_vals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; n++)
    {
        result_vals.push_back(m_layers.back()[n].get_output_val());
    }
}