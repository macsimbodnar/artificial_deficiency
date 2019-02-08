#include <vector>
#include <cassert>
#include <string>
#include <iostream>
#include "log.hpp"
#include "net.hpp"
#include "training_data.hpp"

// NOTE(max):
// http://www.millermattson.com/dave/?p=54

/*
void show_vector_vals(const char *label, std::vector<double> &v)
{
    log_inf("%s ", label);
    for (unsigned i = 0; i < v.size(); i++)
    {
        log_inf("%f ", v[i]);
    }
    log_inf("\n");
}
*/

void show_vector_vals(std::string label, std::vector<double> &v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}


int main(int argc, char *argv[])
{
    log_inf("Start\n");

    ai::TrainingData train_data("/tmp/training_data.txt");
    
    // Note(max): creating the net
    std::vector<unsigned> topology;
    train_data.get_topology(topology);
    ai::Net my_net(topology);

    // Training 
    std::vector<double> input_vals, target_vals, result_vals;
    int training_pass = 0;

    while (!train_data.is_EOF())
    {
        training_pass++;
        log_inf("Pass %d\n", training_pass);

        // Get input data and feed it forward
        if (train_data.get_next_inputs(input_vals) != topology[0])
        {
            break;
        }

        show_vector_vals("Inputs:", input_vals);
        my_net.feed_forward(input_vals);

        // Collect the net's actual results
        my_net.get_result(result_vals);
        show_vector_vals("Outputs:", result_vals);

        // Train the net what the outputs should have been
        train_data.get_target_outputs(target_vals);
        show_vector_vals("Targets:", target_vals);

        //log_deb("target vals: %d, topology back: %d\n", 
        //            (unsigned) target_vals.size(), topology.back());
        assert(target_vals.size() == topology.back());
        
        my_net.back_prop(target_vals);

        // Report how well the training is working
        log_inf("Net recent average error: %lf\n\n", 
                        my_net.get_recent_average_error());
    }
    
    log_inf("End\n");
    return 0;
} 