#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

namespace ai
{

class TrainingData
{
private:
    std::ifstream m_training_data_file;

public:
    TrainingData(const std::string filename);
    ~TrainingData();

    bool is_EOF() { return m_training_data_file.eof(); };
    void get_topology(std::vector<unsigned> &topology);

    // Return the number of input values read from the file
    unsigned get_next_inputs(std::vector<double> &input_vals);
    unsigned get_target_outputs(std::vector<double> &target_output_values);
};


}