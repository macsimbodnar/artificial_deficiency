#include "training_data.hpp"
#include "log.hpp"

ai::TrainingData::TrainingData(const std::string filename)
{
    m_training_data_file.open(filename.c_str());
}


ai::TrainingData::~TrainingData() 
{
    m_training_data_file.close();
}


void ai::TrainingData::get_topology(std::vector<unsigned> &topology)
{
    std::string line, label;
    log_inf("Passo da qui\n");
    getline(m_training_data_file, line);
    std::stringstream ss(line);

    ss >> label;
    if (this->is_EOF() || label.compare("topology:") != 0)
    {
        abort();
    }

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    unsigned size = topology.size();
    log_inf("Topology size: %d [", size);
    for (int i = 0; i < size; i++)
    {
        unsigned val = topology[i];
        log_inf(" %d", val);
    }
    log_inf(" ]\n");
}


unsigned ai::TrainingData::get_next_inputs(std::vector<double> &input_vals)
{
    input_vals.clear();

    std::string line;
    getline(m_training_data_file, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double one_value;
        while (ss >> one_value)
        {
            input_vals.push_back(one_value);
        }
    }


    return input_vals.size();
}


unsigned ai::TrainingData::get_target_outputs(
                                std::vector<double> &target_output_values)
{
    target_output_values.clear();

    std::string line;
    getline(m_training_data_file, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double one_value;
        while (ss >> one_value)
        {
            //log_inf("target: %f", one_value);
            target_output_values.push_back(one_value);
        }
    }

    return target_output_values.size();
}
