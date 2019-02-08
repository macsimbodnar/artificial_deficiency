echo "Start build ..."

g++ training_data.cpp neuron.cpp net.cpp main.cpp -o build/ai

g++ make_training_data.cpp -o build/generate_samples

echo "                  ... end build"