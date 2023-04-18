use ndarray::prelude::*;
use ndarray::Array;
// use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

struct NeuralNet {
    layers: Vec<Layer>,
}

impl NeuralNet {
    fn new(
        num_inputs: usize,
        num_outputs: usize,
        num_hidden_layers: usize,
        num_nodes_per_layer: usize,
    ) -> Self {
        let mut layers: Vec<Layer> = vec![];
        // Create the first layer with the actual number of features from the dataset as the number of inputs
        layers.push(Layer::new(num_inputs, num_nodes_per_layer));

        for _ in 1..num_hidden_layers {
            // Create the rest of the hidden layers with the input as the number of nodes in the previous layer
            layers.push(Layer::new(num_nodes_per_layer, num_nodes_per_layer));
        }

        // Create the output layer with the number of nodes in the previous layer as the number of inputs
        layers.push(Layer::new(num_nodes_per_layer, num_outputs));
        NeuralNet { layers }
    }

    // fn feed_forward(&mut self, input: &Array<f64, Ix1>) -> &Array<f64, Ix1> {
    //     let mut output = input;
    //     for layer in &mut self.layers {
    //         output = layer.forward(output);
    //     }
    //     output
    // }
}

struct Layer {
    weights: Array<f64, Ix2>,
    biases: Array<f64, Ix1>,
    output: Array<f64, Ix1>,
}

impl Layer {
    fn new(num_inputs: usize, num_outputs: usize) -> Self {
        let mut weights = Array::random((num_inputs, num_outputs), StandardNormal);
        let mut biases = Array::random((num_outputs,), StandardNormal);
        let mut output = Array::zeros((num_outputs,));
        Layer {
            weights,
            biases,
            output,
        }
    }

    /// The layer processes the input data and returns the output. Specifically, it is the sum of the product
    /// of the input and the weights, plus the biases.
    fn forward(&mut self, input: &Array<f64, Ix1>) -> &Array<f64, Ix1> {
        self.output = input.dot(&self.weights) + &self.biases;
        &self.output

        // ReLU activation function
        // output.mapv(|x| x.max(0.0))
    }
}

struct Relu {
    output: Array<f64, Ix1>,
}

impl Relu {
    fn new() -> Self {
        Relu {
            output: Array::zeros((1,)),
        }
    }

    /// Takes weighted and biased neuron outputs and applies the ReLU activation function to them.
    fn forward(&mut self, input: &Array<f64, Ix1>) -> &Array<f64, Ix1> {
        self.output = input.mapv(|x| x.max(0.0));
        &self.output
    }
}

// TODO: Implement softmax activation function

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_feed_forward() {
        let input = array![1.0, 2.0];

        let mut layer1 = Layer::new(2, 3);
        let mut activation1 = Relu::new();

        println!("Layer 1 weights: {}\n", layer1.weights);
        println!("Layer 1 biases: {}\n", layer1.biases);
        println!("Layer 1 output before:\n {}", layer1.output);

        layer1.forward(&input);
        println!("Layer 1 output after:\n {}", layer1.output);

        activation1.forward(&layer1.output);
        println!("Relu output after activation:\n {}", activation1.output);
    }
}
