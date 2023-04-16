extern crate ndarray;

use ndarray::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
}

struct Node {
    weights: Vec<(f32, Option<Rc<RefCell<Node>>>)>,
    bias: f32,
    activation_function: ActivationFunction,
    output: f32,
    error: f32,
}

impl Node {
    fn new(
        weights: Vec<(f32, Option<Rc<RefCell<Node>>>)>,
        bias: f32,
        activation_function: ActivationFunction,
        output: f32,
        error: f32,
    ) -> Node {
        Node {
            weights,
            bias,
            activation_function,
            output,
            error,
        }
    }
}

pub struct NeuralNet {
    input_nodes: Vec<Rc<RefCell<Node>>>,
    num_hidden_layers: u32,
    nodes_per_layer: u8,
}

impl NeuralNet {
    pub fn new(num_inputs: u32, num_hidden_layers: u32, nodes_per_layer: u8) -> Self {
        let mut input_nodes: Vec<Rc<RefCell<Node>>> = vec![];

        // Create input nodes
        for _ in 0..num_inputs {
            let node = Node::new(
                vec![(1.0, None); nodes_per_layer as usize],
                0.0,
                ActivationFunction::ReLU,
                1.0,
                0.0,
            );

            input_nodes.push(Rc::new(RefCell::new(node)));
        }

        // Create hidden layers

        Self {
            input_nodes,
            num_hidden_layers,
            nodes_per_layer,
        }
    }
}

// Calculate the sum of squared residuals. This is how far off the network is from the target, the actual value in reality. The lower the better.
fn calculate_ssr(observed: f32, predicted: f32) -> f32 {
    (observed - predicted).powi(2)
}

// Get a random number from a standard normal distribution, with a mean of 0 and a standard deviation of 1.
fn rand_normal_dist() -> f32 {
    let mut rng = thread_rng();
    let x = StandardNormal.sample(&mut rng);
    println!("Random value from standard normal distribution: {}", x);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup() {
        let net = NeuralNet::new(1, 1, 2);

        let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];
        assert_eq!(a.ndim(), 2); // get the number of dimensions of array a
        assert_eq!(a.len(), 8); // get the number of elements in array a
        assert_eq!(a.shape(), [2, 4]); // get the shape of array a
        assert_eq!(a.is_empty(), false); // check if the array has zero elements

        let b = array![1, 2, 3, 4,];
        let c = array![1, 2, 3, 4,];
        let d = b.dot(&c);
        assert_eq!(d, 30);
    }
}
