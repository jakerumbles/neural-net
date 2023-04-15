use std::{cell::RefCell, rc::Rc};

struct Node {
    weights: Vec<f32>,
    bias: f32,
    activation_function: fn(f32) -> f32,
    output: f32,
    error: f32,
}

impl Node {
    fn new(
        weights: Vec<f32>,
        bias: f32,
        activation_function: fn(f32) -> f32,
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
    hidden_layers: u128,
    input_nodes: Vec<Rc<Node>>,
}

impl NeuralNet {
    pub fn new(num_inputs: u32) -> Self {
        let mut input_nodes: Vec<Rc<RefCell<Node>>> = vec![];
        // for i in (0..num_inputs) {

        // }

        Self { hidden_layers: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
