const tf = require('@tensorflow/tfjs-node');

class AdvantageNormalization extends tf.layers.Layer {
    // Static property to get the class name
    static className = 'AdvantageNormalization';

    constructor() {
        super({});
    }

    // Method to compute the output shape of the layer based on the input shape
    computeOutputShape(inputShape) {
        // AdvantageNormalization layer receives two input tensors:
        // one of shape [null, 1, y] and one of shape [null, x, y]
        // and reduces this to one output tensor of shape [null, x, y].
        return inputShape[1]; // Return the shape of the second input tensor
    }

    // Method to perform the forward pass of the layer
    call(inputs) {
        // tf.tidy() is used to free any intermediate tensors created during this method
        return tf.tidy(() => {
            const val = inputs[0]; // Extract the first input tensor (value tensor)
            const adv = inputs[1]; // Extract the second input tensor (advantage tensor)
            const mean = tf.mean(adv, 1, true); // Compute the mean of the advantage tensor along axis 1, keeping the dimensions
            const centered = adv.sub(mean); // Subtract the mean from the advantage tensor to center it
            return centered.add(val); // Add the value tensor to the centered advantage tensor and return the result
        });
    }

    // Method to return the class name of this layer
    getClassName() {
        return 'AdvantageNormalization';
    }
}

// Register the custom layer with TensorFlow.js for serialization
tf.serialization.registerClass(AdvantageNormalization);

// Export the AdvantageNormalization class for use in other modules
module.exports = AdvantageNormalization;