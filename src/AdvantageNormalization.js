const tf = require('@tensorflow/tfjs-node');

class AdvantageNormalization extends tf.layers.Layer {
    static className = 'AdvantageNormalization';

    constructor() {
        super({});
    }

    computeOutputShape(inputShape) {
        // AdvantageNormalization layer gets two input tensors:
        // one of shape [null, 1, y] and one of shape [null, x, y]
        // and reduces this to one output tensor of shape [null, x, y].
        return inputShape[1];
    }

    call(inputs) {
        return tf.tidy(() => {
            const val = inputs[0];
            const adv = inputs[1];
            const mean = tf.mean(adv, 1, true);
            const centered = adv.sub(mean);
            return centered.add(val);
        });
    }

    getClassName() {
        return 'AdvantageNormalization';
    }
}

tf.serialization.registerClass(AdvantageNormalization);

module.exports = AdvantageNormalization;