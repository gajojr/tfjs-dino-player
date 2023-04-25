class Memory {
    constructor(bufferSize, epsilon = 0.01, alpha = 0.5) {
        this.bufferSize = bufferSize;
        this.buffer = new Array(bufferSize).fill(null);
        this.pointer = 0;
        this.count = 0;
        this.sumPriorities = 0;
        this.epsilon = epsilon;
        this.alpha = alpha;
        this.priorities = new Array(bufferSize).fill(0);
    }

    add(experience, priority) {
        if (this.buffer[this.pointer] === null) {
            this.count++;
        }
        const oldPriority = this.priorities[this.pointer];
        const newPriority = Math.pow(priority + this.epsilon, this.alpha);
        this.priorities[this.pointer] = newPriority;
        this.sumPriorities = this.sumPriorities + newPriority - oldPriority;
        this.buffer[this.pointer] = { experience, priority };
        this.pointer = (this.pointer + 1) % this.bufferSize;
    }

    sample(batchSize, priorityScale = 1) {
        const priorities = this.priorities.map(
            (p) => Math.pow(p, priorityScale) / this.sumPriorities
        );
        const cumulativeProbabilities = new Array(this.bufferSize).fill(0);
        cumulativeProbabilities[0] = priorities[0];
        for (let i = 1; i < this.bufferSize; i++) {
            cumulativeProbabilities[i] =
                cumulativeProbabilities[i - 1] + priorities[i];
        }
        const samples = [];
        const sampleIndices = [];
        for (let i = 0; i < batchSize; i++) {
            const sampleProbability = Math.random();
            const sampleIndex = this.binarySearch(cumulativeProbabilities, sampleProbability);
            const { experience } = this.buffer[sampleIndex];
            // const samplePriority = priorities[sampleIndex];
            samples.push(experience);
            sampleIndices.push(sampleIndex);
        }
        return { samples, sampleIndices };
    }

    binarySearch(array, value) {
        let start = 0;
        let end = array.length - 1;
        while (start <= end) {
            const mid = Math.floor((start + end) / 2);
            if (array[mid] === value) {
                return mid;
            } else if (array[mid] < value) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return start;
    }

    updatePriorities(sampleIndices, priorities) {
        for (let i = 0; i < sampleIndices.length; i++) {
            const index = sampleIndices[i];
            const oldPriority = this.priorities[index];
            const newPriority = Math.pow(priorities[i] + this.epsilon, this.alpha);
            this.priorities[index] = newPriority;
            this.sumPriorities = this.sumPriorities + newPriority - oldPriority;
        }
    }
}

module.exports = Memory;