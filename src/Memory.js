// The Memory class is a data structure used to store and manage the agent's experiences during training.
class Memory {
    // Constructor initializes a new memory buffer with specified size, epsilon, and alpha parameters.
    constructor(bufferSize, epsilon = 0.01, alpha = 0.5) {
        this.bufferSize = bufferSize; // Set the maximum size of the memory buffer
        this.buffer = new Array(bufferSize).fill(null); // Initialize the memory buffer with null values
        this.pointer = 0; // Pointer to the current position in the buffer where the next experience will be stored
        this.count = 0; // Count of the total experiences currently stored in the buffer
        this.sumPriorities = 0; // Sum of the priorities of all experiences in the buffer
        this.epsilon = epsilon; // Small value to ensure that no experience has a priority of 0
        this.alpha = alpha; // Exponent for priority calculations
        this.priorities = new Array(bufferSize).fill(0); // Array to store the priorities of experiences in the buffer
    }

    // The add method stores a new experience in the memory buffer.
    add(experience) {
        if (this.buffer[this.pointer] === null) {
            this.count++; // Increment the count of total experiences if a new slot is being filled
        }
        const oldPriority = this.priorities[this.pointer]; // Get the old priority of the experience at the current pointer position
        const newPriority = 10; // Set a high initial priority for new experiences to ensure they are sampled at least once
        this.sumPriorities = this.sumPriorities + newPriority - oldPriority; // Update the sum of priorities
        this.priorities[this.pointer] = newPriority; // Update the priority of the experience at the current pointer position
        this.buffer[this.pointer] = experience; // Store the new experience at the current pointer position
        this.pointer = (this.pointer + 1) % this.bufferSize; // Increment the pointer (circular buffer)
    }

    // The sample method randomly samples a batch of experiences from the memory buffer based on their priorities.
    sample(batchSize, priorityScale = 1) {
        const priorities = this.priorities.map(
            (p) => Math.pow(p, priorityScale) / this.sumPriorities // Calculate the scaled priority of each experience
        );
        const cumulativeProbabilities = new Array(this.bufferSize).fill(0); // Initialize an array to store cumulative probabilities
        cumulativeProbabilities[0] = priorities[0]; // Set the first value in cumulative probabilities array
        for (let i = 1; i < this.bufferSize; i++) {
            cumulativeProbabilities[i] =
                cumulativeProbabilities[i - 1] + priorities[i]; // Calculate cumulative probabilities
        }
        const samples = []; // Initialize an array to store the sampled experiences
        const sampleIndices = []; // Initialize an array to store the indices of the sampled experiences
        for (let i = 0; i < batchSize; i++) {
            const sampleProbability = Math.random(); // Generate a random number for sampling
            const sampleIndex = this.binarySearch(cumulativeProbabilities, sampleProbability); // Find the index of the experience to be sampled
            const experience = this.buffer[sampleIndex]; // Get the sampled experience
            samples.push(experience); // Store the sampled experience
            sampleIndices.push(sampleIndex); // Store the index of the sampled experience
        }
        return { samples, sampleIndices }; // Return the sampled experiences and their indices
    }

    // The binarySearch method finds the index of a value in a sorted array using the binary search algorithm.
    binarySearch(array, value) {
        let start = 0; // Initialize the start of the search range
        let end = array.length - 1; // Initialize the end of the search range
        while (start <= end) {
            const mid = Math.floor((start + end) / 2); // Calculate the midpoint of the search range
            if (array[mid] === value) {
                return mid; // If the value is found, return the index
            } else if (array[mid] < value) {
                start = mid + 1; // If the value is greater, update the start of the search range
            } else {
                end = mid - 1; // If the value is smaller, update the end of the search range
            }
        }
        return start; // Return the start index if the value is not found (the index where the value should be)
    }

    // The updatePriorities method updates the priorities of the specified experiences in the memory buffer.
    updatePriorities(sampleIndices, priorities) {
        for (let i = 0; i < sampleIndices.length; i++) {
            const index = sampleIndices[i]; // Get the index of the experience to be updated
            const oldPriority = this.priorities[index]; // Get the old priority of the experience
            const newPriority = Math.pow(priorities[i] + this.epsilon, this.alpha); // Calculate the new priority of the experience
            this.priorities[index] = newPriority; // Update the priority of the experience
            this.sumPriorities = this.sumPriorities + newPriority - oldPriority; // Update the sum of priorities
        }
    }
}

module.exports = Memory; // Export the Memory class for use in other modules