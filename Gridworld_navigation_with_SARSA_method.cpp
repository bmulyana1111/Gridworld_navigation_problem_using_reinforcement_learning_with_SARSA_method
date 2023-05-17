#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Define the gridworld size
const int GRID_SIZE = 5;

// Define the SARSA agent class
class SARSA {
public:
    SARSA(float alpha, float gamma, float epsilon) :
        alpha(alpha), gamma(gamma), epsilon(epsilon), rng(std::random_device{}()) {
        // Initialize the Q-table with all values set to zero
        qTable.resize(GRID_SIZE, std::vector<float>(GRID_SIZE, 0.0f));
    }

    // Choose the best action for a given state
    int getBestAction(int state) const {
        int bestAction = 0;
        float bestValue = qTable[state / GRID_SIZE][state % GRID_SIZE];
        for (int action = 1; action < 4; ++action) {
            int nextState = getNextState(state, action);
            float value = qTable[nextState / GRID_SIZE][nextState % GRID_SIZE];
            if (value > bestValue) {
                bestAction = action;
                bestValue = value;
            }
        }
        return bestAction;
    }

    // Choose an action using epsilon-greedy exploration
    int chooseAction(int state) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < epsilon) {
            // Choose a random action with probability epsilon
            return std::uniform_int_distribution<int>(0, 3)(rng);
        }
        else {
            // Choose the best action for the current state with probability 1 - epsilon
            return getBestAction(state);
        }
    }

    // Update the Q-table based on the observed reward and next state
    void updateQTable(int state, int action, float reward, int nextState, int nextAction) {
        float qValue = qTable[state / GRID_SIZE][state % GRID_SIZE];
        float nextQValue = qTable[nextState / GRID_SIZE][nextState % GRID_SIZE];
        qTable[state / GRID_SIZE][state % GRID_SIZE] +=
            alpha * (reward + gamma * nextQValue - qValue);
    }

    // Decay the exploration rate
    void decayExplorationRate() {
        epsilon *= 0.99f;
    }

    // Print the Q-table
    void printQTable() const {
        std::cout << "Q-Table:\n";
        for (int i = 0; i < GRID_SIZE; ++i) {
            for (int j = 0; j < GRID_SIZE; ++j) {
                std::cout << qTable[i][j] << " ";
            }
            std::cout << "\n";
        }
    }

private:
    // Q-table
    std::vector<std::vector<float>> qTable;

    // Learning rate
    float alpha;

    // Discount factor
    float gamma;

    // Exploration rate
    float epsilon;

    // Random number generator
    std::mt19937 rng;

    // Get the next state given the current state and action
    int getNextState(int state, int action) const {
        int row = state / GRID_SIZE;
        int col = state % GRID_SIZE;

        switch (action) {
            case 0: // Up
                row = std::max(row - 1, 0);
                break;
            case 1: // Down
                row = std::min(row + 1, GRID_SIZE - 1);
                break;
            case 2: // Left
                col = std::max(col - 1, 0);
                break;
            case 3: // Right
                col = std::min(col + 1, GRID_SIZE - 1);
                break;
        }

        return row * GRID_SIZE + col;
    }
};

int main() {
    // Create an instance of the SARSA agent
    SARSA agent(0.5f, 0.9f, 0.1f);

    // Run the SARSA loop for a specified number of episodes
    const int numEpisodes = 100;
    for (int episode = 0; episode < numEpisodes; ++episode) {
        // Start in a random state
        int state = std::uniform_int_distribution<int>(0, GRID_SIZE * GRID_SIZE - 1)(agent.rng);

        // Choose the first action
        int action = agent.chooseAction(state);

        // Perform actions until reaching the terminal state
        while (state != GRID_SIZE * GRID_SIZE - 1) {
            // Get the next state and reward
            int nextState = agent.getNextState(state, action);
            float reward = (nextState == GRID_SIZE * GRID_SIZE - 1) ? 1.0f : 0.0f;

            // Choose the next action
            int nextAction = agent.chooseAction(nextState);

            // Update the Q-table
            agent.updateQTable(state, action, reward, nextState, nextAction);

            // Transition to the next state-action pair
            state = nextState;
            action = nextAction;
        }

        // Decay the exploration rate after each episode
        agent.decayExplorationRate();
    }

    // Print the learned Q-table
    agent.printQTable();

    return 0;
}
