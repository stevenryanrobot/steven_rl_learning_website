// Simple Q-Learning implementation for grid world
class QLearningAgent {
    constructor(states, actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.states = states;
        this.actions = actions;
        this.alpha = alpha; // learning rate
        this.gamma = gamma; // discount factor
        this.epsilon = epsilon; // exploration rate
        
        // Initialize Q-table with zeros
        this.qTable = {};
        states.forEach(state => {
            this.qTable[state] = {};
            actions.forEach(action => {
                this.qTable[state][action] = 0;
            });
        });
    }
    
    chooseAction(state) {
        // Epsilon-greedy policy
        if (Math.random() < this.epsilon) {
            // Explore: random action
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        } else {
            // Exploit: best known action
            let bestAction = this.actions[0];
            let bestValue = this.qTable[state][bestAction];
            
            this.actions.forEach(action => {
                if (this.qTable[state][action] > bestValue) {
                    bestValue = this.qTable[state][action];
                    bestAction = action;
                }
            });
            
            return bestAction;
        }
    }
    
    learn(state, action, reward, nextState) {
        const currentQ = this.qTable[state][action];
        const maxNextQ = Math.max(...this.actions.map(a => this.qTable[nextState][a]));
        const target = reward + this.gamma * maxNextQ;
        this.qTable[state][action] = currentQ + this.alpha * (target - currentQ);
    }
}

// Grid world visualization
class GridWorld {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.agentPos = {x: 0, y: 0};
        this.goalPos = {x: width-1, y: height-1};
        this.obstacles = [];
    }
    
    getState() {
        return `${this.agentPos.x},${this.agentPos.y}`;
    }
    
    getReward() {
        if (this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y) {
            return 10; // Goal reached
        }
        return -1; // Step penalty
    }
    
    isTerminal() {
        return this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y;
    }
    
    move(action) {
        const oldPos = {...this.agentPos};
        
        switch(action) {
            case 'up':
                if (this.agentPos.y > 0) this.agentPos.y--;
                break;
            case 'down':
                if (this.agentPos.y < this.height-1) this.agentPos.y++;
                break;
            case 'left':
                if (this.agentPos.x > 0) this.agentPos.x--;
                break;
            case 'right':
                if (this.agentPos.x < this.width-1) this.agentPos.x++;
                break;
        }
        
        // Check for obstacles
        const obstacleIndex = this.obstacles.findIndex(obs => 
            obs.x === this.agentPos.x && obs.y === this.agentPos.y
        );
        if (obstacleIndex !== -1) {
            this.agentPos = oldPos; // Hit obstacle, stay in place
        }
    }
    
    reset() {
        this.agentPos = {x: 0, y: 0};
    }
}

// DOM interaction
document.addEventListener('DOMContentLoaded', function() {
    const demoBtn = document.getElementById('demoBtn');
    const gridContainer = document.getElementById('gridWorld');
    
    if (demoBtn && gridContainer) {
        demoBtn.addEventListener('click', runDemo);
    }
    
    function runDemo() {
        const grid = new GridWorld(5, 5);
        const states = [];
        for (let x = 0; x < 5; x++) {
            for (let y = 0; y < 5; y++) {
                states.push(`${x},${y}`);
            }
        }
        const actions = ['up', 'down', 'left', 'right'];
        const agent = new QLearningAgent(states, actions);
        
        // Run training episodes
        for (let episode = 0; episode < 100; episode++) {
            grid.reset();
            let steps = 0;
            
            while (!grid.isTerminal() && steps < 100) {
                const state = grid.getState();
                const action = agent.chooseAction(state);
                grid.move(action);
                const reward = grid.getReward();
                const nextState = grid.getState();
                
                agent.learn(state, action, reward, nextState);
                steps++;
            }
        }
        
        // Show final policy
        updateGridDisplay(grid, agent);
    }
    
    function updateGridDisplay(grid, agent) {
        gridContainer.innerHTML = '';
        const cellSize = 60;
        
        for (let y = 0; y < grid.height; y++) {
            for (let x = 0; x < grid.width; x++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;
                
                const state = `${x},${y}`;
                if (x === grid.goalPos.x && y === grid.goalPos.y) {
                    cell.textContent = '🎯';
                    cell.style.backgroundColor = '#4CAF50';
                } else if (x === 0 && y === 0) {
                    cell.textContent = '🤖';
                    cell.style.backgroundColor = '#2196F3';
                } else {
                    // Show best action as arrow
                    const qValues = agent.qTable[state];
                    let bestAction = 'right';
                    let bestValue = qValues[bestAction];
                    
                    ['up', 'down', 'left', 'right'].forEach(action => {
                        if (qValues[action] > bestValue) {
                            bestValue = qValues[action];
                            bestAction = action;
                        }
                    });
                    
                    const arrows = {
                        'up': '↑',
                        'down': '↓', 
                        'left': '←',
                        'right': '→'
                    };
                    cell.textContent = arrows[bestAction];
                    cell.style.backgroundColor = '#f5f5f5';
                }
                
                gridContainer.appendChild(cell);
            }
            gridContainer.appendChild(document.createElement('br'));
        }
    }
});

// Algorithm comparison data
const algorithms = {
    'Q-Learning': {
        type: 'Model-free, Off-policy',
        description: 'Learns action-value function Q(s,a) directly from experience',
        pros: ['Simple to implement', 'Guaranteed convergence', 'Works well for discrete spaces'],
        cons: ['Struggles with large state spaces', 'Requires discretization']
    },
    'Deep Q-Networks (DQN)': {
        type: 'Model-free, Off-policy',
        description: 'Uses neural networks to approximate Q-function for high-dimensional inputs',
        pros: ['Handles continuous/large state spaces', 'End-to-end learning', 'Successful in complex domains'],
        cons: ['Computationally expensive', 'Can be unstable', 'Requires careful hyperparameter tuning']
    },
    'Policy Gradient': {
        type: 'Model-free, On-policy',
        description: 'Directly optimizes policy parameters using gradient ascent',
        pros: ['Naturally handles stochastic policies', 'Good for continuous action spaces'],
        cons: ['High variance', 'Sample inefficient', 'Convergence can be slow']
    },
    'Actor-Critic': {
        type: 'Model-free, On-policy',
        description: 'Combines value-based and policy-based methods',
        pros: ['Lower variance than pure policy gradient', 'More sample efficient'],
        cons: ['More complex to implement', 'Requires balancing two networks']
    },
    'PPO': {
        type: 'Model-free, On-policy',
        description: 'Uses clipped probability ratios for stable policy updates',
        pros: ['Stable and reliable', 'Good sample efficiency', 'Widely used in practice'],
        cons: ['Still requires many samples', 'Hyperparameter sensitive']
    }
};

// Interactive algorithm selector
function showAlgorithmDetails(algoName) {
    const detailsDiv = document.getElementById('algorithmDetails');
    if (!detailsDiv || !algorithms[algoName]) return;
    
    const algo = algorithms[algoName];
    detailsDiv.innerHTML = `
        <h3>${algoName}</h3>
        <p><strong>Type:</strong> ${algo.type}</p>
        <p><strong>Description:</strong> ${algo.description}</p>
        <p><strong>Pros:</strong></p>
        <ul>${algo.pros.map(p => `<li>${p}</li>`).join('')}</ul>
        <p><strong>Cons:</strong></p>
        <ul>${algo.cons.map(c => `<li>${c}</li>`).join('')}</ul>
    `;
}