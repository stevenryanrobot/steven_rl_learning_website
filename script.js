// 强化学习交互演示套件

// ============================================
// 1. Q-Learning 网格世界
// ============================================

class QLearningAgent {
    constructor(states, actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.states = states;
        this.actions = actions;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.qTable = {};
        states.forEach(state => {
            this.qTable[state] = {};
            actions.forEach(action => {
                this.qTable[state][action] = 0;
            });
        });
    }
    
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        } else {
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

class GridWorld {
    constructor(width = 5, height = 5) {
        this.width = width;
        this.height = height;
        this.agentPos = {x: 0, y: 0};
        this.goalPos = {x: width-1, y: height-1};
        this.obstacles = [{x: 1, y: 1}, {x: 2, y: 1}, {x: 3, y: 2}, {x: 1, y: 3}];
    }
    
    getState() { return `${this.agentPos.x},${this.agentPos.y}`; }
    
    getReward() {
        if (this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y) {
            return 10;
        }
        return -0.1;
    }
    
    isTerminal() {
        return this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y;
    }
    
    move(action) {
        const oldPos = {...this.agentPos};
        switch(action) {
            case 'up': if (this.agentPos.y > 0) this.agentPos.y--; break;
            case 'down': if (this.agentPos.y < this.height-1) this.agentPos.y++; break;
            case 'left': if (this.agentPos.x > 0) this.agentPos.x--; break;
            case 'right': if (this.agentPos.x < this.width-1) this.agentPos.x++; break;
        }
        const hitObstacle = this.obstacles.some(obs => obs.x === this.agentPos.x && obs.y === this.agentPos.y);
        if (hitObstacle) {
            this.agentPos = oldPos;
            return -1;
        }
        return 0;
    }
    
    reset() { this.agentPos = {x: 0, y: 0}; }
}

class GridWorldVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.cellSize = 80;
        this.grid = new GridWorld(5, 5);
        this.states = [];
        for (let x = 0; x < this.grid.width; x++) {
            for (let y = 0; y < this.grid.height; y++) {
                this.states.push(`${x},${y}`);
            }
        }
        this.actions = ['up', 'down', 'left', 'right'];
        this.agent = new QLearningAgent(this.states, this.actions);
        this.isTraining = false;
        this.episodeCount = 0;
        this.stepCount = 0;
        this.currentReward = 0;
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('startBtn');
        const resetBtn = document.getElementById('resetBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('startBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        if (this.isTraining) this.train();
    }
    
    train() {
        if (!this.isTraining) return;
        for (let ep = 0; ep < 5; ep++) {
            this.grid.reset();
            let steps = 0;
            let episodeReward = 0;
            while (!this.grid.isTerminal() && steps < 100) {
                const state = this.grid.getState();
                const action = this.agent.chooseAction(state);
                const moveReward = this.grid.move(action);
                const reward = this.grid.getReward() + moveReward;
                const nextState = this.grid.getState();
                this.agent.learn(state, action, reward, nextState);
                episodeReward += reward;
                steps++;
            }
            this.episodeCount++;
            this.stepCount += steps;
            this.currentReward = episodeReward;
        }
        this.updateStats();
        this.draw();
        if (this.isTraining) requestAnimationFrame(() => this.train());
    }
    
    reset() {
        this.isTraining = false;
        this.episodeCount = 0;
        this.stepCount = 0;
        this.currentReward = 0;
        this.agent = new QLearningAgent(this.states, this.actions);
        const btn = document.getElementById('startBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        this.updateStats();
        this.draw();
    }
    
    updateStats() {
        const epEl = document.getElementById('episodeCount');
        const stepEl = document.getElementById('stepCount');
        const rewardEl = document.getElementById('currentReward');
        if (epEl) epEl.textContent = this.episodeCount;
        if (stepEl) stepEl.textContent = this.stepCount;
        if (rewardEl) rewardEl.textContent = this.currentReward.toFixed(2);
    }
    
    draw() {
        if (!this.ctx) return;
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        for (let y = 0; y < this.grid.height; y++) {
            for (let x = 0; x < this.grid.width; x++) {
                const cellX = x * this.cellSize;
                const cellY = y * this.cellSize;
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fillRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
                const isObstacle = this.grid.obstacles.some(obs => obs.x === x && obs.y === y);
                if (isObstacle) {
                    this.ctx.fillStyle = '#9e9e9e';
                    this.ctx.fillRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
                    this.ctx.fillStyle = '#fff';
                    this.ctx.font = '30px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🚫', cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                if (x === this.grid.goalPos.x && y === this.grid.goalPos.y) {
                    this.ctx.fillStyle = '#4CAF50';
                    this.ctx.fillRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
                    this.ctx.font = '30px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🎯', cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                if (x === this.grid.agentPos.x && y === this.grid.agentPos.y) {
                    this.ctx.fillStyle = '#2196F3';
                    this.ctx.beginPath();
                    this.ctx.arc(cellX + this.cellSize/2, cellY + this.cellSize/2, this.cellSize/3, 0, Math.PI * 2);
                    this.ctx.fill();
                    this.ctx.font = '24px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🤖', cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                if (this.episodeCount > 0 && !isObstacle && !(x === this.grid.goalPos.x && y === this.grid.goalPos.y)) {
                    const state = `${x},${y}`;
                    const qValues = this.agent.qTable[state];
                    let bestAction = 'right';
                    let bestValue = qValues[bestAction];
                    this.actions.forEach(action => {
                        if (qValues[action] > bestValue) {
                            bestValue = qValues[action];
                            bestAction = action;
                        }
                    });
                    const arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'};
                    this.ctx.fillStyle = 'rgba(33, 150, 243, 0.6)';
                    this.ctx.font = 'bold 36px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText(arrows[bestAction], cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                this.ctx.strokeStyle = '#ddd';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
            }
        }
    }
}

// ============================================
// 2. 多臂老虎机 (Multi-Armed Bandit)
// ============================================

class MultiArmedBandit {
    constructor(numBandits = 5) {
        this.numBandits = numBandits;
        this.trueProbabilities = [];
        for (let i = 0; i < numBandits; i++) {
            this.trueProbabilities.push(0.1 + Math.random() * 0.8);
        }
        this.counts = new Array(numBandits).fill(0);
        this.rewards = new Array(numBandits).fill(0);
        this.totalPulls = 0;
        this.totalReward = 0;
        this.epsilon = 0.1;
    }
    
    pull(banditIndex) {
        const reward = Math.random() < this.trueProbabilities[banditIndex] ? 1 : 0;
        this.counts[banditIndex]++;
        this.rewards[banditIndex] += reward;
        this.totalPulls++;
        this.totalReward += reward;
        return reward;
    }
    
    getEstimatedValues() {
        return this.counts.map((count, i) => count > 0 ? this.rewards[i] / count : 0);
    }
    
    chooseAction() {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numBandits);
        } else {
            const values = this.getEstimatedValues();
            return values.indexOf(Math.max(...values));
        }
    }
    
    reset() {
        for (let i = 0; i < this.numBandits; i++) {
            this.trueProbabilities[i] = 0.1 + Math.random() * 0.8;
        }
        this.counts.fill(0);
        this.rewards.fill(0);
        this.totalPulls = 0;
        this.totalReward = 0;
    }
}

class BanditVisualizer {
    constructor() {
        this.bandit = new MultiArmedBandit(5);
        this.container = document.getElementById('bandits');
        this.isAutoPlaying = false;
        this.autoPlayInterval = null;
        this.setupUI();
        this.setupEventListeners();
    }
    
    setupUI() {
        if (!this.container) return;
        this.container.innerHTML = '';
        for (let i = 0; i < this.bandit.numBandits; i++) {
            const machine = document.createElement('div');
            machine.className = 'bandit-machine';
            machine.innerHTML = `
                <span class="emoji">🎰</span>
                <span class="label">机器 ${i+1}</span>
                <span class="stats">
                    <div>次数：<span id="count-${i}">0</span></div>
                    <div>胜率：<span id="rate-${i}">0%</span></div>
                </span>
            `;
            machine.addEventListener('click', () => this.pullMachine(i));
            this.container.appendChild(machine);
        }
        this.updateDisplay();
    }
    
    setupEventListeners() {
        const playBtn = document.getElementById('banditPlayBtn');
        const autoBtn = document.getElementById('banditAutoBtn');
        const resetBtn = document.getElementById('banditResetBtn');
        if (playBtn) playBtn.addEventListener('click', () => this.playOnce());
        if (autoBtn) autoBtn.addEventListener('click', () => this.toggleAutoPlay());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    pullMachine(index) {
        const reward = this.bandit.pull(index);
        this.updateDisplay();
        if (reward > 0) {
            const machine = this.container.children[index];
            machine.style.transform = 'scale(1.2)';
            setTimeout(() => machine.style.transform = 'scale(1)', 200);
        }
    }
    
    playOnce() {
        const action = this.bandit.chooseAction();
        this.pullMachine(action);
    }
    
    toggleAutoPlay() {
        const btn = document.getElementById('banditAutoBtn');
        if (this.isAutoPlaying) {
            clearInterval(this.autoPlayInterval);
            this.isAutoPlaying = false;
            if (btn) btn.textContent = '自动播放';
        } else {
            this.isAutoPlaying = true;
            if (btn) btn.textContent = '停止播放';
            this.autoPlayInterval = setInterval(() => this.playOnce(), 200);
        }
    }
    
    reset() {
        this.isAutoPlaying = false;
        if (this.autoPlayInterval) clearInterval(this.autoPlayInterval);
        this.bandit.reset();
        const btn = document.getElementById('banditAutoBtn');
        if (btn) btn.textContent = '自动播放';
        this.setupUI();
    }
    
    updateDisplay() {
        const values = this.bandit.getEstimatedValues();
        for (let i = 0; i < this.bandit.numBandits; i++) {
            const countEl = document.getElementById(`count-${i}`);
            const rateEl = document.getElementById(`rate-${i}`);
            if (countEl) countEl.textContent = this.bandit.counts[i];
            if (rateEl) rateEl.textContent = (values[i] * 100).toFixed(0) + '%';
        }
        const totalPullsEl = document.getElementById('totalPulls');
        const totalRewardEl = document.getElementById('totalReward');
        if (totalPullsEl) totalPullsEl.textContent = this.bandit.totalPulls;
        if (totalRewardEl) totalRewardEl.textContent = this.bandit.totalReward;
    }
}

// ============================================
// 3. DQN 演示 (简化版)
// ============================================

class DQNDemo {
    constructor() {
        this.canvas = document.getElementById('dqnCanvas');
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.isTraining = false;
        this.episode = 0;
        this.loss = 0.5;
        this.epsilon = 1.0;
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('dqnStartBtn');
        const resetBtn = document.getElementById('dqnResetBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('dqnStartBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        if (this.isTraining) this.train();
    }
    
    train() {
        if (!this.isTraining) return;
        for (let i = 0; i < 10; i++) {
            this.episode++;
            this.loss = Math.max(0.01, this.loss * 0.98);
            this.epsilon = Math.max(0.01, this.epsilon * 0.995);
            this.animateNetwork();
        }
        this.updateStats();
        this.draw();
        if (this.isTraining) requestAnimationFrame(() => this.train());
    }
    
    reset() {
        this.isTraining = false;
        this.episode = 0;
        this.loss = 0.5;
        this.epsilon = 1.0;
        const btn = document.getElementById('dqnStartBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        this.updateStats();
        this.draw();
    }
    
    updateStats() {
        const lossEl = document.getElementById('dqnLoss');
        const epEl = document.getElementById('dqnEpisode');
        const epsEl = document.getElementById('dqnEpsilon');
        if (lossEl) lossEl.textContent = this.loss.toFixed(4);
        if (epEl) epEl.textContent = this.episode;
        if (epsEl) epsEl.textContent = this.epsilon.toFixed(4);
    }
    
    animateNetwork() {
        const neurons = document.querySelectorAll('.neuron');
        neurons.forEach((neuron, i) => {
            if (Math.random() < 0.3) {
                neuron.classList.add('active');
                setTimeout(() => neuron.classList.remove('active'), 200);
            }
        });
    }
    
    draw() {
        if (!this.ctx) return;
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        this.ctx.fillStyle = '#667eea';
        this.ctx.font = 'bold 20px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('DQN 训练可视化', w/2, h/2 - 20);
        this.ctx.font = '14px Arial';
        this.ctx.fillStyle = '#666';
        this.ctx.fillText('观察右侧神经网络的激活状态', w/2, h/2 + 20);
        // 绘制训练曲线
        this.ctx.strokeStyle = '#764ba2';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(50, h - 50);
        for (let i = 0; i < 100; i++) {
            const x = 50 + i * 4;
            const y = h - 50 - Math.sin(i * 0.1) * 30 - this.loss * 50;
            this.ctx.lineTo(x, y);
        }
        this.ctx.stroke();
    }
}

// ============================================
// 4. 策略梯度演示
// ============================================

class PolicyGradientDemo {
    constructor() {
        this.canvas = document.getElementById('policyCanvas');
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.isTraining = false;
        this.episode = 0;
        this.avgReward = 0;
        this.policy = [0.25, 0.25, 0.25, 0.25]; // [up, down, left, right]
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('policyStartBtn');
        const resetBtn = document.getElementById('policyResetBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('policyStartBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        if (this.isTraining) this.train();
    }
    
    train() {
        if (!this.isTraining) return;
        for (let i = 0; i < 5; i++) {
            this.episode++;
            // 模拟策略更新
            const bestAction = Math.floor(Math.random() * 4);
            for (let j = 0; j < 4; j++) {
                if (j === bestAction) {
                    this.policy[j] = Math.min(0.9, this.policy[j] + 0.05);
                } else {
                    this.policy[j] = Math.max(0.05, this.policy[j] - 0.02);
                }
            }
            // 归一化
            const sum = this.policy.reduce((a, b) => a + b, 0);
            this.policy = this.policy.map(p => p / sum);
            this.avgReward = this.avgReward * 0.95 + Math.random() * 2;
        }
        this.updateDisplay();
        this.updateStats();
        this.draw();
        if (this.isTraining) requestAnimationFrame(() => this.train());
    }
    
    reset() {
        this.isTraining = false;
        this.episode = 0;
        this.avgReward = 0;
        this.policy = [0.25, 0.25, 0.25, 0.25];
        const btn = document.getElementById('policyStartBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        this.updateDisplay();
        this.updateStats();
        this.draw();
    }
    
    updateDisplay() {
        const labels = ['up', 'down', 'left', 'right'];
        labels.forEach((label, i) => {
            const bar = document.getElementById(`bar-${label}`);
            const prob = document.getElementById(`prob-${label}`);
            if (bar) bar.style.width = (this.policy[i] * 100) + '%';
            if (prob) prob.textContent = (this.policy[i] * 100).toFixed(1) + '%';
        });
    }
    
    updateStats() {
        const epEl = document.getElementById('policyEpisode');
        const rewardEl = document.getElementById('policyReward');
        if (epEl) epEl.textContent = this.episode;
        if (rewardEl) rewardEl.textContent = this.avgReward.toFixed(2);
    }
    
    draw() {
        if (!this.ctx) return;
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        this.ctx.fillStyle = '#667eea';
        this.ctx.font = 'bold 20px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('策略梯度训练', w/2, h/2 - 20);
        this.ctx.font = '14px Arial';
        this.ctx.fillStyle = '#666';
        this.ctx.fillText('观察左侧策略分布的变化', w/2, h/2 + 20);
    }
}

// ============================================
// 演示切换器
// ============================================

function initDemoTabs() {
    const tabs = document.querySelectorAll('.demo-tab');
    const panels = document.querySelectorAll('.demo-panel');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const demoName = tab.getAttribute('data-demo');
            
            // 更新标签状态
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // 更新面板显示
            panels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === `${demoName}-demo`) {
                    panel.classList.add('active');
                }
            });
            
            // 初始化对应的演示
            setTimeout(() => {
                if (demoName === 'gridworld' && !window.gridVisualizer) {
                    window.gridVisualizer = new GridWorldVisualizer('gridCanvas');
                } else if (demoName === 'bandit' && !window.banditVisualizer) {
                    window.banditVisualizer = new BanditVisualizer();
                } else if (demoName === 'dqn' && !window.dqnDemo) {
                    window.dqnDemo = new DQNDemo();
                } else if (demoName === 'policy' && !window.policyDemo) {
                    window.policyDemo = new PolicyGradientDemo();
                }
            }, 100);
        });
    });
}

// ============================================
// 页面加载初始化
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing RL Demos...');
    
    // 初始化演示切换
    initDemoTabs();
    
    // 初始化网格世界（默认显示）
    const gridCanvas = document.getElementById('gridCanvas');
    if (gridCanvas) {
        window.gridVisualizer = new GridWorldVisualizer('gridCanvas');
    }
    
    console.log('RL Demos initialized successfully!');
});
