// 强化学习交互演示套件 - 第三阶段
// 包含：超参数调节、训练曲线、模型保存、新环境

// ============================================
// 全局工具函数
// ============================================

function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
        return true;
    } catch (e) {
        console.error('Save failed:', e);
        return false;
    }
}

function loadFromLocalStorage(key) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (e) {
        console.error('Load failed:', e);
        return null;
    }
}

// ============================================
// 1. Q-Learning 网格世界（增强版）
// ============================================

class QLearningAgent {
    constructor(states, actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.states = states;
        this.actions = actions;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.qTable = {};
        this.initQTable();
    }
    
    initQTable() {
        this.qTable = {};
        this.states.forEach(state => {
            this.qTable[state] = {};
            this.actions.forEach(action => {
                this.qTable[state][action] = 0;
            });
        });
    }
    
    setParams(alpha, gamma, epsilon) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
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
    
    toJSON() {
        return {
            qTable: this.qTable,
            alpha: this.alpha,
            gamma: this.gamma,
            epsilon: this.epsilon
        };
    }
    
    fromJSON(data) {
        this.qTable = data.qTable;
        this.alpha = data.alpha;
        this.gamma = data.gamma;
        this.epsilon = data.epsilon;
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
    constructor(canvasId, chartId) {
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
        
        // 超参数
        this.alpha = 0.1;
        this.gamma = 0.9;
        this.epsilon = 0.1;
        this.trainSpeed = 1;
        
        // 训练状态
        this.isTraining = false;
        this.episodeCount = 0;
        this.stepCount = 0;
        this.currentReward = 0;
        this.episodeRewards = [];
        this.avgRewards = [];
        
        // 图表
        this.chartCtx = document.getElementById(chartId);
        this.rewardChart = null;
        if (this.chartCtx) {
            this.initChart();
        }
        
        this.setupEventListeners();
        this.setupHyperparams();
        this.draw();
    }
    
    initChart() {
        this.rewardChart = new Chart(this.chartCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Reward',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }, {
                    label: 'Average Reward (10)',
                    data: [],
                    borderColor: '#764ba2',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                animation: false,
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: 'Episode' }
                    },
                    y: {
                        display: true,
                        title: { display: true, text: 'Reward' }
                    }
                },
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: '训练曲线' }
                }
            }
        });
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('startBtn');
        const resetBtn = document.getElementById('resetBtn');
        const saveBtn = document.getElementById('saveBtn');
        const loadBtn = document.getElementById('loadBtn');
        
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
        if (saveBtn) saveBtn.addEventListener('click', () => this.saveModel());
        if (loadBtn) loadBtn.addEventListener('click', () => this.loadModel());
    }
    
    setupHyperparams() {
        const lrSlider = document.getElementById('lrSlider');
        const gammaSlider = document.getElementById('gammaSlider');
        const epsilonSlider = document.getElementById('epsilonSlider');
        const speedSlider = document.getElementById('speedSlider');
        
        if (lrSlider) {
            lrSlider.addEventListener('input', (e) => {
                this.alpha = parseFloat(e.target.value);
                document.getElementById('lrValue').textContent = this.alpha.toFixed(2);
                this.agent.setParams(this.alpha, this.gamma, this.epsilon);
            });
        }
        
        if (gammaSlider) {
            gammaSlider.addEventListener('input', (e) => {
                this.gamma = parseFloat(e.target.value);
                document.getElementById('gammaValue').textContent = this.gamma.toFixed(2);
                this.agent.setParams(this.alpha, this.gamma, this.epsilon);
            });
        }
        
        if (epsilonSlider) {
            epsilonSlider.addEventListener('input', (e) => {
                this.epsilon = parseFloat(e.target.value);
                document.getElementById('epsilonValue').textContent = this.epsilon.toFixed(2);
                this.agent.setParams(this.alpha, this.gamma, this.epsilon);
            });
        }
        
        if (speedSlider) {
            speedSlider.addEventListener('input', (e) => {
                this.trainSpeed = parseInt(e.target.value);
                document.getElementById('speedValue').textContent = this.trainSpeed + 'x';
            });
        }
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
        
        for (let s = 0; s < this.trainSpeed; s++) {
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
            this.episodeRewards.push(episodeReward);
            
            // 计算 10 轮平均
            const last10 = this.episodeRewards.slice(-10);
            const avg = last10.reduce((a, b) => a + b, 0) / last10.length;
            this.avgRewards.push(avg);
        }
        
        this.updateStats();
        this.updateChart();
        this.draw();
        
        if (this.isTraining) {
            requestAnimationFrame(() => this.train());
        }
    }
    
    updateStats() {
        const epEl = document.getElementById('episodeCount');
        const stepEl = document.getElementById('stepCount');
        const rewardEl = document.getElementById('currentReward');
        const avgEl = document.getElementById('avgReward');
        
        if (epEl) epEl.textContent = this.episodeCount;
        if (stepEl) stepEl.textContent = this.stepCount;
        if (rewardEl) rewardEl.textContent = this.currentReward.toFixed(2);
        if (avgEl && this.avgRewards.length > 0) {
            avgEl.textContent = this.avgRewards[this.avgRewards.length - 1].toFixed(2);
        }
    }
    
    updateChart() {
        if (!this.rewardChart) return;
        
        this.rewardChart.data.labels = this.episodeRewards.map((_, i) => i + 1);
        this.rewardChart.data.datasets[0].data = this.episodeRewards;
        this.rewardChart.data.datasets[1].data = this.avgRewards;
        this.rewardChart.update();
    }
    
    saveModel() {
        const data = {
            qTable: this.agent.toJSON(),
            episodeCount: this.episodeCount,
            episodeRewards: this.episodeRewards,
            timestamp: Date.now()
        };
        const saved = saveToLocalStorage('gridworld_model', data);
        if (saved) {
            alert('✅ 模型已保存！');
        } else {
            alert('❌ 保存失败');
        }
    }
    
    loadModel() {
        const data = loadFromLocalStorage('gridworld_model');
        if (data) {
            this.agent.fromJSON(data.qTable);
            this.episodeCount = data.episodeCount || 0;
            this.episodeRewards = data.episodeRewards || [];
            this.avgRewards = [];
            // 重新计算平均奖励
            for (let i = 0; i < this.episodeRewards.length; i++) {
                const last10 = this.episodeRewards.slice(Math.max(0, i - 9), i + 1);
                const avg = last10.reduce((a, b) => a + b, 0) / last10.length;
                this.avgRewards.push(avg);
            }
            this.updateStats();
            this.updateChart();
            this.draw();
            alert('✅ 模型已加载！');
        } else {
            alert('❌ 未找到保存的模型');
        }
    }
    
    reset() {
        this.isTraining = false;
        this.episodeCount = 0;
        this.stepCount = 0;
        this.currentReward = 0;
        this.episodeRewards = [];
        this.avgRewards = [];
        this.agent.initQTable();
        
        const btn = document.getElementById('startBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        
        this.updateStats();
        if (this.rewardChart) {
            this.rewardChart.data.labels = [];
            this.rewardChart.data.datasets[0].data = [];
            this.rewardChart.data.datasets[1].data = [];
            this.rewardChart.update();
        }
        this.draw();
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
// 2. CartPole 倒立摆环境
// ============================================

class CartPoleEnv {
    constructor() {
        this.gravity = 9.8;
        this.cartMass = 1.0;
        this.poleMass = 0.1;
        this.poleLength = 0.5;
        this.forceMag = 10.0;
        this.dt = 0.02;
        this.reset();
    }
    
    reset() {
        this.x = 0; // 小车位置
        this.xDot = 0; // 小车速度
        this.theta = 0; // 杆子角度（0 为垂直）
        this.thetaDot = 0; // 杆子角速度
        this.steps = 0;
        return this.getState();
    }
    
    getState() {
        return [this.x, this.xDot, this.theta, this.thetaDot];
    }
    
    step(action) {
        const force = action === 1 ? this.forceMag : -this.forceMag;
        
        const cosTheta = Math.cos(this.theta);
        const sinTheta = Math.sin(this.theta);
        
        const totalMass = this.cartMass + this.poleMass;
        const poleMassLength = this.poleMass * this.poleLength;
        
        const temp = (force + poleMassLength * this.thetaDot * this.thetaDot * sinTheta) / totalMass;
        const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) / 
            (this.poleLength * (4.0/3.0 - this.poleMass * cosTheta * cosTheta / totalMass));
        const xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;
        
        this.x += this.dt * this.xDot;
        this.xDot += this.dt * xAcc;
        this.theta += this.dt * this.thetaDot;
        this.thetaDot += this.dt * thetaAcc;
        
        this.steps++;
        
        // 检查是否结束
        const done = (
            this.x < -2.4 || this.x > 2.4 ||
            this.theta < -0.2 || this.theta > 0.2 ||
            this.steps > 500
        );
        
        const reward = done ? 0 : 1;
        return { state: this.getState(), reward, done };
    }
}

class CartPoleVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.env = new CartPoleEnv();
        
        this.isTraining = false;
        this.episode = 0;
        this.bestScore = 0;
        this.currentSteps = 0;
        
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('cartpoleStartBtn');
        const resetBtn = document.getElementById('cartpoleResetBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('cartpoleStartBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        if (this.isTraining) this.train();
    }
    
    train() {
        if (!this.isTraining) return;
        
        // 简单随机策略演示
        let state = this.env.reset();
        let episodeReward = 0;
        
        while (true) {
            const action = Math.random() > 0.5 ? 1 : 0;
            const result = this.env.step(action);
            episodeReward += result.reward;
            
            if (result.done) {
                this.episode++;
                this.currentSteps = this.env.steps;
                if (this.currentSteps > this.bestScore) {
                    this.bestScore = this.currentSteps;
                }
                this.updateStats();
                this.env.reset();
                break;
            }
        }
        
        this.draw();
        
        if (this.isTraining) {
            setTimeout(() => this.train(), 50);
        }
    }
    
    reset() {
        this.isTraining = false;
        this.episode = 0;
        this.bestScore = 0;
        this.currentSteps = 0;
        this.env.reset();
        
        const btn = document.getElementById('cartpoleStartBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        
        this.updateStats();
        this.draw();
    }
    
    updateStats() {
        const epEl = document.getElementById('cartpoleEpisode');
        const stepsEl = document.getElementById('cartpoleSteps');
        const bestEl = document.getElementById('cartpoleBest');
        const angleEl = document.getElementById('cartpoleAngle');
        
        if (epEl) epEl.textContent = this.episode;
        if (stepsEl) stepsEl.textContent = this.currentSteps;
        if (bestEl) bestEl.textContent = this.bestScore;
        if (angleEl) angleEl.textContent = (this.env.theta * 180 / Math.PI).toFixed(1) + '°';
    }
    
    draw() {
        if (!this.ctx) return;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        
        const centerX = w / 2;
        const groundY = h - 50;
        const scale = 80;
        
        // 绘制地面
        this.ctx.fillStyle = '#8B4513';
        this.ctx.fillRect(0, groundY, w, h - groundY);
        
        // 绘制小车
        const cartX = centerX + this.env.x * scale;
        const cartY = groundY - 30;
        this.ctx.fillStyle = '#2196F3';
        this.ctx.fillRect(cartX - 40, cartY, 80, 30);
        
        // 绘制轮子
        this.ctx.fillStyle = '#333';
        this.ctx.beginPath();
        this.ctx.arc(cartX - 30, cartY + 30, 10, 0, Math.PI * 2);
        this.ctx.arc(cartX + 30, cartY + 30, 10, 0, Math.PI * 2);
        this.ctx.fill();
        
        // 绘制杆子
        const poleLength = 100;
        const poleEndX = cartX + poleLength * Math.sin(this.env.theta);
        const poleEndY = cartY - poleLength * Math.cos(this.env.theta);
        
        this.ctx.strokeStyle = '#FF5722';
        this.ctx.lineWidth = 8;
        this.ctx.beginPath();
        this.ctx.moveTo(cartX, cartY);
        this.ctx.lineTo(poleEndX, poleEndY);
        this.ctx.stroke();
        
        // 绘制杆子顶端的球
        this.ctx.fillStyle = '#FF5722';
        this.ctx.beginPath();
        this.ctx.arc(poleEndX, poleEndY, 15, 0, Math.PI * 2);
        this.ctx.fill();
        
        // 绘制边界线
        this.ctx.strokeStyle = '#f44336';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - 2.4 * scale, groundY);
        this.ctx.lineTo(centerX - 2.4 * scale, groundY - 150);
        this.ctx.moveTo(centerX + 2.4 * scale, groundY);
        this.ctx.lineTo(centerX + 2.4 * scale, groundY - 150);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        // 标签
        this.ctx.fillStyle = '#666';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('边界', centerX - 2.4 * scale, groundY - 160);
        this.ctx.fillText('边界', centerX + 2.4 * scale, groundY - 160);
    }
}

// ============================================
// 3. MountainCar 山地车环境
// ============================================

class MountainCarEnv {
    constructor() {
        this.minPosition = -1.2;
        this.maxPosition = 0.6;
        this.maxSpeed = 0.07;
        this.goalPosition = 0.5;
        this.force = 0.001;
        this.gravity = 0.0025;
        this.reset();
    }
    
    reset() {
        this.position = -0.5 + Math.random() * 0.1;
        this.velocity = 0;
        this.steps = 0;
        this.successCount = 0;
        return this.getState();
    }
    
    getState() {
        return [this.position, this.velocity];
    }
    
    step(action) {
        // action: 0=left, 1=noop, 2=right
        const force = action === 2 ? this.force : (action === 0 ? -this.force : 0);
        
        this.velocity += force - this.gravity * Math.cos(3 * this.position);
        this.velocity = Math.max(-this.maxSpeed, Math.min(this.maxSpeed, this.velocity));
        this.position += this.velocity;
        this.position = Math.max(this.minPosition, Math.min(this.maxPosition, this.position));
        
        if (this.position === this.minPosition) {
            this.velocity = 0;
        }
        
        this.steps++;
        
        const done = this.position >= this.goalPosition || this.steps > 200;
        const reward = done && this.position >= this.goalPosition ? 1 : 0;
        
        if (done && this.position >= this.goalPosition) {
            this.successCount++;
        }
        
        return { state: this.getState(), reward, done };
    }
}

class MountainCarVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.env = new MountainCarEnv();
        
        this.isTraining = false;
        this.episode = 0;
        
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('mountaincarStartBtn');
        const resetBtn = document.getElementById('mountaincarResetBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('mountaincarStartBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        if (this.isTraining) this.train();
    }
    
    train() {
        if (!this.isTraining) return;
        
        let state = this.env.reset();
        
        while (true) {
            // 简单策略：根据位置决定动作
            let action = 1;
            if (this.env.position < 0) {
                action = 0; // 向左加速积累动量
            } else {
                action = 2; // 向右冲向山顶
            }
            
            const result = this.env.step(action);
            
            if (result.done) {
                this.episode++;
                this.updateStats();
                break;
            }
        }
        
        this.draw();
        
        if (this.isTraining) {
            setTimeout(() => this.train(), 100);
        }
    }
    
    reset() {
        this.isTraining = false;
        this.episode = 0;
        this.env.reset();
        
        const btn = document.getElementById('mountaincarStartBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        
        this.updateStats();
        this.draw();
    }
    
    updateStats() {
        const epEl = document.getElementById('mountaincarEpisode');
        const posEl = document.getElementById('mountaincarPos');
        const velEl = document.getElementById('mountaincarVel');
        const successEl = document.getElementById('mountaincarSuccess');
        
        if (epEl) epEl.textContent = this.episode;
        if (posEl) posEl.textContent = this.env.position.toFixed(2);
        if (velEl) velEl.textContent = this.env.velocity.toFixed(3);
        if (successEl) successEl.textContent = this.env.successCount;
    }
    
    draw() {
        if (!this.ctx) return;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        
        const padding = 50;
        const drawWidth = w - padding * 2;
        const drawHeight = h - padding * 2;
        
        // 绘制山地曲线
        this.ctx.strokeStyle = '#4CAF50';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        for (let x = 0; x <= drawWidth; x++) {
            const normX = (x / drawWidth) * (this.env.maxPosition - this.env.minPosition) + this.env.minPosition;
            const height = Math.cos(3 * normX);
            const y = padding + drawHeight/2 * (1 - height);
            
            if (x === 0) {
                this.ctx.moveTo(x + padding, y);
            } else {
                this.ctx.lineTo(x + padding, y);
            }
        }
        this.ctx.stroke();
        
        // 绘制小车
        const carX = padding + ((this.env.position - this.env.minPosition) / (this.env.maxPosition - this.env.minPosition)) * drawWidth;
        const carHeight = Math.cos(3 * this.env.position);
        const carY = padding + drawHeight/2 * (1 - carHeight);
        
        this.ctx.fillStyle = '#FF5722';
        this.ctx.beginPath();
        this.ctx.arc(carX, carY, 15, 0, Math.PI * 2);
        this.ctx.fill();
        
        // 绘制目标位置
        const goalX = padding + ((this.env.goalPosition - this.env.minPosition) / (this.env.maxPosition - this.env.minPosition)) * drawWidth;
        this.ctx.fillStyle = '#FFD700';
        this.ctx.fillRect(goalX - 5, carY - 30, 10, 30);
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🚩', goalX, carY - 35);
        
        // 标签
        this.ctx.fillStyle = '#666';
        this.ctx.fillText('起点', padding, carY + 30);
        this.ctx.fillText('目标', goalX, carY + 30);
    }
}

// ============================================
// 4. Cliff Walking 悬崖行走环境
// ============================================

class CliffWalkingEnv {
    constructor(width = 12, height = 4) {
        this.width = width;
        this.height = height;
        this.start = {x: 0, y: height - 1};
        this.goal = {x: width - 1, y: height - 1};
        this.cliff = [];
        for (let x = 1; x < width - 1; x++) {
            this.cliff.push({x, y: height - 1});
        }
        this.reset();
    }
    
    reset() {
        this.agentPos = {...this.start};
        this.steps = 0;
        this.falls = 0;
        this.success = false;
        return this.getState();
    }
    
    getState() {
        return `${this.agentPos.x},${this.agentPos.y}`;
    }
    
    move(action) {
        const oldPos = {...this.agentPos};
        
        switch(action) {
            case 'up': if (this.agentPos.y > 0) this.agentPos.y--; break;
            case 'down': if (this.agentPos.y < this.height-1) this.agentPos.y++; break;
            case 'left': if (this.agentPos.x > 0) this.agentPos.x--; break;
            case 'right': if (this.agentPos.x < this.width-1) this.agentPos.x++; break;
        }
        
        // 检查是否掉下悬崖
        const onCliff = this.cliff.some(c => c.x === this.agentPos.x && c.y === this.agentPos.y);
        if (onCliff) {
            this.agentPos = {...this.start};
            this.falls++;
            return -100; // 巨大惩罚
        }
        
        // 检查是否到达目标
        if (this.agentPos.x === this.goal.x && this.agentPos.y === this.goal.y) {
            this.success = true;
            return 10;
        }
        
        this.steps++;
        return -1; // 每步惩罚
    }
}

class CliffWalkingVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.env = new CliffWalkingEnv(12, 4);
        
        this.states = [];
        for (let x = 0; x < this.env.width; x++) {
            for (let y = 0; y < this.env.height; y++) {
                this.states.push(`${x},${y}`);
            }
        }
        this.actions = ['up', 'down', 'left', 'right'];
        this.agent = new QLearningAgent(this.states, this.actions);
        
        this.isTraining = false;
        this.episode = 0;
        this.totalFalls = 0;
        this.totalSuccess = 0;
        this.episodeRewards = [];
        
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('cliffStartBtn');
        const resetBtn = document.getElementById('cliffResetBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.toggleTraining());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('cliffStartBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        if (this.isTraining) this.train();
    }
    
    train() {
        if (!this.isTraining) return;
        
        for (let i = 0; i < 5; i++) {
            this.env.reset();
            let episodeReward = 0;
            
            while (!this.env.success && this.env.steps < 100) {
                const state = this.env.getState();
                const action = this.agent.chooseAction(state);
                const reward = this.env.move(action);
                const nextState = this.env.getState();
                this.agent.learn(state, action, reward, nextState);
                episodeReward += reward;
            }
            
            this.episode++;
            this.totalFalls += this.env.falls;
            if (this.env.success) this.totalSuccess++;
            this.episodeRewards.push(episodeReward);
        }
        
        this.updateStats();
        this.draw();
        
        if (this.isTraining) {
            requestAnimationFrame(() => this.train());
        }
    }
    
    reset() {
        this.isTraining = false;
        this.episode = 0;
        this.totalFalls = 0;
        this.totalSuccess = 0;
        this.episodeRewards = [];
        this.agent.initQTable();
        this.env.reset();
        
        const btn = document.getElementById('cliffStartBtn');
        if (btn) {
            btn.textContent = '开始训练';
            btn.style.background = '#ff6b6b';
        }
        
        this.updateStats();
        this.draw();
    }
    
    updateStats() {
        const epEl = document.getElementById('cliffEpisode');
        const fallsEl = document.getElementById('cliffFalls');
        const rewardEl = document.getElementById('cliffReward');
        const successEl = document.getElementById('cliffSuccess');
        
        if (epEl) epEl.textContent = this.episode;
        if (fallsEl) fallsEl.textContent = this.totalFalls;
        if (rewardEl) {
            const avg = this.episodeRewards.length > 0 ? 
                this.episodeRewards.slice(-10).reduce((a,b) => a+b, 0) / 10 : 0;
            rewardEl.textContent = avg.toFixed(1);
        }
        if (successEl) {
            const rate = this.episode > 0 ? (this.totalSuccess / this.episode * 100) : 0;
            successEl.textContent = rate.toFixed(0) + '%';
        }
    }
    
    draw() {
        if (!this.ctx) return;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        
        const cellW = w / this.env.width;
        const cellH = h / this.env.height;
        
        for (let y = 0; y < this.env.height; y++) {
            for (let x = 0; x < this.env.width; x++) {
                const cellX = x * cellW;
                const cellY = y * cellH;
                
                // 绘制单元格
                this.ctx.fillStyle = '#fff';
                this.ctx.fillRect(cellX + 1, cellY + 1, cellW - 2, cellH - 2);
                
                // 悬崖
                const onCliff = this.env.cliff.some(c => c.x === x && c.y === y);
                if (onCliff) {
                    this.ctx.fillStyle = '#f44336';
                    this.ctx.fillRect(cellX + 1, cellY + 1, cellW - 2, cellH - 2);
                    this.ctx.fillStyle = '#fff';
                    this.ctx.font = '20px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('⚠️', cellX + cellW/2, cellY + cellH/2);
                }
                
                // 起点
                if (x === this.env.start.x && y === this.env.start.y) {
                    this.ctx.fillStyle = '#4CAF50';
                    this.ctx.fillRect(cellX + 1, cellY + 1, cellW - 2, cellH - 2);
                    this.ctx.fillStyle = '#fff';
                    this.ctx.font = '20px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🚩', cellX + cellW/2, cellY + cellH/2);
                }
                
                // 目标
                if (x === this.env.goal.x && y === this.env.goal.y) {
                    this.ctx.fillStyle = '#FFD700';
                    this.ctx.fillRect(cellX + 1, cellY + 1, cellW - 2, cellH - 2);
                    this.ctx.fillStyle = '#333';
                    this.ctx.font = '20px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🏁', cellX + cellW/2, cellY + cellH/2);
                }
                
                // 智能体
                if (x === this.env.agentPos.x && y === this.env.agentPos.y) {
                    this.ctx.fillStyle = '#2196F3';
                    this.ctx.beginPath();
                    this.ctx.arc(cellX + cellW/2, cellY + cellH/2, cellW/4, 0, Math.PI * 2);
                    this.ctx.fill();
                }
                
                // 策略箭头
                if (this.episode > 0 && !onCliff) {
                    const state = `${x},${y}`;
                    const qValues = this.agent.qTable[state];
                    if (qValues) {
                        let bestAction = 'right';
                        let bestValue = qValues[bestAction];
                        this.actions.forEach(action => {
                            if (qValues[action] > bestValue) {
                                bestValue = qValues[action];
                                bestAction = action;
                            }
                        });
                        const arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'};
                        this.ctx.fillStyle = 'rgba(33, 150, 243, 0.5)';
                        this.ctx.font = 'bold 24px Arial';
                        this.ctx.textAlign = 'center';
                        this.ctx.textBaseline = 'middle';
                        this.ctx.fillText(arrows[bestAction], cellX + cellW/2, cellY + cellH/2);
                    }
                }
                
                // 网格线
                this.ctx.strokeStyle = '#ddd';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(cellX + 1, cellY + 1, cellW - 2, cellH - 2);
            }
        }
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
            
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            panels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === `${demoName}-demo`) {
                    panel.classList.add('active');
                }
            });
            
            setTimeout(() => {
                if (demoName === 'gridworld' && !window.gridVisualizer) {
                    window.gridVisualizer = new GridWorldVisualizer('gridCanvas', 'rewardChart');
                } else if (demoName === 'bandit' && !window.banditVisualizer) {
                    window.banditVisualizer = new BanditVisualizer();
                } else if (demoName === 'dqn' && !window.dqnDemo) {
                    window.dqnDemo = new DQNDemo();
                } else if (demoName === 'policy' && !window.policyDemo) {
                    window.policyDemo = new PolicyGradientDemo();
                } else if (demoName === 'cartpole' && !window.cartpoleViz) {
                    window.cartpoleViz = new CartPoleVisualizer('cartpoleCanvas');
                } else if (demoName === 'mountaincar' && !window.mountaincarViz) {
                    window.mountaincarViz = new MountainCarVisualizer('mountaincarCanvas');
                } else if (demoName === 'cliffwalking' && !window.cliffViz) {
                    window.cliffViz = new CliffWalkingVisualizer('cliffCanvas');
                }
            }, 100);
        });
    });
}

// ============================================
// 页面加载初始化
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing RL Demos Phase 3...');
    
    initDemoTabs();
    
    const gridCanvas = document.getElementById('gridCanvas');
    if (gridCanvas) {
        window.gridVisualizer = new GridWorldVisualizer('gridCanvas', 'rewardChart');
    }
    
    console.log('Phase 3 RL Demos initialized successfully!');
});
