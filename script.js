// 强化学习交互演示 - Q-Learning 网格世界

// Q-Learning 智能体类
class QLearningAgent {
    constructor(states, actions, alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.states = states;
        this.actions = actions;
        this.alpha = alpha; // 学习率
        this.gamma = gamma; // 折扣因子
        this.epsilon = epsilon; // 探索率
        
        // 初始化 Q 表
        this.qTable = {};
        states.forEach(state => {
            this.qTable[state] = {};
            actions.forEach(action => {
                this.qTable[state][action] = 0;
            });
        });
    }
    
    // 选择动作（epsilon-greedy 策略）
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
            // 探索：随机动作
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        } else {
            // 利用：最优动作
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
    
    // 学习更新
    learn(state, action, reward, nextState) {
        const currentQ = this.qTable[state][action];
        const maxNextQ = Math.max(...this.actions.map(a => this.qTable[nextState][a]));
        const target = reward + this.gamma * maxNextQ;
        this.qTable[state][action] = currentQ + this.alpha * (target - currentQ);
    }
}

// 网格世界环境
class GridWorld {
    constructor(width = 5, height = 5) {
        this.width = width;
        this.height = height;
        this.agentPos = {x: 0, y: 0};
        this.goalPos = {x: width-1, y: height-1};
        this.obstacles = [
            {x: 1, y: 1},
            {x: 2, y: 1},
            {x: 3, y: 2},
            {x: 1, y: 3}
        ];
    }
    
    getState() {
        return `${this.agentPos.x},${this.agentPos.y}`;
    }
    
    getReward() {
        if (this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y) {
            return 10; // 到达目标
        }
        return -0.1; // 每步惩罚，鼓励最短路径
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
        
        // 检查障碍物
        const hitObstacle = this.obstacles.some(obs => 
            obs.x === this.agentPos.x && obs.y === this.agentPos.y
        );
        if (hitObstacle) {
            this.agentPos = oldPos; // 撞到障碍物，留在原地
            return -1; // 额外惩罚
        }
        
        return 0;
    }
    
    reset() {
        this.agentPos = {x: 0, y: 0};
    }
}

// 可视化控制器
class GridWorldVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error('Canvas not found:', canvasId);
            return;
        }
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
        
        if (startBtn) {
            startBtn.addEventListener('click', () => this.toggleTraining());
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.reset());
        }
    }
    
    toggleTraining() {
        this.isTraining = !this.isTraining;
        const btn = document.getElementById('startBtn');
        if (btn) {
            btn.textContent = this.isTraining ? '暂停训练' : '开始训练';
            btn.style.background = this.isTraining ? '#ff9800' : '#ff6b6b';
        }
        
        if (this.isTraining) {
            this.train();
        }
    }
    
    train() {
        if (!this.isTraining) return;
        
        // 每帧训练多个 episode 以加速
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
        
        // 继续训练
        if (this.isTraining) {
            requestAnimationFrame(() => this.train());
        }
    }
    
    reset() {
        this.isTraining = false;
        this.episodeCount = 0;
        this.stepCount = 0;
        this.currentReward = 0;
        
        // 重置 Q 表
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
        
        // 清空画布
        this.ctx.fillStyle = '#f5f5f5';
        this.ctx.fillRect(0, 0, w, h);
        
        // 绘制网格
        for (let y = 0; y < this.grid.height; y++) {
            for (let x = 0; x < this.grid.width; x++) {
                const cellX = x * this.cellSize;
                const cellY = y * this.cellSize;
                
                // 绘制单元格背景
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fillRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
                
                // 绘制障碍物
                const isObstacle = this.grid.obstacles.some(obs => 
                    obs.x === x && obs.y === y
                );
                if (isObstacle) {
                    this.ctx.fillStyle = '#9e9e9e';
                    this.ctx.fillRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
                    this.ctx.fillStyle = '#fff';
                    this.ctx.font = '30px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🚫', cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                
                // 绘制目标
                if (x === this.grid.goalPos.x && y === this.grid.goalPos.y) {
                    this.ctx.fillStyle = '#4CAF50';
                    this.ctx.fillRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
                    this.ctx.font = '30px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText('🎯', cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                
                // 绘制智能体
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
                
                // 绘制策略箭头（如果已经训练过）
                if (this.episodeCount > 0 && !isObstacle && 
                    !(x === this.grid.goalPos.x && y === this.grid.goalPos.y)) {
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
                    
                    const arrows = {
                        'up': '↑',
                        'down': '↓',
                        'left': '←',
                        'right': '→'
                    };
                    
                    this.ctx.fillStyle = 'rgba(33, 150, 243, 0.6)';
                    this.ctx.font = 'bold 36px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText(arrows[bestAction], cellX + this.cellSize/2, cellY + this.cellSize/2);
                }
                
                // 绘制网格线
                this.ctx.strokeStyle = '#ddd';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(cellX + 2, cellY + 2, this.cellSize - 4, this.cellSize - 4);
            }
        }
        
        // 绘制图例
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillStyle = '#333';
        this.ctx.fillText('🤖 智能体', 10, h - 45);
        this.ctx.fillText('🎯 目标', 10, h - 25);
        this.ctx.fillText('🚫 障碍物', w - 100, h - 45);
        this.ctx.fillText('↑↓←→ 策略', w - 100, h - 25);
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Grid World Demo...');
    
    // 检查 canvas 是否存在
    const canvas = document.getElementById('gridCanvas');
    if (canvas) {
        window.gridVisualizer = new GridWorldVisualizer('gridCanvas');
        console.log('Grid World Demo initialized successfully!');
    } else {
        console.error('Grid canvas not found in DOM');
    }
});
