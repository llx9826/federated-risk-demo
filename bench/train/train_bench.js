#!/usr/bin/env node
/**
 * 联邦训练性能基准测试
 * 支持参与方 N≥2，网格搜索，记录通信量、AUC/KS曲线
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const yargs = require('yargs');

// 配置参数
const DEFAULT_CONFIG = {
    n: 5000,
    participants: 2,
    epsilon: [Infinity, 8, 5, 3],
    maxRounds: 20,
    serverUrl: 'http://localhost:8002',
    timeout: 120000,
    retries: 3,
    outputDir: 'reports/bench/train',
    gridSearch: {
        learningRate: [0.1, 0.3],
        maxDepth: [3, 6],
        subsample: [0.8, 1.0]
    }
};

class TrainingBenchmark {
    constructor(config) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.results = {
            config: this.config,
            timestamp: new Date().toISOString(),
            experiments: [],
            performance: {
                convergence: [],
                communication: [],
                privacy: [],
                scalability: []
            },
            errors: []
        };
    }

    // 生成训练配置组合
    generateGridConfigs() {
        const configs = [];
        const { gridSearch } = this.config;
        
        // 生成所有参数组合
        const learningRates = gridSearch.learningRate || [0.1];
        const maxDepths = gridSearch.maxDepth || [6];
        const subsamples = gridSearch.subsample || [0.8];
        
        for (const lr of learningRates) {
            for (const depth of maxDepths) {
                for (const subsample of subsamples) {
                    for (const epsilon of this.config.epsilon) {
                        configs.push({
                            learning_rate: lr,
                            max_depth: depth,
                            subsample: subsample,
                            epsilon: epsilon,
                            max_rounds: this.config.maxRounds,
                            participants: this.config.participants
                        });
                    }
                }
            }
        }
        
        // 限制配置数量（≤30组）
        if (configs.length > 30) {
            console.log(`生成了${configs.length}个配置，随机选择30个进行测试`);
            return this.shuffleArray(configs).slice(0, 30);
        }
        
        return configs;
    }

    shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    // 启动联邦训练
    async startTraining(trainConfig) {
        const taskName = `bench_${Date.now()}_eps${trainConfig.epsilon}`;
        
        try {
            const response = await axios.post(
                `${this.config.serverUrl}/train/federated`,
                {
                    task_name: taskName,
                    participants: this.generateParticipantList(trainConfig.participants),
                    algorithm: 'secure_boost',
                    privacy_config: {
                        epsilon: trainConfig.epsilon,
                        delta: 1e-5
                    },
                    model_config: {
                        learning_rate: trainConfig.learning_rate,
                        max_depth: trainConfig.max_depth,
                        subsample: trainConfig.subsample,
                        objective: 'binary:logistic',
                        eval_metric: 'auc'
                    },
                    max_rounds: trainConfig.max_rounds,
                    early_stopping_rounds: 5,
                    verbose: false
                },
                {
                    timeout: this.config.timeout,
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            return {
                success: true,
                taskId: response.data.task_id,
                taskName: taskName,
                startTime: Date.now()
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                taskName: taskName
            };
        }
    }

    generateParticipantList(count) {
        const participants = [];
        for (let i = 0; i < count; i++) {
            participants.push(`party_${String.fromCharCode(97 + i)}`);
        }
        return participants;
    }

    // 监控训练进度
    async monitorTraining(taskId, startTime) {
        const monitoringData = {
            rounds: [],
            communication: [],
            performance: [],
            convergence: false
        };
        
        let completed = false;
        let attempts = 0;
        const maxAttempts = Math.ceil(this.config.timeout / 5000); // 每5秒检查一次
        
        while (!completed && attempts < maxAttempts) {
            try {
                await new Promise(resolve => setTimeout(resolve, 5000)); // 等待5秒
                
                const response = await axios.get(
                    `${this.config.serverUrl}/train/status/${taskId}`,
                    { timeout: 10000 }
                );
                
                const status = response.data;
                
                if (status.status === 'completed') {
                    completed = true;
                    
                    // 获取详细结果
                    const resultResponse = await axios.get(
                        `${this.config.serverUrl}/train/result/${taskId}`,
                        { timeout: 10000 }
                    );
                    
                    monitoringData.finalResult = resultResponse.data;
                    monitoringData.totalTime = Date.now() - startTime;
                    
                } else if (status.status === 'failed') {
                    throw new Error(`训练失败: ${status.error || 'Unknown error'}`);
                } else if (status.status === 'running') {
                    // 记录中间状态
                    if (status.current_round !== undefined) {
                        monitoringData.rounds.push({
                            round: status.current_round,
                            timestamp: Date.now(),
                            metrics: status.metrics || {}
                        });
                    }
                }
                
                attempts++;
            } catch (error) {
                if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
                    console.log(`连接失败，重试中... (${attempts}/${maxAttempts})`);
                    attempts++;
                    continue;
                }
                throw error;
            }
        }
        
        if (!completed) {
            throw new Error('训练超时');
        }
        
        return monitoringData;
    }

    // 分析训练结果
    analyzeTrainingResult(monitoringData, trainConfig) {
        const result = monitoringData.finalResult;
        
        if (!result || !result.performance_metrics) {
            return {
                error: 'No performance metrics available',
                config: trainConfig
            };
        }
        
        const analysis = {
            config: trainConfig,
            performance: {
                auc: result.performance_metrics.auc || 0,
                ks: result.performance_metrics.ks || 0,
                precision: result.performance_metrics.precision || 0,
                recall: result.performance_metrics.recall || 0,
                f1_score: result.performance_metrics.f1_score || 0
            },
            efficiency: {
                totalTime: monitoringData.totalTime,
                rounds: result.training_summary?.rounds_completed || 0,
                communicationCost: result.training_summary?.communication_cost || 0,
                convergenceRound: this.findConvergenceRound(monitoringData.rounds)
            },
            privacy: {
                epsilon: trainConfig.epsilon,
                privacyBudgetUsed: result.privacy_metrics?.privacy_budget_used || 0,
                differentialPrivacy: trainConfig.epsilon !== Infinity
            },
            scalability: {
                participants: trainConfig.participants,
                timePerRound: monitoringData.totalTime / (result.training_summary?.rounds_completed || 1),
                communicationPerParticipant: (result.training_summary?.communication_cost || 0) / trainConfig.participants
            }
        };
        
        // 计算收敛曲线
        if (monitoringData.rounds && monitoringData.rounds.length > 0) {
            analysis.convergence = this.analyzeConvergence(monitoringData.rounds);
        }
        
        return analysis;
    }

    findConvergenceRound(rounds) {
        if (!rounds || rounds.length === 0) return null;
        
        // 寻找AUC不再显著提升的轮次
        let bestAuc = 0;
        let convergenceRound = null;
        let stableRounds = 0;
        
        for (const round of rounds) {
            const auc = round.metrics.auc || 0;
            
            if (auc > bestAuc + 0.001) { // 显著提升阈值
                bestAuc = auc;
                stableRounds = 0;
            } else {
                stableRounds++;
                if (stableRounds >= 3 && !convergenceRound) {
                    convergenceRound = round.round;
                }
            }
        }
        
        return convergenceRound;
    }

    analyzeConvergence(rounds) {
        const aucCurve = rounds.map(r => r.metrics.auc || 0);
        const ksCurve = rounds.map(r => r.metrics.ks || 0);
        
        return {
            aucCurve,
            ksCurve,
            finalAuc: aucCurve[aucCurve.length - 1] || 0,
            finalKs: ksCurve[ksCurve.length - 1] || 0,
            maxAuc: Math.max(...aucCurve),
            maxKs: Math.max(...ksCurve),
            convergenceStability: this.calculateStability(aucCurve.slice(-5)) // 最后5轮的稳定性
        };
    }

    calculateStability(values) {
        if (values.length < 2) return 0;
        
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        return 1 - (stdDev / mean); // 稳定性分数，越接近1越稳定
    }

    // 多参与方扩展性测试
    async runScalabilityTest() {
        console.log('开始多参与方扩展性测试...');
        
        const scalabilityResults = [];
        const participantCounts = [2, 3, 5]; // 测试不同参与方数量
        
        for (const participants of participantCounts) {
            console.log(`测试 ${participants} 参与方...`);
            
            const testConfig = {
                learning_rate: 0.1,
                max_depth: 6,
                subsample: 0.8,
                epsilon: 5, // 使用中等隐私保护
                max_rounds: 10, // 减少轮次以节省时间
                participants: participants
            };
            
            try {
                const startResult = await this.startTraining(testConfig);
                if (!startResult.success) {
                    throw new Error(startResult.error);
                }
                
                const monitoringData = await this.monitorTraining(startResult.taskId, startResult.startTime);
                const analysis = this.analyzeTrainingResult(monitoringData, testConfig);
                
                scalabilityResults.push(analysis);
                
            } catch (error) {
                console.error(`${participants} 参与方测试失败:`, error.message);
                scalabilityResults.push({
                    config: testConfig,
                    error: error.message
                });
            }
        }
        
        return scalabilityResults;
    }

    // 隐私-效用权衡分析
    async runPrivacyUtilityAnalysis() {
        console.log('开始隐私-效用权衡分析...');
        
        const privacyResults = [];
        const epsilonValues = [Infinity, 8, 5, 3]; // 不同隐私预算
        
        const baseConfig = {
            learning_rate: 0.1,
            max_depth: 6,
            subsample: 0.8,
            max_rounds: 15,
            participants: 2
        };
        
        for (const epsilon of epsilonValues) {
            console.log(`测试 ε=${epsilon} 隐私预算...`);
            
            const testConfig = { ...baseConfig, epsilon };
            
            try {
                const startResult = await this.startTraining(testConfig);
                if (!startResult.success) {
                    throw new Error(startResult.error);
                }
                
                const monitoringData = await this.monitorTraining(startResult.taskId, startResult.startTime);
                const analysis = this.analyzeTrainingResult(monitoringData, testConfig);
                
                privacyResults.push(analysis);
                
            } catch (error) {
                console.error(`ε=${epsilon} 测试失败:`, error.message);
                privacyResults.push({
                    config: testConfig,
                    error: error.message
                });
            }
        }
        
        return privacyResults;
    }

    // 运行完整基准测试
    async runBenchmark() {
        console.log('开始联邦训练基准测试...');
        console.log(`配置: n=${this.config.n}, participants=${this.config.participants}`);
        
        try {
            // 生成网格搜索配置
            const gridConfigs = this.generateGridConfigs();
            console.log(`生成 ${gridConfigs.length} 个训练配置`);
            
            // 执行网格搜索
            console.log('执行网格搜索...');
            const gridResults = [];
            
            for (let i = 0; i < gridConfigs.length; i++) {
                const config = gridConfigs[i];
                console.log(`[${i+1}/${gridConfigs.length}] 测试配置: lr=${config.learning_rate}, depth=${config.max_depth}, ε=${config.epsilon}`);
                
                try {
                    const startResult = await this.startTraining(config);
                    if (!startResult.success) {
                        throw new Error(startResult.error);
                    }
                    
                    const monitoringData = await this.monitorTraining(startResult.taskId, startResult.startTime);
                    const analysis = this.analyzeTrainingResult(monitoringData, config);
                    
                    gridResults.push(analysis);
                    
                } catch (error) {
                    console.error(`配置 ${i+1} 失败:`, error.message);
                    gridResults.push({
                        config: config,
                        error: error.message
                    });
                }
            }
            
            this.results.experiments = gridResults;
            
            // 可扩展性测试
            const scalabilityResults = await this.runScalabilityTest();
            this.results.performance.scalability = scalabilityResults;
            
            // 隐私-效用分析
            const privacyResults = await this.runPrivacyUtilityAnalysis();
            this.results.performance.privacy = privacyResults;
            
            // 生成报告
            this.generateReport();
            
            console.log('联邦训练基准测试完成');
            return this.results;
            
        } catch (error) {
            console.error('基准测试失败:', error);
            this.results.errors.push({
                type: 'benchmark_failure',
                message: error.message,
                timestamp: new Date().toISOString()
            });
            throw error;
        }
    }

    // 生成报告
    generateReport() {
        const successfulExperiments = this.results.experiments.filter(exp => !exp.error);
        
        if (successfulExperiments.length === 0) {
            console.log('没有成功的实验，无法生成报告');
            return;
        }
        
        // 找到最佳配置
        const bestExperiment = successfulExperiments.reduce((best, current) => {
            return (current.performance.auc > best.performance.auc) ? current : best;
        });
        
        const report = {
            summary: {
                totalExperiments: this.results.experiments.length,
                successfulExperiments: successfulExperiments.length,
                bestAuc: bestExperiment.performance.auc,
                bestKs: bestExperiment.performance.ks,
                bestConfig: bestExperiment.config,
                averageTrainingTime: this.calculateAverageTime(successfulExperiments),
                privacyUtilityTradeoff: this.analyzePrivacyUtilityTradeoff()
            },
            detailed: this.results
        };
        
        // 保存报告
        this.saveReport(report);
        
        // 打印摘要
        console.log('\n=== 联邦训练基准测试报告 ===');
        console.log(`成功实验: ${report.summary.successfulExperiments}/${report.summary.totalExperiments}`);
        console.log(`最佳AUC: ${report.summary.bestAuc.toFixed(4)}`);
        console.log(`最佳KS: ${report.summary.bestKs.toFixed(4)}`);
        console.log(`平均训练时间: ${report.summary.averageTrainingTime.toFixed(2)}ms`);
        console.log(`最佳配置: lr=${bestExperiment.config.learning_rate}, depth=${bestExperiment.config.max_depth}, ε=${bestExperiment.config.epsilon}`);
    }

    calculateAverageTime(experiments) {
        const times = experiments.map(exp => exp.efficiency.totalTime).filter(t => t > 0);
        return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
    }

    analyzePrivacyUtilityTradeoff() {
        const privacyResults = this.results.performance.privacy;
        if (!privacyResults || privacyResults.length === 0) {
            return null;
        }
        
        const tradeoff = privacyResults.map(result => ({
            epsilon: result.config.epsilon,
            auc: result.performance.auc,
            utilityLoss: result.performance.auc < 1 ? (1 - result.performance.auc) : 0
        }));
        
        return tradeoff;
    }

    saveReport(report) {
        const outputDir = this.config.outputDir || 'reports/bench/train';
        
        // 确保输出目录存在
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `train_bench_${timestamp}.json`;
        const filepath = path.join(outputDir, filename);
        
        fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
        console.log(`报告保存至: ${filepath}`);
        
        // 同时保存为最新报告
        const latestPath = path.join(outputDir, 'latest.json');
        fs.writeFileSync(latestPath, JSON.stringify(report, null, 2));
    }
}

// 命令行接口
if (require.main === module) {
    const argv = yargs
        .option('n', {
            alias: 'samples',
            type: 'number',
            default: 5000,
            description: '训练样本数量'
        })
        .option('participants', {
            type: 'number',
            default: 2,
            description: '参与方数量'
        })
        .option('epsilon', {
            type: 'array',
            default: [Infinity, 8, 5, 3],
            description: '隐私预算列表'
        })
        .option('max-rounds', {
            type: 'number',
            default: 20,
            description: '最大训练轮次'
        })
        .option('output', {
            type: 'string',
            default: 'reports/bench/train',
            description: '输出目录'
        })
        .option('server', {
            type: 'string',
            default: 'http://localhost:8002',
            description: '训练服务地址'
        })
        .help()
        .argv;

    const config = {
        n: argv.n,
        participants: argv.participants,
        epsilon: argv.epsilon,
        maxRounds: argv.maxRounds,
        outputDir: argv.output,
        serverUrl: argv.server
    };

    const benchmark = new TrainingBenchmark(config);
    
    benchmark.runBenchmark()
        .then(() => {
            console.log('基准测试成功完成');
            process.exit(0);
        })
        .catch((error) => {
            console.error('基准测试失败:', error);
            process.exit(1);
        });
}

module.exports = TrainingBenchmark;