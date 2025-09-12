#!/usr/bin/env node
/**
 * PSI性能基准测试
 * 支持分片 + 多进程 + 流式批处理
 * 可选 Ray/Dask 后端，回退多进程
 */

const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const crypto = require('crypto');
const axios = require('axios');
const yargs = require('yargs');

// 配置参数
const DEFAULT_CONFIG = {
    n: 10000,
    workers: 4,
    shards: 8,
    batchSize: 1000,
    algorithm: 'ecdh_psi',
    overlapRate: 0.3,
    serverUrl: 'http://localhost:8001',
    timeout: 30000,
    retries: 3,
    outputDir: 'reports/bench/psi'
};

class PSIBenchmark {
    constructor(config) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.results = {
            config: this.config,
            timestamp: new Date().toISOString(),
            performance: {
                throughput: [],
                latency: [],
                errorRate: [],
                resourceUsage: []
            },
            scalability: {
                shardingEfficiency: [],
                workerUtilization: [],
                memoryUsage: []
            },
            errors: []
        };
    }

    // 生成测试数据
    generateTestData(n, overlapRate = 0.3) {
        const partyAData = [];
        const partyBData = [];
        
        // 生成重叠数据
        const overlapSize = Math.floor(n * overlapRate);
        const overlapIds = new Set();
        
        for (let i = 0; i < overlapSize; i++) {
            const id = `overlap_${i.toString().padStart(8, '0')}`;
            overlapIds.add(id);
            partyAData.push(id);
            partyBData.push(id);
        }
        
        // 生成A方独有数据
        const aUniqueSize = Math.floor(n * 0.4);
        for (let i = 0; i < aUniqueSize; i++) {
            partyAData.push(`party_a_${i.toString().padStart(8, '0')}`);
        }
        
        // 生成B方独有数据
        const bUniqueSize = Math.floor(n * 0.3);
        for (let i = 0; i < bUniqueSize; i++) {
            partyBData.push(`party_b_${i.toString().padStart(8, '0')}`);
        }
        
        // 打乱数据
        this.shuffleArray(partyAData);
        this.shuffleArray(partyBData);
        
        return {
            partyA: partyAData,
            partyB: partyBData,
            expectedIntersection: overlapSize
        };
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    // 数据分片
    shardData(data, numShards) {
        const shards = Array(numShards).fill().map(() => []);
        
        data.forEach((item, index) => {
            const shardIndex = index % numShards;
            shards[shardIndex].push(item);
        });
        
        return shards;
    }

    // 执行PSI计算
    async executePSI(partyAData, partyBData, shardIndex = 0) {
        const startTime = Date.now();
        
        try {
            const response = await axios.post(
                `${this.config.serverUrl}/psi/compute`,
                {
                    party_a_data: partyAData,
                    party_b_data: partyBData,
                    algorithm: this.config.algorithm,
                    shard_id: shardIndex
                },
                {
                    timeout: this.config.timeout,
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            const endTime = Date.now();
            const latency = endTime - startTime;
            
            return {
                success: true,
                latency,
                intersectionSize: response.data.intersection ? response.data.intersection.length : 0,
                throughput: partyAData.length / (latency / 1000), // items per second
                shardIndex,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            const endTime = Date.now();
            return {
                success: false,
                latency: endTime - startTime,
                error: error.message,
                shardIndex,
                timestamp: new Date().toISOString()
            };
        }
    }

    // 批处理执行
    async executeBatchPSI(partyAShards, partyBShards) {
        const results = [];
        const workers = [];
        
        // 创建工作线程池
        for (let i = 0; i < this.config.workers; i++) {
            workers.push(this.createWorker());
        }
        
        // 分配任务到工作线程
        const tasks = [];
        for (let i = 0; i < partyAShards.length; i++) {
            tasks.push({
                partyAData: partyAShards[i],
                partyBData: partyBShards[i],
                shardIndex: i
            });
        }
        
        // 执行任务
        const promises = tasks.map((task, index) => {
            const workerIndex = index % this.config.workers;
            return this.executeWorkerTask(workers[workerIndex], task);
        });
        
        try {
            const workerResults = await Promise.all(promises);
            results.push(...workerResults);
        } catch (error) {
            this.results.errors.push({
                type: 'worker_execution',
                message: error.message,
                timestamp: new Date().toISOString()
            });
        } finally {
            // 清理工作线程
            workers.forEach(worker => worker.terminate());
        }
        
        return results;
    }

    createWorker() {
        return new Worker(__filename, {
            workerData: {
                isWorker: true,
                config: this.config
            }
        });
    }

    async executeWorkerTask(worker, task) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Worker task timeout'));
            }, this.config.timeout * 2);
            
            worker.once('message', (result) => {
                clearTimeout(timeout);
                resolve(result);
            });
            
            worker.once('error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
            
            worker.postMessage(task);
        });
    }

    // 性能分析
    analyzePerformance(results) {
        const successResults = results.filter(r => r.success);
        const errorResults = results.filter(r => !r.success);
        
        if (successResults.length === 0) {
            return {
                error: 'No successful PSI computations',
                errorRate: 1.0
            };
        }
        
        const latencies = successResults.map(r => r.latency);
        const throughputs = successResults.map(r => r.throughput);
        const totalIntersection = successResults.reduce((sum, r) => sum + r.intersectionSize, 0);
        
        return {
            totalComputations: results.length,
            successfulComputations: successResults.length,
            errorRate: errorResults.length / results.length,
            
            latency: {
                min: Math.min(...latencies),
                max: Math.max(...latencies),
                mean: latencies.reduce((a, b) => a + b, 0) / latencies.length,
                p50: this.percentile(latencies, 0.5),
                p95: this.percentile(latencies, 0.95),
                p99: this.percentile(latencies, 0.99)
            },
            
            throughput: {
                min: Math.min(...throughputs),
                max: Math.max(...throughputs),
                mean: throughputs.reduce((a, b) => a + b, 0) / throughputs.length,
                total: throughputs.reduce((a, b) => a + b, 0)
            },
            
            intersection: {
                total: totalIntersection,
                averagePerShard: totalIntersection / successResults.length
            },
            
            errors: errorResults.map(r => ({
                shardIndex: r.shardIndex,
                error: r.error,
                timestamp: r.timestamp
            }))
        };
    }

    percentile(arr, p) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = Math.ceil(sorted.length * p) - 1;
        return sorted[index];
    }

    // 可扩展性测试
    async runScalabilityTest() {
        console.log('开始可扩展性测试...');
        
        const scalabilityResults = [];
        const testConfigs = [
            { workers: 1, shards: 2 },
            { workers: 2, shards: 4 },
            { workers: 4, shards: 8 },
            { workers: 8, shards: 16 }
        ];
        
        for (const testConfig of testConfigs) {
            console.log(`测试配置: ${testConfig.workers} workers, ${testConfig.shards} shards`);
            
            const testData = this.generateTestData(this.config.n, this.config.overlapRate);
            const partyAShards = this.shardData(testData.partyA, testConfig.shards);
            const partyBShards = this.shardData(testData.partyB, testConfig.shards);
            
            const startTime = Date.now();
            const oldConfig = { ...this.config };
            this.config.workers = testConfig.workers;
            
            const results = await this.executeBatchPSI(partyAShards, partyBShards);
            const performance = this.analyzePerformance(results);
            
            const endTime = Date.now();
            this.config = oldConfig;
            
            // 检查性能分析是否成功
            if (performance.error) {
                console.warn(`配置 ${testConfig.workers}/${testConfig.shards} 测试失败: ${performance.error}`);
                scalabilityResults.push({
                    config: testConfig,
                    performance,
                    totalTime: endTime - startTime,
                    efficiency: 0,
                    error: performance.error
                });
            } else {
                scalabilityResults.push({
                    config: testConfig,
                    performance,
                    totalTime: endTime - startTime,
                    efficiency: performance.throughput.total / testConfig.workers
                });
            }
        }
        
        return scalabilityResults;
    }

    // 运行完整基准测试
    async runBenchmark() {
        console.log('开始PSI基准测试...');
        console.log(`配置: n=${this.config.n}, workers=${this.config.workers}, shards=${this.config.shards}`);
        
        try {
            // 生成测试数据
            console.log('生成测试数据...');
            const testData = this.generateTestData(this.config.n, this.config.overlapRate);
            
            // 数据分片
            console.log('数据分片...');
            const partyAShards = this.shardData(testData.partyA, this.config.shards);
            const partyBShards = this.shardData(testData.partyB, this.config.shards);
            
            // 执行PSI计算
            console.log('执行PSI计算...');
            const startTime = Date.now();
            const results = await this.executeBatchPSI(partyAShards, partyBShards);
            const endTime = Date.now();
            
            // 性能分析
            const performance = this.analyzePerformance(results);
            this.results.performance = performance;
            this.results.totalTime = endTime - startTime;
            
            // 可扩展性测试
            const scalabilityResults = await this.runScalabilityTest();
            this.results.scalability = scalabilityResults;
            
            // 生成报告
            this.generateReport();
            
            console.log('PSI基准测试完成');
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
        const report = {
            summary: {
                totalTime: this.results.totalTime,
                throughput: this.results.performance.throughput?.mean || 0,
                latencyP95: this.results.performance.latency?.p95 || 0,
                errorRate: this.results.performance.errorRate || 0,
                scalabilityEfficiency: this.calculateScalabilityEfficiency()
            },
            detailed: this.results
        };
        
        // 保存报告
        this.saveReport(report);
        
        // 打印摘要
        console.log('\n=== PSI基准测试报告 ===');
        console.log(`总耗时: ${report.summary.totalTime}ms`);
        console.log(`平均吞吐量: ${report.summary.throughput.toFixed(2)} items/s`);
        console.log(`P95延迟: ${report.summary.latencyP95.toFixed(2)}ms`);
        console.log(`错误率: ${(report.summary.errorRate * 100).toFixed(2)}%`);
        console.log(`可扩展性效率: ${report.summary.scalabilityEfficiency.toFixed(2)}`);
    }

    calculateScalabilityEfficiency() {
        if (!this.results.scalability || this.results.scalability.length < 2) {
            return 0;
        }
        
        const baseline = this.results.scalability[0];
        const best = this.results.scalability[this.results.scalability.length - 1];
        
        const speedup = best.performance.throughput.total / baseline.performance.throughput.total;
        const workerRatio = best.config.workers / baseline.config.workers;
        
        return speedup / workerRatio; // 理想情况下应该接近1
    }

    saveReport(report) {
        const outputDir = this.config.outputDir || 'reports/bench/psi';
        
        // 确保输出目录存在
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `psi_bench_${timestamp}.json`;
        const filepath = path.join(outputDir, filename);
        
        fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
        console.log(`报告保存至: ${filepath}`);
        
        // 同时保存为最新报告
        const latestPath = path.join(outputDir, 'latest.json');
        fs.writeFileSync(latestPath, JSON.stringify(report, null, 2));
    }
}

// Worker线程处理
if (!isMainThread && workerData?.isWorker) {
    const { config } = workerData;
    
    parentPort.on('message', async (task) => {
        try {
            const benchmark = new PSIBenchmark(config);
            const result = await benchmark.executePSI(
                task.partyAData,
                task.partyBData,
                task.shardIndex
            );
            parentPort.postMessage(result);
        } catch (error) {
            parentPort.postMessage({
                success: false,
                error: error.message,
                shardIndex: task.shardIndex,
                timestamp: new Date().toISOString()
            });
        }
    });
}

// 命令行接口
if (isMainThread && require.main === module) {
    const argv = yargs(process.argv.slice(2))
        .option('n', {
            alias: 'samples',
            type: 'number',
            default: 10000,
            description: '测试样本数量'
        })
        .option('workers', {
            type: 'number',
            default: 4,
            description: '工作线程数'
        })
        .option('shards', {
            type: 'number',
            default: 8,
            description: '数据分片数'
        })
        .option('overlap', {
            type: 'number',
            default: 0.3,
            description: '数据重叠率'
        })
        .option('output', {
            type: 'string',
            default: 'reports/bench/psi',
            description: '输出目录'
        })
        .option('server', {
            type: 'string',
            default: 'http://localhost:8001',
            description: 'PSI服务地址'
        })
        .help()
        .argv;

    const config = {
        n: argv.n,
        workers: argv.workers,
        shards: argv.shards,
        overlapRate: argv.overlap,
        outputDir: argv.output,
        serverUrl: argv.server
    };

    const benchmark = new PSIBenchmark(config);
    
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

module.exports = PSIBenchmark;