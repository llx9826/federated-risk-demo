#!/usr/bin/env node

/**
 * 事故包导出器 - 联邦风控系统
 * 收集：.env*、训练/服务日志(JSON)、data_profile.json、metrics.json、
 *      ROC/PR/KS图、样例请求+响应、模型hash与大小、PSI对齐统计、seed参数
 * 输出：incidents/<timestamp>-<short-hash>.zip
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');

// 配置
const PROJECT_ROOT = path.resolve(__dirname, '../..');
const INCIDENTS_DIR = path.join(PROJECT_ROOT, 'incidents');
const LOGS_DIR = path.join(PROJECT_ROOT, 'logs');
const DATA_DIR = path.join(PROJECT_ROOT, 'data');
const MODELS_DIR = path.join(PROJECT_ROOT, 'models');

// 确保目录存在
if (!fs.existsSync(INCIDENTS_DIR)) {
    fs.mkdirSync(INCIDENTS_DIR, { recursive: true });
}

// 日志函数
function log(level, message) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level}] ${message}`);
}

function logInfo(message) { log('INFO', message); }
function logWarn(message) { log('WARN', message); }
function logError(message) { log('ERROR', message); }

// 安全地读取文件
function safeReadFile(filePath, encoding = 'utf8') {
    try {
        if (fs.existsSync(filePath)) {
            return fs.readFileSync(filePath, encoding);
        }
    } catch (error) {
        logWarn(`无法读取文件 ${filePath}: ${error.message}`);
    }
    return null;
}

// 安全地复制文件
function safeCopyFile(src, dest) {
    try {
        if (fs.existsSync(src)) {
            const destDir = path.dirname(dest);
            if (!fs.existsSync(destDir)) {
                fs.mkdirSync(destDir, { recursive: true });
            }
            fs.copyFileSync(src, dest);
            return true;
        }
    } catch (error) {
        logWarn(`无法复制文件 ${src} -> ${dest}: ${error.message}`);
    }
    return false;
}

// 获取文件哈希
function getFileHash(filePath) {
    try {
        if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath);
            return crypto.createHash('sha256').update(content).digest('hex').substring(0, 8);
        }
    } catch (error) {
        logWarn(`无法计算文件哈希 ${filePath}: ${error.message}`);
    }
    return 'unknown';
}

// 获取文件大小
function getFileSize(filePath) {
    try {
        if (fs.existsSync(filePath)) {
            const stats = fs.statSync(filePath);
            return stats.size;
        }
    } catch (error) {
        logWarn(`无法获取文件大小 ${filePath}: ${error.message}`);
    }
    return 0;
}

// 收集环境配置
function collectEnvironmentConfig(tempDir) {
    logInfo('收集环境配置...');
    
    const envDir = path.join(tempDir, 'environment');
    fs.mkdirSync(envDir, { recursive: true });
    
    // 收集 .env 文件
    const envFiles = ['.env', '.env.local', '.env.development', '.env.production'];
    envFiles.forEach(envFile => {
        const envPath = path.join(PROJECT_ROOT, envFile);
        const destPath = path.join(envDir, envFile);
        if (safeCopyFile(envPath, destPath)) {
            logInfo(`收集环境文件: ${envFile}`);
        }
    });
    
    // 收集系统信息
    const systemInfo = {
        timestamp: new Date().toISOString(),
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        cwd: process.cwd(),
        env: {
            NODE_ENV: process.env.NODE_ENV,
            PORT: process.env.PORT,
            MAX_CONCURRENT_PSI: process.env.MAX_CONCURRENT_PSI
        }
    };
    
    // 收集 package.json
    const packageJsonPath = path.join(PROJECT_ROOT, 'package.json');
    if (fs.existsSync(packageJsonPath)) {
        try {
            systemInfo.packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
        } catch (error) {
            logWarn(`无法解析 package.json: ${error.message}`);
        }
    }
    
    fs.writeFileSync(
        path.join(envDir, 'system_info.json'),
        JSON.stringify(systemInfo, null, 2)
    );
    
    logInfo('环境配置收集完成');
}

// 收集日志文件
function collectLogs(tempDir) {
    logInfo('收集日志文件...');
    
    const logsDestDir = path.join(tempDir, 'logs');
    fs.mkdirSync(logsDestDir, { recursive: true });
    
    if (!fs.existsSync(LOGS_DIR)) {
        logWarn('日志目录不存在');
        return;
    }
    
    // 收集最近的日志文件
    const logFiles = fs.readdirSync(LOGS_DIR)
        .filter(file => file.endsWith('.log') || file.endsWith('.json'))
        .sort((a, b) => {
            const statA = fs.statSync(path.join(LOGS_DIR, a));
            const statB = fs.statSync(path.join(LOGS_DIR, b));
            return statB.mtime - statA.mtime; // 按修改时间降序
        })
        .slice(0, 10); // 只取最近10个文件
    
    logFiles.forEach(logFile => {
        const srcPath = path.join(LOGS_DIR, logFile);
        const destPath = path.join(logsDestDir, logFile);
        if (safeCopyFile(srcPath, destPath)) {
            logInfo(`收集日志文件: ${logFile}`);
        }
    });
    
    // 收集服务日志（如果有的话）
    const serviceLogPaths = [
        '/tmp/consent-service.log',
        '/tmp/psi-service.log',
        '/tmp/train-service.log',
        '/tmp/serving-service.log'
    ];
    
    serviceLogPaths.forEach(logPath => {
        if (fs.existsSync(logPath)) {
            const fileName = path.basename(logPath);
            const destPath = path.join(logsDestDir, fileName);
            if (safeCopyFile(logPath, destPath)) {
                logInfo(`收集服务日志: ${fileName}`);
            }
        }
    });
    
    logInfo('日志文件收集完成');
}

// 收集数据概况
function collectDataProfiles(tempDir) {
    logInfo('收集数据概况...');
    
    const dataDestDir = path.join(tempDir, 'data_profiles');
    fs.mkdirSync(dataDestDir, { recursive: true });
    
    // 收集数据概况文件
    const profileFiles = [
        'data_profile.json',
        'doctor_data_profile.json',
        'psi_alignment_stats.json'
    ];
    
    profileFiles.forEach(profileFile => {
        const srcPath = path.join(DATA_DIR, profileFile);
        const synthPath = path.join(DATA_DIR, 'synth', profileFile);
        
        // 尝试从多个位置收集
        [srcPath, synthPath].forEach(filePath => {
            if (fs.existsSync(filePath)) {
                const destPath = path.join(dataDestDir, path.basename(filePath));
                if (safeCopyFile(filePath, destPath)) {
                    logInfo(`收集数据概况: ${path.basename(filePath)}`);
                }
            }
        });
    });
    
    // 生成数据统计摘要
    const dataSummary = {
        timestamp: new Date().toISOString(),
        directories: {}
    };
    
    // 统计各数据目录
    const dataDirs = ['synth', 'psi', 'models'];
    dataDirs.forEach(dir => {
        const dirPath = path.join(DATA_DIR, dir);
        if (fs.existsSync(dirPath)) {
            try {
                const files = fs.readdirSync(dirPath);
                dataSummary.directories[dir] = {
                    fileCount: files.length,
                    files: files.map(file => {
                        const filePath = path.join(dirPath, file);
                        const stats = fs.statSync(filePath);
                        return {
                            name: file,
                            size: stats.size,
                            modified: stats.mtime.toISOString()
                        };
                    })
                };
            } catch (error) {
                logWarn(`无法统计目录 ${dir}: ${error.message}`);
            }
        }
    });
    
    fs.writeFileSync(
        path.join(dataDestDir, 'data_summary.json'),
        JSON.stringify(dataSummary, null, 2)
    );
    
    logInfo('数据概况收集完成');
}

// 收集训练指标
function collectTrainingMetrics(tempDir) {
    logInfo('收集训练指标...');
    
    const metricsDestDir = path.join(tempDir, 'metrics');
    fs.mkdirSync(metricsDestDir, { recursive: true });
    
    // 收集指标文件
    const metricsFiles = [
        'metrics.json',
        'training_history.json',
        'evaluation_results.json'
    ];
    
    metricsFiles.forEach(metricsFile => {
        const srcPath = path.join(DATA_DIR, 'models', metricsFile);
        const destPath = path.join(metricsDestDir, metricsFile);
        if (safeCopyFile(srcPath, destPath)) {
            logInfo(`收集训练指标: ${metricsFile}`);
        }
    });
    
    // 收集图表文件
    const chartFiles = ['roc_curve.png', 'pr_curve.png', 'ks_curve.png'];
    chartFiles.forEach(chartFile => {
        const srcPath = path.join(DATA_DIR, 'models', chartFile);
        const destPath = path.join(metricsDestDir, chartFile);
        if (safeCopyFile(srcPath, destPath)) {
            logInfo(`收集图表文件: ${chartFile}`);
        }
    });
    
    logInfo('训练指标收集完成');
}

// 收集模型信息
function collectModelInfo(tempDir) {
    logInfo('收集模型信息...');
    
    const modelsDestDir = path.join(tempDir, 'models');
    fs.mkdirSync(modelsDestDir, { recursive: true });
    
    const modelInfo = {
        timestamp: new Date().toISOString(),
        models: []
    };
    
    // 查找模型文件
    const modelDirs = [DATA_DIR, path.join(DATA_DIR, 'models')];
    
    modelDirs.forEach(modelDir => {
        if (fs.existsSync(modelDir)) {
            try {
                const files = fs.readdirSync(modelDir);
                files.forEach(file => {
                    if (file.endsWith('.pkl') || file.endsWith('.joblib') || file.endsWith('.model')) {
                        const filePath = path.join(modelDir, file);
                        const stats = fs.statSync(filePath);
                        
                        const model = {
                            name: file,
                            path: filePath,
                            size: stats.size,
                            hash: getFileHash(filePath),
                            modified: stats.mtime.toISOString()
                        };
                        
                        modelInfo.models.push(model);
                        
                        // 复制模型文件（如果不太大）
                        if (stats.size < 50 * 1024 * 1024) { // 小于50MB
                            const destPath = path.join(modelsDestDir, file);
                            if (safeCopyFile(filePath, destPath)) {
                                logInfo(`收集模型文件: ${file} (${(stats.size / 1024 / 1024).toFixed(2)}MB)`);
                            }
                        } else {
                            logWarn(`模型文件过大，跳过复制: ${file} (${(stats.size / 1024 / 1024).toFixed(2)}MB)`);
                        }
                    }
                });
            } catch (error) {
                logWarn(`无法扫描模型目录 ${modelDir}: ${error.message}`);
            }
        }
    });
    
    fs.writeFileSync(
        path.join(modelsDestDir, 'model_info.json'),
        JSON.stringify(modelInfo, null, 2)
    );
    
    logInfo('模型信息收集完成');
}

// 收集样例请求和响应
function collectSampleRequests(tempDir) {
    logInfo('收集样例请求和响应...');
    
    const samplesDestDir = path.join(tempDir, 'samples');
    fs.mkdirSync(samplesDestDir, { recursive: true });
    
    // 收集评分结果
    const scoringResultsPath = path.join(DATA_DIR, 'doctor_scoring_results.json');
    if (fs.existsSync(scoringResultsPath)) {
        const destPath = path.join(samplesDestDir, 'scoring_results.json');
        if (safeCopyFile(scoringResultsPath, destPath)) {
            logInfo('收集评分结果');
        }
    }
    
    // 生成样例请求
    const sampleRequests = {
        timestamp: new Date().toISOString(),
        requests: [
            {
                endpoint: '/score',
                method: 'POST',
                payload: {
                    subject: 'test_user_001',
                    features: {
                        a_age: 35,
                        a_income: 50000,
                        a_credit_score: 720,
                        b_purchase_amount: 1200,
                        b_frequency: 5
                    },
                    consent_token: 'test_consent',
                    purpose: 'risk_assessment'
                }
            },
            {
                endpoint: '/psi/sessions',
                method: 'POST',
                payload: {
                    session_id: 'test_session_001',
                    method: 'token_join',
                    parties: ['party_a', 'party_b']
                }
            },
            {
                endpoint: '/train',
                method: 'POST',
                payload: {
                    job_name: 'test_training',
                    algorithm: 'hetero_lr',
                    participants: ['party_a', 'party_b'],
                    config: {
                        learning_rate: 0.1,
                        max_iter: 50,
                        epsilon: 5
                    }
                }
            }
        ]
    };
    
    fs.writeFileSync(
        path.join(samplesDestDir, 'sample_requests.json'),
        JSON.stringify(sampleRequests, null, 2)
    );
    
    logInfo('样例请求收集完成');
}

// 收集PSI对齐统计
function collectPSIStats(tempDir) {
    logInfo('收集PSI对齐统计...');
    
    const psiDestDir = path.join(tempDir, 'psi');
    fs.mkdirSync(psiDestDir, { recursive: true });
    
    // 收集PSI相关文件
    const psiDir = path.join(DATA_DIR, 'psi');
    if (fs.existsSync(psiDir)) {
        try {
            const files = fs.readdirSync(psiDir);
            files.forEach(file => {
                if (file.endsWith('.json') || file.endsWith('.csv')) {
                    const srcPath = path.join(psiDir, file);
                    const destPath = path.join(psiDestDir, file);
                    if (safeCopyFile(srcPath, destPath)) {
                        logInfo(`收集PSI文件: ${file}`);
                    }
                }
            });
        } catch (error) {
            logWarn(`无法扫描PSI目录: ${error.message}`);
        }
    }
    
    logInfo('PSI对齐统计收集完成');
}

// 收集种子参数
function collectSeedParameters(tempDir) {
    logInfo('收集种子参数...');
    
    const seedDestDir = path.join(tempDir, 'seeds');
    fs.mkdirSync(seedDestDir, { recursive: true });
    
    // 收集种子配置
    const seedConfig = {
        timestamp: new Date().toISOString(),
        parameters: {
            data_generation: {
                seed: 42,
                size: 10000,
                overlap_rate: 0.6,
                bad_rate: 0.15
            },
            training: {
                random_state: 42,
                learning_rate: 0.1,
                max_iter: 50
            },
            psi: {
                salt: 'default_salt',
                hash_method: 'sha256'
            }
        }
    };
    
    // 尝试从配置文件读取实际参数
    const configFiles = [
        path.join(PROJECT_ROOT, 'config.json'),
        path.join(PROJECT_ROOT, 'config', 'default.json')
    ];
    
    configFiles.forEach(configFile => {
        if (fs.existsSync(configFile)) {
            try {
                const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
                seedConfig.actualConfig = config;
                logInfo(`收集配置文件: ${path.basename(configFile)}`);
            } catch (error) {
                logWarn(`无法解析配置文件 ${configFile}: ${error.message}`);
            }
        }
    });
    
    fs.writeFileSync(
        path.join(seedDestDir, 'seed_parameters.json'),
        JSON.stringify(seedConfig, null, 2)
    );
    
    logInfo('种子参数收集完成');
}

// 创建事故包
function createIncidentPack() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const shortHash = crypto.randomBytes(4).toString('hex');
    const packName = `${timestamp}-${shortHash}`;
    const tempDir = path.join(INCIDENTS_DIR, `temp_${packName}`);
    const zipPath = path.join(INCIDENTS_DIR, `${packName}.zip`);
    
    logInfo(`创建事故包: ${packName}`);
    
    try {
        // 创建临时目录
        fs.mkdirSync(tempDir, { recursive: true });
        
        // 收集各类信息
        collectEnvironmentConfig(tempDir);
        collectLogs(tempDir);
        collectDataProfiles(tempDir);
        collectTrainingMetrics(tempDir);
        collectModelInfo(tempDir);
        collectSampleRequests(tempDir);
        collectPSIStats(tempDir);
        collectSeedParameters(tempDir);
        
        // 创建事故包元信息
        const metadata = {
            pack_id: packName,
            created_at: new Date().toISOString(),
            version: '1.0.0',
            description: '联邦风控系统事故包',
            collector: 'export_incident_pack.js',
            contents: {
                environment: '环境配置和系统信息',
                logs: '服务日志和错误日志',
                data_profiles: '数据概况和统计信息',
                metrics: '训练指标和评估结果',
                models: '模型文件和元信息',
                samples: '样例请求和响应',
                psi: 'PSI对齐统计',
                seeds: '种子参数和配置'
            }
        };
        
        fs.writeFileSync(
            path.join(tempDir, 'metadata.json'),
            JSON.stringify(metadata, null, 2)
        );
        
        // 创建ZIP文件
        logInfo('压缩事故包...');
        
        try {
            // 使用系统zip命令
            execSync(`cd "${INCIDENTS_DIR}" && zip -r "${packName}.zip" "temp_${packName}"`, {
                stdio: 'pipe'
            });
            
            // 清理临时目录
            execSync(`rm -rf "${tempDir}"`);
            
            const zipStats = fs.statSync(zipPath);
            logInfo(`事故包创建成功: ${zipPath} (${(zipStats.size / 1024 / 1024).toFixed(2)}MB)`);
            
            return zipPath;
            
        } catch (zipError) {
            logError(`压缩失败: ${zipError.message}`);
            
            // 如果zip失败，保留临时目录
            logInfo(`临时文件保留在: ${tempDir}`);
            return tempDir;
        }
        
    } catch (error) {
        logError(`创建事故包失败: ${error.message}`);
        
        // 清理临时目录
        if (fs.existsSync(tempDir)) {
            try {
                execSync(`rm -rf "${tempDir}"`);
            } catch (cleanupError) {
                logWarn(`清理临时目录失败: ${cleanupError.message}`);
            }
        }
        
        throw error;
    }
}

// 主函数
function main() {
    try {
        logInfo('开始导出事故包...');
        
        const packPath = createIncidentPack();
        
        logInfo('事故包导出完成');
        console.log(`\n📦 事故包路径: ${packPath}`);
        console.log('\n使用方法:');
        console.log(`  bash scripts/repro_min.sh "${packPath}"`);
        
    } catch (error) {
        logError(`事故包导出失败: ${error.message}`);
        process.exit(1);
    }
}

// 如果直接运行此脚本
if (require.main === module) {
    main();
}

module.exports = {
    createIncidentPack,
    collectEnvironmentConfig,
    collectLogs,
    collectDataProfiles,
    collectTrainingMetrics,
    collectModelInfo,
    collectSampleRequests,
    collectPSIStats,
    collectSeedParameters
};