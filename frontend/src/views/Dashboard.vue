<template>
  <div class="dashboard">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1 class="page-title">{{ $t('dashboard.title') }}</h1>
      <div class="page-actions">
        <el-button @click="refreshData" :loading="loading" type="primary">
          <el-icon><Refresh /></el-icon>
          {{ $t('common.refresh') }}
        </el-button>
      </div>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-grid">
      <div class="stat-card" v-for="stat in statistics" :key="stat.key">
        <div class="stat-icon" :style="{ backgroundColor: stat.color }">
          <el-icon :size="24">
            <component :is="stat.icon" />
          </el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ formatNumber(stat.value) }}</div>
          <div class="stat-label">{{ $t(stat.label) }}</div>
          <div class="stat-change" :class="stat.change >= 0 ? 'positive' : 'negative'">
            <el-icon><component :is="stat.change >= 0 ? 'ArrowUp' : 'ArrowDown'" /></el-icon>
            {{ Math.abs(stat.change) }}%
          </div>
        </div>
      </div>
    </div>

    <!-- 图表区域 -->
    <div class="charts-grid">
      <!-- 系统状态图表 -->
      <div class="chart-card">
        <div class="chart-header">
          <h3>{{ $t('dashboard.systemStatus') }}</h3>
          <el-select v-model="statusTimeRange" size="small" style="width: 120px">
            <el-option label="1小时" value="1h" />
            <el-option label="6小时" value="6h" />
            <el-option label="24小时" value="24h" />
            <el-option label="7天" value="7d" />
          </el-select>
        </div>
        <div class="chart-content" ref="systemStatusChart"></div>
      </div>

      <!-- 性能指标图表 -->
      <div class="chart-card">
        <div class="chart-header">
          <h3>{{ $t('dashboard.performanceMetrics') }}</h3>
          <el-select v-model="performanceTimeRange" size="small" style="width: 120px">
            <el-option label="1小时" value="1h" />
            <el-option label="6小时" value="6h" />
            <el-option label="24小时" value="24h" />
            <el-option label="7天" value="7d" />
          </el-select>
        </div>
        <div class="chart-content" ref="performanceChart"></div>
      </div>
    </div>

    <!-- 活动和告警 -->
    <div class="activity-grid">
      <!-- 最近活动 -->
      <div class="activity-card">
        <div class="activity-header">
          <h3>{{ $t('dashboard.recentActivities') }}</h3>
          <el-link type="primary" @click="$router.push('/audit')">查看全部</el-link>
        </div>
        <div class="activity-list">
          <div class="activity-item" v-for="activity in recentActivities" :key="activity.id">
            <div class="activity-icon" :class="activity.type">
              <el-icon><component :is="activity.icon" /></el-icon>
            </div>
            <div class="activity-content">
              <div class="activity-title">{{ activity.title }}</div>
              <div class="activity-desc">{{ activity.description }}</div>
              <div class="activity-time">{{ formatTime(activity.timestamp) }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 系统告警 -->
      <div class="alert-card">
        <div class="alert-header">
          <h3>{{ $t('dashboard.alerts') }}</h3>
          <el-badge :value="alerts.filter(a => !a.acknowledged).length" class="alert-badge">
            <el-link type="primary" @click="$router.push('/system/alerts')">管理告警</el-link>
          </el-badge>
        </div>
        <div class="alert-list">
          <div class="alert-item" v-for="alert in alerts.slice(0, 5)" :key="alert.id" :class="alert.level">
            <div class="alert-icon">
              <el-icon><component :is="getAlertIcon(alert.level)" /></el-icon>
            </div>
            <div class="alert-content">
              <div class="alert-title">{{ alert.title }}</div>
              <div class="alert-desc">{{ alert.message }}</div>
              <div class="alert-time">{{ formatTime(alert.timestamp) }}</div>
            </div>
            <div class="alert-actions" v-if="!alert.acknowledged">
              <el-button size="small" @click="acknowledgeAlert(alert.id)">确认</el-button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快速操作 -->
    <div class="quick-actions">
      <h3>{{ $t('dashboard.quickActions') }}</h3>
      <div class="action-grid">
        <div class="action-item" v-for="action in quickActions" :key="action.key" @click="handleQuickAction(action)">
          <div class="action-icon" :style="{ backgroundColor: action.color }">
            <el-icon :size="20">
              <component :is="action.icon" />
            </el-icon>
          </div>
          <div class="action-label">{{ $t(action.label) }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, nextTick } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  Refresh, ArrowUp, ArrowDown, User, DataAnalysis, 
  Monitor, Warning, InfoFilled, CircleCheck,
  Plus, Setting, Document, Upload
} from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { formatNumber, formatTime } from '@/utils'

const store = useStore()
const router = useRouter()

// 响应式数据
const loading = ref(false)
const statusTimeRange = ref('24h')
const performanceTimeRange = ref('24h')
const systemStatusChart = ref(null)
const performanceChart = ref(null)

// 图表实例
let statusChartInstance = null
let performanceChartInstance = null

// 统计数据
const statistics = reactive([
  {
    key: 'totalUsers',
    label: 'dashboard.totalUsers',
    value: 1248,
    change: 12.5,
    color: '#409EFF',
    icon: 'User'
  },
  {
    key: 'activeUsers',
    label: 'dashboard.activeUsers',
    value: 892,
    change: 8.2,
    color: '#67C23A',
    icon: 'User'
  },
  {
    key: 'totalTasks',
    label: 'dashboard.totalTasks',
    value: 3456,
    change: -2.1,
    color: '#E6A23C',
    icon: 'DataAnalysis'
  },
  {
    key: 'runningTasks',
    label: 'dashboard.runningTasks',
    value: 127,
    change: 15.3,
    color: '#F56C6C',
    icon: 'Monitor'
  }
])

// 最近活动
const recentActivities = reactive([
  {
    id: 1,
    type: 'success',
    icon: 'CircleCheck',
    title: '联邦训练任务完成',
    description: '任务 FL-2024-001 已成功完成，准确率达到 92.5%',
    timestamp: new Date(Date.now() - 5 * 60 * 1000)
  },
  {
    id: 2,
    type: 'info',
    icon: 'InfoFilled',
    title: 'PSI任务启动',
    description: '隐私求交任务 PSI-2024-015 已开始执行',
    timestamp: new Date(Date.now() - 15 * 60 * 1000)
  },
  {
    id: 3,
    type: 'warning',
    icon: 'Warning',
    title: '同意即将过期',
    description: '用户 user123 的数据使用同意将在3天后过期',
    timestamp: new Date(Date.now() - 30 * 60 * 1000)
  }
])

// 系统告警
const alerts = reactive([
  {
    id: 1,
    level: 'critical',
    title: 'CPU使用率过高',
    message: '训练节点 node-01 CPU使用率达到 95%',
    timestamp: new Date(Date.now() - 10 * 60 * 1000),
    acknowledged: false
  },
  {
    id: 2,
    level: 'warning',
    title: '内存使用率告警',
    message: 'PSI服务内存使用率超过 80%',
    timestamp: new Date(Date.now() - 25 * 60 * 1000),
    acknowledged: false
  }
])

// 快速操作
const quickActions = reactive([
  {
    key: 'createConsent',
    label: 'consent.create',
    icon: 'Plus',
    color: '#409EFF',
    route: '/consent/create'
  },
  {
    key: 'createPSI',
    label: 'psi.create',
    icon: 'Plus',
    color: '#67C23A',
    route: '/psi/create'
  },
  {
    key: 'createTraining',
    label: 'training.create',
    icon: 'Plus',
    color: '#E6A23C',
    route: '/training/create'
  },
  {
    key: 'systemSettings',
    label: 'nav.settings',
    icon: 'Setting',
    color: '#909399',
    route: '/system/settings'
  }
])

// 方法
const refreshData = async () => {
  loading.value = true
  try {
    // 刷新统计数据
    await store.dispatch('system/fetchSystemInfo')
    await store.dispatch('system/fetchSystemStatus')
    
    // 更新图表
    updateCharts()
    
    ElMessage.success('数据刷新成功')
  } catch (error) {
    console.error('Failed to refresh data:', error)
    ElMessage.error('数据刷新失败')
  } finally {
    loading.value = false
  }
}

const getAlertIcon = (level) => {
  const icons = {
    critical: 'Warning',
    warning: 'Warning',
    info: 'InfoFilled'
  }
  return icons[level] || 'InfoFilled'
}

const acknowledgeAlert = async (alertId) => {
  try {
    const alert = alerts.find(a => a.id === alertId)
    if (alert) {
      alert.acknowledged = true
      ElMessage.success('告警已确认')
    }
  } catch (error) {
    console.error('Failed to acknowledge alert:', error)
    ElMessage.error('确认告警失败')
  }
}

const handleQuickAction = (action) => {
  if (action.route) {
    router.push(action.route)
  }
}

// 初始化系统状态图表
const initSystemStatusChart = () => {
  if (!systemStatusChart.value) return
  
  statusChartInstance = echarts.init(systemStatusChart.value)
  
  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: ['CPU使用率', '内存使用率', '磁盘使用率']
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: generateTimeLabels()
    },
    yAxis: {
      type: 'value',
      max: 100,
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [
      {
        name: 'CPU使用率',
        type: 'line',
        data: generateRandomData(),
        smooth: true,
        itemStyle: { color: '#409EFF' }
      },
      {
        name: '内存使用率',
        type: 'line',
        data: generateRandomData(),
        smooth: true,
        itemStyle: { color: '#67C23A' }
      },
      {
        name: '磁盘使用率',
        type: 'line',
        data: generateRandomData(),
        smooth: true,
        itemStyle: { color: '#E6A23C' }
      }
    ]
  }
  
  statusChartInstance.setOption(option)
}

// 初始化性能指标图表
const initPerformanceChart = () => {
  if (!performanceChart.value) return
  
  performanceChartInstance = echarts.init(performanceChart.value)
  
  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: ['请求数', '响应时间', '错误率']
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: generateTimeLabels()
    },
    yAxis: [
      {
        type: 'value',
        name: '请求数',
        position: 'left'
      },
      {
        type: 'value',
        name: '响应时间(ms)',
        position: 'right'
      }
    ],
    series: [
      {
        name: '请求数',
        type: 'bar',
        data: generateRandomData(100, 1000),
        itemStyle: { color: '#409EFF' }
      },
      {
        name: '响应时间',
        type: 'line',
        yAxisIndex: 1,
        data: generateRandomData(50, 500),
        smooth: true,
        itemStyle: { color: '#67C23A' }
      },
      {
        name: '错误率',
        type: 'line',
        data: generateRandomData(0, 10),
        smooth: true,
        itemStyle: { color: '#F56C6C' }
      }
    ]
  }
  
  performanceChartInstance.setOption(option)
}

// 更新图表
const updateCharts = () => {
  if (statusChartInstance) {
    statusChartInstance.setOption({
      xAxis: {
        data: generateTimeLabels()
      },
      series: [
        { data: generateRandomData() },
        { data: generateRandomData() },
        { data: generateRandomData() }
      ]
    })
  }
  
  if (performanceChartInstance) {
    performanceChartInstance.setOption({
      xAxis: {
        data: generateTimeLabels()
      },
      series: [
        { data: generateRandomData(100, 1000) },
        { data: generateRandomData(50, 500) },
        { data: generateRandomData(0, 10) }
      ]
    })
  }
}

// 生成时间标签
const generateTimeLabels = () => {
  const labels = []
  const now = new Date()
  for (let i = 23; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000)
    labels.push(time.getHours().toString().padStart(2, '0') + ':00')
  }
  return labels
}

// 生成随机数据
const generateRandomData = (min = 0, max = 100) => {
  const data = []
  for (let i = 0; i < 24; i++) {
    data.push(Math.floor(Math.random() * (max - min + 1)) + min)
  }
  return data
}

// 窗口大小变化处理
const handleResize = () => {
  if (statusChartInstance) {
    statusChartInstance.resize()
  }
  if (performanceChartInstance) {
    performanceChartInstance.resize()
  }
}

// 生命周期
onMounted(async () => {
  await nextTick()
  initSystemStatusChart()
  initPerformanceChart()
  
  // 监听窗口大小变化
  window.addEventListener('resize', handleResize)
  
  // 初始化数据
  refreshData()
})

onUnmounted(() => {
  // 销毁图表实例
  if (statusChartInstance) {
    statusChartInstance.dispose()
  }
  if (performanceChartInstance) {
    performanceChartInstance.dispose()
  }
  
  // 移除事件监听
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.dashboard {
  padding: 24px;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 24px;
  margin-bottom: 24px;
}

.stat-card {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  line-height: 1;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin: 4px 0;
}

.stat-change {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.stat-change.positive {
  color: #67C23A;
}

.stat-change.negative {
  color: #F56C6C;
}

.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.chart-card {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.chart-content {
  height: 300px;
}

.activity-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.activity-card,
.alert-card {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.activity-header,
.alert-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.activity-header h3,
.alert-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.activity-list,
.alert-list {
  max-height: 300px;
  overflow-y: auto;
}

.activity-item,
.alert-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.activity-item:last-child,
.alert-item:last-child {
  border-bottom: none;
}

.activity-icon,
.alert-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  flex-shrink: 0;
}

.activity-icon.success {
  background-color: #67C23A;
}

.activity-icon.info {
  background-color: #409EFF;
}

.activity-icon.warning {
  background-color: #E6A23C;
}

.alert-item.critical .alert-icon {
  background-color: #F56C6C;
}

.alert-item.warning .alert-icon {
  background-color: #E6A23C;
}

.alert-item.info .alert-icon {
  background-color: #409EFF;
}

.activity-content,
.alert-content {
  flex: 1;
}

.activity-title,
.alert-title {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 4px;
}

.activity-desc,
.alert-desc {
  font-size: 12px;
  color: #606266;
  margin-bottom: 4px;
}

.activity-time,
.alert-time {
  font-size: 12px;
  color: #909399;
}

.alert-actions {
  flex-shrink: 0;
}

.quick-actions {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.quick-actions h3 {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 16px 0;
}

.action-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.action-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
}

.action-item:hover {
  border-color: #409EFF;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}

.action-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.action-label {
  font-size: 12px;
  color: #606266;
  text-align: center;
}

@media (max-width: 768px) {
  .charts-grid,
  .activity-grid {
    grid-template-columns: 1fr;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>