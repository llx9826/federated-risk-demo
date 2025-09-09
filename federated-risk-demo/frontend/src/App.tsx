import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, App as AntdApp } from 'antd'
import { useAppStore } from '@store/app'

import AppLayout from '@components/Layout/AppLayout'
import Dashboard from '@pages/Dashboard'
import PSIPage from '@pages/PSI'
import ConsentPage from '@pages/Consent'
import TrainingPage from '@pages/Training'
import InferencePage from '@pages/Inference'
import AuditPage from '@pages/Audit'
import SettingsPage from '@pages/Settings'

const { Content } = Layout

function App() {
  const { theme } = useAppStore()

  return (
    <AntdApp>
      <div className="App" data-theme={theme}>
        <AppLayout>
          <Content style={{ margin: 0, minHeight: 'calc(100vh - 64px)' }}>
            <Routes>
              {/* 默认重定向到仪表板 */}
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              
              {/* 主要页面 */}
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/psi" element={<PSIPage />} />
              <Route path="/consent" element={<ConsentPage />} />
              <Route path="/training" element={<TrainingPage />} />
              <Route path="/inference" element={<InferencePage />} />
              <Route path="/audit" element={<AuditPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              
              {/* 404页面 */}
              <Route path="*" element={<div>页面未找到</div>} />
            </Routes>
          </Content>
        </AppLayout>
      </div>
    </AntdApp>
  )
}

export default App