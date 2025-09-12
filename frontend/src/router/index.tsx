import React from 'react'
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom'

// 页面组件
import Dashboard from '@/pages/Dashboard'
import Layout from '@/components/Layout/index'
import Consent from '@/pages/Consent'
import PSI from '@/pages/PSI'
import Training from '@/pages/Training'
import Inference from '@/pages/Inference'
import Audit from '@/pages/Audit'
import Settings from '@/pages/Settings'

// 路由配置
export const router = createBrowserRouter([
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: '/',
    element: <Layout><Outlet /></Layout>,
    children: [
      {
        path: 'dashboard',
        element: <Dashboard />,
      },
      {
        path: 'consent',
        element: <Consent />,
      },
      {
        path: 'psi',
        element: <PSI />,
      },
      {
        path: 'federated',
        element: <Training />,
      },
      {
        path: 'models',
        element: <Inference />,
      },
      {
        path: 'audit',
        element: <Audit />,
      },
      {
        path: 'settings',
        element: <Settings />,
      },
    ],
  },
  {
    path: '*',
    element: (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">404</h1>
          <p className="text-gray-600 mb-8">页面不存在</p>
          <a
            href="/dashboard"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
          >
            返回首页
          </a>
        </div>
      </div>
    ),
  },
])

export default router