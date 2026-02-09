'use client'

import React, { useState, useEffect } from 'react'
import { ChevronRight, ChevronDown, Folder, File, RotateCw } from 'lucide-react'

interface FileNode {
  name: string
  type: 'file' | 'folder'
  path: string
  children?: FileNode[]
}

export default function Sidebar() {
  const [fileStructure, setFileStructure] = useState<FileNode[] | null>(null)
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['materials', 'projectAcme', 'projectFacebook']))
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const fetchFileStructure = async () => {
    try {
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiBaseUrl}/api/file-structure`)
      const data = await response.json()
      setFileStructure(data)
      setLoading(false)
      setRefreshing(false)
    } catch (error) {
      console.error('Failed to fetch file structure:', error)
      setFileStructure(getDefaultStructure())
      setLoading(false)
      setRefreshing(false)
    }
  }

  const handleRefresh = () => {
    setRefreshing(true)
    fetchFileStructure()
  }

  useEffect(() => {
    fetchFileStructure()
  }, [])

  const getDefaultStructure = (): FileNode[] => [
    {
      name: 'materials',
      type: 'folder',
      path: 'materials',
      children: [
        {
          name: 'projectAcme',
          type: 'folder',
          path: 'materials/projectAcme',
          children: [
            {
              name: 'concrete',
              type: 'folder',
              path: 'materials/projectAcme/concrete',
              children: [
                {
                  name: 'concrete_prices.csv',
                  type: 'file',
                  path: 'materials/projectAcme/concrete/concrete_prices.csv',
                },
              ],
            },
            {
              name: 'metal',
              type: 'folder',
              path: 'materials/projectAcme/metal',
              children: [
                {
                  name: 'metal_prices.csv',
                  type: 'file',
                  path: 'materials/projectAcme/metal/metal_prices.csv',
                },
              ],
            },
            {
              name: 'stone',
              type: 'folder',
              path: 'materials/projectAcme/stone',
              children: [
                {
                  name: 'stone_prices.csv',
                  type: 'file',
                  path: 'materials/projectAcme/stone/stone_prices.csv',
                },
              ],
            },
            {
              name: 'wood',
              type: 'folder',
              path: 'materials/projectAcme/wood',
              children: [
                {
                  name: 'wood_prices.csv',
                  type: 'file',
                  path: 'materials/projectAcme/wood/wood_prices.csv',
                },
              ],
            },
          ],
        },
        {
          name: 'projectFacebook',
          type: 'folder',
          path: 'materials/projectFacebook',
          children: [
            {
              name: 'concrete',
              type: 'folder',
              path: 'materials/projectFacebook/concrete',
              children: [
                {
                  name: 'concrete_prices.csv',
                  type: 'file',
                  path: 'materials/projectFacebook/concrete/concrete_prices.csv',
                },
              ],
            },
            {
              name: 'metal',
              type: 'folder',
              path: 'materials/projectFacebook/metal',
              children: [
                {
                  name: 'metal_prices.csv',
                  type: 'file',
                  path: 'materials/projectFacebook/metal/metal_prices.csv',
                },
              ],
            },
            {
              name: 'stone',
              type: 'folder',
              path: 'materials/projectFacebook/stone',
              children: [
                {
                  name: 'stone_prices.csv',
                  type: 'file',
                  path: 'materials/projectFacebook/stone/stone_prices.csv',
                },
              ],
            },
            {
              name: 'wood',
              type: 'folder',
              path: 'materials/projectFacebook/wood',
              children: [
                {
                  name: 'wood_prices.csv',
                  type: 'file',
                  path: 'materials/projectFacebook/wood/wood_prices.csv',
                },
              ],
            },
          ],
        },
      ],
    },
  ]

  const toggleFolder = (path: string) => {
    setExpandedFolders((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(path)) {
        newSet.delete(path)
      } else {
        newSet.add(path)
      }
      return newSet
    })
  }

  const handleFileClick = async (filePath: string, fileName: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/download?path=${encodeURIComponent(filePath)}`)
      
      if (!response.ok) {
        throw new Error('Download failed')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = fileName
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download file:', error)
      alert('Failed to download file')
    }
  }

  const renderNode = (node: FileNode, level: number = 0): React.ReactNode => {
    const isExpanded = expandedFolders.has(node.path)
    const hasChildren = node.children && node.children.length > 0

    return (
      <div key={node.path}>
        <div
          className="flex items-center gap-2 px-3 py-2 hover:bg-gray-100 cursor-pointer group"
          style={{ paddingLeft: `${12 + level * 16}px` }}
          onClick={() => {
            if (node.type === 'folder') {
              toggleFolder(node.path)
            }
          }}
        >
          {node.type === 'folder' ? (
            <>
              <div className="w-4 h-4 flex items-center justify-center">
                {hasChildren ? (
                  isExpanded ? (
                    <ChevronDown size={16} className="text-gray-600" />
                  ) : (
                    <ChevronRight size={16} className="text-gray-600" />
                  )
                ) : null}
              </div>
              <Folder size={16} className="text-blue-500 flex-shrink-0" />
              <span className="text-sm text-gray-700 group-hover:text-gray-900 truncate">
                {node.name}
              </span>
            </>
          ) : (
            <>
              <div className="w-4 h-4" />
              <File size={16} className="text-gray-400 flex-shrink-0" />
              <span 
                className="text-sm text-blue-600 group-hover:text-blue-800 truncate cursor-pointer hover:underline"
                onClick={(e) => {
                  e.stopPropagation()
                  handleFileClick(node.path, node.name)
                }}
              >
                {node.name}
              </span>
            </>
          )}
        </div>
        {node.type === 'folder' && isExpanded && hasChildren && (
          <div>
            {node.children!.map((child) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="w-64 bg-gray-50 border-r border-gray-200 overflow-y-auto flex flex-col">
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Data Structure</h2>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          title="Refresh file structure"
          className="p-1.5 hover:bg-gray-200 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RotateCw 
            size={18} 
            className={`text-gray-600 ${refreshing ? 'animate-spin' : ''}`}
          />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="p-4 text-center text-gray-500">Loading...</div>
        ) : fileStructure ? (
          <div>
            {fileStructure.map((node) => renderNode(node))}
          </div>
        ) : (
          <div className="p-4 text-center text-gray-500">Failed to load structure</div>
        )}
      </div>
    </div>
  )
}
