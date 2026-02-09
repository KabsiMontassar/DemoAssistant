'use client'

import React, { useState, useRef, useEffect } from 'react'
import ChatInterface from '@/components/ChatInterface'
import SourceCitation from '@/components/SourceCitation'
import axios from 'axios'
import { BarChart3 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources: SourceType[]
  timestamp: Date
}

interface SourceType {
  file_path: string
  content_snippet: string
  relevance_score: number
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [useWebSearch, setUseWebSearch] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [systemStatus, setSystemStatus] = useState<'checking' | 'healthy' | 'error'>('checking')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Check system health on mount
  useEffect(() => {
    checkHealth()
    const healthInterval = setInterval(checkHealth, 30000) // Check every 30s
    return () => clearInterval(healthInterval)
  }, [])

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000,
      })
      setSystemStatus(response.data.status === 'healthy' ? 'healthy' : 'error')
      setError(null)
    } catch (err) {
      setSystemStatus('error')
      setError('Backend service unavailable. Please check if the backend is running.')
    }
  }

  const handleSendMessage = async (query: string) => {
    if (!query.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      sources: [],
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setLoading(true)
    setError(null)

    try {
      // Call backend API
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        query: query,
        use_web_search: useWebSearch,
      })

      // Add assistant message with sources
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.response,
        sources: response.data.sources || [],
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
      setSystemStatus('healthy')
    } catch (err) {
      const errorMessage = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Failed to process query'

      // Add error message
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${errorMessage}`,
        sources: [],
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, errorMsg])
      setError(errorMessage)
      setSystemStatus('error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-white relative">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white shadow-sm flex-shrink-0">
        <div className="px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-gray-900">
                Material Pricing AI Assistant
              </h1>
              <span
                className={`h-3 w-3 rounded-full ${
                  systemStatus === 'healthy'
                    ? 'bg-green-500'
                    : systemStatus === 'error'
                    ? 'bg-red-500'
                    : 'bg-yellow-500'
                }`}
                title={`System: ${systemStatus}`}
              />
            </div>
            <div className="flex items-center gap-4">
            </div>
          </div>
          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto bg-gray-50 pb-32">
        <div className="max-w-4xl mx-auto px-4 py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="mb-4">
                <BarChart3 size={64} className="text-blue-500 mx-auto" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Material Pricing Assistant
              </h2>
              <p className="text-gray-600 max-w-md">
                Ask questions about material pricing and project details. The system
                will search through material documents and provide sourced answers.
              </p>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`mb-6 flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-2xl rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-blue-500 text-white rounded-br-none'
                        : 'bg-white border border-gray-200 rounded-bl-none shadow-sm'
                    }`}
                  >
                    {message.role === 'user' ? (
                      <p className="text-sm text-white">{message.content}</p>
                    ) : (
                      <div className="text-sm text-gray-900 prose prose-sm max-w-none">
                        <ReactMarkdown
                          components={{
                            h2: ({node, ...props}) => <h2 className="text-base font-bold mt-3 mb-2 text-gray-900" {...props} />,
                            h3: ({node, ...props}) => <h3 className="text-sm font-semibold mt-2 mb-1 text-gray-800" {...props} />,
                            p: ({node, ...props}) => <p className="text-sm text-gray-700 mb-2" {...props} />,
                            ul: ({node, ...props}) => <ul className="list-disc list-inside text-sm text-gray-700 mb-2 space-y-1" {...props} />,
                            ol: ({node, ...props}) => <ol className="list-decimal list-inside text-sm text-gray-700 mb-2 space-y-1" {...props} />,
                            li: ({node, ...props}) => <li className="text-sm text-gray-700" {...props} />,
                            strong: ({node, ...props}) => <strong className="font-semibold text-gray-900" {...props} />,
                            em: ({node, ...props}) => <em className="italic text-gray-700" {...props} />,
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    )}

                    {/* Sources for assistant messages */}
                    {message.role === 'assistant' && message.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <p className="text-xs font-semibold text-gray-600 mb-2">
                          Sources:
                        </p>
                        <div className="space-y-2">
                          {message.sources.map((source, idx) => (
                            <SourceCitation key={idx} source={source} />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>

      {/* Input Area - Floating */}
      <div className="fixed bottom-10 left-64 right-0">
        <div className="max-w-4xl mx-auto px-4 py-4 w-full">
          <ChatInterface
            onSendMessage={handleSendMessage}
            disabled={loading || systemStatus === 'error'}
            isLoading={loading}
            useWebSearch={useWebSearch}
            onWebSearchToggle={setUseWebSearch}
          />
        </div>
      </div>
    </div>
  )
}
