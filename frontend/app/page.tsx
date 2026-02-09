'use client'

import React, { useState, useRef, useEffect } from 'react'
import ChatInterface from '@/components/ChatInterface'
import SourceCitation from '@/components/SourceCitation'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Globe } from 'lucide-react'

const API_BASE_URL = 'http://localhost:8000'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources: SourceType[]
  timestamp: Date
  usedWebSearch?: boolean
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
      usedWebSearch: useWebSearch,
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
        usedWebSearch: response.data.web_search_used,
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
        usedWebSearch: useWebSearch,
      }

      setMessages((prev) => [...prev, errorMsg])
      setError(errorMessage)
      setSystemStatus('error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col font-sans">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-blue-100 flex-shrink-0 z-10 sticky top-0">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">

              <h1 className="text-xl font-bold text-blue-900 tracking-tight">
                Atlas
              </h1>
            </div>
          </div>
          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <p className="text-sm text-red-700 font-medium">{error}</p>
            </div>
          )}
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto w-full">
        <div className="max-w-3xl mx-auto px-6 py-8 pb-32">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
              <p className="text-blue-600/80 max-w-md leading-relaxed font-medium">
                Ready to analyze material pricing, project specifications, and historical data.
              </p>
            </div>
          ) : (
            <div className="space-y-8">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex w-full ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-2xl ${message.role === 'user'
                      ? 'bg-blue-100 border border-blue-400 text-[#1e3a8a] rounded-2xl rounded-tr-sm px-6 py-4 shadow-md shadow-blue-900/5'
                      : 'w-full pl-0 ' // Assistant message takes full width but no background
                      }`}
                  >
                    {message.role === 'user' ? (
                      <div className="flex flex-col gap-1">
                        <p className="text-[15px] leading-relaxed font-medium">{message.content}</p>
                        {message.usedWebSearch && (
                          <div className="flex items-center gap-1 mt-1 opacity-80 self-end">
                            <Globe size={10} strokeWidth={3} />
                            <span className="text-[9px] font-bold uppercase tracking-widest text-blue-700/70">Search ON</span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-slate-800">
                        {/* Avatar for Assistant */}
                        <div className="flex items-center gap-3 mb-4">
                          <span className="text-sm font-bold text-blue-900">Atlas</span>
                          <span className="text-xs text-blue-400 font-medium">
                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                          {message.usedWebSearch && (
                            <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-50 border border-blue-100 shadow-sm ml-2">
                              <Globe size={12} className="text-blue-600" />
                              <span className="text-[10px] font-bold text-blue-700 uppercase tracking-tight">Search ON</span>
                            </div>
                          )}
                        </div>

                        <div className="prose prose-slate max-w-none prose-p:leading-relaxed prose-headings:font-bold prose-strong:font-bold prose-li:marker:text-blue-600">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-6 mb-3 text-slate-900 border-b border-blue-100 pb-1" {...props} />,
                              h3: ({ node, ...props }) => <h3 className="text-base font-bold mt-4 mb-2 text-slate-800" {...props} />,
                              p: ({ node, ...props }) => <p className="mb-3 text-[15px] text-slate-700 leading-relaxed" {...props} />,
                              ul: ({ node, ...props }) => <ul className="list-disc ml-4 mb-4 space-y-1" {...props} />,
                              li: ({ node, ...props }) => <li className="text-slate-700 pl-1" {...props} />,
                              strong: ({ node, ...props }) => <strong className="font-bold text-slate-900" {...props} />,
                              a: ({ node, ...props }) => <a className="text-blue-600 hover:underline" {...props} />,
                              table: ({ node, ...props }) => (
                                <div className="overflow-x-auto my-6 border border-blue-100 rounded-lg shadow-sm">
                                  <table className="min-w-full divide-y divide-blue-100 bg-white" {...props} />
                                </div>
                              ),
                              thead: ({ node, ...props }) => <thead className="bg-blue-50/50" {...props} />,
                              th: ({ node, ...props }) => <th className="px-4 py-2.5 text-left text-[13px] font-bold text-blue-900 uppercase tracking-wider" {...props} />,
                              td: ({ node, ...props }) => <td className="px-4 py-2.5 text-[14px] text-slate-700 border-t border-blue-50" {...props} />,
                              tbody: ({ node, ...props }) => <tbody className="divide-y divide-blue-50" {...props} />,
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {/* Sources for assistant messages */}
                    {message.role === 'assistant' && message.sources.length > 0 && (
                      <div className="mt-5 pt-4 border-t border-slate-200">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="h-1.5 w-1.5 rounded-full bg-blue-600" />
                          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                            Verified Sources
                          </p>
                        </div>
                        <div className="grid gap-2">
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
            </div>
          )}
        </div>
      </div>

      {/* Input Area - Fixed at bottom of container */}
      <div className="w-full z-20">
        <div className="max-w-3xl mx-auto ">
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
