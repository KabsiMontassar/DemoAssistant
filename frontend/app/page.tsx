'use client'

import React, { useState, useRef, useEffect } from 'react'
import ChatInterface from '@/components/ChatInterface'
import SourceCitation from '@/components/SourceCitation'
import axios from 'axios'
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
    <div className="h-full flex flex-col relative z-20">
      {/* Header */}
      <header className=" bg-white/40 backdrop-blur-md flex-shrink-0">
        <div className="px-8 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                Atlas
              </h1>
            </div>
          </div>
          {error && (
            <div className="mt-3 p-3 bg-red-50/80 border border-red-200/50 rounded-xl">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto pb-40 px-4">
        <div className="max-w-4xl mx-auto py-10">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center mt-20">
              <h2 className="text-xl font-semibold text-blue-900 mb-2">Welcome to Atlas</h2>
              <p className="text-slate-500 max-w-sm leading-relaxed">
                Your high-performance material pricing assistant. Ask me anything about project specs or material costs.
              </p>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`mb-8 flex ${message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                >
                  <div
                    className={`max-w-2xl rounded-3xl px-6 py-5 ${message.role === 'user'
                      ? 'bg-blue-200 text-white rounded-tr-none shadow-lg shadow-blue-200'
                      : 'bg-slate-50/50 border border-slate-100 rounded-tl-none'
                      }`}
                  >
                    {message.role === 'user' ? (
                      <p className="text-sm leading-relaxed">{message.content}</p>
                    ) : (
                      <div className="text-sm text-slate-800 prose prose-sm max-w-none prose-headings:text-slate-900 prose-strong:text-slate-900">
                        <ReactMarkdown
                          components={{
                            h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-6 mb-3" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="text-base font-semibold mt-4 mb-2" {...props} />,
                            p: ({ node, ...props }) => <p className="mb-4 leading-relaxed" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc ml-4 mb-4 space-y-2" {...props} />,
                            li: ({ node, ...props }) => <li className="text-slate-700" {...props} />,
                            strong: ({ node, ...props }) => <strong className="font-bold text-slate-900" {...props} />,
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    )}

                    {/* Sources for assistant messages */}
                    {message.role === 'assistant' && message.sources.length > 0 && (
                      <div className="mt-6 pt-6 border-t border-slate-200/60">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="h-1 w-1 rounded-full bg-blue-400" />
                          <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                            Verified Sources
                          </p>
                        </div>
                        <div className="space-y-3">
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

      {/* Input Area - Absolute inside the flex container */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white/80 to-transparent pt-10 pb-8 px-8 z-30">
        <div className="max-w-4xl mx-auto">
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
